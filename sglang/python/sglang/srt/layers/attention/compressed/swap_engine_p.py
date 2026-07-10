"""
swap_engine_p.py — Path-P c4 KV offload engine (pure imitation, HiSparse-free)
==============================================================================

Path P implements "offload non-recalled CSA c4 attention-KV to CPU + shrink the
GPU c4 pool" WITHOUT touching any HiSparse class. It is a self-contained,
parallel c4 swap path whose byte-layout has been validated offline against the
real sglang pool/quant/store (swap_infra/stage3b_P_bytelayout_test.py, 0 mismatch).

CORE INVARIANT (validated)
--------------------------
The c4 attention-KV pool buffer is [n_pages, row_bytes] uint8; each ROW packs 64
c4-tokens with an intricate 584-byte/token layout. A token at compressed loc L
lives in page (L // 64) at in-page offset (L % 64). If we copy a WHOLE page-row
verbatim into reserved slot-page S, the token's new loc becomes

        new_loc = S * 64 + (L % 64)

— identical in-page offset → byte-identical to the full-pool baseline. We only
ever remap the PAGE NUMBER; intra-page bytes are never reconstructed.

WHAT THIS ENGINE OWNS (per c4 compress-layer)
---------------------------------------------
  * host_mirror[layer]: CPU (pinned) [n_logical_pages, row_bytes] uint8 — the
    full c4 history, byte-identical to what the GPU store kernel would produce.
  * the SHRUNK GPU c4_kv_pool itself is the reserved region (no extra tensor).
  * page_to_slot: dict[logical_c4_page -> reserved slot-page]  (append-only).

THREE INTERCEPTION POINTS (wired in deepseek_v4 store + indexer read paths)
---------------------------------------------------------------------------
  1. prefill c4 attention-KV store: written to host_mirror (CPU) instead of the
     shrunk GPU pool, using the real _set_k_and_s_torch layout (CPU == GPU triton,
     verified). The indexer-K (132B scoring) buffer stays FULL-size GPU-resident
     and is NOT touched here (it is cheap and scored every cycle).
  2. indexer topk read-remap (every retriever cycle): selected compressed locs →
     reserved slot locs via page_to_slot; newly recalled pages swapped in
     (host->device whole-row copy, append-only).
  3. decode new-token c4 store: each new c4 chunk gets a reserved slot appended
     (the map_last_loc analog) so attention can read it. See KNOWN-RISK below.

GLOBAL vs PER-LAYER cell allocation (2026-06-24)
------------------------------------------------
EAGER path (SGLANG_PATHP_CUDAGRAPH unset): each c4 layer picks its OWN divergent native
top-512 (no Phase-1 mask), so cells are PER-LAYER (chunk_to_cell[layer] / next_cell[layer]
/ free_cells[layer]) — layers never share or evict each other. recall_chunks(layer, locs)
handles one layer.

GLOBAL path (Phase-2, SGLANG_PATHP_CUDAGRAPH=1): the Level-1 Memory Indexer produces ONE
resident_set shared by ALL 21 c4 layers (the mask confines every layer's top-512 to it). So
cell allocation is GLOBAL (_g_chunk_to_cell / _g_next_cell / _g_free_cells): chunk c maps to
the SAME cell in EVERY layer's reserve_buf. recall_resident_global(resident_set) does ONE
torch.unique + ONE D2H .tolist() + ONE lut scatter across all layers + 21 sync-free byte
copies. This REPLACES the old 21x per-layer recall loop (21x unique + 21x D2H sync + 21x
Python alloc, ~9.4ms/cycle serializing the scheduler loop → barrier=16 watchdog kill). The
shared cell keeps lut UNIFORM across layers (lut[0][c]>=0 ⟺ resident in ALL layers), so
reading layer-0's lut for the miss set is exact. Single-machine chunked-prefill also routes
through the global allocator (recall_chunks → _recall_global), so prefill/decode share one
free-list and never collide on cells.

KNOWN-RISK (honest)
-------------------
Point 3 (decode new-token remap) is implemented but lightly exercised on a live
server. Points 1+2 are backed by the offline byte-layout test AND a live
multi-request needle pass: ~194K needle no-crash+FOUND, and 5 distinct sequential
long requests match baseline per-case.

CONCURRENCY VALUE PROVEN (2026-06-21, simplified engine)
--------------------------------------------------------
Iso-memory test at the SAME 131K c4 budget (c4_kv_pool 1536 MiB):
  * baseline (no swap): N=1 works, N=4 -> CUDA illegal memory access, whole
    server aborts (256K logical loc exceeds the 131K physical c4 pool; native
    c4_out_loc=full_loc//4 is hard-bound to the physical pool, no offload).
  * Path-P: N=16 concurrent 256K all succeed, needle 16/16 (4.09M tokens
    in-flight = 31x the physical c4 pool), evictions=2.8M, no reserve-exceed.
So Path-P serves >=16x the concurrency baseline can at the same c4 memory.
KEY: prefill is SERIALIZED (sglang chunked prefill runs ONE long request's
prefill at a time), so the ~43K full-history prefill peak is a ONE-TIME reserve
floor, NOT multiplied by N. Decode requests ride on their ~6K keep-set. The
reserve must be sized SMALL to force LRU eviction (large reserve -> no eviction
-> degenerates to baseline residency). Value config = small reserve + heavy evict.
chunk-granular recall is load-bearing here: in decode the ~4K retriever-selected
chunks scatter across ~954 of 968 history pages, so PAGE-granular would pin ~10x
more reserve than chunk-granular (whole 37440B rows for 1-chunk-per-page hits).

LIVE BUGS FOUND + FIXED (2026-06-20, not catchable by single-request offline test)
----------------------------------------------------------------------------------
  a. recall+remap must run in PREFILL too, not decode-only: chunked prefill
     (chunk 2+) sparse-reads earlier chunks' c4 history, which we offloaded to
     the mirror. Gating recall to decode left the shrunk GPU pool empty during
     prefill -> garbage. (Fix lives in deepseek_v4_backend_radix.forward.)
  b. cross-request stale GPU bytes: `_copied` permanently marked a chunk resident,
     but a later request reusing the same pool loc rewrites the mirror with fresh
     bytes while recall SKIPS re-copying (chunk in _copied) -> stale prev-request
     bytes served. Fix: store_prefill_to_mirror discards touched chunks from
     `_copied` after (re)writing the mirror, forcing recall to re-materialize.
  c. reserve exhaustion under high-concurrency long-ctx: the original append-only
     slot allocator filled the reserve permanently. Fix: LRU eviction (chunk_to_cell
     is an OrderedDict; selected chunks move_to_end; when full, evict the
     least-recently-selected NON-selected chunk, reset its lut to -1, clear it from
     all layers' _copied; bytes stay in the CPU mirror, re-copied if reselected).
     Result: 256K x12 -> 12/12, x16 -> 16/16. The instantaneous protected set
     (~one prefill 43K + decoders' 6K each) stays under a 131K reserve, so eviction
     never has to drop a currently-selected chunk.
"""
from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Dict, Optional

import torch

from sglang.srt.layers.attention.compressed.pathp_profile import prof

C4_PAGE_SIZE = 64  # c4-tokens per page (== model page_size 256 // 4)
_DBG = os.environ.get("SGLANG_SWAP_P_DBG", "0") == "1"


def _is_cuda_graph_capturing() -> bool:
    """True if the current CUDA stream is in graph-capture mode. Used to suppress debug
    host-syncs (.item()/.sum()) in code paths (remap) that run INSIDE the captured graph
    under Phase-2 — those syncs are illegal during capture and abort it."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False

# ── intra-page byte layout of one c4 chunk (validated byte-exact against the real
# _set_k_and_s_torch in swap_infra/stage3b_P_chunk_copy_test.py, 0 mismatch) ──
# A page row packs C4_PAGE_SIZE chunks. Each chunk occupies TWO segments:
#   * SEG1: 576 contiguous bytes at  off*576           (448B fp8 nope + 128B bf16 rope)
#   * scale: 8 bytes at  SCALE_BASE + off*8            (7B scale + 1 pad)
# These let us copy a SINGLE chunk (not a whole 64-chunk page) into an arbitrary
# dense reserve cell, killing the ~24x page-fragmentation of whole-page recall.
_SEG1_BYTES = 448 + 64 * 2          # 576: nope(fp8,448) + rope(bf16,64*2)
_SCALE_STRIDE = 448 // 64 + 1       # 8: scale_dim(7) + pad(1)
_SCALE_BASE = C4_PAGE_SIZE * _SEG1_BYTES   # 36864: scale region starts after all SEG1s


def _row_bytes(qk_nope_head_dim: int = 448, qk_rope_head_dim: int = 64) -> int:
    bpt = (
        qk_nope_head_dim
        + qk_rope_head_dim * 2          # bf16 rope
        + qk_nope_head_dim // 64        # scale dim (7)
        + 1                             # scale_pad
    )
    nonpad = C4_PAGE_SIZE * bpt
    return math.ceil(nonpad / 576) * 576


class SwapEngineP:
    """Per-server c4 offload engine. One CPU mirror per c4 compress-layer.

    Single-request, append-only first step (matches AppendOnlySwapIn). Multi-
    request sharing of the reserved region is a follow-up (see module doc).
    """

    def __init__(
        self,
        c4_kv_pool,                 # the SHRUNK DeepSeekV4SingleKVPool (reserved region)
        n_logical_pages: int,       # full-history page capacity for the mirror
        c4_layer_num: int,
        device: str,
        qk_nope_head_dim: int = 448,
        qk_rope_head_dim: int = 64,
        reserve_pages: int = 0,     # >0: engine owns a SEPARATE reserve buffer (true
                                    # dual-pool); 0: reuse c4_kv_pool as reserve (legacy
                                    # single-pool, only safe single-request).
    ):
        self.pool = c4_kv_pool
        self.device = device
        self.c4_layer_num = c4_layer_num
        self.row_bytes = _row_bytes(qk_nope_head_dim, qk_rope_head_dim)
        self.n_logical_pages = n_logical_pages

        # ── TRUE DUAL-POOL (reserve_pages>0): the engine owns a SEPARATE reserve GPU
        # buffer, physically disjoint from c4_kv_pool. The NIXL transfer lands the
        # full c4 history in c4_kv_pool (the "landing" pool, indexed by prealloc
        # positions we don't control); ingest copies it to the CPU mirror; decode
        # recall copies the keep-set from the mirror into THIS reserve buffer, which
        # attention reads via get_extra_key_buffer. Because landing (c4_kv_pool) and
        # reserve (this buffer) are separate allocations, the two index spaces never
        # collide under concurrency. reserve_buf[layer] is [reserve_pages, row_bytes].
        if reserve_pages > 0:
            self.reserve_buf = []
            for _ in range(c4_layer_num):
                b = torch.zeros((reserve_pages, self.row_bytes),
                                dtype=torch.uint8, device=device)
                self.reserve_buf.append(b)
            self.reserve_pages = reserve_pages
            self._own_reserve = True
        else:
            self.reserve_buf = c4_kv_pool.kv_buffer   # legacy: share landing pool
            self.reserve_pages = c4_kv_pool.kv_buffer[0].shape[0]
            self._own_reserve = False

        # CPU pinned host mirror, one buffer per c4 compress-layer.
        self.host_mirror = []
        for _ in range(c4_layer_num):
            t = torch.zeros((n_logical_pages, self.row_bytes), dtype=torch.uint8)
            if device != "cpu":
                t = t.pin_memory()
            self.host_mirror.append(t)

        # logical-chunk -> reserved CELL, maintained as an LRU (OrderedDict:
        # oldest first, most-recently-selected last). A CELL is one chunk-slot in
        # the reserved region: cell_id in [0, reserve_pages*64), addressed as
        # (cell//64) page-row, (cell%64) in-page offset.
        #
        # PER-LAYER cell space (2026-06-23): chunk_to_cell / chunk_cell_lut / next_cell
        # / full are now PER c4-layer (list indexed by compress_layer_id), matching the
        # already-per-layer reserve_buf and _copied. RATIONALE: the 18 non-target c4
        # layers each pick their OWN native top-512 chunk subset (different per layer),
        # so a SINGLE shared cell pool had to hold the UNION across 21 layers
        # (~21x20480 > capacity) -> thrash: layer B's recall evicted layer A's
        # just-copied chunks, and _evict_one cleared the victim from ALL layers' _copied,
        # forcing re-copy every step (measured: recall_copy = 83% of decode hot path,
        # ~39万 evictions/window). With per-layer cells each layer's selection
        # (~20480 << per-layer cap) fits with ~19x headroom -> ZERO eviction in steady
        # state -> `if not todo: return 0` early-exit finally bites -> recall_copy ~= 0.
        #
        # PAGE -> CHUNK granularity (2026-06-20): whole-page recall put a whole
        # 64-chunk row on GPU even when the retriever selected 1 scattered chunk in
        # it -> measured 24x blow-up (a 256K req's ~2550-chunk keep-set occupied 975
        # pages). Chunk-granular recall copies ONLY the selected chunk's two byte
        # segments into a dense cell, so reserve demand ~= actual keep-set chunks.
        # Evicted cells' bytes stay safe in the CPU mirror, re-copied if reselected.
        self.chunk_to_cell: list = [OrderedDict() for _ in range(c4_layer_num)]
        self.cell_capacity = self.reserve_pages * C4_PAGE_SIZE
        self.next_cell = [0] * c4_layer_num    # per-layer high-water mark
        # per-layer FREE-LIST of recall cells reclaimed when a request's pool slot is reused
        # (release_req_pool_locs). recall allocation prefers free cells before next_cell++,
        # so next_cell does NOT grow unbounded across requests (the cross-request bug:
        # without this, every request's prefill recall bumped next_cell forever → reserve
        # exhausted after a few requests → later requests' chunks not resident → needle MISS).
        self.free_cells: list = [[] for _ in range(c4_layer_num)]
        self.full = [False] * c4_layer_num     # per-layer: reserve filled at least once
        self.n_evictions = 0

        # ── Phase-2 GLOBAL cell allocator (2026-06-24) ──────────────────────────────
        # When the resident-mask capturer drives recall (SGLANG_PATHP_CUDAGRAPH=1), the
        # Level-1 Memory Indexer produces ONE global resident_set shared by all 21 c4
        # layers (Phase-1 mask confines every layer's native top-512 to it). So cell
        # allocation is GLOBAL: chunk c maps to the SAME cell index in EVERY layer's
        # (separate) reserve_buf. This collapses the old 21x-redundant per-layer recall
        # (21x torch.unique + 21x D2H .tolist() sync + 21x Python alloc, ~9.4ms/cycle that
        # serialized the scheduler loop → barrier=16 watchdog kill) into ONE alloc plan +
        # 21 sync-free byte copies. The bytes are still per-layer-correct (each layer copies
        # its own mirror→its own reserve_buf at the shared cell). Eager mode (env off) keeps
        # per-layer cells — there each layer picks its OWN divergent top-512 (no mask), so a
        # shared cell pool would thrash. See module doc + plan.
        self._global_mode = os.environ.get("SGLANG_PATHP_CUDAGRAPH", "0") == "1"
        self._g_chunk_to_cell: OrderedDict = OrderedDict()  # logical chunk -> shared cell (LRU)
        self._g_next_cell = 0                               # global high-water mark
        self._g_free_cells: list = []                       # reclaimed shared cells
        self._g_orphan_miss: dict = {}                      # chunk -> consecutive reclaim-misses (grace)
        # PAGE-RECALL (SGLANG_PATHP_PAGE_RECALL=1, 2026-06-30): when the Level-1 selects WHOLE
        # pages (resident_mask_capturer._select_resident_pages), recall at page-block granularity:
        # each logical page -> one ALIGNED 64-cell reserve block, page-LRU eviction (free 64 cells
        # at once). _g_page_to_block is 64x smaller than _g_chunk_to_cell and eviction is per-page
        # (not per scattered chunk) → kills the chunk-granular eviction storm. The chunk->cell LUT
        # the attention read uses is still filled per-chunk (cell = block*64 + chunk%64), so the
        # remap/attention contract is unchanged. Blocks live in the SAME [0, recall_cell_capacity)
        # cell space (block b spans cells [b*64, b*64+64)); _g_next_block is the page high-water.
        self._page_recall = os.environ.get("SGLANG_PATHP_PAGE_RECALL", "0") == "1"
        self._g_page_to_block: OrderedDict = OrderedDict()  # logical page -> reserve block (LRU)
        self._g_next_block = 0                              # page-block high-water (in blocks)
        self._g_free_blocks: list = []                      # reclaimed reserve blocks
        # Per-layer vectorized LUT for O(1) remap: chunk_cell_lut[layer][logical_chunk]
        # = cell_id, or -1 if not resident. int32 (cell < cell_capacity < 2^31). The
        # cell id IS the flat (page*64 + offset) index the attention kernel reads, so
        # remap returns it directly (no *64). +1 sentinel row for clamp safety.
        self.chunk_cell_lut = torch.full(
            (c4_layer_num, n_logical_pages * C4_PAGE_SIZE + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # NOTE: residency is tracked SOLELY by chunk_cell_lut (>=0 == resident+fresh).
        # The old per-layer `_copied` sets were removed (2026-06-23): recall now derives
        # the miss set from the lut on GPU, and all invalidation paths clear the lut.
        # DEBUG: poison GPU c4 source rows after PD ingest (falsification test).
        self._poison_after_ingest = (
            os.environ.get("SGLANG_SWAP_P_POISON", "0") == "1"
        )

        # ── Phase-2 cuda-graph: DECODE-RESIDENT region ──────────────────────────────
        # decode-generated c4 chunks are AUTOREGRESSIVE: later decode tokens must attend
        # to earlier decode tokens, so a decode chunk must be readable from the step it
        # forms (NOT deferred to the next cycle). Under cuda graph the decode store runs
        # IN the capture path → only no-host-sync GPU ops allowed. So we carve a fixed
        # "decode-resident" sub-region at the TAIL of each layer's reserve cell space, and
        # map each decode chunk to a DETERMINISTIC cell (no next_cell++/Python alloc):
        #     cell = decode_base + req_slot * max_decode_chunks + (c4_out_loc - prefill_chunks[req])
        # In-graph: _set_k_and_s_triton writes the pack into reserve[cell] (reuses the
        # validated GPU kernel → no byte-layout reimpl), and a GPU scatter sets
        # chunk_cell_lut[c4_out_loc] = cell so remap finds it the very next step. The
        # recall LRU (front region) never touches these cells. Sized by env:
        #   SGLANG_PATHP_DECODE_RESIDENT_CHUNKS = max decode c4 chunks per request (default
        #     2048 = 8192 decode tokens) ; concurrency = max_running_requests.
        self._decode_resident_chunks = int(
            os.environ.get("SGLANG_PATHP_DECODE_RESIDENT_CHUNKS", "2048")
        )
        self._decode_max_conc = int(
            os.environ.get("SGLANG_PATHP_DECODE_MAX_CONC", "64")
        )
        _decode_region = self._decode_resident_chunks * self._decode_max_conc
        # decode region sits at the tail; recall LRU is capped to the front so they never
        # collide. cell ids in [decode_base, decode_base+_decode_region) are decode-resident.
        # PLUS one SINK cell at the very tail: invalid decode rows (seq%4!=0 → no new chunk,
        # c4_out_loc=0) are funneled here under cuda graph's FIXED-SHAPE requirement (we
        # store ALL B rows unconditionally — no host-sync .nonzero()/.any() filter — and
        # torch.where invalid rows' dst to this throwaway cell so they corrupt nothing).
        self.decode_base = max(0, self.cell_capacity - _decode_region - 1)
        self.sink_cell = self.cell_capacity - 1
        # recall LRU must stop before decode_base.
        self.recall_cell_capacity = self.decode_base
        # sentinel lut COLUMN (last, the +1 row): invalid rows scatter their lut write here
        # so they never overwrite a real chunk's lut entry. remap never queries this column.
        self.lut_sentinel_col = self.chunk_cell_lut.shape[1] - 1
        # per req_pool_idx -> prefill c4 chunk count (= prefill_len//4), filled at the
        # request's prefill boundary (reset_req_residency). decode chunk local index =
        # c4_out_loc - prefill_chunks[req]. -1 = unknown (req not yet seen at prefill).
        self._prefill_chunks: dict[int, int] = {}
        # per req_pool_idx -> decode slot (0..max_conc-1), assigned round-robin at prefill.
        self._req_decode_slot: dict[int, int] = {}
        self._next_decode_slot = 0
        if _DBG:
            print(f"[P-DECODE-REGION] cell_capacity={self.cell_capacity} "
                  f"decode_base={self.decode_base} decode_region={_decode_region} "
                  f"sink_cell={self.sink_cell} "
                  f"(per_req={self._decode_resident_chunks} x conc={self._decode_max_conc}) "
                  f"recall_cap={self.recall_cell_capacity}", flush=True)

        # ── Path-I (index-K offload) ────────────────────────────────────────────────
        # OPT-IN (attach_index_k_offload). When enabled, the index-K pool (used by the
        # Lightning Indexer for page scoring) is ALSO offloaded: a CPU pinned mirror holds
        # the full-history index-K page rows (opaque INDEX_PAGE_BYTES each), and a SHRUNK
        # GPU reserve holds only the recalled pages — swapped in LOCKSTEP with the c4 page
        # recall (same selected pages, same page->block map _g_page_to_block). The scoring
        # then reads the reserve at reserve-block page ids (indexer.py rewrites
        # resident_id_page_table full-history-phys-page -> reserve block). See Phase I-0
        # byte-equivalence proof (swap_infra/indexk_reserve_equiv.py). Default OFF: these
        # stay None and every index-K path is byte-identical to today.
        self.index_k_offload = False
        self.index_host_mirror = None     # list[clid] CPU pinned [n_logical_pages, INDEX_PAGE_BYTES]; None for full(target) layers
        self.index_reserve_buf = None     # list[clid] GPU [recall_blocks, INDEX_PAGE_BYTES]; None for full(target) layers
        self.index_page_bytes = 0
        self.index_full_layers = set()    # compress-layer-ids kept full-history on GPU (target layers)

    def attach_index_k_offload(self, index_page_bytes: int, full_layers=None):
        """Path-I SELECTIVE: allocate the index-K CPU mirror (full history) + GPU reserve
        (recall region only) for the OFFLOAD layers ONLY. index_page_bytes = the
        DeepSeekV4IndexerPool page row size (e.g. 8448 = 64*128 fp8 + 64*4 scale). The reserve
        holds `recall_cell_capacity//C4_PAGE_SIZE` page-blocks (same block space as the c4 page
        recall, so _g_page_to_block maps 1:1).

        full_layers = set of c4 compress-layer-ids that stay FULL history on GPU (the retriever
        target layers, scored in full every cycle by Level-1). For those layers we allocate NO
        mirror and NO reserve (index_host_mirror[clid] / index_reserve_buf[clid] = None) — the
        indexer scores their full pool directly (byte-unchanged) and the PD transfer leaves them
        in VRAM. Offload layers (the other 18) get a mirror + reserve. Called once at engine
        setup when SGLANG_PATHP_INDEX_K_OFFLOAD=1."""
        self.index_page_bytes = int(index_page_bytes)
        self.index_full_layers = set(full_layers) if full_layers else set()
        recall_blocks = self.recall_cell_capacity // C4_PAGE_SIZE
        # +1 SINK block: the decode store funnels not-yet-placeable rows (page not resident) to
        # the LAST cell. Give the reserve one EXTRA block beyond the recall region so that sink
        # cell can never collide with a real recalled page-block (recall only uses blocks
        # [0, recall_blocks); the sink lives in block recall_blocks). Scoring never addresses the
        # sink block (resident_id_page_table_reserve only holds recall blocks < recall_blocks).
        self._index_recall_blocks = recall_blocks
        self._index_total_blocks = recall_blocks + 1
        # Per compress-layer-id lists; offload layers get a buffer, full (target) layers None.
        self.index_host_mirror = []
        self.index_reserve_buf = []
        _n_off = 0
        for _clid in range(self.c4_layer_num):
            if _clid in self.index_full_layers:
                self.index_host_mirror.append(None)   # full-history layer: no offload
                self.index_reserve_buf.append(None)
                continue
            _n_off += 1
            # Allocate the pinned mirror DIRECTLY (empty(pin_memory=True) + zero_), NOT
            # zeros()->pin_memory() which builds a non-pinned tensor first then COPIES to a
            # pinned one = 2x transient host RAM. At 1M the index mirror is ~36 GB/rank ×8
            # ranks; the 2x transient (~584 GB peak) collided with weights and host-OOM'd the
            # boot. Direct pinned alloc halves the peak. (c4 mirror is 14x smaller so its
            # zeros()->pin pattern is harmless; only index-K needs this.)
            if self.device != "cpu":
                t = torch.empty((self.n_logical_pages, self.index_page_bytes),
                                dtype=torch.uint8, pin_memory=True)
                t.zero_()
            else:
                t = torch.zeros((self.n_logical_pages, self.index_page_bytes), dtype=torch.uint8)
            self.index_host_mirror.append(t)
            b = torch.zeros((self._index_total_blocks, self.index_page_bytes),
                            dtype=torch.uint8, device=self.device)
            self.index_reserve_buf.append(b)
        self.index_k_offload = True
        if _DBG:
            print(f"[Path-I] index-K offload ON (SELECTIVE): offload_layers={_n_off}/"
                  f"{self.c4_layer_num} full(target)_layers={sorted(self.index_full_layers)} "
                  f"mirror [{self.n_logical_pages}, {self.index_page_bytes}] (CPU pinned), "
                  f"reserve [{self._index_total_blocks} blk (={recall_blocks} recall +1 sink), "
                  f"{self.index_page_bytes}B] (GPU) per offload layer", flush=True)

    def index_mirror_buf_info(self, compress_layer_id: int):
        """(data_ptr, nbytes, item_len) of this layer's index-K CPU mirror — the DRAM
        transfer destination for Path-I (parallel to mirror_buf_info for c4). The P->D NIXL
        WRITE lands the full-history index-K straight into this mirror (no GPU landing)."""
        m = self.index_host_mirror[compress_layer_id]
        return m.data_ptr(), m.nbytes, self.index_page_bytes

    def store_index_decode_to_reserve(
        self, compress_layer_id: int, req_pool_indices, c4_out_loc,
        index_k, index_k_scale,
    ) -> None:
        """Path-I decode store (OFFLOAD layers only): write the NEW c4 chunk's index-K into the
        index reserve at the cell the PAGE-BLOCK scoring will read it from.

        UNLIKE c4 classical (whose attention reads chunk-granular via chunk_cell_lut, so its
        decode store targets a separate decode-region cell), the index-K Level-2 scoring reads
        PAGE-BLOCK granular: resident_id_page_table_reserve[bi] holds reserve BLOCK ids, and the
        deep_gemm kernel gathers whole 64-chunk blocks. So a decode chunk at logical loc must
        live at reserve cell = recall_block(page(loc))*64 + loc%64, i.e. INSIDE the recall block
        of its own page — NOT a scattered decode-region cell. That recall block is exactly what
        the shared chunk_cell_lut records for the page's chunk-0 (recall wrote lut[:,page*64+o]=
        block*64+o), so cell = chunk_cell_lut[clid][page*64] + loc%64.

        FIXED-SHAPE, GPU-only (cuda-graph safe): store ALL B rows; rows whose page is not yet
        resident (lut<0), whose seq%4!=0 (loc=0 → no new chunk), or unknown are funneled to the
        sink cell (throwaway). The tail (recency) page is force-resident and never evicted, so a
        decode chunk landing in it has a stable block and its write persists across cycles (the
        recall never re-copies a resident page from the mirror). Uses index_buf_accessor.SetKAndS
        so bytes match the native index-K store exactly. No-op for full (target) layers — their
        index_reserve_buf[clid] is None and they still write the full pool natively."""
        if not self.index_k_offload:
            return
        if (self.index_reserve_buf is None
                or compress_layer_id >= len(self.index_reserve_buf)
                or self.index_reserve_buf[compress_layer_id] is None):
            return  # full (target) layer: not offloaded, keep native full-pool store
        if getattr(self, "chunk_cell_lut", None) is None:
            return
        from sglang.srt.layers.attention.nsa import index_buf_accessor
        ps = C4_PAGE_SIZE
        loc = c4_out_loc.to(self.device, dtype=torch.int64)         # [B] logical chunk loc (0 if none)
        page = loc // ps
        off = loc % ps
        lut0 = self.chunk_cell_lut[0]                               # layer-uniform block map
        page_chunk0 = (page * ps).clamp_(0, lut0.numel() - 1)
        block_cell = lut0[page_chunk0].to(torch.int64)             # recall block's chunk-0 cell (=block*64), -1 if not resident
        real_cell = block_cell + off
        n_res_cells = self.index_reserve_buf[compress_layer_id].shape[0] * ps
        valid = (loc > 0) & (block_cell >= 0) & (real_cell < n_res_cells)
        # FAIL-LOUD: a decode chunk with loc>0 whose page is NOT resident (block_cell<0) can't
        # be placed page-aligned → it would be scored as zeros for THIS offload layer. For
        # realistic needle answers (short gen, all decode chunks in the force-resident last
        # prefill page) this never happens; count it so the agreement test can assert 0 and a
        # long-gen run surfaces it instead of silently diverging. GPU counter, read out-of-graph.
        if getattr(self, "_index_decode_dropped", None) is None:
            self._index_decode_dropped = torch.zeros((), dtype=torch.int64, device=self.device)
        dropped = ((loc > 0) & (block_cell < 0)).sum()
        self._index_decode_dropped += dropped.to(torch.int64)
        cell = torch.where(valid, real_cell,
                           torch.full_like(real_cell, n_res_cells - 1))  # sink = last cell
        cell = cell.clamp(0, n_res_cells - 1).contiguous()
        index_buf_accessor.SetKAndS.execute(
            self.pool, buf=self.index_reserve_buf[compress_layer_id], loc=cell,
            index_k=index_k, index_k_scale=index_k_scale,
        )

    def _copy_index_pages(self, new_pages, new_blocks):
        """Path-I SELECTIVE: copy the given logical PAGES' index-K rows mirror->reserve, for the
        OFFLOAD c4 layers ONLY (full/target layers have index_reserve_buf[clid]=None and are
        skipped — their full history is on GPU already), at page-block granularity (block b ==
        the c4 recall's block for this page). new_pages / new_blocks are Python lists (same ones
        _recall_global_pages just allocated). Whole-page opaque row copy (INDEX_PAGE_BYTES) —
        byte-exact, no split."""
        if not self.index_k_offload or not new_pages:
            return
        # offload layers = those with a real reserve buffer (target layers are None).
        off_clids = [c for c in range(self.c4_layer_num)
                     if self.index_reserve_buf[c] is not None]
        if not off_clids:
            return
        src_pg = torch.tensor(new_pages, dtype=torch.int64)     # [P] logical page ids
        dst_blk = torch.tensor(new_blocks, dtype=torch.int64)   # [P] reserve block ids
        n_mirror = self.n_logical_pages
        n_res = self.index_reserve_buf[off_clids[0]].shape[0]
        ok = (src_pg >= 0) & (src_pg < n_mirror) & (dst_blk >= 0) & (dst_blk < n_res)
        if not bool(ok.all()):
            src_pg = src_pg[ok]
            dst_blk = dst_blk[ok]
            if src_pg.numel() == 0:
                return
        dst_blk_dev = dst_blk.to(self.device)
        # gather offload layers' mirror rows on CPU, batch the H2D across those layers only.
        rows_list = [self.index_host_mirror[clid][src_pg] for clid in off_clids]
        all_rows = torch.stack(rows_list).to(self.device, non_blocking=True)  # [Loff,P,bytes]
        for _i, clid in enumerate(off_clids):
            self.index_reserve_buf[clid][dst_blk_dev] = all_rows[_i]


    def store_prefill_to_mirror(self, compress_layer_id: int, loc: torch.Tensor, pack):
        """Write prefill c4 KV into the CPU mirror at logical loc (NOT GPU pool).

        loc: per-extend-token compressed locs (= full_loc//4). NOTE: sglang sets
        loc=0 for the 3/4 of tokens that are NOT c4-boundaries (seq%4!=0), so loc
        has many duplicate 0s. The native pool tolerates this via last-write-wins
        in _set_k_and_s_torch's Python loop. We replicate that by deduping loc to
        keep the LAST occurrence of each loc (+ its pack row), so the mirror ends
        up byte-identical to the native pool.
        pack: NopeFp8RopeBf16Pack from quant_to_nope_fp8_rope_bf16_pack_triton.
        """
        from sglang.srt.layers.attention.nsa.index_buf_accessor_v4 import (
            NopeFp8RopeBf16Pack,
            _set_k_and_s_torch,
        )

        loc_cpu = loc.detach().to("cpu", dtype=torch.int64).contiguous()
        # dedup keep-last (VECTORIZED): for each distinct loc keep its LAST
        # occurrence row, matching native _set_k_and_s_torch last-write-wins.
        # scatter_reduce amax over the inverse-index groups picks the max (= last)
        # original row per distinct loc; sorting gives the same ascending row set
        # the old Python loop produced (byte-identical mirror, no Python per-row).
        n = loc_cpu.numel()
        _, inverse = torch.unique(loc_cpu, return_inverse=True)
        rows = torch.arange(n, dtype=torch.int64)
        last_pos = torch.full((int(inverse.max()) + 1,), -1, dtype=torch.int64)
        last_pos.scatter_reduce_(0, inverse, rows, reduce="amax", include_self=True)
        keep_rows = torch.sort(last_pos).values
        loc_dedup = loc_cpu[keep_rows].contiguous()
        # Index on the ORIGINAL device tensors (fp8 CPU indexing is unsupported),
        # then move the selected rows to CPU.
        kr_dev = keep_rows.to(pack.k_nope_fp8.device)
        pack_cpu = NopeFp8RopeBf16Pack(
            k_nope_fp8=pack.k_nope_fp8.detach()[kr_dev].to("cpu").contiguous(),
            k_rope_bf16=pack.k_rope_bf16.detach()[kr_dev].to("cpu").contiguous(),
            scale_k_nope_ue8m0=pack.scale_k_nope_ue8m0.detach()[kr_dev].to("cpu").contiguous(),
        )
        _set_k_and_s_torch(
            self.host_mirror[compress_layer_id], loc_dedup, pack_cpu, C4_PAGE_SIZE
        )
        # CROSS-REQUEST CORRECTNESS: the mirror chunk(s) we just (re)wrote now hold
        # fresh bytes. If a previous request had already copied those chunks into a GPU
        # cell, the lut still marks them resident and recall would SKIP re-copying ->
        # stale prev-request bytes. Invalidate the touched chunks in THIS layer's lut
        # (set -1) so the next recall re-materializes fresh bytes. The cell stays owned
        # in chunk_to_cell; recall reuses it and re-sets lut after the byte (re)copy.
        touched = torch.unique(loc_dedup).to(torch.int64)
        touched = touched[(touched >= 0) & (touched < self.chunk_cell_lut.shape[1])]
        if touched.numel():
            self.chunk_cell_lut[compress_layer_id, touched.to(self.device)] = -1

    # ── point 2: every layer's attention (after store) ──────────────────────────
    def recall_chunks(self, compress_layer_id: int, compress_locs: torch.Tensor) -> int:
        """Swap-in for ONE c4 layer (EAGER path) — or, under Phase-2 global mode, the
        ALL-LAYER global core (the compress_layer_id is then only a routing hint; the core
        makes the chunks resident in EVERY layer at a shared cell).

        EAGER (self._global_mode == False): per-layer LRU chunk-granular swap-in. Each c4
        layer has its own LRU + lut + cell high-water (layers never evict each other). The
        miss set (`todo`) is computed on GPU from this layer's lut (resident iff
        lut[layer][c] >= 0), so the steady-state path does NO full-selection D2H / Python loop.

        GLOBAL (Phase-2): delegate to _recall_global — see its docstring. Used by single-
        machine chunked-prefill (called per layer with that layer's divergent top-512); the
        core dedups against the shared _g_chunk_to_cell so each distinct chunk is copied to
        all 21 layers exactly once, and lut stays uniform across layers (the invariant the
        decode side-band's single-D2H miss check relies on)."""
        if self._global_mode:
            return self._recall_global(compress_locs)
        if compress_locs.numel() == 0:
            return 0
        lut = self.chunk_cell_lut[compress_layer_id]
        c2c = self.chunk_to_cell[compress_layer_id]
        # GPU-side: distinct valid chunks, and which of them are NOT yet resident.
        with prof.region("recall_d2h"):
            chunks_t = torch.unique(compress_locs)
            chunks_t = chunks_t[chunks_t >= 0]
            if chunks_t.numel() == 0:
                return 0
            # resident iff this layer's lut entry >= 0; miss = the ones to (re)copy.
            cl = chunks_t.clamp(0, lut.numel() - 1)
            miss_t = chunks_t[lut[cl] < 0]
            if miss_t.numel() == 0:
                return 0
            todo = miss_t.to("cpu").tolist()   # only the (small) miss set crosses to CPU
        if not todo:
            return 0

        # ── VECTORIZED FAST PATH (2026-06-24): the original per-chunk Python loop placed
        # cells one at a time; here, when there is enough high-water headroom to place ALL
        # misses without eviction (the common case), allocate a contiguous cell block with
        # GPU ops and ZERO Python per-chunk loop. Fall back to the LRU Python loop only when
        # eviction is genuinely required (near-capacity).
        nc = self.next_cell[compress_layer_id]
        n_free = len(self.free_cells[compress_layer_id])
        reuse = [c for c in todo if c in c2c]
        new = [c for c in todo if c not in c2c]
        if reuse:
            ru = torch.tensor(reuse, dtype=torch.int64, device=self.device)
            ru_cells = torch.tensor([c2c[c] for c in reuse], dtype=torch.int32, device=self.device)
            lut[ru.clamp(0, lut.numel() - 1)] = ru_cells
        if new:
            n_new = len(new)
            if n_free == 0 and nc + n_new <= self.recall_cell_capacity:
                cells = list(range(nc, nc + n_new))
                self.next_cell[compress_layer_id] = nc + n_new
                new_t = torch.tensor(new, dtype=torch.int64, device=self.device)
                cells_t = torch.tensor(cells, dtype=torch.int32, device=self.device)
                lut[new_t.clamp(0, lut.numel() - 1)] = cells_t
                for c, cell in zip(new, cells):
                    c2c[c] = cell
            else:
                protected = None
                for c in new:
                    if self.free_cells[compress_layer_id]:
                        cell = self.free_cells[compress_layer_id].pop()
                    elif self.next_cell[compress_layer_id] < self.recall_cell_capacity:
                        cell = self.next_cell[compress_layer_id]
                        self.next_cell[compress_layer_id] += 1
                    else:
                        if protected is None:
                            protected = set(chunks_t.to("cpu").tolist())
                        cell = self._evict_one(compress_layer_id, protected)
                        if cell is None:
                            if not self.full[compress_layer_id]:
                                print(f"[P-LRU] L{compress_layer_id} selection exceeds reserve "
                                      f"cells (>{self.cell_capacity}); some masked", flush=True)
                            self.full[compress_layer_id] = True
                            break
                    c2c[c] = cell
                    lut[c] = cell
        # footprint probe: resident chunks (cells) vs the topk selection size.
        self._peak_cells = max(getattr(self, "_peak_cells", 0), len(c2c))
        self._foot_n = getattr(self, "_foot_n", 0) + 1
        if _DBG or (compress_layer_id == 0 and self._foot_n % 20 == 0):
            print(f"[P-FOOT] L0 sel_chunks={int(chunks_t.numel())} "
                  f"resident_cells={len(c2c)} peak_cells={self._peak_cells} "
                  f"cell_cap={self.cell_capacity} evictions={self.n_evictions}",
                  flush=True)

        copy_list = [c for c in todo if c in c2c]
        if not copy_list:
            return 0
        with prof.region("recall_copy"):
            self._copy_chunks_layer(
                compress_layer_id,
                torch.tensor(copy_list, dtype=torch.int64),
                torch.tensor([c2c[c] for c in copy_list], dtype=torch.int64),
            )
        prof.count("recall_chunks", len(copy_list))
        # NOTE: NO per-call torch.cuda.synchronize() — the copies are queued on the default
        # stream; the next forward reads the reserve on the same stream so program order
        # guarantees the writes land first. (The old per-call sync, ×21 layers ×N reqs, was
        # the high-concurrency bottleneck.)
        return len(copy_list)

    def recall_resident_global(self, kept_locs: torch.Tensor) -> int:
        """Phase-2 side-band entry: swap the ONE global resident_set into ALL 21 c4 layers
        in a SINGLE allocation pass. This REPLACES the old `for clid in 21: recall_chunks(...)`
        loop (which redid torch.unique + a D2H .tolist() sync + a Python alloc 21× → ~9.4ms/
        cycle serializing the scheduler loop → barrier=16 watchdog kill). See _recall_global.

        PAGE-RECALL: route to the page-block allocator (whole-page residency → aligned 64-cell
        blocks + page-LRU). kept_locs are the selected pages' chunk locs (contiguous 64-runs)."""
        if self._page_recall:
            return self._recall_global_pages(kept_locs)
        return self._recall_global(kept_locs)

    def _recall_global(self, compress_locs: torch.Tensor) -> int:
        """ALL-LAYER global swap-in core. The Level-1 Memory Indexer produces ONE resident_set
        shared by every c4 layer (Phase-1 mask confines each layer's native top-512 to it), so
        cell allocation is GLOBAL: chunk c maps to the SAME cell in EVERY layer's reserve_buf.

        ONE torch.unique + ONE D2H .tolist() (miss set) + ONE lut scatter across all layers +
        21 sync-free byte copies. Bytes are still per-layer-correct (each layer copies its own
        mirror→its own reserve_buf at the shared cell). The shared cell makes lut UNIFORM across
        layers, so reading layer-0's lut to compute the miss set is exact (lut[0][c]>=0 ⟺ c is
        resident in ALL layers, because this core always writes all layers atomically).
        """
        if compress_locs.numel() == 0:
            return 0
        lut0 = self.chunk_cell_lut[0]
        with prof.region("recall_d2h"):
            u = torch.unique(compress_locs)
            u = u[u >= 0]
            if u.numel() == 0:
                return 0
            miss = u[lut0[u.clamp(0, lut0.numel() - 1)] < 0]
            if miss.numel() == 0:
                return 0
            todo = miss.to("cpu").tolist()        # the ONE D2H sync per cycle (whole batch)
        if not todo:
            return 0

        c2c = self._g_chunk_to_cell
        all_chunks: list = []
        all_cells: list = []
        with prof.region("recall_alloc"):
            # FAST PATH (vectorized, no Python per-chunk loop): the common case once the
            # reserve is sized so eviction never fires. `todo` are all genuinely-new chunks
            # (todo=miss=lut0<0); if none already own a cell (reuse empty) and a contiguous
            # high-water block fits, assign cells = [nc, nc+n) in one shot + bulk dict update.
            # `move_to_end` (LRU touch) is only needed to pick eviction victims — irrelevant
            # when we're not evicting — so we skip it on this path.
            n_todo = len(todo)
            nc = self._g_next_cell
            reuse_any = any(c in c2c for c in todo) if n_todo else False
            if (
                n_todo
                and not reuse_any
                and not self._g_free_cells
                and not getattr(self, "_g_full", False)
                and nc + n_todo <= self.recall_cell_capacity
            ):
                cells_list = list(range(nc, nc + n_todo))
                self._g_next_cell = nc + n_todo
                c2c.update(zip(todo, cells_list))     # bulk dict insert, no per-key Python body
                all_chunks = todo
                all_cells = cells_list
            else:
                # SLOW PATH (Python per-chunk): reuse re-validation + eviction fallback. Rare
                # once sized right; kept correct for the over-capacity / desync cases.
                reuse = [c for c in todo if c in c2c]
                for c in reuse:
                    cell = c2c[c]
                    c2c.move_to_end(c)
                    all_chunks.append(c)
                    all_cells.append(cell)
                new = [c for c in todo if c not in c2c]
                if new:
                    n_new = len(new)
                    nc = self._g_next_cell
                    if not self._g_free_cells and nc + n_new <= self.recall_cell_capacity:
                        # contiguous high-water block, no eviction.
                        self._g_next_cell = nc + n_new
                        for off, c in enumerate(new):
                            cell = nc + off
                            c2c[c] = cell
                            all_chunks.append(c)
                            all_cells.append(cell)
                    else:
                        # First consume free cells + the contiguous high-water block (O(1) each),
                        # then BATCH-evict the remainder in one pass (PERF: the old per-chunk
                        # _evict_one_global did O(dict) scan + 21-layer GPU scatter PER chunk →
                        # multi-minute side-band stall at run=70 → the residual hang).
                        evict_protected = None
                        evict_cells = []          # pre-evicted cells to hand out
                        evict_i = 0
                        for _ci, c in enumerate(new):
                            if self._g_free_cells:
                                cell = self._g_free_cells.pop()
                            elif self._g_next_cell < self.recall_cell_capacity:
                                cell = self._g_next_cell
                                self._g_next_cell += 1
                            else:
                                # at capacity → need eviction. Batch-evict ALL still-needed
                                # victims on the FIRST time we hit this (one dict scan + one
                                # 21-layer scatter), then hand them out one by one.
                                if evict_protected is None:
                                    evict_protected = set(u.to("cpu").tolist())  # resident_set
                                    evict_cells = self._evict_batch_global(
                                        evict_protected, len(new) - _ci
                                    )
                                    evict_i = 0
                                if evict_i < len(evict_cells):
                                    cell = evict_cells[evict_i]
                                    evict_i += 1
                                else:
                                    if not self._g_full:
                                        print(f"[P-LRU-G] global selection exceeds reserve cells "
                                              f"(>{self.recall_cell_capacity}); some masked", flush=True)
                                    self._g_full = True
                                    break
                            c2c[c] = cell
                            all_chunks.append(c)
                            all_cells.append(cell)

        if not all_chunks:
            return 0
        # bounds filter ONCE on CPU (shared across layers; no GPU sync) → reused by every layer.
        chunks_cpu = torch.tensor(all_chunks, dtype=torch.int64)
        cells_cpu = torch.tensor(all_cells, dtype=torch.int64)
        mirror_chunks = self.n_logical_pages * C4_PAGE_SIZE
        reserve_cells = self.reserve_pages * C4_PAGE_SIZE
        safe = (
            (chunks_cpu >= 0) & (chunks_cpu < mirror_chunks)
            & (cells_cpu >= 0) & (cells_cpu < reserve_cells)
        )
        if not bool(safe.all()):   # CPU tensors → .all() is pure CPU, no device sync
            if not getattr(self, "_oob_warned", False):
                self._oob_warned = True
                print(f"[P-COPY-G] skipped {int((~safe).sum())} out-of-range chunks "
                      f"(reserve over-subscribed?); masked not crashed", flush=True)
            chunks_cpu = chunks_cpu[safe]
            cells_cpu = cells_cpu[safe]
        if chunks_cpu.numel() == 0:
            return 0
        # ORDER MATTERS for async recall (Part-2): copy the BYTES first, THEN mark the chunk
        # resident in the lut. A chunk must never be "resident" (lut>=0) before its KV bytes
        # are present, else a concurrent forward could remap→cell→read garbage. With copy-
        # before-lut, the worst race is a forward seeing lut<0 (not-yet-resident) → remap
        # returns -1 → chunk masked-out → GRACEFUL (self-heals next step). Both ops are on the
        # default stream so they're GPU-ordered; this ordering makes the partial state safe.
        chunks_dev = chunks_cpu.to(self.device).clamp_(0, self.chunk_cell_lut.shape[1] - 1)
        # copy bytes for ALL 21 layers — NO inter-layer synchronize (same default stream).
        with prof.region("recall_copy"):
            self._copy_chunks_all_layers(chunks_cpu, cells_cpu)
        # ONE lut scatter across ALL layers (broadcast cells over the layer dim), AFTER bytes.
        self.chunk_cell_lut[:, chunks_dev] = cells_cpu.to(self.device, dtype=torch.int32)
        prof.count("recall_chunks", chunks_cpu.numel() * self.c4_layer_num)
        self._peak_cells = max(getattr(self, "_peak_cells", 0), len(c2c))
        self._foot_n = getattr(self, "_foot_n", 0) + 1
        if _DBG or self._foot_n % 50 == 0:
            print(f"[P-FOOT-G] sel_chunks={int(u.numel())} copied={int(chunks_cpu.numel())} "
                  f"resident_cells={len(c2c)} peak_cells={self._peak_cells} "
                  f"recall_cap={self.recall_cell_capacity} evictions={self.n_evictions}",
                  flush=True)
            prof.dump()   # print accumulated recall_d2h/alloc/copy split (no-op if PROFILE off)
        return int(chunks_cpu.numel())

    def _recall_global_pages(self, kept_locs: torch.Tensor) -> int:
        """PAGE-RECALL all-layer swap-in: the Level-1 selected WHOLE pages, so recall at
        page-block granularity. kept_locs = selected pages' physical chunk locs (contiguous
        64-runs). For each logical page not yet resident, allocate an ALIGNED 64-cell reserve
        block (high-water _g_next_block, or page-LRU evict), copy ALL 64 of its chunks
        (mirror→reserve, reusing the byte-exact _copy_chunks_all_layers), and scatter
        chunk_cell_lut[:, page*64+[0..63]] = block*64 + [0..63] for all layers in ONE op.

        WHY page-block (vs chunk-granular _recall_global): _g_page_to_block is 64x smaller and
        eviction frees a whole 64-cell block per op → no scattered-chunk eviction storm. The
        chunk→cell LUT the attention read uses is unchanged (per-chunk cell = block*64+offset).
        Blocks live in [0, recall_cell_capacity); block b spans cells [b*64, b*64+64)."""
        if kept_locs.numel() == 0:
            return 0
        ps = C4_PAGE_SIZE
        # logical pages of the kept chunks (ONE D2H of the small unique page set).
        with prof.region("recall_d2h"):
            pages_t = torch.unique(kept_locs.to(torch.int64) // ps)
            pages_t = pages_t[pages_t >= 0]
            if pages_t.numel() == 0:
                return 0
            # miss = pages whose page-block isn't allocated yet (check the page's chunk-0 lut).
            lut0 = self.chunk_cell_lut[0]
            chunk0 = (pages_t * ps).clamp(0, lut0.numel() - 1)
            miss = pages_t[lut0[chunk0] < 0]
            todo_pages = miss.to("cpu").tolist()
        if not todo_pages:
            return 0
        p2b = self._g_page_to_block
        max_blocks = self.recall_cell_capacity // ps    # page-blocks that fit in the recall region
        new_pages: list = []
        new_blocks: list = []
        with prof.region("recall_alloc"):
            for pg in todo_pages:
                if pg in p2b:
                    p2b.move_to_end(pg)                 # LRU touch (already resident, shouldn't hit miss)
                    continue
                if self._g_free_blocks:
                    blk = self._g_free_blocks.pop()
                elif self._g_next_block < max_blocks:
                    blk = self._g_next_block
                    self._g_next_block += 1
                else:
                    # page-LRU evict: oldest page not in the CURRENT selection (pages_t).
                    blk = self._evict_one_page(set(pages_t.to("cpu").tolist()))
                    if blk is None:
                        if not self._g_full:
                            print(f"[P-PAGE-LRU] page-blocks exhausted (>{max_blocks}); "
                                  f"some pages masked", flush=True)
                        self._g_full = True
                        break
                p2b[pg] = blk
                new_pages.append(pg)
                new_blocks.append(blk)
        if not new_pages:
            return 0
        # expand each new page -> its 64 (chunk_loc, cell) pairs. chunk = page*64+off,
        # cell = block*64+off. Vectorized: outer pages, inner offsets.
        pg_t = torch.tensor(new_pages, dtype=torch.int64)            # [P]
        blk_t = torch.tensor(new_blocks, dtype=torch.int64)          # [P]
        off = torch.arange(ps, dtype=torch.int64)                    # [64]
        src_chunk = (pg_t.unsqueeze(1) * ps + off.unsqueeze(0)).reshape(-1)   # [P*64]
        dst_cell = (blk_t.unsqueeze(1) * ps + off.unsqueeze(0)).reshape(-1)   # [P*64]
        # copy bytes for ALL 21 layers (byte-exact, shared as_strided gather), THEN set lut
        # (copy-before-lut keeps the partial state safe; see _recall_global rationale).
        with prof.region("recall_copy"):
            self._copy_chunks_all_layers(src_chunk, dst_cell)
        # Path-I: copy the SAME new pages' index-K rows mirror->index-reserve (block b == the
        # c4 block for this page, so scoring reads the reserve at the same block ids). No-op
        # when index-K offload is off. new_pages/new_blocks are this cycle's freshly allocated
        # pages (already computed above); index-K copy at page-block granularity.
        if self.index_k_offload:
            self._copy_index_pages(new_pages, new_blocks)
        chunks_dev = src_chunk.to(self.device).clamp_(0, self.chunk_cell_lut.shape[1] - 1)
        self.chunk_cell_lut[:, chunks_dev] = dst_cell.to(self.device, dtype=torch.int32)
        prof.count("recall_chunks", src_chunk.numel() * self.c4_layer_num)
        self._peak_cells = max(getattr(self, "_peak_cells", 0), len(p2b) * ps)
        self._foot_n = getattr(self, "_foot_n", 0) + 1
        if _DBG or self._foot_n % 50 == 0:
            print(f"[P-FOOT-PG] sel_pages={int(pages_t.numel())} new_pages={len(new_pages)} "
                  f"copied_chunks={int(src_chunk.numel())} resident_pages={len(p2b)} "
                  f"max_blocks={max_blocks} evictions={self.n_evictions}", flush=True)
            prof.dump()
        return int(src_chunk.numel())

    def _evict_one_page(self, protected_pages: set) -> Optional[int]:
        """Page-LRU eviction: free the least-recently-used page-block whose logical page is NOT
        in the current selection. Clears that page's 64 chunks in EVERY layer's lut (orphans the
        bytes in the mirror; re-copied if reselected). Returns the freed block, or None if all
        resident pages are protected."""
        p2b = self._g_page_to_block
        victim_pg = None
        for pg in p2b:                                  # oldest-first (OrderedDict order)
            if pg not in protected_pages:
                victim_pg = pg
                break
        if victim_pg is None:
            return None
        blk = p2b.pop(victim_pg)
        ps = C4_PAGE_SIZE
        # clear the victim page's 64 chunks in every layer's lut (vectorized).
        cols = (victim_pg * ps + torch.arange(ps, device=self.device, dtype=torch.int64))
        cols = cols.clamp(0, self.chunk_cell_lut.shape[1] - 1)
        self.chunk_cell_lut[:, cols] = -1
        self.n_evictions += 1
        return blk

    def _copy_chunks_layer(
        self, compress_layer_id: int, src_chunk: torch.Tensor, dst_cell: torch.Tensor
    ) -> None:
        """Vectorized two-segment byte copy for ONE layer: mirror[src_chunk] -> reserve[dst_cell].
        SEG1 (576B nope+rope contiguous) + scale (8B at page tail), both addressed as flat
        uint8 offsets (validated byte-exact offline). src_chunk / dst_cell are CPU int64 tensors
        (same length); callers pass bounds-safe tensors (the global path filters once for all
        layers; the eager path filters here)."""
        mirror = self.host_mirror[compress_layer_id].view(-1)      # CPU uint8 flat
        buf = self.reserve_buf[compress_layer_id].view(-1)         # GPU reserve flat
        rb = self.row_bytes
        # bounds guard (CPU comparisons → no device sync). Skip OOB rows instead of crashing.
        mirror_chunks = self.n_logical_pages * C4_PAGE_SIZE
        reserve_cells = self.reserve_pages * C4_PAGE_SIZE
        ok = (
            (src_chunk >= 0) & (src_chunk < mirror_chunks)
            & (dst_cell >= 0) & (dst_cell < reserve_cells)
        )
        if not bool(ok.all()):
            if not getattr(self, "_oob_warned", False):
                self._oob_warned = True
                print(f"[P-COPY] L{compress_layer_id} skipped {int((~ok).sum())} "
                      f"out-of-range chunks (reserve over-subscribed?); masked not crashed",
                      flush=True)
            src_chunk = src_chunk[ok]
            dst_cell = dst_cell[ok]
        if src_chunk.numel() == 0:
            return
        src_pg, src_off = src_chunk // C4_PAGE_SIZE, src_chunk % C4_PAGE_SIZE
        dst_pg, dst_off = dst_cell // C4_PAGE_SIZE, dst_cell % C4_PAGE_SIZE
        ar1 = torch.arange(_SEG1_BYTES)
        ars = torch.arange(_SCALE_STRIDE)
        # SEG1 (nope+rope): [base + off*576 : +576]
        s1_src = (src_pg * rb + src_off * _SEG1_BYTES).unsqueeze(1) + ar1
        s1_dst = (dst_pg * rb + dst_off * _SEG1_BYTES).unsqueeze(1) + ar1
        # scale: [base + SCALE_BASE + off*8 : +8]
        sc_src = (src_pg * rb + _SCALE_BASE + src_off * _SCALE_STRIDE).unsqueeze(1) + ars
        sc_dst = (dst_pg * rb + _SCALE_BASE + dst_off * _SCALE_STRIDE).unsqueeze(1) + ars
        # gather from CPU mirror, scatter to GPU reserve (one H2D move per segment).
        seg1 = mirror[s1_src.reshape(-1)].to(self.device, non_blocking=True)
        buf[s1_dst.reshape(-1).to(self.device)] = seg1
        scl = mirror[sc_src.reshape(-1)].to(self.device, non_blocking=True)
        buf[sc_dst.reshape(-1).to(self.device)] = scl

    def _copy_chunks_all_layers(
        self, src_chunk: torch.Tensor, dst_cell: torch.Tensor
    ) -> None:
        """ROW-granular all-layer copy: mirror[src_chunk] -> reserve[dst_cell] for every
        c4 layer, byte-identical to 21x `_copy_chunks_layer` (validated offline in
        swap_infra/test_asstride_gather.py) but ~700x cheaper.

        WHY this exists (measured 2026-06-28): the global cell allocation makes the
        chunk->cell map IDENTICAL across all 21 layers, yet the old per-layer loop rebuilt
        the index tensors 21x AND expanded each chunk into 576 individual byte indices
        (`start.unsqueeze(1)+arange(576)`) -> 23 MB of int64 indices to move 3 MB of data
        (8:1 index:data, 21x redundant). recall measured 207ms/call = 14 MB/s effective
        (700x below PCIe) -> the cost was pure CPU index-build + launch overhead, NOT
        bandwidth. Fix: build page/offset indices ONCE (shared across layers) and gather
        whole 576B/8B ROWS via as_strided (N indices, not N*576). The 2-D advanced index
        `view[pg, off]` copies each contiguous row in one shot."""
        if src_chunk.numel() == 0:
            return
        rb = self.row_bytes
        mirror_chunks = self.n_logical_pages * C4_PAGE_SIZE
        reserve_cells = self.reserve_pages * C4_PAGE_SIZE
        ok = (
            (src_chunk >= 0) & (src_chunk < mirror_chunks)
            & (dst_cell >= 0) & (dst_cell < reserve_cells)
        )
        if not bool(ok.all()):
            if not getattr(self, "_oob_warned", False):
                self._oob_warned = True
                print(f"[P-COPY-ALL] skipped {int((~ok).sum())} out-of-range chunks "
                      f"(reserve over-subscribed?); masked not crashed", flush=True)
            src_chunk = src_chunk[ok]
            dst_cell = dst_cell[ok]
            if src_chunk.numel() == 0:
                return
        # page/in-page offset — built ONCE, reused by every layer (cell map is global).
        src_pg = (src_chunk // C4_PAGE_SIZE)
        src_off = (src_chunk % C4_PAGE_SIZE)
        dst_pg = (dst_cell // C4_PAGE_SIZE).to(self.device)
        dst_off = (dst_cell % C4_PAGE_SIZE).to(self.device)
        rp = self.reserve_pages
        # Batch the H2D: gather all 21 layers' rows on CPU, stack, do ONE .to(device) per
        # segment (was 21 separate H2D + 21 kernel launches/segment = the dominant recall GPU
        # cost competing with decode). Then scatter each layer's slice on GPU. src_pg/src_off
        # are CPU; the CPU advanced-index gather is unavoidable (mirror is the offload target),
        # but batching the transfer cuts launch/sync overhead ~21x.
        N = src_chunk.numel()
        s1_list = []
        sc_list = []
        for clid in range(self.c4_layer_num):
            mirror = self.host_mirror[clid]                    # CPU [n_logical_pages, rb]
            np_ = mirror.shape[0]
            m_s1 = mirror.as_strided((np_, C4_PAGE_SIZE, _SEG1_BYTES), (rb, _SEG1_BYTES, 1), 0)
            s1_list.append(m_s1[src_pg, src_off])              # CPU [N,576]
            m_sc = mirror.as_strided((np_, C4_PAGE_SIZE, _SCALE_STRIDE), (rb, _SCALE_STRIDE, 1), _SCALE_BASE)
            sc_list.append(m_sc[src_pg, src_off])              # CPU [N,8]
        # ONE H2D per segment for all layers (stacked [21, N, *]).
        all_s1 = torch.stack(s1_list).to(self.device, non_blocking=True)   # [L,N,576]
        all_sc = torch.stack(sc_list).to(self.device, non_blocking=True)   # [L,N,8]
        for clid in range(self.c4_layer_num):
            buf = self.reserve_buf[clid]                       # GPU [reserve_pages, rb]
            b_s1 = buf.as_strided((rp, C4_PAGE_SIZE, _SEG1_BYTES), (rb, _SEG1_BYTES, 1), 0)
            b_s1[dst_pg, dst_off] = all_s1[clid]
            b_sc = buf.as_strided((rp, C4_PAGE_SIZE, _SCALE_STRIDE), (rb, _SCALE_STRIDE, 1), _SCALE_BASE)
            b_sc[dst_pg, dst_off] = all_sc[clid]

    def _evict_one_global(self, protected: set) -> Optional[int]:
        """Phase-2 global LRU eviction: evict the least-recently-used GLOBAL chunk NOT in the
        current resident_set, freeing its shared cell (cleared in EVERY layer's lut). Bytes
        stay in the CPU mirror, re-copied if reselected. Returns the freed cell, or None if
        every mapped chunk is protected."""
        c2c = self._g_chunk_to_cell
        victim = None
        for c in c2c:                              # oldest-first
            if c not in protected:
                victim = c
                break
        if victim is None:
            return None
        cell = c2c.pop(victim)
        self.chunk_cell_lut[:, victim] = -1        # orphan bytes in every layer
        self.n_evictions += 1
        return cell

    def _evict_batch_global(self, protected: set, n_needed: int) -> list:
        """Batched LRU eviction (PERF FIX 2026-06-29): evict up to n_needed least-recently-used
        GLOBAL chunks NOT in protected, in ONE pass + ONE 21-layer lut scatter. The per-chunk
        _evict_one_global did a full O(dict) scan AND a separate 21-layer GPU scatter PER victim
        — at run=70 with a churned c2c this serialized into a multi-minute side-band stall (the
        residual run=70 hang: side-band stuck in _evict_one_global → main prev.join() forever).
        Same victims (oldest-first, skip protected), same freed cells — just collected once and
        the lut cleared in a single vectorized scatter. Returns the freed cells (may be < n_needed
        if the rest are all protected)."""
        c2c = self._g_chunk_to_cell
        victims = []
        cells = []
        for c in c2c:                              # oldest-first (OrderedDict insertion order)
            if c not in protected:
                victims.append(c)
                if len(victims) >= n_needed:
                    break
        if not victims:
            return []
        for c in victims:
            cells.append(c2c.pop(c))
        self.n_evictions += len(victims)
        # ONE scatter clears all victims' lut entries across every layer (vs 21 per victim).
        vidx = torch.tensor(victims, dtype=torch.long, device=self.device)
        vidx = vidx[(vidx >= 0) & (vidx < self.chunk_cell_lut.shape[1])]
        if vidx.numel() > 0:
            self.chunk_cell_lut[:, vidx] = -1
        return cells

    def _evict_one(self, compress_layer_id: int, protected: set) -> Optional[int]:
        """Evict the least-recently-inserted mapped CHUNK that is NOT protected, in
        THIS LAYER's cell space, freeing its reserved cell for reuse. Returns the freed
        cell id, or None if every mapped chunk is protected (selection >= reserve cells).

        Per-layer (2026-06-23): only touches this layer's chunk_to_cell + lut, so layers
        never evict each other. In steady state (selection << per-layer cap) this is
        never reached — eviction is a safety fallback for the rare over-cap layer."""
        c2c = self.chunk_to_cell[compress_layer_id]
        lut = self.chunk_cell_lut[compress_layer_id]
        victim = None
        for c in c2c:                          # oldest-first iteration
            if c not in protected:
                victim = c
                break
        if victim is None:
            return None
        cell = c2c.pop(victim)
        lut[victim] = -1                       # bytes orphaned -> not resident
        self.n_evictions += 1
        return cell


    def remap_compressed_locs(
        self, compress_layer_id: int, compress_locs: torch.Tensor
    ) -> torch.Tensor:
        """Logical compressed locs (chunk ids) -> THIS LAYER's reserved-region flat cell
        indices. Fully vectorized: gather cell from chunk_cell_lut[layer]. The cell id IS
        the flat (slot_page*64 + offset) index the attention kernel reads, so it is
        returned directly. Non-resident / negative -> -1 (read path treats <0 as
        masked-out, like the topk padding sentinel)."""
        lut = self.chunk_cell_lut[compress_layer_id]
        locs = compress_locs.to(self.device, dtype=torch.int64)
        valid = locs >= 0
        idx = torch.where(valid, locs, torch.zeros_like(locs))
        idx = idx.clamp_(0, lut.numel() - 1)
        cells = lut[idx].to(torch.int64)                        # -1 where not resident
        resident = valid & (cells >= 0)
        # DBG block does host syncs (.sum()/.item()) — ILLEGAL inside cuda graph capture.
        # Phase-2 runs remap IN the captured decode forward, so guard the debug print to
        # only fire eager (capture-mode off). _maybe_capturing() is cheap + capture-safe.
        if _DBG and not _is_cuda_graph_capturing():
            n_sel = int(valid.sum())
            n_res = int(resident.sum())
            if n_sel:
                print(f"[P-DBG remap] L{compress_layer_id} selected={n_sel} "
                      f"resident={n_res} masked={n_sel - n_res} "
                      f"next_cell={self.next_cell[compress_layer_id]} "
                      f"full={self.full[compress_layer_id]}", flush=True)
        out = torch.where(resident, cells, torch.full_like(locs, -1))
        return out.to(compress_locs.device, dtype=compress_locs.dtype)

    # ── point 3: decode new-token c4 chunk -> reserved cell (map_last_loc analog) ─
    def store_decode(self, compress_layer_id: int, out_loc: torch.Tensor, pack) -> None:
        """Store decode-step new c4 chunk(s): write the authoritative bytes into the
        CPU mirror, then materialize each touched chunk into a reserved cell + copy
        its bytes onto the GPU so attention reads it immediately. Reuses the
        validated chunk-granular path (recall_chunks) for cell alloc + byte copy.

        out_loc: logical compressed chunk locs written this decode step (device).
        """
        if out_loc.numel() == 0:
            return
        # 1. authoritative copy in the mirror at the logical loc (also invalidates
        #    these chunks in this layer's lut so the recall below re-copies fresh bytes).
        self.store_prefill_to_mirror(compress_layer_id, out_loc, pack)
        # 2. materialize the freshly-written chunks into GPU cells (alloc + byte
        #    copy + mark resident), identical to a recall of just these chunks.
        self.recall_chunks(compress_layer_id, out_loc)

    # ── Phase-2 cuda-graph: decode chunk -> DETERMINISTIC decode-resident cell ──────
    # prefill_chunks / decode_slot are kept as GPU tensors indexed by req_pool_idx so the
    # in-graph decode store is FULLY VECTORIZED (no .item()/dict lookup = no host sync).
    def _ensure_decode_tensors(self, max_req_pool: int):
        if getattr(self, "_decode_t_ready", False):
            return
        self._prefill_chunks_t = torch.full(
            (max_req_pool,), -1, dtype=torch.int64, device=self.device
        )
        self._decode_slot_t = torch.full(
            (max_req_pool,), -1, dtype=torch.int64, device=self.device
        )
        self._max_req_pool = max_req_pool
        self._decode_t_ready = True

    def register_decode_req(self, req_pool_idx: int, prefill_chunks: int,
                            max_req_pool: int = 4096) -> None:
        """Called at a request's PREFILL boundary (out of graph, host sync OK). Record its
        prefill c4 chunk count + assign a round-robin decode slot into GPU tensors, so the
        in-graph decode store reads them by gather (no host sync). max_req_pool is a fixed
        upper bound on req_pool_idx (req_to_token pool size, ~hundreds) — sized generously
        so the tensors are allocated once and no real req_pool_idx is ever clamped."""
        self._ensure_decode_tensors(max_req_pool)
        ri = int(req_pool_idx)
        if ri < 0 or ri >= self._max_req_pool:
            return
        self._prefill_chunks_t[ri] = int(prefill_chunks)
        # assign a slot only once per (re)used pool idx; round-robin host-side counter.
        if int(self._decode_slot_t[ri].item()) < 0:
            self._decode_slot_t[ri] = self._next_decode_slot % self._decode_max_conc
            self._next_decode_slot += 1

    @torch.no_grad()
    def store_decode_to_reserve(
        self, compress_layer_id: int, req_pool_indices: torch.Tensor,
        c4_out_loc: torch.Tensor, pack
    ) -> None:
        """IN-GRAPH-SAFE decode store (FIXED-SHAPE, NO host sync): unconditionally store ALL
        B rows every step (cuda graph requires fixed shape — no .nonzero()/.any() filter).
        Rows that did NOT form a new c4 chunk this step (seq%4!=0 → c4_out_loc=0, or unknown
        req, or span overflow) are funneled by torch.where to a throwaway SINK cell + a
        SENTINEL lut column, so they corrupt no real data. Valid rows go to their
        DETERMINISTIC decode-resident cell (cell = decode_base + slot*max_decode + local) via
        the validated triton _set_k_and_s kernel (reused → no byte reimpl), and set
        chunk_cell_lut[c4_out_loc]=cell so remap finds them from the NEXT step (autoregressive
        readability). All gather/scatter/triton on fixed-address buffers → capturable.

        req_pool_indices: [B] int. c4_out_loc: [B] int (0 where no new chunk). pack:
        NopeFp8RopeBf16Pack with B rows."""
        from sglang.srt.layers.attention.nsa.index_buf_accessor_v4 import SetKAndS

        if not getattr(self, "_decode_t_ready", False):
            return
        ri = req_pool_indices.to(self.device, dtype=torch.int64)
        loc = c4_out_loc.to(self.device, dtype=torch.int64)
        ri_c = ri.clamp(0, self._max_req_pool - 1)
        pc = self._prefill_chunks_t[ri_c]            # [B] prefill chunk count, -1 if unknown
        slot = self._decode_slot_t[ri_c]             # [B] decode slot, -1 if unknown
        local = loc - pc                             # [B] local decode chunk index
        real_cell = self.decode_base + slot * self._decode_resident_chunks + local
        # valid: real new chunk this step (loc>0), known req (pc>=0, slot>=0), within span.
        valid = (loc > 0) & (pc >= 0) & (slot >= 0) & (local >= 0) \
            & (local < self._decode_resident_chunks)
        # FIXED-SHAPE funnel: invalid rows -> sink cell (throwaway) so all B rows store
        # unconditionally with no host-sync filter. Valid rows -> their deterministic cell.
        cell = torch.where(valid, real_cell, torch.full_like(real_cell, self.sink_cell))
        cell = cell.clamp(0, self.cell_capacity - 1).contiguous()
        SetKAndS.execute(
            pool=self.pool,
            buf=self.reserve_buf[compress_layer_id],
            loc=cell,
            nope_fp8_rope_bf16_pack=pack,
        )
        # lut scatter: valid rows write lut[c4_out_loc]=cell; invalid rows write the SENTINEL
        # column (never queried by remap) so they overwrite no real chunk's mapping.
        lut_col = torch.where(
            valid, loc.clamp(0, self.lut_sentinel_col),
            torch.full_like(loc, self.lut_sentinel_col),
        )
        self.chunk_cell_lut[compress_layer_id, lut_col] = cell.to(torch.int32)


    # ── PD point: ingest P->D transferred c4 history into the CPU mirror ─────────
    def page_size_chunks(self) -> int:
        """c4 chunks per page (== C4_PAGE_SIZE). Exposed for the PD ingest hook so
        callers need not import the module constant."""
        return C4_PAGE_SIZE

    def get_reserve_buffer(self, compress_layer_id: int):
        """The GPU buffer attention must read for c4 when Path-P is active. In true
        dual-pool mode this is the engine's OWN reserve (disjoint from c4_kv_pool);
        recall_chunks writes keep-set cells here, remap returns cell indices into it.
        In legacy single-pool mode it is c4_kv_pool's buffer (shared)."""
        return self.reserve_buf[compress_layer_id]

    def mirror_buf_info(self, compress_layer_id: int):
        """(data_ptr, nbytes, item_len) of this layer's CPU pinned mirror. Used by
        Path-P-B (NIXL c4 direct-to-mirror): the D-side disaggregation registers the
        mirror as the c4 transfer DRAM destination, so P's NIXL WRITE lands the c4
        history straight into the host mirror (no GPU landing). The transfer indexes
        it by decode_index = token_pos//256 == mirror c4-page index (row_bytes each)."""
        m = self.host_mirror[compress_layer_id]
        return m.data_ptr(), m.nbytes, self.row_bytes

    def discard_copied_pages(self, logical_pages: torch.Tensor) -> int:
        """Path-P-B: after NIXL writes fresh c4 history directly into the mirror for a
        request, invalidate those pages' chunks in EVERY layer's lut (set -1) so the
        next recall re-materializes them into the reserve (no GPU->mirror copy needed;
        the transfer already populated the mirror). Returns #pages touched."""
        pages = torch.unique(logical_pages.detach().to("cpu", dtype=torch.int64))
        pages = pages[(pages >= 0) & (pages < self.n_logical_pages)]
        if pages.numel() == 0:
            return 0
        # chunks = pages*64 + [0..63], vectorized; clear all layers' lut at those cols.
        base = (pages * C4_PAGE_SIZE).view(-1, 1)
        chunks = (base + torch.arange(C4_PAGE_SIZE)).reshape(-1)
        chunks = chunks[chunks < self.chunk_cell_lut.shape[1]].to(self.device)
        self.chunk_cell_lut[:, chunks] = -1
        return int(pages.numel())

    def release_req_pool_locs(self, pool_locs: torch.Tensor) -> int:
        """Reclaim recall cells when a request's pool slot is (re)used by a new request
        (called at the prefill boundary from reset_req_residency). The chunk_to_cell map +
        next_cell high-water are keyed by GLOBAL pool_loc and were NEVER released when a
        request finished — so next_cell grew unbounded across requests until the reserve was
        exhausted and later requests' chunks could not become resident (→ needle MISS). Here
        we pop each given pool_loc's cell out of EVERY layer's chunk_to_cell and return it to
        that layer's free_cells list (recall reuses free cells before bumping next_cell), and
        clear the lut. This bounds reserve occupancy to the live (currently-prefilling +
        decoding) requests, matching the design.

        pool_locs: the c4 chunk locs (= the new request's pool slot range) to free. Out of
        graph (prefill boundary) → host syncs OK. Idempotent (already-absent locs skipped).
        Iterates over the (small) RESIDENT set ∩ released locs, not the full pool_locs, so
        it is cheap even when pool_locs spans the whole 64K-chunk history."""
        locs = torch.unique(pool_locs.detach().to("cpu", dtype=torch.int64))
        locs = locs[(locs >= 0) & (locs < self.chunk_cell_lut.shape[1])]
        if locs.numel() == 0:
            return 0
        loc_set = set(locs.tolist())
        n_freed = 0
        if self._page_recall:
            # PAGE-RECALL: cells are owned in page-blocks (_g_page_to_block). Free each released
            # loc's PAGE-block once, return it to _g_free_blocks, and clear that page's 64 lut
            # entries. (A finished request's pages return to the page free-list, bounding
            # page-block occupancy to live requests — same purpose as the chunk path.)
            ps = C4_PAGE_SIZE
            p2b = self._g_page_to_block
            pages = {int(c) // ps for c in loc_set}
            to_free = [pg for pg in pages if pg in p2b]
            for pg in to_free:
                blk = p2b.pop(pg, None)
                if blk is not None:
                    self._g_free_blocks.append(int(blk))
                    n_freed += 1
                    cols = (pg * ps + torch.arange(ps, dtype=torch.int64))
                    cols = cols[(cols >= 0) & (cols < self.chunk_cell_lut.shape[1])]
                    self.chunk_cell_lut[:, cols.to(self.device)] = -1
            return n_freed
        if self._global_mode:
            # GLOBAL: one shared chunk_to_cell + free-list across all layers (a chunk owns
            # the SAME cell in every layer). Pop the released locs once and reclaim the cell.
            c2c = self._g_chunk_to_cell
            free = self._g_free_cells
            to_free = loc_set.intersection(c2c.keys()) if len(c2c) < len(loc_set) else \
                [c for c in loc_set if c in c2c]
            for c in list(to_free):
                cell = c2c.pop(c, None)
                if cell is not None:
                    free.append(int(cell))
                    n_freed += 1
        else:
            for layer in range(self.c4_layer_num):
                c2c = self.chunk_to_cell[layer]
                free = self.free_cells[layer]
                # intersect with the (typically small) resident set, not the full released range.
                to_free = loc_set.intersection(c2c.keys()) if len(c2c) < len(loc_set) else \
                    [c for c in loc_set if c in c2c]
                for c in list(to_free):
                    cell = c2c.pop(c, None)
                    if cell is not None:
                        free.append(int(cell))
                        n_freed += 1
        # clear these locs in every layer's lut (not resident anymore until re-recalled).
        self.chunk_cell_lut[:, locs.to(self.device)] = -1
        return n_freed

    def reclaim_orphans_global(self, resident_mask: torch.Tensor) -> int:
        """PROACTIVE orphan reclaim using the AUTHORITATIVE per-chunk residency mask (v4,
        2026-06-30). Frees every reserve cell whose chunk's resident_chunk_mask bit is False.

        WHY THIS SOURCE (v1/v2/v3 all failed on the live-set source): the global
        `resident_chunk_mask` is set per-request EVERY cycle by _scatter_logical_keep
        (mask[pool_loc]=keep), so mask[c]==True ⟺ chunk c is in SOME active request's CURRENT
        resident set — the exact ground truth, with no cap (resident_pool_loc_buf was capped at
        resident_buf=2048 << real ~5991) and no per-call timing gap (union_locs only had the 1
        due req/call). A chunk in c2c with mask==False is genuinely no longer wanted by anyone.

        Root cause addressed (measured run~41): without reclaim, the shifting recency tail orphans
        cells; _g_chunk_to_cell grew to the full reserve (983039) though only ~150K live -> every
        new chunk forced an eviction scan -> recall ballooned to ~75ms/cycle. Reclaiming by the
        true mask keeps c2c == working set, eviction never fires, and NO live chunk is freed.

        resident_mask: bool[c4_logical_size+1] global mask (capturer.resident_chunk_mask). One
        gather of mask at the c2c keys (GPU) + one D2H of the dead subset. Out of graph."""
        if not self._global_mode:
            return 0
        c2c = self._g_chunk_to_cell
        if len(c2c) == 0:
            return 0
        keys = list(c2c.keys())
        kt = torch.tensor(keys, dtype=torch.long, device=resident_mask.device)
        kt_c = kt.clamp(0, resident_mask.shape[0] - 1)
        # dead = c2c chunks whose mask bit is False (no active request keeps them)
        dead_mask = ~resident_mask[kt_c]
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        if dead_idx.numel() == 0:
            return 0
        dead = kt[dead_idx].to("cpu").tolist()
        free = self._g_free_cells
        for c in dead:
            cell = c2c.pop(c, None)
            if cell is not None:
                free.append(int(cell))
        dead_t = kt[dead_idx]
        dead_t = dead_t[(dead_t >= 0) & (dead_t < self.chunk_cell_lut.shape[1])]
        if dead_t.numel() > 0:
            self.chunk_cell_lut[:, dead_t.to(self.device)] = -1
        return len(dead)

    def ingest_transferred_history(
        self, src_c4_pool, logical_pages: torch.Tensor
    ) -> int:
        """PD decode-server hook: after NIXL transfers a request's full c4 history
        into the (full-size) GPU c4 pool, copy those page-rows into the CPU mirror
        so decode-time recall_chunks/remap can pull the keep-set into the reserve.

        On D the transfer lands at GPU c4 physical loc == logical loc (c4_out_loc =
        full_loc//4, full pool not shrunk), so this is an IDENTITY-indexed whole-row
        copy: mirror[layer][pg] = src_c4_pool.kv_buffer[layer][pg]. After ingest the
        mirror holds the authoritative bytes; the touched pages are invalidated in
        every layer's lut so the next recall re-materializes them.

        src_c4_pool: the GPU DeepSeekV4SingleKVPool the transfer wrote into (full c4
            pool). For the mechanism test this is the same pool object as self.pool
            (no shrink); once D's c4 pool is shrunk, this is the separate full pool.
        logical_pages: 1-D int tensor of c4 logical page numbers this request owns
            (= unique(req_to_token c4 locs // C4_PAGE_SIZE)).
        Returns #pages ingested.
        """
        pages = torch.unique(logical_pages.detach().to("cpu", dtype=torch.int64))
        pages = pages[(pages >= 0) & (pages < self.n_logical_pages)]
        if pages.numel() == 0:
            return 0
        # chunks covered by these pages (mirror is page-row layout, but recall works at
        # CHUNK granularity: chunk = page*64 + offset).
        for layer in range(self.c4_layer_num):
            gpu_buf = src_c4_pool.kv_buffer[layer]            # [n_pages, row_bytes] uint8
            rows = gpu_buf[pages].to("cpu", non_blocking=False)
            self.host_mirror[layer][pages] = rows
            # DEBUG falsification: poison the GPU source rows after mirroring, so a
            # needle that still survives PROVES decode read via mirror->reserve recall
            # (not the stale in-place GPU copy). Only when SGLANG_SWAP_P_POISON=1.
            if self._poison_after_ingest:
                gpu_buf[pages] = 0
        # invalidate these pages' chunks in EVERY layer's lut so the next recall
        # re-materializes fresh bytes (vectorized, replaces the per-chunk Python loop).
        base = (pages * C4_PAGE_SIZE).view(-1, 1)
        chunks = (base + torch.arange(C4_PAGE_SIZE)).reshape(-1)
        chunks = chunks[chunks < self.chunk_cell_lut.shape[1]].to(self.device)
        self.chunk_cell_lut[:, chunks] = -1
        if self.device != "cpu":
            torch.cuda.synchronize()
        return int(pages.numel())

