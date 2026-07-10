"""
resident_mask_capturer.py — Phase-1 of cuda-graph-compatible two-level recall.
=============================================================================

The two-level FlashMemory recall has ONE host-sync-free in-graph primitive and
ALL the host-sync work pushed to a side-band that fires at cycle boundaries:

  * IN GRAPH (every decode step, all 21 c4 layers, pure GPU — captured & replayed):
      read the global static `resident_chunk_mask` buffer, gather residency for
      this layer's per-req logical chunks via the static page_table, and mask the
      native indexer logits of non-resident chunks to -inf BEFORE topk_transform_512.
      => the native top-512 is selected FROM the resident_set, exactly the spec.

  * SIDE-BAND (cycle boundary only, ~1 step in `interval`, OUT of graph — host
      syncs legal): run the Level-1 Memory Indexer (trained retriever) over the
      target layers, ensemble + threshold to a fresh resident_set, and scatter it
      into the global `resident_chunk_mask`. Applies to the NEXT window (n→n+1
      latency, spec-sanctioned: the retriever predicts the next ~64 steps anyway,
      recent/sink cover the newest chunks).

This mirrors sglang's own IndexerTopkCapturer (indexer_topk_capturer.py): a fixed
device buffer written in-graph and consumed in `on_forward_end` out-of-graph. The
key difference: WE only need the in-graph READ to be graph-safe; the WRITE (Level-1
scoring) is the side-band, so this file owns the static buffer + the mask gather
helper, and (Strategy A) the side-band scoring trigger.

Phase-1 scope (see swap_infra/CUDAGRAPH_COMPAT_DESIGN.md):
  * masking-only, FULL c4 residency (no Path-P offload yet) — so there is no
    recall / decode-store host-sync to relocate; only the Level-1 scoring is
    side-banded. This isolates "cuda-graph compatibility" as the single variable.

Coordinate systems (load-bearing, do NOT confuse):
  * `logits` in the indexer is [B, max_blk] in per-req LOGICAL chunk order (column
    j = the req's j-th c4 chunk).
  * `resident_chunk_mask` is a GLOBAL bool[c4_logical_size] indexed by POOL LOC
    (compressed loc = page*page_size + offset), shared by all 21 layers (pool locs
    are request-disjoint, so one global array is correct).
  * map logical (bi, j) -> pool loc via the static page_table:
        pool_loc = page_table[bi, j // page_size] * page_size + (j % page_size)
    then gather resident_chunk_mask[pool_loc]. This is the same page_table gather
    topk_transform_512 already does internally, so it is graph-safe.

Enabled via env (read once at capturer creation):
  SGLANG_PATHP_CUDAGRAPH=1            # master switch for the graph-compatible path
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_HEAD_DIM = 128


def pathp_cudagraph_enabled() -> bool:
    return os.environ.get("SGLANG_PATHP_CUDAGRAPH", "0") == "1"


class ResidentMaskCapturer:
    """Owns the global static `resident_chunk_mask` GPU buffer + the in-graph mask
    helper + (Strategy A) the side-band Level-1 scoring trigger.

    One server-lifetime instance, fixed-address buffers (so the captured graph can
    read them and the side-band can overwrite them between replays). Created in
    model_runner.init_resident_mask_capturer when SGLANG_PATHP_CUDAGRAPH=1.
    """

    def __init__(
        self,
        c4_logical_size: int,
        c4_page_size: int,
        device: str,
        interval: int = 64,
        hidden_dim: int = 4096,
        max_bs: int = 64,
    ):
        self._enabled = True
        self.c4_logical_size = int(c4_logical_size)
        self.c4_page_size = int(c4_page_size)
        self.device = device
        self.interval = int(interval)
        self.hidden_dim = int(hidden_dim)
        self.max_bs = int(max_bs)

        # ── the ONE global static buffer the captured graph READS every step ──
        # bool[c4_logical_size+1]: True = chunk (at this pool loc) is in the
        # resident_set. +1 sentinel row for clamp safety (like swap_engine_p lut).
        # Initialized ALL-TRUE so that before the first cycle finalizes (cold start
        # / warmup capture) nothing is masked — identical to native (full attention).
        self.resident_chunk_mask = torch.ones(
            self.c4_logical_size + 1, dtype=torch.bool, device=device
        )

        # ── Strategy A side-band: lazily build the InlineRetrieverHook to REUSE its
        # trained _TrainedScorer (per target layer) + _finalize_resident (ensemble +
        # threshold + recent/sink + capacity) — so the Level-1 math is identical to the
        # eager path, no divergence. We only drive its scorers from the side-band; its
        # __call__ is never used here.
        self._hook = None  # lazy (needs SGLANG_RETRIEVER_INLINE_CKPT etc.)
        self.target_layers = []
        self._init_hook()

        # ── target-layer hidden capture buffer (graph WRITES, side-band READS) ──
        # [n_target, max_bs, hidden]. The captured graph copies each target layer's
        # decode hidden state here (fixed address); on_forward_end reads it to run
        # Level-1. Only target layers write; non-target layers don't touch it.
        n_target = max(1, len(self.target_layers))
        self.target_hidden_buf = torch.zeros(
            (n_target, self.max_bs, self.hidden_dim), dtype=torch.bfloat16, device=device
        )
        # map compress_layer_id -> row in target_hidden_buf
        self._target_row = {lid: i for i, lid in enumerate(self.target_layers)}

        # ── side-band input buffers (graph WRITES via capture_side_band_inputs, side-band
        # READS post-replay). page_table + c4_seq_lens are needed to (a) extract c4-indexer-K
        # and (b) map logical->pool_loc. We capture them into OUR fixed buffers rather than
        # read backend.forward_metadata, which in PREP_IN_CUDA_GRAPH decode is a raw stub
        # (DSV4MetadataRawDecode, real_metadata=None) post-forward. Lazily sized on first
        # capture (n_pages known only at runtime). req_pool_indices come from forward_batch
        # (stable in on_forward_end).
        self.page_table_buf = None      # [max_bs, n_pages] int32, lazy
        self.c4_seq_lens_buf = None     # [max_bs] int64, lazy
        self.positions_buf = None       # [max_bs] int64, lazy
        self._sb_captured_step = False  # whether THIS forward already captured pt/seql

        # per req_pool_idx -> step counter, for per-req cycle cadence in the side-band.
        self._req_step: dict[int, int] = {}
        self._cycle_count = 0

        # ── cycle-phase stagger (env SGLANG_PATHP_STAGGER_CYCLE=1) ──
        # A batch of requests that LAND together (transfer drains in bursts, or a barrier
        # releases them together) all get _req_step=0 → they all hit `step % interval == 0`
        # on the SAME decode steps → `due=N/cycle` spikes → N recalls fire in ONE step →
        # N× the swap bandwidth at once → stall. In PRODUCTION requests arrive staggered so
        # their cycle boundaries spread across the interval (due≈N/interval per step, smooth).
        # This gate reproduces that: assign each request a round-robin PHASE in [0,interval)
        # so its cadence is `(step - phase) % interval == 0` — spreading recalls evenly.
        # The FIRST finalize still fires immediately (step 0) for correctness; only the
        # steady-state cadence is phase-shifted. Default off = original (step-0-aligned).
        self._stagger_cycle = os.environ.get("SGLANG_PATHP_STAGGER_CYCLE", "0") == "1"
        self._req_phase: dict[int, int] = {}
        self._next_phase = 0

        # ── Step-1/2: score-resident-only (env SGLANG_PATHP_SCORE_RESIDENT=1) ──
        # Instead of scoring the FULL c4 history every step then masking (apply_mask_in_graph),
        # the in-graph indexer scores ONLY the resident_set. The side-band (here, every
        # `interval` steps, out-of-graph) packs each request's resident indexer-K into a
        # contiguous PAGED scratch buffer (one per c4 layer) with an IDENTITY page_table, so
        # the kernel reads it as if a tiny full history. logits become [B, RESIDENT_BUF]
        # (BUF<=2048) → indexer FLOP O(buf) not O(context), and the full-width mask is gone.
        # Buffers are FIXED-ADDRESS (lazy-alloc on first build), refreshed side-band, read by
        # the captured graph on replay. All gate-off (default) = original full-history path.
        self._score_resident = os.environ.get("SGLANG_PATHP_SCORE_RESIDENT", "0") == "1"
        # PAGE-RECALL (SGLANG_PATHP_PAGE_RECALL=1, 2026-06-30): recall WHOLE c4 pages
        # (64 chunks each) chosen by per-page score = #(sigmoid>thresh) chunks in the page,
        # top-K_max densest pages. The indexer then scores those pages with a REAL page_table
        # over the live index-K pool (no scratch packing — see _score_paged_layer proof). The
        # per-req buffers (resident_id_page_table / resident_count_buf / resident_pool_loc_buf /
        # valid_mask_buf) are reused VERBATIM from the score-resident path; only their POPULATION
        # (page-select vs scratch-pack) and the indexer's KV source (live pool vs scratch) differ.
        # Requires score_resident on (shares its buffer plumbing). Gate off = unchanged.
        self._page_recall = os.environ.get("SGLANG_PATHP_PAGE_RECALL", "0") == "1"
        self._page_kmax = int(os.environ.get("SGLANG_PATHP_PAGE_KMAX", "96"))  # max pages/req
        # Path-I: index-K offload. When on, the indexer scores the SHRUNK index-K reserve
        # (resident_id_page_table_reserve holds reserve-block ids); when off, unchanged
        # (indexer reads resident_id_page_table = full-history phys pages).
        self._index_k_offload = os.environ.get("SGLANG_PATHP_INDEX_K_OFFLOAD", "0") == "1"
        self.resident_buf = int(os.environ.get("SGLANG_PATHP_RESIDENT_BUF", "2048"))
        # PAGE-RECALL sizes the buffer to K_max whole pages (K_max*64 cols); the score-resident
        # default (2048) is for the scratch path. Page-recall needs exactly K_max pages/req.
        if self._page_recall:
            self.resident_buf = self._page_kmax * self.c4_page_size
        # round BUF up to a whole number of c4 pages so the scratch is page-aligned.
        self._resident_pages = (self.resident_buf + self.c4_page_size - 1) // self.c4_page_size
        self.resident_buf = self._resident_pages * self.c4_page_size
        self.resident_kv_scratch = None     # dict clid -> [max_bs*pages, 64, 1, 132] uint8, lazy
        self.resident_count_buf = None      # [max_bs] int32, lazy: per-req resident chunk count
        self.resident_pool_loc_buf = None   # [max_bs, BUF] int64, lazy: compacted-col -> phys pool loc
        self.resident_id_page_table = None  # [max_bs, pages] int32, lazy: identity page table
        self.resident_id_page_table_reserve = None  # [max_bs, pages] int32, lazy: Path-I reserve-block ids
        self.resident_dg_meta = None        # deep_gemm SM-schedule meta, rebuilt side-band/cycle
        self._rs_kv_view = None             # cached (n_pages_view, 64, 132) shape probe per clid
        self._pending_count = {}            # bi -> staged K; published to resident_count_buf LAST
                                            # (after col_to_cell refresh) to keep count>0 ⟺ fresh cells

        # ── Part-2: ASYNC recall (env SGLANG_PATHP_ASYNC_RECALL=1, default ON when score-
        # resident) ── The batched recall (Part-1) still does torch.unique + a .tolist() D2H
        # sync + a Python alloc + as_strided H2D copies = ~33ms, and it runs SYNCHRONOUSLY on
        # the scheduler thread between forwards → the WHOLE batch stalls ~33ms every cycle
        # boundary (measured). The resident_set finalized at cycle N feeds the NEXT 64-step
        # window (n→n+1 latency is already spec-sanctioned), and recall (33ms) ≪ window (~1s),
        # so we run recall in a BACKGROUND THREAD: the scheduler returns immediately and
        # launches the next forward while the thread does unique/sync/alloc/copy. A
        # remap-miss during the brief in-flight window is GRACEFUL (remap returns -1 →
        # chunk masked-out → self-heals as copies land; recency-tail needle is always
        # resident so quality is unaffected). We join the previous cycle's thread at the
        # start of the next cycle (recall ≪ interval, so it's always already done → no wait).
        self._async_recall = (
            self._score_resident
            and os.environ.get("SGLANG_PATHP_ASYNC_RECALL", "1") == "1"
            and device != "cpu"
        )
        self._recall_thread: Optional[threading.Thread] = None

        # ── side-band phase timing (env SGLANG_PATHP_SB_TIMING=1): CUDA-event ms per phase
        # accumulated over cycles, printed every 50 cycles. Localizes the run=80 stall
        # (scoring / finalize / scatter / recall / scratch / meta). Zero overhead when off.
        self._sb_timing = os.environ.get("SGLANG_PATHP_SB_TIMING", "0") == "1"
        self._sb_acc = {}      # phase -> accumulated ms
        self._sb_n = 0         # number of on_forward_end decode calls timed

        # DELAY-AUDIT: optional per-cycle resident-set dump (env SGLANG_RESIDENT_DUMP).
        self._dump_path = os.environ.get("SGLANG_RESIDENT_DUMP", "") or None
        self._dump_f = None
        if self._dump_path is not None:
            # rank-suffix so 8 TP ranks don't clobber; only rank with data matters.
            from sglang.srt.distributed import get_tensor_model_parallel_rank as _tpr
            try:
                rk = _tpr()
            except Exception:
                rk = 0
            self._dump_path = f"{self._dump_path}.tp{rk}"
            self._dump_f = open(self._dump_path, "w")
            logger.info(f"[ResidentMask] DELAY-AUDIT dump → {self._dump_path}")

        size_mb = self.resident_chunk_mask.numel() / _MB
        logger.info(
            f"[ResidentMask] init: c4_logical_size={self.c4_logical_size} "
            f"page_size={self.c4_page_size} interval={self.interval} "
            f"mask_buffer={size_mb:.1f}MB (bool) target_layers={self.target_layers} "
            f"hidden_buf={list(self.target_hidden_buf.shape)} device={device}"
        )

    def _init_hook(self):
        """Build the InlineRetrieverHook (reused for scorers + finalize). Requires the
        SGLANG_RETRIEVER_INLINE_CKPT env (same as eager inline path). If unset, the
        capturer stays mask-only (all-True, no Level-1) — V1 behavior."""
        ckpt = os.environ.get("SGLANG_RETRIEVER_INLINE_CKPT", "")
        if not ckpt or not os.path.exists(ckpt):
            logger.warning(
                "[ResidentMask] no SGLANG_RETRIEVER_INLINE_CKPT — side-band Level-1 "
                "disabled (mask stays all-True = native). Set it to enable two-level."
            )
            return
        from sglang.srt.layers.attention.compressed.inline_retriever_hook import (
            InlineRetrieverHook,
        )
        self._hook = InlineRetrieverHook()
        self.target_layers = list(self._hook.target_layers)
        # honor the hook's interval (env SGLANG_RETRIEVER_INTERVAL) as the cycle period.
        self.interval = int(self._hook.interval)


    def is_enabled(self) -> bool:
        return self._enabled

    @torch.no_grad()
    def apply_mask_in_graph(
        self,
        logits: torch.Tensor,        # [B, max_blk] per-req LOGICAL chunk order
        page_table: torch.Tensor,    # [B, n_pages] static (capture-fixed) buffer
        c4_seq_lens: torch.Tensor,   # [B] or [B,1] per-req c4 chunk count
    ) -> torch.Tensor:
        """PURE GPU, graph-safe. Mask non-resident chunks' logits to -inf so the
        downstream native topk_transform_512 selects ONLY from the resident_set.

        All tensors are fixed-address / replay-refreshed; ops are gather + masked_fill
        with an arange<seq_lens padding guard. No .item()/.tolist()/sync — capturable.
        """
        B, max_blk = logits.shape
        ps = self.c4_page_size
        device = logits.device

        # per-req logical chunk index j over [0, max_blk)
        j = torch.arange(max_blk, device=device)                  # [max_blk]
        page_of_j = j // ps                                        # [max_blk]
        off_of_j = j % ps                                          # [max_blk]
        # gather the physical page for each (bi, j): page_table[bi, j//ps]
        # clamp page index to page_table width to stay in-bounds for padding cols.
        n_pages = page_table.shape[1]
        page_of_j_c = page_of_j.clamp(0, n_pages - 1)
        phys_page = page_table[:, page_of_j_c]                     # [B, max_blk]
        pool_loc = phys_page.to(torch.long) * ps + off_of_j       # [B, max_blk]
        pool_loc = pool_loc.clamp(0, self.c4_logical_size)        # sentinel-safe
        resident = self.resident_chunk_mask[pool_loc]             # [B, max_blk] bool

        # padding guard: columns j >= this req's c4_seq_len are not real chunks;
        # leave them as-is (topk already ignores them via c4_seq_lens), so treat as
        # resident (don't add spurious -inf that could shift topk padding behavior).
        if c4_seq_lens.dim() == 2:
            seql = c4_seq_lens[:, 0]
        else:
            seql = c4_seq_lens
        valid = j.unsqueeze(0) < seql.unsqueeze(1)                # [B, max_blk]
        drop = valid & (~resident)
        return logits.masked_fill(drop, float("-inf"))

    @torch.no_grad()
    def capture_side_band_inputs(
        self,
        compress_layer_id: int,
        x: torch.Tensor,
        page_table: torch.Tensor,
        c4_seq_lens: torch.Tensor,
        positions: torch.Tensor = None,
    ) -> None:
        """IN-GRAPH (target layers only): copy this layer's decode hidden + (once per
        forward) page_table / c4_seq_lens / positions into fixed buffers so the side-band
        can read them post-replay WITHOUT touching backend.forward_metadata (a raw stub
        in PREP_IN_CUDA_GRAPH decode). Pure GPU copy_ into fixed addresses — capturable.
        x: [B, hidden]. No-op if this layer isn't a target."""
        row = self._target_row.get(compress_layer_id)
        if row is None:
            return
        B = min(x.shape[0], self.max_bs)
        self.target_hidden_buf[row, :B].copy_(x[:B])

        # page_table / c4_seq_lens / positions: capture once per forward (first target
        # layer to arrive). They're identical across the 21 layers for a given step.
        if compress_layer_id == self.target_layers[0]:
            self._capture_meta(page_table, c4_seq_lens, positions, B)

    @torch.no_grad()
    def _capture_meta(self, page_table, c4_seq_lens, positions, B):
        # lazy-allocate fixed buffers sized to the capture-time shapes
        n_pages = page_table.shape[1]
        if self.page_table_buf is None or self.page_table_buf.shape[1] != n_pages:
            self.page_table_buf = torch.zeros(
                (self.max_bs, n_pages), dtype=page_table.dtype, device=self.device
            )
            self.c4_seq_lens_buf = torch.zeros(
                self.max_bs, dtype=torch.int64, device=self.device
            )
            self.positions_buf = torch.zeros(
                self.max_bs, dtype=torch.int64, device=self.device
            )
        self.page_table_buf[:B].copy_(page_table[:B])
        seql = c4_seq_lens[:, 0] if c4_seq_lens.dim() == 2 else c4_seq_lens
        self.c4_seq_lens_buf[:B].copy_(seql[:B].to(torch.int64))
        if positions is not None:
            self.positions_buf[:B].copy_(positions[:B].to(torch.int64))

    # kept for backward-compat with any caller using the old name
    @torch.no_grad()
    def capture_target_hidden(self, compress_layer_id: int, x: torch.Tensor) -> None:
        row = self._target_row.get(compress_layer_id)
        if row is None:
            return
        B = min(x.shape[0], self.max_bs)
        self.target_hidden_buf[row, :B].copy_(x[:B])

    @torch.no_grad()
    def on_forward_begin(self, forward_batch, token_to_kv_pool):
        """PRE-FORWARD (out of graph, host syncs legal). CHURN-CRASH FIX (2026-06-29):
        reset score-resident per-row buffers for any decode row whose occupant changed since
        last step, BEFORE the in-graph indexer of this step reads them.

        Why here and not on_forward_end: the decode event loop runs filter_batch (row compaction
        on request finish) + merge (new transferred reqs) in get_next_disagg_decode_batch_to_run,
        THEN run_batch→forward. So by the time the forward's captured indexer reads
        resident_count_buf[bi] / col_to_cell_buf[bi], the row layout has ALREADY changed — a
        reused/shifted row would carry the previous occupant's count>0 + stale reserve-cell
        pointers → illegal access. on_forward_end is one step too late (crash already happened in
        the forward). The prefill-side reset_req_residency never fires on the D PD-decode server
        (pure decode, no prefill/extend), so this is the ONLY cross-request cleanup on D.

        Ours-only, out of graph, no captured/baseline code touched."""
        if not self._score_resident or self.resident_count_buf is None:
            return
        if forward_batch.forward_mode is None or not forward_batch.forward_mode.is_decode():
            return
        B = int(forward_batch.req_pool_indices.shape[0])
        if B <= 0:
            return
        # FAST PATH: detect whether any decode row changed occupant since last step. This is a
        # cheap D2H of B ints; compare to the host-side snapshot. On the vast majority of steps
        # (steady decode) NOTHING changed → return immediately: NO join, NO reset, NO GPU work.
        # We only pay the (potentially slow) side-band join + reset on the RARE steps where the
        # batch composition actually changed (a request finished → filter_batch compacted rows,
        # or a new transferred req was admitted). Joining every step instead stalls the scheduler
        # hot path behind the heavy churn-cycle recall → detokenizer heartbeat death (observed).
        req_ids = forward_batch.req_pool_indices.tolist()
        prev = getattr(self, "_prev_row_occupant", None)
        if prev is not None and len(prev) == B and prev == req_ids[:B]:
            return                                  # no composition change → fast path
        # composition CHANGED: join the in-flight side-band (so it can't overwrite our reset with
        # the previous occupant's data) THEN reset the changed rows. Bounded to change events, so
        # even a slow churn-cycle join happens only when the batch actually churned, not per step.
        prev_thr = getattr(self, "_recall_thread", None)
        if prev_thr is not None and prev_thr.is_alive():
            prev_thr.join()
        self._cached_req_ids = req_ids
        self._reset_changed_rows(req_ids, B, token_to_kv_pool)

    @torch.no_grad()
    def on_forward_end(self, forward_batch, can_run_graph, backend, token_to_kv_pool):
        """SIDE-BAND (out of graph, host syncs legal). For each decode req at a cycle
        boundary (its step % interval == 0), reconstruct the Level-1 scoring inputs
        (hidden + page_table + c4_seq_lens + positions ALL from OUR captured buffers,
        c4-indexer-K from the pool) and run the trained Memory Indexer → ensemble →
        threshold → scatter the resident_set into the global resident_chunk_mask.
        Applies to the NEXT window (n→n+1, spec-sanctioned). No-op if Level-1 disabled."""
        self._ofe_calls = getattr(self, "_ofe_calls", 0) + 1
        if self._hook is None:
            if self._ofe_calls <= 3:
                logger.info(f"[ResidentMask] on_forward_end #{self._ofe_calls}: hook=None, skip")
            return
        if not forward_batch.forward_mode.is_decode():
            return
        self._ofe_decode_calls = getattr(self, "_ofe_decode_calls", 0) + 1
        # OFE main-thread wall-clock profiler (SGLANG_PATHP_OFE_PROF=1): measures the per-step
        # MAIN-THREAD cost of on_forward_end (the thing OFE_SKIP=1 eliminates → 83→112 tok/s at
        # run=1). Host time, not GPU. Accumulates per-section, prints every 200 decode calls.
        _ofe_prof = os.environ.get("SGLANG_PATHP_OFE_PROF", "0") == "1"
        _t0 = time.perf_counter() if _ofe_prof else 0.0
        if _ofe_prof and not hasattr(self, "_ofe_prof_acc"):
            self._ofe_prof_acc = {}
        def _ofe_tic(name, t_start):
            if not _ofe_prof:
                return
            dt = (time.perf_counter() - t_start) * 1e3
            a = self._ofe_prof_acc.setdefault(name, [0.0, 0])
            a[0] += dt; a[1] += 1
        # PROBE (SGLANG_PATHP_OFE_SKIP=1): skip ALL per-step side-band host work to measure
        # whether the every-step .tolist() D2H sync (line below) is the per-step throughput
        # tax. Output is WRONG (resident_set never updates) — speed-isolation only.
        if os.environ.get("SGLANG_PATHP_OFE_SKIP", "0") == "1":
            return
        # inputs from OUR captured buffers (filled in-graph by capture_side_band_inputs),
        # NOT backend.forward_metadata (a raw stub in PREP_IN_CUDA_GRAPH decode).
        if self.page_table_buf is None:
            if self._ofe_decode_calls <= 3:
                logger.info("[ResidentMask] on_forward_end decode: side-band buffers not "
                            "captured yet (no target-layer capture ran?)")
            return
        page_table = self.page_table_buf
        c4_seq_lens = self.c4_seq_lens_buf
        positions = self.positions_buf
        # Per-step req-id resolution WITHOUT a D2H sync every step. The .tolist() below
        # forces the host to wait for the in-graph forward to finish before it can build
        # the next batch — serializing graph replay (measured: ~170 agg tok/s tax at run=27,
        # log throughput oscillating 333-723 = repeated sync stalls). req_pool_indices order
        # is STABLE across a decode run; it only changes when batch composition changes,
        # which we detect sync-free via .shape[0]. So we sync the id list at most once per
        # `interval` steps OR when B changes, and advance step counters purely on host in
        # between. (A request finishing/joining changes B → forces a refresh → correctness.)
        B = int(forward_batch.req_pool_indices.shape[0])   # host metadata, no sync
        self._ofe_global_step = getattr(self, "_ofe_global_step", 0) + 1
        # CHURN-CRASH FIX (2026-06-29): the score-resident per-row buffers (resident_count_buf
        # / col_to_cell_buf / resident_pool_loc_buf / scratch row) are POSITIONAL by decode row
        # bi. filter_batch COMPACTS rows on request finish (req_pool_indices = [keep_indices]),
        # and on the D (PD-decode) server reset_req_residency NEVER fires (it lives in the
        # prefill/extend branch; D runs pure decode). So a reused/shifted row keeps the previous
        # occupant's count>0 + stale reserve-cell pointers → in-graph gather derefs a freed cell
        # → CUDA illegal access (only under churn = high concurrency; run=1 never reuses row 0).
        # The OLD full-history+mask path keyed off GLOBAL pool_loc (permutation-invariant) → was
        # immune. FIX: when score-resident is active, sync req ids EVERY step (the cached-id
        # micro-opt is unsafe here — a same-B occupant swap between interval boundaries would
        # advance _req_step for the wrong ri AND leave stale per-row buffers) and reset any row
        # whose occupant changed before the due loop. Out-of-graph host code, ours-only.
        # req-id resolution for the due loop: cached, refreshed when B changes or every
        # interval (the original sync-frugal cadence). on_forward_begin already syncs + resets
        # changed rows BEFORE the forward, so the per-row buffers are safe regardless of this
        # cache; we do NOT force a per-step sync here (that added a per-step D2H that serialized
        # the 70-way decode behind a full-batch sync → scheduler stall under churn).
        _cached = getattr(self, "_cached_req_ids", None)
        if (_cached is None or len(_cached) != B
                or (self._ofe_global_step % self.interval) == 0):
            req_ids = forward_batch.req_pool_indices.tolist()   # the ONE periodic sync
            self._cached_req_ids = req_ids
        else:
            req_ids = _cached
        # NOTE: the stale-row RESET runs in on_forward_begin (BEFORE the forward reads the
        # buffers), not here — on_forward_end is too late (the in-graph indexer of THIS step
        # already read resident_count_buf with the new row layout). See on_forward_begin.

        # which reqs are at a cycle boundary this step (per-req cadence)
        due = []
        for bi in range(B):
            ri = req_ids[bi]
            step = self._req_step.get(ri, 0)
            # step 0 always fires (first finalize, correctness). Steady-state cadence is
            # phase-shifted per request when stagger is on, so a batch landing together
            # doesn't all recall on the same steps (see _stagger_cycle in __init__).
            phase = self._req_phase.get(ri, 0) if self._stagger_cycle else 0
            if step == 0 or (step % self.interval) == (phase % self.interval):
                due.append(bi)
            self._req_step[ri] = step + 1
        _ofe_tic("perstep_dueloop", _t0)
        if self._ofe_decode_calls <= 5:
            logger.info(
                f"[ResidentMask] on_forward_end decode #{self._ofe_decode_calls}: "
                f"B={B} due={len(due)} target_layers={self.target_layers}"
            )
        if not due:
            _ofe_tic("total", _t0)
            if _ofe_prof and self._ofe_decode_calls % 200 == 0:
                parts = " ".join(f"{k}={v[0]/max(v[1],1):.3f}ms(n{v[1]})"
                                 for k, v in sorted(self._ofe_prof_acc.items()))
                logger.info(f"[OFE-PROF] avg main-thread ms/call: {parts}")
            return

        # c4 seq lens to host once (small: B ints)
        _t_d2h = time.perf_counter() if _ofe_prof else 0.0
        if c4_seq_lens.dim() == 2:
            seql_t = c4_seq_lens[:, 0]
        else:
            seql_t = c4_seq_lens
        seql_cpu = seql_t.to("cpu", dtype=torch.int64).tolist()
        _ofe_tic("due_d2h_tolist", _t_d2h)

        # ── Part-2 FULL-ASYNC side-band ── The side-band (score 14ms + finalize + scatter +
        # scratch + recall) feeds the NEXT 64-step window (n→n+1, spec-sanctioned) and is
        # ~20ms ≪ window (~1s), yet it ran SYNCHRONOUSLY on the scheduler thread, and with
        # due≈B/step (transfer-spread arrival) it blocked ~every step → the dominant stall.
        # We snapshot the 3 LIVE input buffers the work reads (hidden / page_table / positions
        # — the next forward overwrites them) and run ALL the work in a background thread on a
        # side stream. The work WRITES live fixed-address buffers (resident_chunk_mask /
        # scratch / count / pool_loc / valid_mask / reserve / lut) that the next forwards read;
        # a concurrent read during the write is GRACEFUL (mask per-chunk bool, scratch bytes,
        # count per-req — worst case a few steps of slightly-stale residency for non-needle
        # chunks; the needle lives in the always-resident recency tail). Join the previous
        # cycle's thread first so ≤1 is ever in flight (work ≪ interval → join is instant).
        if self._async_recall:
            prev = self._recall_thread
            _t_join = time.perf_counter() if _ofe_prof else 0.0
            if prev is not None and prev.is_alive():
                prev.join()
            _ofe_tic("due_prevjoin", _t_join)
            _t_clone = time.perf_counter() if _ofe_prof else 0.0
            hb = self.target_hidden_buf.clone()
            pt = page_table.clone()
            pos = positions.clone() if positions is not None else None
            due_snap = list(due)
            ri_snap = list(req_ids)
            seql_snap = list(seql_cpu)
            _ofe_tic("due_clone_dispatch", _t_clone)
            # the bg thread does NOT inherit this rank's CUDA device (defaults to cuda:0) —
            # under TP each rank lives on its own device, so pin it explicitly or every tensor
            # op hits "two devices cuda:N vs cuda:0". Capture the real ordinal from a buffer.
            dev_idx = self.resident_chunk_mask.device.index
            def _bg_sideband():
                try:
                    if dev_idx is not None:
                        torch.cuda.set_device(dev_idx)
                    with torch.cuda.stream(self._get_recall_stream()):
                        self._sideband_work(due_snap, ri_snap, seql_snap, hb, pt, pos,
                                            token_to_kv_pool)
                except Exception as e:
                    if self._cycle_count < 5:
                        logger.warning(f"[ResidentMask] async side-band failed: {e}")
                finally:
                    self._cycle_count += 1
            self._recall_thread = threading.Thread(target=_bg_sideband, daemon=True)
            self._recall_thread.start()
            self._sb_report(B, len(due))
            _ofe_tic("total", _t0)
            if _ofe_prof and self._ofe_decode_calls % 200 == 0:
                parts = " ".join(f"{k}={v[0]/max(v[1],1):.3f}ms(n{v[1]})"
                                 for k, v in sorted(self._ofe_prof_acc.items()))
                logger.info(f"[OFE-PROF] avg main-thread ms/call: {parts}")
            return

        # ── SYNC path (async off): run the side-band inline (original behavior) ──
        self._sideband_work(due, req_ids, seql_cpu, self.target_hidden_buf, page_table,
                            positions, token_to_kv_pool)
        self._sb_report(B, len(due))
        self._cycle_count += 1

    @torch.no_grad()
    def _sideband_work(self, due, req_ids, seql_cpu, hidden_buf, page_table, positions,
                       token_to_kv_pool):
        """The Level-1 side-band body: for each due req, score (retriever) → finalize →
        scatter mask → build scratch, then ONE batched recall + meta/valid-mask refresh.
        Reads hidden/page_table/positions from the passed (possibly snapshotted) buffers so
        it can run on a background thread while the next forward overwrites the live ones.
        Writes the live fixed-address publish buffers (mask/scratch/count/pool_loc/valid/
        reserve/lut)."""
        # PROBE (SGLANG_PATHP_SKIP_SBWORK=1): skip the ENTIRE per-req side-band body (score +
        # finalize + scatter + scratch + recall) but keep the bg-thread dispatch machinery. If
        # run=1 jumps to ~112, the tax is 100% inside this body (GPU work + .item()/.tolist
        # syncs contending with main decode); narrows vs OFE_SKIP (which skips dispatch too).
        if os.environ.get("SGLANG_PATHP_SKIP_SBWORK", "0") == "1":
            return
        # PROBE (SGLANG_PATHP_SB_SLEEP_MS=N): replace the real body with a pure host sleep of N
        # ms (NO CUDA calls) to test whether the bg-thread tax is from CUDA calls holding the
        # GIL/driver lock (→ pure sleep should NOT stall main decode → run=1 stays ~112) vs the
        # bg thread holding the GIL during Python work (→ sleep WOULD stall → run<112).
        _sb_sleep = os.environ.get("SGLANG_PATHP_SB_SLEEP_MS", "0")
        if _sb_sleep != "0":
            time.sleep(float(_sb_sleep) / 1000.0)
            return
        # ASYNC FIX (2026-06-29): this whole body runs on the BG thread but holds the GIL
        # through its Python orchestration (3 score layers + 21 scratch-copy layers + scatter/
        # recall). While held, the MAIN decode thread can't grab the GIL to launch its next
        # decode graph → run=1 83 vs 112 ceiling (pure-sleep bg = 112, proving it's GIL/lock
        # contention, not GPU). _sb_yield() does time.sleep(0) (releases the GIL momentarily) at
        # block boundaries so the main thread can interleave its per-step graph launch. The
        # side-band feeds the NEXT 64-step window (~768ms budget) so spreading it out is free.
        # Tunable via SGLANG_PATHP_SB_YIELD_MS (default 0 = sleep(0) pure GIL yield).
        _sb_yield_on = os.environ.get("SGLANG_PATHP_SB_YIELD", "1") == "1"
        _sb_yield_ms = float(os.environ.get("SGLANG_PATHP_SB_YIELD_MS", "0"))
        def _sb_yield():
            if _sb_yield_on:
                time.sleep(_sb_yield_ms / 1000.0)
        self._sb_yield_fn = _sb_yield   # so _build_resident_scratch's 21-layer loop can yield too
        batched_kept = []
        for bi in due:
            ri = req_ids[bi]
            n_blk = int(seql_cpu[bi])
            if n_blk <= 0:
                continue
            # ── per target layer: score this req's history ──
            # PAGED mode (SGLANG_PATHP_PAGED_SCORE, default "1"): score the index-K pool
            # DIRECTLY via the deep_gemm kernel (real page_table, NO _extract_compressed_k
            # gather) — same math as the native indexer. Validated 2026-06-29 vs the old
            # fp32 einsum path: resident-set jaccard 0.973 (threshold-set recall 0.971); the
            # fp8(q) rounding only flips ~3% of borderline low-confidence chunks. "validate"
            # runs BOTH and logs the overlap; "0" = old einsum+extract fallback.
            _paged_mode = os.environ.get("SGLANG_PATHP_PAGED_SCORE", "1")
            partial = {}
            partial_einsum = {} if _paged_mode == "validate" else None
            _t = self._sb_tic()
            for lid in self.target_layers:
                row = self._target_row[lid]
                x_bi = hidden_buf[row, bi : bi + 1]   # [1, hidden]
                pos = (
                    positions[bi : bi + 1]
                    if positions is not None
                    else torch.tensor([n_blk], device=self.device)
                )
                if _paged_mode in ("1", "validate"):
                    lg_p = self._score_paged_layer(
                        token_to_kv_pool, lid, bi, n_blk, page_table, x_bi, pos
                    )
                    if lg_p is not None:
                        partial[lid] = lg_p
                _sb_yield()   # release GIL between score layers so main decode can launch
                if _paged_mode in ("0", "validate"):
                    ek = self._extract_compressed_k(
                        token_to_kv_pool, lid, bi, n_blk, page_table
                    )
                    if ek is None:
                        continue
                    k_fp8, k_scale = ek
                    k_fp8_v = k_fp8.contiguous().view(torch.float8_e4m3fn)
                    k_scale_v = k_scale.contiguous().view(torch.float32).squeeze(-1)
                    scorer = self._hook.scorers[lid]
                    lg = scorer.forward(x_bi, k_fp8_v, k_scale_v, pos.to(torch.int64))[0].float()
                    if _paged_mode == "validate":
                        partial_einsum[lid] = lg
                    else:
                        partial[lid] = lg
            self._sb_toc("score", _t)
            # PROBE (SGLANG_PATHP_SCORE_ONLY=1): run the score (touches CUDA / occupies GPU) but
            # SKIP finalize+scratch+scatter+recall+publish (writes NOTHING the decode graph reads).
            # Discriminates the run=1 tax: if run=1 -> ~112, the tax is the buffer-WRITE / cross-
            # stream sync (FIXABLE via proper event mgmt); if still ~83, it's the score's GPU
            # occupancy (needs work-reduction / MPS). Output wrong — speed-isolation only.
            if os.environ.get("SGLANG_PATHP_SCORE_ONLY", "0") == "1":
                continue
            # validation: compare resident-set selection (paged vs einsum) for this req.
            if _paged_mode == "validate" and len(partial) == len(self.target_layers) \
                    and len(partial_einsum) == len(self.target_layers):
                self._log_paged_overlap(partial, partial_einsum, n_blk, ri)
            if len(partial) != len(self.target_layers):
                if self._cycle_count < 3:
                    logger.info(
                        f"[ResidentMask] cycle skip req={ri} n_blk={n_blk}: "
                        f"partial={list(partial.keys())} (need {self.target_layers})"
                    )
                continue
            # ── ensemble + threshold + recent/sink + capacity (reuse hook logic) ──
            # PAGE-RECALL: select top-K_max densest c4 pages instead; the returned keep is the
            # page-expanded chunk mask (all chunks of selected pages) and the score buffers
            # (id_page_table/pool_loc/count) are populated for the live-pool indexer. The
            # chunk-granular scratch path is skipped entirely (no _build_resident_scratch).
            _t = self._sb_tic()
            if self._page_recall:
                self._ensure_page_recall_buffers(token_to_kv_pool)
                keep = self._select_resident_pages(
                    token_to_kv_pool, bi, ri, n_blk, partial, page_table
                )  # bool[n_blk], page-expanded; also staged count + id_page_table + pool_loc
                self._sb_toc("pageselect", _t)
            else:
                keep = self._hook._finalize_resident(partial, n_blk, self.device, ri)  # bool[n_blk]
                self._sb_toc("finalize", _t)
            # ASYNC FIX (2026-06-29): keep.sum().item() is a D2H SYNC — on the bg thread it
            # blocks holding the CUDA driver lock + GIL, stalling the main decode thread's next
            # graph launch (run=1: 83 vs 112 ceiling). It was used ONLY for the throttled log
            # line below, so compute it ONLY when we actually log (≤5 cycles or every 200).
            _do_log = self._cycle_count < 5 or self._cycle_count % 200 == 0
            if _do_log:
                n_keep = int(keep.sum().item())
                logger.info(
                    f"[ResidentMask] cycle#{self._cycle_count} req={ri} n_blk={n_blk} "
                    f"resident={n_keep} ({n_keep/max(n_blk,1):.1%}) under cuda graph"
                )
            if self._dump_path is not None:
                kid = keep.nonzero(as_tuple=True)[0].to("cpu").tolist()
                self._dump_f.write(json.dumps({
                    "cycle": self._cycle_count, "req": int(ri), "step": self._req_step[ri] - 1,
                    "n_blk": n_blk, "n_keep": int(keep.sum().item()),
                    "resident_ids": kid,
                }) + "\n")
                self._dump_f.flush()
            # ── Step-1: build scratch FIRST (scratch + pool_loc), then scatter mask; finally
            # publish resident_count LAST so the in-graph indexer never sees a new count with
            # a stale scratch (count is the gate _score_resident_active reads).
            # PAGE-RECALL already populated id_page_table/pool_loc/staged-count in
            # _select_resident_pages above and needs NO scratch (scores the live pool).
            if self._score_resident and not self._page_recall:
                _t = self._sb_tic()
                self._build_resident_scratch(
                    token_to_kv_pool, bi, ri, n_blk, keep, page_table
                )
                self._sb_toc("scratch", _t)
            _sb_yield()   # release GIL after the 21-layer scratch copy burst
            # PROBE (SGLANG_PATHP_SKIP_AFTER_SCRATCH=1): do score+finalize+scratch (writes scratch
            # + pool_loc), but SKIP scatter(mask)+recall+publish(col_to_cell/count/meta/valid).
            # Bisects the per-step tax: combined with SCORE_ONLY(=112) and full(=83), if this
            # gives 112 the tax is in scatter/recall/publish (not scratch); if 83, scratch is it.
            if os.environ.get("SGLANG_PATHP_SKIP_AFTER_SCRATCH", "0") == "1":
                continue
            # ── map per-req LOGICAL keep -> GLOBAL pool locs, scatter into mask ──
            _t = self._sb_tic()
            kept_locs = self._scatter_logical_keep(bi, n_blk, keep, page_table)
            self._sb_toc("scatter", _t)
            _sb_yield()
            if kept_locs is not None and kept_locs.numel() > 0:
                batched_kept.append(kept_locs)
        # ── Phase-2 BATCHED recall over the union of all due reqs' kept locs (one call). ──
        union_locs = None
        if batched_kept:
            union_locs = (batched_kept[0] if len(batched_kept) == 1
                          else torch.cat(batched_kept))
            _t = self._sb_tic()
            # PROBE (SGLANG_PATHP_SKIP_RECALL=1): skip ONLY the recall (H2D + CPU gather + D2H
            # .tolist) to bisect whether the bg-thread tax that OFE_SKIP eliminated is the recall
            # (the heavy/syncing part) vs score/scatter. Output wrong (reserve not refreshed) —
            # speed-isolation only.
            if os.environ.get("SGLANG_PATHP_SKIP_RECALL", "0") != "1":
                self._recall_resident_to_reserve(token_to_kv_pool, union_locs)
            self._sb_toc("recall", _t)
        # Path-I: after recall populated chunk_cell_lut for the recalled pages, remap this
        # cycle's resident_id_page_table (full-history phys pages) -> reserve-block ids into
        # resident_id_page_table_reserve, which the in-graph indexer reads under index-K
        # offload. No-op when offload off. Must be AFTER recall (needs fresh lut).
        if self._index_k_offload and self._page_recall:
            _t = self._sb_tic()
            self._remap_index_page_table(token_to_kv_pool, due)
            self._sb_toc("idx_remap", _t)
        # ── ORPHAN RECLAIM v4 (2026-06-30): free reserve cells whose chunk is no longer resident
        # for ANY request, using the AUTHORITATIVE per-chunk mask (resident_chunk_mask). v1-v3
        # failed on the live-set source: resident_pool_loc_buf is CAPPED at resident_buf (2048 <<
        # real ~5991), and union_locs only had the 1 due req/call → both freed live chunks → recall
        # re-copied (17-27K/cycle). resident_chunk_mask[c]==True ⟺ chunk c is kept by SOME active
        # request (set per-req every cycle by _scatter_logical_keep) = ground truth, no cap, no
        # per-call gap. Without reclaim the shifting recency tail fills the reserve to capacity
        # (983039) → eviction storm → recall ~75ms/cycle (the dominant cost). Gated by RECLAIM_ORPHANS.
        if os.environ.get("SGLANG_PATHP_RECLAIM_ORPHANS", "0") == "1":
            eng = getattr(token_to_kv_pool, "_swap_engine_p", None)
            if (eng is not None and hasattr(eng, "reclaim_orphans_global")
                    and self.resident_chunk_mask is not None):
                try:
                    _t = self._sb_tic()
                    n_rec = eng.reclaim_orphans_global(self.resident_chunk_mask)
                    self._sb_toc("reclaim", _t)
                    if self._cycle_count < 80 and n_rec > 0:
                        logger.info(f"[ResidentMask] reclaimed {n_rec} orphan reserve cells")
                except Exception as e:
                    if self._cycle_count < 5:
                        logger.warning(f"[ResidentMask] reclaim_orphans failed: {e}")
        # ── (legacy resident_pool_loc_buf reclaim path removed: it was capped → freed live chunks) ──
        # ── Step-2: rebuild deep_gemm meta + validity mask once for this batch. ──
        B = self.resident_count_buf.shape[0] if self.resident_count_buf is not None else 0
        if self._score_resident and self.resident_count_buf is not None and B > 0:
            # PUBLISH-ORDER (concurrency OOB fix): build col_to_cell + valid_mask FIRST (from
            # the STAGED counts / pool_loc, NOT the live resident_count_buf), THEN publish the
            # staged counts LAST. This guarantees that when the in-graph decode (default stream)
            # observes resident_count_buf[bi]>0, the matching col_to_cell_buf[bi] cells are
            # already written → no stale-cell read → no CUDA illegal access under request churn.
            # FUSED-REMAP: col->reserve-cell table (reads pool_loc + lut; count-independent).
            if os.environ.get("SGLANG_PATHP_FUSED_REMAP", "0") == "1":
                self._refresh_col_to_cell(token_to_kv_pool)
            # validity mask + dg meta from the STAGED counts (so they match what we publish).
            self._publish_pending_counts()       # writes resident_count_buf from _pending_count
            _t = self._sb_tic()
            self._rebuild_resident_dg_meta(B)
            self._sb_toc("meta", _t)
            # validity mask [max_bs, buf] = (col < resident_count): precomputed ONCE per
            # cycle here so the in-graph topk skips rebuilding arange()+compare ×21/step
            # (measured 3.2ms→1.1ms/step at the topk block). resident_count is frozen for
            # the next `interval` steps, so this mask is valid for the whole window. Fixed
            # address (the captured graph baked it) — refresh via copy_, never realloc.
            self._refresh_valid_mask()
        else:
            # score-resident off but counts may have been staged (defensive): drop them.
            self._pending_count.clear()

    @torch.no_grad()
    def _reset_changed_rows(self, req_ids, B, token_to_kv_pool):
        """CHURN-CRASH FIX (D PD-decode): reset score-resident per-row buffers for any decode
        row whose occupant changed since last step. On D, prefill/extend (and thus
        reset_req_residency) never runs, and filter_batch compacts rows on request finish, so a
        reused/shifted row would otherwise carry the previous occupant's count>0 + stale
        col_to_cell reserve-cell pointers → in-graph gather derefs a freed cell → illegal access.

        A row is 'changed' if (a) it now holds a different req_pool_idx than last step (finish+
        reuse OR compaction shift), or (b) it's newly within B (batch grew). For each changed
        row: zero its count, blank its pool_loc/col_to_cell to the -1 sentinel, and force its new
        occupant's cadence to step 0 so the next side-band re-finalizes it from scratch (recency-
        tail / SWA covers the ≤1-cycle gap, same guarantee as the prefill-side reset). Also clear
        the new occupant's stale _req_step so a leftover counter can't delay its first finalize.

        Pure GPU scatter on the (usually 0-few) changed rows + a host dict update. Out of graph.
        """
        if self.resident_count_buf is None:
            return
        prev = getattr(self, "_prev_row_occupant", None)
        changed = []
        for bi in range(min(B, self.max_bs)):
            ri = req_ids[bi]
            if prev is None or bi >= len(prev) or prev[bi] != ri:
                changed.append(bi)
        # PHANTOM-ROW FIX (the actual hang root, 2026-06-29): the deep_gemm score kernel is given
        # a side-band-built schedule_meta computed over resident_count_buf[:max_bs] (all 128 rows)
        # but runs with q_fp8[:_B] / count[:_B] at the live (padded) batch _B. If ANY row j>=_B
        # still has a stale count>0 (a finished request's row now beyond the shrunk batch), the
        # 128-row meta schedules SM blocks for phantom row j → the kernel reads context_lens[j]
        # OUT OF BOUNDS of its _B-row count tensor → GPU hang (surfaces as the next .tolist() D2H
        # spinning forever; observed at run=4 churn, never at run=1 where no row is ever vacated).
        # So ALSO zero every row at/after the current batch size. Cheap (one slice fill).
        if B < self.max_bs:
            # release any leaked cells held by rows now beyond the live batch (finished reqs
            # whose row wasn't reused by a shift), then zero them. Idempotent (locs already -1
            # are skipped). Bounds _g_chunk_to_cell to live requests → keeps eviction scan cheap.
            eng_t = getattr(token_to_kv_pool, "_swap_engine_p", None) if token_to_kv_pool else None
            if eng_t is not None and hasattr(eng_t, "release_req_pool_locs"):
                tail_locs = self.resident_pool_loc_buf[B:]
                tail_locs = tail_locs[tail_locs >= 0]
                if tail_locs.numel() > 0:
                    try:
                        eng_t.release_req_pool_locs(tail_locs)
                    except Exception:
                        pass
            self.resident_count_buf[B:] = 0
            self.resident_pool_loc_buf[B:] = -1
            if getattr(self, "col_to_cell_buf", None) is not None:
                self.col_to_cell_buf[B:] = -1
            # PAGE-RECALL: also zero stale page ids for phantom rows (defensive; count=0 already
            # gates these rows off the score-resident path, but keep page_table consistent).
            if self._page_recall and self.resident_id_page_table is not None:
                self.resident_id_page_table[B:] = 0
        if changed:
            idx = torch.tensor(changed, dtype=torch.long, device=self.device)
            # CELL-LEAK FIX (residual run=70 hang, 2026-06-29): on D the prefill-side
            # release_req_pool_locs never runs, so a FINISHED request's reserve cells were never
            # returned to the engine free-list → _g_chunk_to_cell grew unbounded (90K+ entries at
            # run=70) → _recall_global's _evict_one_global does a Python `for c in c2c` linear scan
            # PER new chunk → O(new × dict_size) → the side-band stalls for minutes → the main
            # thread's prev.join() in on_forward_end waits forever (py-spy: side-band stuck in
            # _evict_one_global). When a row's occupant changes, the OLD occupant's resident pool
            # locs are still in resident_pool_loc_buf[bi] (before we overwrite to -1) — release
            # them so their cells return to the free-list, bounding c2c to live requests.
            eng = getattr(token_to_kv_pool, "_swap_engine_p", None) if token_to_kv_pool else None
            if eng is not None and hasattr(eng, "release_req_pool_locs"):
                old_locs = self.resident_pool_loc_buf[idx]          # [n_changed, buf], -1 padded
                old_locs = old_locs[old_locs >= 0]
                if old_locs.numel() > 0:
                    try:
                        eng.release_req_pool_locs(old_locs)
                    except Exception as e:
                        if self._cycle_count < 5:
                            logger.warning(f"[ResidentMask] release_req_pool_locs failed: {e}")
            self.resident_count_buf[idx] = 0
            self.resident_pool_loc_buf[idx] = -1
            if getattr(self, "col_to_cell_buf", None) is not None:
                self.col_to_cell_buf[idx] = -1
            if self._page_recall and self.resident_id_page_table is not None:
                self.resident_id_page_table[idx] = 0
            # the in-graph indexer gates score-resident on resident_count_buf[bi]>0, so count=0
            # alone makes the row fall back to the full-history path (safe) until re-finalized.
            for bi in changed:
                ri = req_ids[bi]
                self._req_step[ri] = 0          # next step is this occupant's step-0 → due → finalize
                # drop any stale staged count for this row (it referred to the old occupant)
                self._pending_count.pop(bi, None)
                # STAGGER (2026-06-30): assign a round-robin cycle PHASE so co-landing requests
                # don't all recall on the same step. On D the prefill-side reset_req_residency
                # (which assigns phase) NEVER runs, so without this every req keeps phase=0 and
                # all recalls align on step%interval==0 → one big synchronized side-band recall
                # dip (measured: throughput swings ~900<->2400 bimodally at run=72 barrier). By
                # spreading phases across [0,interval) the dips smear into a steady ~1/interval-
                # per-step trickle → sustained throughput rises toward the no-recall-step ceiling
                # (~2400 = 1.8x baseline) instead of being pulled to the dip floor. Mirrors the
                # prefill-side stagger; gated by SGLANG_PATHP_STAGGER_CYCLE (default off).
                if self._stagger_cycle:
                    self._req_phase[ri] = self._next_phase % self.interval
                    self._next_phase += 1
            if self._cycle_count < 50:
                logger.info(
                    f"[ResidentMask] reset {len(changed)} changed decode rows "
                    f"(churn): rows={changed[:8]}{'...' if len(changed) > 8 else ''}"
                )
        # META CONSISTENCY (the hang root, 2026-06-29): the deep_gemm score kernel is a
        # PERSISTENT grid-scheduled kernel — it reads schedule_meta (built from the count
        # distribution) and the count tensor, and its SM blocks loop until the scheduled work
        # is consumed. The meta is normally rebuilt only at 64-step CYCLE boundaries (side-band),
        # but a request turnover changes resident_count MID-window (we just zeroed rows above).
        # A meta that no longer matches the live counts makes the persistent kernel wait forever
        # for work that the (now-smaller) counts never produce → cuda-graph replay() hangs (py-
        # spy: stuck in graphs.py replay; never crashes — it's a deadlock, not an OOB). So
        # whenever we mutate counts here, REBUILD the meta + valid_mask to match, keeping the
        # frozen-for-the-window invariant (meta ⟺ count) intact. Out of graph, fixed-address.
        if changed or B < self.max_bs:
            self._rebuild_resident_dg_meta(self.resident_count_buf.shape[0])
            self._refresh_valid_mask()
        # snapshot occupants for next step's diff
        self._prev_row_occupant = list(req_ids[:B])

    @torch.no_grad()
    def _publish_pending_counts(self):
        """Publish staged per-req K into resident_count_buf (the gate the in-graph indexer reads
        for score-resident). Called LAST in _sideband_work — AFTER col_to_cell is refreshed — so
        count>0 ⟺ col_to_cell cells are fresh (no stale-cell OOB under request churn). Rows not
        in this cycle's due set keep their previous count (sticky); reset clears finished slots."""
        if not self._pending_count or self.resident_count_buf is None:
            self._pending_count.clear()
            return
        for bi, K in self._pending_count.items():
            if 0 <= bi < self.resident_count_buf.shape[0]:
                self.resident_count_buf[bi] = K
        self._pending_count.clear()

    def _get_recall_stream(self):
        """Lazy dedicated CUDA stream for async recall (Part-2). Its .tolist() D2H sync and
        H2D copies run here so they don't serialize the main stream's queued decode forwards.
        Created on THIS rank's device (not the bg thread's default cuda:0).

        ASYNC FIX (2026-06-29): SGLANG_PATHP_RECALL_STREAM_PRIO sets this stream's CUDA priority
        so the GPU scheduler favors the MAIN decode stream (default prio 0) over the side-band's
        GPU work. "low" (default) = least-priority (decode preempts side-band at the HW-queue
        level → fixes the per-step ~3ms decode-graph completion delay caused by side-band kernels
        sharing the GPU context); "high" = greatest-priority; "none" = default (old behavior)."""
        s = getattr(self, "_recall_stream", None)
        if s is None:
            dev_idx = self.resident_chunk_mask.device.index
            _prio_mode = os.environ.get("SGLANG_PATHP_RECALL_STREAM_PRIO", "none")
            if _prio_mode == "none":
                s = torch.cuda.Stream(device=dev_idx)
            else:
                # CUDA stream priority: LOWER int = HIGHER priority. Typical range [-N, 0];
                # 0 = least priority (default stream). "low"=least(0)/large, "high"=greatest(-1).
                _prio = 0 if _prio_mode == "low" else -1
                try:
                    s = torch.cuda.Stream(device=dev_idx, priority=_prio)
                    logger.info(f"[ResidentMask] recall stream prio={_prio_mode} (val={_prio})")
                except Exception as e:
                    logger.warning(f"[ResidentMask] recall stream prio failed ({e}); default")
                    s = torch.cuda.Stream(device=dev_idx)
            self._recall_stream = s
        return s

    @torch.no_grad()
    def _refresh_valid_mask(self):
        """Per-cycle: valid_mask_buf[r, j] = (j < resident_count_buf[r]) for all rows.
        Read by the in-graph score-resident topk to mask padded columns. Fixed-address."""
        buf = self.resident_buf
        if getattr(self, "valid_mask_buf", None) is None or self.valid_mask_buf.shape[1] != buf:
            self.valid_mask_buf = torch.zeros(
                (self.max_bs, buf), dtype=torch.bool, device=self.device
            )
            self._valid_cols = torch.arange(buf, device=self.device).unsqueeze(0)  # [1,buf]
        cnt = self.resident_count_buf.unsqueeze(1)            # [max_bs,1]
        self.valid_mask_buf.copy_(self._valid_cols < cnt)     # broadcast → [max_bs,buf]

    @torch.no_grad()
    def _remap_index_page_table(self, token_to_kv_pool, due):
        """Path-I (index-K offload): rewrite each due req's resident_id_page_table (K_max
        full-history PHYSICAL c4 page ids) into resident_id_page_table_reserve (K_max
        RESERVE-block ids), using the c4 recall's chunk_cell_lut. For a physical page P, its
        chunk-0 logical loc is P*64; chunk_cell_lut[0][P*64] = the reserve CELL that recall
        copied page P's chunk-0 into; the reserve BLOCK = cell//64 (page-block recall copies
        the whole page into one aligned 64-cell block, so cell = block*64 + offset). The
        index-K reserve stores whole pages at block granularity, so the indexer's page_table
        entry for page P must be that reserve block. Non-resident / dummy pages (lut<0) map to
        block 0 (a safe dummy; its cols are masked out by valid_mask/count, same as page 0 in
        the full-history path). Out of graph (side-band), fixed address (copy_ into the
        pre-alloc buffer the captured indexer baked). Runs AFTER recall (needs fresh lut)."""
        eng = getattr(token_to_kv_pool, "_swap_engine_p", None)
        if (eng is None or getattr(eng, "chunk_cell_lut", None) is None
                or self.resident_id_page_table_reserve is None):
            return
        ps = self.c4_page_size
        lut0 = eng.chunk_cell_lut[0]                              # [n_logical_chunks] int32, layer-uniform
        # phys pages this cycle for the due reqs; convert each to its reserve block.
        for bi in due:
            phys = self.resident_id_page_table[bi].to(torch.int64)   # [pages] full-history phys page ids
            chunk0 = (phys * ps).clamp_(0, lut0.numel() - 1)          # each page's chunk-0 loc
            cell = lut0[chunk0].to(torch.int64)                       # reserve cell (or -1 if not resident)
            block = torch.where(cell >= 0, cell // ps, torch.zeros_like(cell))  # reserve block; -1->0 dummy
            self.resident_id_page_table_reserve[bi] = block.to(torch.int32)

    @torch.no_grad()
    def _refresh_col_to_cell(self, token_to_kv_pool):
        """FUSED-REMAP per-cycle: col_to_cell_buf[bi, j] = reserve cell for the resident chunk
        at compacted column j of req bi, or -1 if non-resident. Composes the two mappings the
        decode used to do per-layer:
            col j --(resident_pool_loc_buf)--> physical pool loc --(chunk_cell_lut[0])--> cell.
        Path-P writes chunk_cell_lut[:, loc]=cell for ALL 21 c4 layers at once (layer-uniform),
        so layer 0's lut is exact for every layer → ONE shared table. Out of graph (side-band),
        fixed address (copy_ into the pre-alloc buffer the captured graph baked). Invalid/padded
        cols (resident_pool_loc_buf=-1) and not-yet-resident locs (lut<0) both resolve to -1,
        which the read path treats as masked-out (same sentinel as topk padding)."""
        eng = getattr(token_to_kv_pool, "_swap_engine_p", None)
        if eng is None or getattr(eng, "chunk_cell_lut", None) is None:
            return
        lut0 = eng.chunk_cell_lut[0]                           # [n_logical_chunks] int32, layer-uniform
        pool_loc = self.resident_pool_loc_buf                  # [max_bs, buf] int64 (-1 padded)
        valid = pool_loc >= 0
        idx = torch.where(valid, pool_loc, torch.zeros_like(pool_loc))
        idx = idx.clamp_(0, lut0.numel() - 1)
        cells = lut0[idx].to(torch.int64)                      # -1 where not resident
        out = torch.where(valid & (cells >= 0), cells,
                          torch.full_like(pool_loc, -1))
        self.col_to_cell_buf.copy_(out)

    def _sb_tic(self):
        """Start a CUDA-event timer (returns (start,end) event pair) if SB timing on."""
        if not self._sb_timing:
            return None
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        return (s, e)

    def _sb_toc(self, phase, ev):
        """Stop the timer + accumulate ms into phase. Syncs the end event (side-band is
        out-of-graph so a sync here is legal). Cheap relative to the phases measured."""
        if ev is None:
            return
        s, e = ev
        e.record()
        e.synchronize()
        self._sb_acc[phase] = self._sb_acc.get(phase, 0.0) + s.elapsed_time(e)

    def _sb_report(self, B, n_due):
        """Print accumulated per-phase ms every 50 timed calls + reset."""
        if not self._sb_timing:
            return
        self._sb_n += 1
        if self._sb_n % 50 != 0:
            return
        tot = sum(self._sb_acc.values())
        parts = " ".join(f"{k}={v/50:.1f}" for k, v in sorted(self._sb_acc.items()))
        logger.info(
            f"[SB-TIMING] over 50 decode on_forward_end calls (last B={B} due={n_due}): "
            f"avg total={tot/50:.1f}ms/call  |  {parts}  (ms/call)"
        )
        self._sb_acc = {}

    @torch.no_grad()
    def _rebuild_resident_dg_meta(self, B):
        """Side-band: compute the deep_gemm paged-MQA SM-schedule metadata for the current
        resident_count[:B], once per cycle (out of graph). The in-graph indexer slices it
        per its own batch. No-op for torch/tilelang/chunk kernels (they ignore meta)."""
        if (os.environ.get("SGLANG_OPT_USE_TILELANG_INDEXER", "0") == "1"
                or os.environ.get("SGLANG_FP8_PAGED_MQA_LOGITS_TORCH", "0") == "1"
                or os.environ.get("SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE", "-1") != "-1"):
            self.resident_dg_meta = None
            return
        try:
            import deep_gemm as _dg
            cnt = self.resident_count_buf[:B].to(torch.int32).unsqueeze(-1)  # [B,1]
            meta = _dg.get_paged_mqa_logits_metadata(
                cnt, self.c4_page_size, _dg.get_num_sms()
            )
            # Copy into a FIXED-address buffer so the captured graph (which baked the
            # capture-time address) keeps reading the live schedule across cycles. A fresh
            # tensor each cycle would change addresses → graph reads stale meta.
            if self.resident_dg_meta is None or self.resident_dg_meta.shape != meta.shape:
                self.resident_dg_meta = torch.empty_like(meta)
            self.resident_dg_meta.copy_(meta)
        except Exception as e:
            if self._cycle_count < 3:
                logger.warning(f"[ResidentMask] resident_dg_meta build failed: {e}")
            self.resident_dg_meta = None

    @torch.no_grad()
    def _recall_resident_to_reserve(self, token_to_kv_pool, kept_locs) -> None:
        """Phase-2: swap the resident_set (GLOBAL pool locs, kept this cycle) into the
        Path-P reserve for ALL 21 c4 layers, so that during the next 64-step window every
        layer's native top-512 (confined to the resident_set by the mask) finds its KV
        resident in the reserve (remap hits ~100%). LRU + 63/64 window overlap → each cycle
        only the delta is actually copied. No-op when Path-P is not active (no engine).

        2026-06-24: ONE global call (recall_resident_global) replaces the old per-layer loop
        `for clid in clids: eng.recall_chunks(clid, kept_locs)`. The resident_set is identical
        for all 21 c4 layers (the mask confines each layer's top-512 to it), so the old loop
        redid torch.unique + a D2H .tolist() sync + a Python alloc 21× — ~9.4ms/cycle that
        serialized the scheduler loop and triggered the barrier=16 watchdog kill. The global
        core does ONE unique + ONE D2H + ONE lut scatter across all layers + 21 sync-free byte
        copies (bytes are still per-layer; only the alloc plan is shared)."""
        eng = getattr(token_to_kv_pool, "_swap_engine_p", None)
        if eng is None or kept_locs is None or kept_locs.numel() == 0:
            return
        n_copied = eng.recall_resident_global(kept_locs)
        self._recall_cycles = getattr(self, "_recall_cycles", 0) + 1
        if self._recall_cycles <= 5 or self._recall_cycles % 200 == 0:
            n_clids = len(self._c4_compress_layer_ids(token_to_kv_pool))
            logger.info(
                f"[ResidentMask] Phase-2 recall #{self._recall_cycles}: "
                f"resident={int(kept_locs.numel())} chunks -> reserve "
                f"({n_clids} c4 layers, global; copied={n_copied} chunk-rows)"
            )

    @torch.no_grad()
    def prebuild_for_capture(self, token_to_kv_pool) -> None:
        """(A) Pre-build ALL score-resident fixed-address buffers BEFORE cuda-graph capture,
        so the captured decode graph bakes the SCORE-RESIDENT branch (not full-history+mask).

        WHY (proven 2026-06-29 via CAP-PROBE): with --skip-server-warmup, capture runs at boot
        when resident_kv_scratch is None → the indexer's _score_resident_active gate is False at
        capture → the graph freezes the full-history-score + apply_mask + remap path forever
        (forward_c4_indexer runs ONLY at capture under PREP_IN_CUDA_GRAPH, then pure-replays).
        The entire score-resident machinery was therefore DEAD under cuda graph. This method
        flips that: by allocating scratch (+ count/pool_loc/col_to_cell/id_pt/valid_mask/meta)
        here, _score_resident_active is True at capture → the graph bakes score-resident +
        (if enabled) fused-remap. Call from model_runner AFTER init_resident_mask_capturer and
        BEFORE init_device_graphs.

        Capture-time SAFETY: count_buf=0 → the captured score-resident path scores 0 valid cols
        → topk → all -1 → all chunks masked-out (no OOB; the capture forward uses dummy batches
        anyway). At runtime the side-band fills count/pool_loc/col_to_cell per cycle and the
        SAME baked graph reads the live buffers. No realloc ever (all writes are copy_/in-place).

        No-op when score-resident is off (gate-off path is unchanged → zero risk)."""
        if not self._score_resident:
            logger.info("[ResidentMask] prebuild_for_capture: score-resident OFF, skip "
                        "(graph will bake full-history+mask path, as before).")
            return
        # GATE (SGLANG_PATHP_PREBUILD_CAPTURE, default OFF = known-good): only pre-build (→ capture
        # score-resident) when explicitly enabled. Default off keeps the proven full-history+mask
        # path captured (night DEMO70 ran ~1180 tok/s stable). Capturing score-resident currently
        # has an UNSOLVED concurrency OOB at request churn (2026-06-29), so it's opt-in for debug.
        if os.environ.get("SGLANG_PATHP_PREBUILD_CAPTURE", "0") != "1":
            logger.info("[ResidentMask] prebuild_for_capture: PREBUILD_CAPTURE!=1, skip "
                        "(graph bakes full-history+mask = known-good). Set =1 to capture "
                        "score-resident (WIP: churn OOB).")
            return
        if self._hook is None:
            logger.warning("[ResidentMask] prebuild_for_capture: no hook (no INLINE_CKPT?) — "
                           "scratch built but Level-1 disabled; captured path scores resident "
                           "scratch (all-masked until a real finalize).")
        # 1. score buffers (fixed addresses). PAGE-RECALL allocates the lean buffer set (no KV
        #    scratch — it scores the live pool with a real page_table); the chunk-granular path
        #    allocates the per-req dense KV scratch.
        if self._page_recall:
            self._ensure_page_recall_buffers(token_to_kv_pool)
        else:
            self._ensure_resident_scratch(token_to_kv_pool)
        # 2. valid_mask_buf (read by the in-graph custom topk; native topk uses count+id_pt).
        self._refresh_valid_mask()
        # 3. dg meta: build for the full max_bs so the captured kernel has a valid schedule
        #    buffer address. count=0 rows → meta still valid (kernel self-schedules 0 work).
        try:
            self._rebuild_resident_dg_meta(self.max_bs)
        except Exception as e:
            logger.warning(f"[ResidentMask] prebuild dg_meta failed (ok, kernel self-sched): {e}")
        clids = list(self.resident_kv_scratch.keys()) if self.resident_kv_scratch else []
        logger.info(
            f"[ResidentMask] prebuild_for_capture DONE: "
            f"{'PAGE-RECALL (live-pool, no scratch)' if self._page_recall else f'scratch for {len(clids)} c4 layers'}, "
            f"buf={self.resident_buf}, count=0 (all-masked until first finalize). "
            f"score-resident WILL be captured (gate=True at capture). "
            f"fused_remap={os.environ.get('SGLANG_PATHP_FUSED_REMAP','0')}"
        )

    @torch.no_grad()
    def _ensure_resident_scratch(self, token_to_kv_pool):
        """Lazy-alloc the score-resident scratch buffers (first build). One PAGED KV
        scratch per c4 layer in the EXACT layout the indexer kernel reads
        (index_k_with_scale_buffer[clid] is [n_pages, page_bytes] uint8; the kernel views
        it [n_pages, 64, 1, 132], indexer.py:378). We give each request `_resident_pages`
        contiguous scratch pages, addressed by an identity page_table."""
        if self.resident_kv_scratch is not None:
            return
        clids = self._c4_compress_layer_ids(token_to_kv_pool)
        self.resident_kv_scratch = {}
        for clid in clids:
            mlid = self._model_layer_of_compress(token_to_kv_pool, clid)
            buf = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=mlid)
            page_bytes = buf.shape[1]           # = 64 * 132 (per-chunk interleaved)
            self.resident_kv_scratch[clid] = torch.zeros(
                (self.max_bs * self._resident_pages, page_bytes),
                dtype=buf.dtype, device=self.device,
            )
        self.resident_count_buf = torch.zeros(self.max_bs, dtype=torch.int32, device=self.device)
        self.resident_pool_loc_buf = torch.full(
            (self.max_bs, self.resident_buf), -1, dtype=torch.int64, device=self.device
        )
        # FUSED-REMAP (SGLANG_PATHP_FUSED_REMAP=1): per-cycle table col -> reserve CELL,
        # collapsing decode's two per-layer mappings (col->phys pool loc via resident_pool_loc_buf,
        # then phys loc->reserve cell via remap_compressed_locs ×21) into ONE in-graph gather.
        # The Path-P cell mapping is GLOBAL/layer-uniform (swap_engine_p._recall_global writes
        # chunk_cell_lut[:, locs]=cell for ALL 21 layers at once), so ONE shared table suffices.
        # Built side-band (out of graph) right after recall fills the lut; fixed-address so the
        # captured graph keeps reading the live cells across cycles. -1 sentinel = non-resident.
        self.col_to_cell_buf = torch.full(
            (self.max_bs, self.resident_buf), -1, dtype=torch.int64, device=self.device
        )
        # identity page_table: row bi -> its own contiguous scratch pages.
        base = (torch.arange(self.max_bs, device=self.device).unsqueeze(1)
                * self._resident_pages)
        cols = torch.arange(self._resident_pages, device=self.device).unsqueeze(0)
        self.resident_id_page_table = (base + cols).to(torch.int32)  # [max_bs, pages]
        logger.info(
            f"[ResidentMask] score-resident scratch: {len(clids)} c4 layers x "
            f"[{self.max_bs * self._resident_pages}, {page_bytes}] uint8, "
            f"buf={self.resident_buf} chunks ({self._resident_pages} pages/req)"
        )

    @torch.no_grad()
    def _ensure_page_recall_buffers(self, token_to_kv_pool):
        """Lazy-alloc the per-req buffers page-recall shares with the score-resident path,
        WITHOUT the dense KV scratch (page-recall scores the LIVE index-K pool with a real
        page_table, so no packed scratch is needed). Buffers (fixed-address, read by the
        captured graph): resident_id_page_table [max_bs, K_max] int32 (the K_max selected
        PHYSICAL c4 page ids per req, padded with a safe page), resident_count_buf [max_bs]
        (K*64 valid cols), resident_pool_loc_buf [max_bs, BUF] (col -> physical chunk loc),
        valid_mask_buf (col < count). resident_kv_scratch stays None on this path."""
        if self.resident_pool_loc_buf is not None:
            return
        self.resident_count_buf = torch.zeros(self.max_bs, dtype=torch.int32, device=self.device)
        self.resident_pool_loc_buf = torch.full(
            (self.max_bs, self.resident_buf), -1, dtype=torch.int64, device=self.device
        )
        # page_table holds K_max PHYSICAL page ids per req (NOT identity). 0-init = page 0 is a
        # safe dummy (its cols are masked out by valid_mask/count, never selected by topk).
        self.resident_id_page_table = torch.zeros(
            (self.max_bs, self._resident_pages), dtype=torch.int32, device=self.device
        )
        # Path-I (index-K offload): a PARALLEL page_table holding RESERVE-block ids (not
        # full-history phys pages). Built side-band from chunk_cell_lut after recall
        # (phys_page -> chunk_cell_lut[0][phys_page*64]//64 = reserve block). The in-graph
        # indexer reads THIS (instead of resident_id_page_table) when index-K offload is on,
        # so scoring gathers the SHRUNK index-K reserve. None/unused when offload off (then
        # the indexer reads resident_id_page_table = full-history phys pages, byte-unchanged).
        self.resident_id_page_table_reserve = torch.zeros(
            (self.max_bs, self._resident_pages), dtype=torch.int32, device=self.device
        )
        logger.info(
            f"[ResidentMask] PAGE-RECALL buffers: K_max={self._page_kmax} pages/req "
            f"(buf={self.resident_buf} cols), id_page_table [{self.max_bs}, "
            f"{self._resident_pages}] int32, NO scratch (scores live pool)."
        )

    @torch.no_grad()
    def _select_resident_pages(self, token_to_kv_pool, bi, ri, n_blk, partial, page_table):
        """PAGE-RECALL (replaces _build_resident_scratch under SGLANG_PATHP_PAGE_RECALL):
        score each c4 page by #(sigmoid>thresh) chunks in it, force the recency-tail + sink
        pages, pick the top-K_max densest pages, and populate the score-resident buffers so
        the in-graph indexer scores ONLY those K pages over the LIVE index-K pool (real
        page_table, no scratch). Returns a page-expanded LOGICAL keep mask (bool[n_blk]) =
        every chunk of every selected page — the SAME set fed to _scatter_logical_keep so the
        attention reserve holds exactly the chunks topk can pick (mask⟺reserve lock-step).

        Side-band (out of graph) → host syncs legal. All math mirrors _finalize_resident's
        ensemble + thr + recent|sink, but aggregated to page granularity."""
        ps = self.c4_page_size
        h = self._hook
        # 1. per-chunk ensembled sigmoid (same as _finalize_resident, recomputed locally so we
        #    don't change that method's signature / the gate-off path).
        sigs = torch.stack([torch.sigmoid(partial[l]) for l in h.target_layers], 0)
        ens = sigs.max(0).values if h.ensemble_mode == "or" else sigs.mean(0)   # [n_blk]
        thr = (ens > h.thresh)                                                   # [n_blk] bool
        n_pages = (n_blk + ps - 1) // ps
        # 2. per-page score = #(thr) chunks in the page. scatter_add over page index.
        page_of = (torch.arange(n_blk, device=self.device) // ps)               # [n_blk]
        page_score = torch.zeros(n_pages, device=self.device)
        page_score.scatter_add_(0, page_of, thr.float())                        # [n_pages]
        # 3. FORCE recency-tail + sink pages (score-independent, == _finalize_resident's
        #    recent|sink but at page granularity): give them +inf so topk always keeps them.
        last_start = max(0, n_blk - h.last_keep)
        tail_page0 = last_start // ps
        page_score[tail_page0:] = float("inf")                                  # recency tail pages
        if h.first_keep > 0:
            sink_page1 = (min(h.first_keep, n_blk) + ps - 1) // ps
            page_score[:sink_page1] = float("inf")                              # sink pages
        # 4. top-K_max densest pages (logical page ids ascending for clean offset math).
        K = min(self._page_kmax, n_pages)
        top_logical = torch.topk(page_score, K, largest=True, sorted=False).indices
        top_logical, _ = torch.sort(top_logical)                                # [K] logical pages
        phys_pages = page_table[bi, top_logical].to(torch.int32)                # [K] physical pages
        # HIGH-CONC SAFETY (2026-07-01): hard-clamp phys page ids to the valid physical-page
        # range of the index-K pool. At a barrier release / churn, the captured page_table_buf
        # row may briefly hold a previous occupant's (or uninitialized) entries before
        # capture_side_band_inputs refreshes it; an out-of-range phys page fed to the in-graph
        # deep_gemm (which gathers the REAL index-K pool by page id) → OOB deref → CUDA illegal
        # access (observed crash at concurrency≥65, stable at ≤60). Clamp turns any stale id
        # into an in-bounds (wrong-but-safe) page 0 read; the chunk is masked out downstream by
        # valid_cols / pool_loc=-1, so correctness is unaffected for real (in-range) pages.
        _max_phys_page = max(0, (self.c4_logical_size // ps) - 1)
        phys_pages = phys_pages.clamp_(0, _max_phys_page)
        # 5. populate score buffers (col j in [0,K*ps): page top_logical[j//ps], offset j%ps).
        self.resident_id_page_table[bi].zero_()
        self.resident_id_page_table[bi, :K] = phys_pages
        cols = torch.arange(K * ps, device=self.device)
        col_phys_page = phys_pages[cols // ps].to(torch.int64)                  # [K*ps]
        col_pool_loc = col_phys_page * ps + (cols % ps)                         # physical chunk loc
        self.resident_pool_loc_buf[bi].fill_(-1)
        self.resident_pool_loc_buf[bi, :K * ps] = col_pool_loc
        # last selected page may be PARTIAL (n_blk not a multiple of ps) — mask its OOB cols so
        # topk never picks a non-existent chunk. valid count = #real chunks among the K*ps cols.
        col_logical_chunk = top_logical[cols // ps].to(torch.int64) * ps + (cols % ps)  # logical idx
        valid_cols = (col_logical_chunk < n_blk)                               # [K*ps] bool
        # invalidate OOB cols in pool_loc (so even if selected they map to -1 sentinel downstream)
        self.resident_pool_loc_buf[bi, :K * ps][~valid_cols] = -1
        # resident_count drives the in-graph valid mask; cols beyond K*ps are padding. We keep
        # count = K*ps (a contiguous prefix) and rely on pool_loc=-1 + the score's own validity
        # for the partial-page OOB cols. Stage count LAST (published with col_to_cell, see caller).
        self._pending_count[bi] = K * ps
        # 6. page-expanded LOGICAL keep (bool[n_blk]) = all real chunks of the K selected pages.
        keep = torch.zeros(n_blk, dtype=torch.bool, device=self.device)
        sel_logical = col_logical_chunk[valid_cols]                            # real logical chunks
        keep[sel_logical.clamp(0, n_blk - 1)] = True
        return keep

    @torch.no_grad()
    def _build_resident_scratch(self, token_to_kv_pool, bi, ri, n_blk, keep, page_table):
        """Pack request bi's RESIDENT indexer-K (logical order) into the contiguous scratch
        for ALL c4 layers, and record compacted-col -> physical pool loc. Side-band (out of
        graph), so host syncs are legal. After this, the in-graph indexer (Step-2) scores
        rows [0, K) of this req's scratch segment instead of the full [0, n_blk) history.

        Byte-layout-correct by construction: we read each layer's pool buffer with the SAME
        flat [n_pages*64, page_row] view the kernel uses (one row = one chunk's 132 bytes),
        gather the resident chunks' rows, and scatter them densely into the scratch's flat
        view. No fp8/scale split — we move whole 132-byte chunk rows, so the scratch is
        byte-identical to what the kernel would have read at the resident chunks' positions."""
        self._ensure_resident_scratch(token_to_kv_pool)
        ps = self.c4_page_size
        # resident logical chunk ids (ascending), capped to BUF (finalize already applied
        # capacity, but guard hard so we never overflow the scratch segment).
        keep_dev = keep.to(self.device, dtype=torch.bool)
        res_logical = keep_dev.nonzero(as_tuple=True)[0]            # [K] logical chunk ids
        K = int(res_logical.numel())
        if K > self.resident_buf:
            res_logical = res_logical[: self.resident_buf]
            K = self.resident_buf
        # PUBLISH-ORDER (concurrency OOB fix 2026-06-29): do NOT write resident_count_buf[bi]
        # here. The captured score-resident path gates on count>0 and the fused path then
        # gathers col_to_cell_buf[bi, raw] — but col_to_cell is refreshed only AFTER the batched
        # recall (end of _sideband_work). Publishing count>0 now (before col_to_cell is rebuilt
        # for this req) opens a window where the in-graph decode (default stream) reads a stale
        # col_to_cell → stale reserve cell → CUDA illegal access under request churn. So STAGE
        # K here and publish resident_count_buf LAST, together with col_to_cell (see
        # _publish_pending_counts called at the end of _sideband_work).
        self._pending_count[bi] = K
        if K == 0:
            return
        # logical chunk -> physical pool row index in the layer buffer's flat [n_pages*64,*]
        # view: phys_page = page_table[bi, c//ps]; row = phys_page*ps + c%ps  (== pool_loc).
        phys_page = page_table[bi, res_logical // ps].to(torch.long)
        src_rows = phys_page * ps + (res_logical % ps)             # [K] physical chunk rows
        self.resident_pool_loc_buf[bi, :K] = src_rows
        if K < self.resident_buf:
            self.resident_pool_loc_buf[bi, K:] = -1
        # dense scratch rows for this req: [seg_base, seg_base+K)
        seg_base = bi * self._resident_pages * ps
        _noalloc = os.environ.get("SGLANG_PATHP_SCRATCH_NOALLOC", "1") == "1"
        dst_rows = (torch.arange(K, device=self.device) + seg_base) if not _noalloc else None
        _dummy_src = os.environ.get("SGLANG_PATHP_SCRATCH_DUMMY_SRC", "0") == "1"
        _srow64 = src_rows.to(torch.int64)
        for clid, scratch in self.resident_kv_scratch.items():
            mlid = self._model_layer_of_compress(token_to_kv_pool, clid)
            buf = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=mlid)
            page_bytes = buf.shape[1]
            src_flat = buf.view(-1, page_bytes // ps)              # [n_pages*64, 132]
            dst_flat = scratch.view(-1, page_bytes // ps)          # [max_bs*pages*64, 132]
            if _dummy_src:
                # PROBE: gather from the SCRATCH itself (a static buffer the decode graph does
                # NOT write) instead of the live index-K pool. Same kernel/shape/scatter, but NO
                # read-dependency on the decode-written pool. If run=1→112, the tax is the cross-
                # stream R/W dep on index_k_with_scale_buffer (fixable: main-stream/snapshot);
                # if still 83, the gather/scatter op itself is the cost. Output wrong.
                dst_flat[dst_rows] = dst_flat[src_rows.clamp(0, dst_flat.shape[0] - 1)]
            elif _noalloc:
                # ASYNC FIX (2026-06-29): the advanced-index gather `src_flat[src_rows]` ALLOCATES
                # a [K,132] temp tensor every call ×21 → CUDA caching-allocator churn that
                # serializes with the decode graph's allocator use EVERY step (microbench: gather
                # is 0.26ms/cycle but its EFFECT is ~200ms/cycle = 800x; the amp is the allocator,
                # not the copy). dst_rows is contiguous (arange(K)+seg_base), so index_select
                # straight INTO the contiguous scratch slice = ZERO temp alloc, byte-identical.
                torch.index_select(src_flat, 0, _srow64,
                                   out=dst_flat[seg_base : seg_base + K])
            else:
                dst_flat[dst_rows] = src_flat[src_rows]                # gather resident chunk rows
            # ASYNC FIX: release the GIL between layers so the main decode thread can launch
            # its next graph during this 21-layer gather burst (bg-thread CUDA-launch contention).
            _yf = getattr(self, "_sb_yield_fn", None)
            if _yf is not None:
                _yf()


    def _c4_compress_layer_ids(self, token_to_kv_pool):
        """All c4 (compress_ratio==4) compress_layer_ids, cached. The resident_set is
        global (shared by all 21 c4 layers); each layer owns its own reserve cell space,
        so recall must run per layer."""
        ids = getattr(self, "_c4_clids", None)
        if ids is None:
            ids = []
            lm = getattr(token_to_kv_pool, "layer_mapping", None)
            if lm is not None:
                for m in lm:
                    if getattr(m, "compress_ratio", None) == 4:
                        cid = getattr(m, "compress_layer_id", None)
                        if cid is not None:
                            ids.append(cid)
            self._c4_clids = ids
        return ids

    @torch.no_grad()
    def _extract_compressed_k(self, token_to_kv_pool, compress_layer_id, bi, n_blocks, page_table):
        """Side-band c4-indexer-K extraction (mirror of InlineRetrieverHook._extract_
        compressed_k, but sourced from the POOL + current metadata, not forward-time
        _last_* refs which graph replay never sets). Returns (k_fp8[n_blk,128] uint8,
        k_scale[n_blk,4] uint8) or None."""
        # layer_mapping: model layer_id -> compress_layer_id; we need the model layer_id
        # whose compress id == compress_layer_id to fetch its index-K buffer.
        model_layer_id = self._model_layer_of_compress(token_to_kv_pool, compress_layer_id)
        if model_layer_id is None:
            return None
        kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=model_layer_id)
        # kv_cache: [n_pages, page_size*(head_dim+4)] uint8-equivalent; mirror the
        # dump_training recipe used by the eager hook.
        page_size = self.c4_page_size
        head_dim = _HEAD_DIM
        n_pages = (n_blocks + page_size - 1) // page_size
        pages = page_table[bi, :n_pages]
        kv_flat = kv_cache.view(kv_cache.shape[0], page_size * (head_dim + 4))
        pages_data = kv_flat[pages.long()]
        SCALE_OFFSET = page_size * head_dim
        k_fp8 = pages_data[:, :SCALE_OFFSET].reshape(-1, head_dim)[:n_blocks]
        k_scale = pages_data[:, SCALE_OFFSET:].reshape(-1, 4)[:n_blocks]
        return k_fp8, k_scale

    @torch.no_grad()
    def _score_paged_layer(self, token_to_kv_pool, lid, bi, n_blk, page_table, x_bi, pos):
        """Score req bi's history at target layer `lid` via the PAGED deep_gemm kernel,
        reading the index-K pool DIRECTLY (real page_table) — NO _extract_compressed_k gather.
        The retriever's q (trained projection) is computed + fp8-quantized by the scorer, then
        fed to fp8_paged_mqa_logits exactly like the native indexer. Returns logits[n_blk] or
        None. (User insight 2026-06-29: native indexer scores the paged pool with zero gather;
        the einsum scorer only needed the gather because einsum wants dense K. Same math.)"""
        try:
            import deep_gemm as _dg
        except Exception:
            return None
        mlid = self._model_layer_of_compress(token_to_kv_pool, lid)
        if mlid is None:
            return None
        scorer = self._hook.scorers[lid]
        # q-side: trained projection -> fp8 + fused weights (q_scale folded in).
        q_fp8, fused_w = scorer.compute_q_fp8(x_bi.float(), pos.to(torch.int64))  # [1,1,H,128],[1,H]
        kv = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=mlid)        # [n_pages, page_bytes]
        ps = self.c4_page_size
        n_pages = (n_blk + ps - 1) // ps
        kv_view = kv.view(kv.shape[0], ps, 1, _HEAD_DIM + 4)                      # [n_pages,64,1,132]
        pt = page_table[bi : bi + 1, :n_pages].to(torch.int32)                    # [1, n_pages]
        seq_lens = torch.tensor([[n_blk]], dtype=torch.int32, device=self.device) # [1,1]
        meta = _dg.get_paged_mqa_logits_metadata(seq_lens, ps, _dg.get_num_sms())
        # The deep_gemm kernel supports ONLY 32 or 64 heads, but the retriever (R930) has 128.
        # logits = sum_h relu(k·q_h^T)·w_h is LINEAR over the head-sum and relu is per-head, so
        # we split the H heads into 64-head groups, run the kernel per group, and SUM the
        # [1,n_blk] results — EXACT, no approximation, still ZERO extract (reads paged pool).
        H = q_fp8.shape[2]
        out = None
        for h0 in range(0, H, 64):
            q_g = q_fp8[:, :, h0:h0 + 64, :].contiguous()
            w_g = fused_w[:, h0:h0 + 64].contiguous()
            lg = _dg.fp8_paged_mqa_logits(q_g, kv_view, w_g, seq_lens, pt, meta, n_blk, False)
            lg = lg[0, :n_blk].float()
            out = lg if out is None else out + lg
        if getattr(scorer, "logit_offset", 0.0):
            out = out + scorer.logit_offset
        return out

    @torch.no_grad()
    def _log_paged_overlap(self, partial_paged, partial_einsum, n_blk, ri):
        """Validation: apply the SAME _finalize_resident to both score paths and log the
        resident-set agreement (paged fp8(q)+kernel vs einsum fp32). Reports overlap on the
        THRESHOLD-selected set only (recent|sink are score-independent → always agree), which
        is the part fp8(q) rounding can actually change. Accumulates across cycles."""
        try:
            keep_p = self._hook._finalize_resident(partial_paged, n_blk, self.device, ri)
            keep_e = self._hook._finalize_resident(partial_einsum, n_blk, self.device, ri)
            # threshold-only selection (exclude recent|sink which are score-independent):
            ar = torch.arange(n_blk, device=self.device)
            forced = (ar >= max(0, n_blk - self._hook.last_keep)) | (ar < min(self._hook.first_keep, n_blk))
            thr_p = keep_p & ~forced
            thr_e = keep_e & ~forced
            inter = int((thr_p & thr_e).sum())
            up = int(thr_p.sum()); ue = int(thr_e.sum())
            union = int((thr_p | thr_e).sum())
            # full-set overlap (incl recent|sink) — what attention actually sees:
            f_inter = int((keep_p & keep_e).sum()); f_union = int((keep_p | keep_e).sum())
            acc = self._paged_ov = getattr(self, "_paged_ov", {"thr_i":0,"thr_u":0,"thr_pe":0,"thr_ee":0,"f_i":0,"f_u":0,"n":0})
            acc["thr_i"]+=inter; acc["thr_u"]+=union; acc["thr_pe"]+=up; acc["thr_ee"]+=ue
            acc["f_i"]+=f_inter; acc["f_u"]+=f_union; acc["n"]+=1
            if acc["n"] % 20 == 0:
                thr_jac = acc["thr_i"]/max(acc["thr_u"],1)
                thr_recall = acc["thr_i"]/max(acc["thr_ee"],1)   # of einsum-selected, how many paged also picks
                f_jac = acc["f_i"]/max(acc["f_u"],1)
                logger.info(
                    f"[PAGED-VALIDATE] n={acc['n']} | THRESHOLD-set: jaccard={thr_jac:.4f} "
                    f"recall(einsum∩paged/einsum)={thr_recall:.4f} (paged_sel={acc['thr_pe']} einsum_sel={acc['thr_ee']}) "
                    f"| FULL resident-set (incl recent/sink): jaccard={f_jac:.4f}"
                )
        except Exception as e:
            if self._cycle_count < 5:
                logger.warning(f"[PAGED-VALIDATE] failed: {e}")

    def _model_layer_of_compress(self, token_to_kv_pool, compress_layer_id):
        """Resolve a c4 compress_layer_id -> model layer_id via the pool's layer_mapping
        (cached). CRITICAL: c4 and c128 layers EACH number their compress_layer_id from
        0, so the ids overlap — only map entries with compress_ratio==4 (else a c128
        layer with the same id shadows the c4 one → get_index_k asserts ratio!=4)."""
        cache = getattr(self, "_compress_to_model", None)
        if cache is None:
            cache = {}
            lm = getattr(token_to_kv_pool, "layer_mapping", None)
            if lm is not None:
                for model_lid, m in enumerate(lm):
                    if getattr(m, "compress_ratio", None) != 4:
                        continue  # only c4 layers carry the indexer-K
                    cid = getattr(m, "compress_layer_id", None)
                    if cid is not None:
                        cache[cid] = model_lid
            self._compress_to_model = cache
        return cache.get(compress_layer_id)

    @torch.no_grad()
    def _scatter_logical_keep(self, bi, n_blk, keep, page_table):
        """Map a per-req LOGICAL keep mask (bool[n_blk]) to GLOBAL pool locs via the
        page_table and write into resident_chunk_mask. First CLEAR this req's pool
        locs (set non-kept to False), then set kept to True — so a chunk dropped this
        cycle actually loses residency. Returns the kept GLOBAL pool_locs (1-D long)
        so the caller can ALSO recall them into the Path-P reserve (Phase-2) — the same
        loc space recall_chunks/remap use, so mask and reserve stay in lock-step."""
        ps = self.c4_page_size
        j = torch.arange(n_blk, device=self.device)
        phys_page = page_table[bi, j // ps].to(torch.long)
        pool_loc = phys_page * ps + (j % ps)
        pool_loc = pool_loc.clamp(0, self.c4_logical_size)
        keep_dev = keep.to(self.device, dtype=torch.bool)
        self.resident_chunk_mask[pool_loc] = keep_dev
        # the kept locs (resident_set in GLOBAL pool-loc space) — for Phase-2 recall.
        return pool_loc[keep_dev]


    # ── Strategy A side-band hook (cycle boundary): scatter a finalized resident_set
    #    (per-req logical keep over pool locs) into the global mask. Called OUT of
    #    graph (on_forward_end), host syncs legal. Phase-1 Strategy B fills the mask
    #    from the existing eager hook instead; this is the Strategy A path.
    @torch.no_grad()
    def scatter_resident(self, pool_locs: torch.Tensor, keep: torch.Tensor) -> None:
        """Update the global mask: resident_chunk_mask[pool_locs] = keep. pool_locs
        are global compressed locs (page*page_size+offset); keep is bool. Out-of-graph.
        """
        pl = pool_locs.to(self.device, dtype=torch.long).clamp(0, self.c4_logical_size)
        self.resident_chunk_mask[pl] = keep.to(self.device, dtype=torch.bool)

    @torch.no_grad()
    def reset_all_resident(self) -> None:
        """Reset to all-resident (= native full attention). Used on (re)init or when
        falling back. Cheap full-buffer fill."""
        self.resident_chunk_mask.fill_(True)

    @torch.no_grad()
    def reset_req_residency(self, req_pool_indices, page_table, c4_seq_lens,
                            token_to_kv_pool=None) -> None:
        """REQUEST-BOUNDARY RESET (called from PREFILL, out of graph — host syncs legal).

        Why: the global resident_chunk_mask + the _req_step cadence counter are BOTH
        keyed by physical pool_loc / req_pool_idx and are NEVER cleared when a request
        finishes and its pool slot is reused. The eager hook is immune (per-req state in
        self._cache, a fresh req gets entry=None=full-attention, finalized SYNCHRONOUSLY
        in the same forward). The graph path side-bands finalize (n→n+1), so a fresh req's
        FIRST decode step(s) read the GLOBAL buffer — which still holds the PREVIOUS
        request's False bits at the reused pool_locs (→ the needle chunk gets masked to
        -inf → wrong answer). And a stale _req_step[ri] (e.g. 100) delays the first
        finalize by up to interval-1 steps (100%64≠0 → no finalize until step 128).

        Fix (mirror eager's "new req = full attention until its own set is finalized"):
        at prefill (request lifecycle start, before the first decode forward), for each
        req in this batch (a) set ALL its c4 pool_locs back to True (full residency =
        native, no -inf), and (b) zero its step counter so the first decode step (step 0)
        finalizes immediately. Idempotent across chunked-prefill chunks; the last chunk
        carries the full c4_seq_len so coverage is complete before decode begins.

        Phase-2 (改造4 + decode register): the Path-P swap engine's chunk_cell_lut is the
        SAME class of global state keyed by reused pool_loc — a fresh req reusing a slot
        would inherit stale lut entries pointing at the previous occupant's cell (→ reads
        old KV). So ALSO clear this req's pool_locs in EVERY layer's lut (reuse the engine's
        page-granular discard_copied_pages). And register the req's prefill_chunks + decode
        slot so the in-graph decode store can compute its deterministic decode-resident cell.

        Pure GPU scatter + a few host ints; prefill is never under cuda graph capture."""
        ps = self.c4_page_size
        ri_list = req_pool_indices.tolist()
        # Invalidate the decode-loop's cached req-id list: a (re)admitted request changes
        # batch composition, so the next on_forward_end MUST re-sync the bi->ri map (else a
        # stale map could mis-assign a due cycle to the wrong req).
        self._cached_req_ids = None
        if c4_seq_lens.dim() == 2:
            seql = c4_seq_lens[:, 0]
        else:
            seql = c4_seq_lens
        seql_cpu = seql.to("cpu", dtype=torch.int64).tolist()
        eng = getattr(token_to_kv_pool, "_swap_engine_p", None) if token_to_kv_pool else None
        B = len(ri_list)
        for bi in range(B):
            ri = int(ri_list[bi])
            n_blk = int(seql_cpu[bi]) if bi < len(seql_cpu) else 0
            if n_blk <= 0:
                # still zero the cadence so a 0-length/edge req starts clean
                self._req_step[ri] = 0
                continue
            j = torch.arange(n_blk, device=self.device)
            phys_page = page_table[bi, j // ps].to(torch.long)
            pool_loc = (phys_page * ps + (j % ps)).clamp(0, self.c4_logical_size)
            self.resident_chunk_mask[pool_loc] = True   # full residency for the new req
            self._req_step[ri] = 0                       # first decode step → immediate finalize
            if self._stagger_cycle:
                # round-robin cycle phase so co-landing requests spread their recalls across
                # the interval (production-like; avoids the due=N spike under batch landing).
                self._req_phase[ri] = self._next_phase % self.interval
                self._next_phase += 1
            # ── Step-1: clear this slot's score-resident scratch bookkeeping so a reused
            # pool slot can't score the previous occupant's resident set before its own
            # first finalize. count=0 → the captured score-resident path scores 0 valid cols
            # → topk all -1 → all c4 chunks masked-out for this req until its step-0 finalize
            # fills the scratch (the recency-tail / SWA covers correctness meanwhile).
            # CRITICAL (concurrency OOB fix 2026-06-29): ALSO clear col_to_cell_buf[bi]. With
            # fused-remap the in-graph decode resolves top-512 cols straight to reserve cells
            # via col_to_cell_buf[bi, raw]. A reused slot would otherwise inherit the PREVIOUS
            # occupant's cells; the async step-0 finalize hasn't refreshed col_to_cell yet, so
            # any raw>=0 (only possible if count>0, but be defensive) would gather a stale cell
            # pointing at a reassigned/invalid reserve slot → CUDA illegal access under churn.
            if self._score_resident and self.resident_count_buf is not None and bi < self.max_bs:
                self.resident_count_buf[bi] = 0
                self.resident_pool_loc_buf[bi].fill_(-1)
                if getattr(self, "col_to_cell_buf", None) is not None:
                    self.col_to_cell_buf[bi].fill_(-1)
            # ── 改造4: clear this req's stale lut entries + register decode bookkeeping ──
            if eng is not None:
                # discard_copied_pages clears EVERY layer's chunk_cell_lut for these pages
                # (page = pool_loc // C4_PAGE_SIZE), so a reused slot can't read the prior
                # request's cell. Pass logical pages this req occupies.
                logical_pages = torch.unique(pool_loc // ps)
                eng.discard_copied_pages(logical_pages)
                # RECLAIM recall cells the previous occupant of this pool slot held at these
                # locs → return them to the free-list so next_cell doesn't grow unbounded
                # across requests (the cross-request reserve-exhaustion bug). Without this,
                # request N's prefill recall kept bumping next_cell forever → reserve full →
                # later requests' needle chunks never became resident → MISS.
                eng.release_req_pool_locs(pool_loc)
                # prefill_chunks[req] = n_blk (c4 chunks produced by prefill) → decode chunk
                # local index = c4_out_loc - n_blk. Register slot for deterministic cell.
                # max_req_pool fixed-large (4096) so req_pool_idx is never clamped.
                eng.register_decode_req(ri, prefill_chunks=n_blk, max_req_pool=4096)


_RESIDENT_MASK_CAPTURER: Optional[ResidentMaskCapturer] = None


def get_resident_mask_capturer() -> Optional[ResidentMaskCapturer]:
    return _RESIDENT_MASK_CAPTURER


def set_resident_mask_capturer(cap: Optional[ResidentMaskCapturer]) -> None:
    global _RESIDENT_MASK_CAPTURER
    _RESIDENT_MASK_CAPTURER = cap
