from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.deepseek_v4 import topk_transform_512, topk_transform_512_v2
from sglang.srt.environ import envs
from sglang.srt.layers.attention.compressed.metadata import (
    PagedCoreMetadata,
    PagedIndexerMetadata,
)
from sglang.srt.layers.attention.indexer_topk_capturer import (
    get_global_indexer_capturer,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.utils import is_hip

# score-resident topk padding sentinels: a large FINITE negative (not -inf, to avoid inf
# arithmetic / nan in the kernel) used to mask out compacted columns >= resident_count so
# padding is never selected; _NEG_HALF is the threshold to detect "this topk pick was a
# padded column" (score below it → emit -1 pool-loc → dropped downstream).
_NEG_LOGIT = -1.0e30
_NEG_HALF = -1.0e29

if TYPE_CHECKING:
    from sglang.srt.layers.attention.compressed.compressor import CompressorBackend
    from sglang.srt.layers.attention.compressed.metadata import DeepseekV4Metadata
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v4 import C4Indexer


if is_hip():
    FP8_DTYPE = torch.float8_e4m3fnuz
    FP8_MAX = torch.finfo(FP8_DTYPE).max
else:
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    for i in range(batch_size):
        q = q_fp8[i, 0]
        q = q.to(torch.float32)
        q_scale = weight[i]
        seq_len = int(seq_lens[i].item())
        assert seq_len <= max_seq_len
        num_pages = (seq_len + block_size - 1) // block_size
        padded_seq_len = num_pages * block_size
        pages = page_table[i, :num_pages]
        kvcache_fp8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
        kvcache = kvcache_fp8[pages]
        SCALE_OFFSET = block_size * head_dim
        kvcache_value = kvcache[..., :SCALE_OFFSET].view(dtype=FP8_DTYPE)
        kvcache_scale = kvcache[..., SCALE_OFFSET:].view(dtype=torch.float32)
        kvcache_value = kvcache_value.to(torch.float32)
        kvcache_scale = kvcache_scale.contiguous()
        kvcache_value = kvcache_value.view(padded_seq_len, head_dim)
        kvcache_scale = kvcache_scale.view(padded_seq_len)
        score = F.linear(kvcache_value, q)
        score = F.relu(score)
        score *= q_scale[None, :]
        score = score.sum(dim=1)
        score *= kvcache_scale
        logits[i, :seq_len] = score[:seq_len]

    return logits


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:

    TOPK = 512
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores[~valid_mask] = float("-inf")

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        padding = torch.zeros(
            (batch_size, TOPK - actual_k), dtype=torch.int32, device=device
        )
        raw_indices = torch.cat([raw_indices, padding], dim=1)

    batch_indices = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, TOPK)
    )
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = torch.arange(TOPK, device=device).unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    if needs_sequential.any():
        sequential_indices = (
            torch.arange(TOPK, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

        raw_indices = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK),
            torch.where(
                sequential_valid,
                sequential_indices,
                torch.tensor(-1, device=device, dtype=torch.int32),
            ),
            raw_indices,
        )
        valid_topk = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK), sequential_valid, valid_topk
        )

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)

    page_indices = torch.where(
        valid_topk, page_indices, torch.tensor(-1, device=device, dtype=torch.int32)
    )

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = torch.where(
            valid_topk, raw_indices, torch.tensor(-1, device=device, dtype=torch.int32)
        )
        out_raw_indices.copy_(raw_indices)


@triton.jit
def _fused_scale_kernel(
    weight_ptr,
    q_scale_ptr,
    out_ptr,
    numel,
    out_scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    qs = tl.load(q_scale_ptr + offs, mask=mask)

    acc = w.to(tl.float32) * out_scale * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    assert weight.is_contiguous() and q_scale.is_contiguous()
    B, H = weight.shape
    numel = B * H
    out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    out = torch.empty((B, H, 1), device=weight.device, dtype=out_dtype)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_scale_kernel[grid](
        weight,
        q_scale,
        out,
        numel,
        out_scale,
        BLOCK=BLOCK,
    )
    return out


class C4IndexerBackend:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DeepseekV4Metadata
        self.debug_use_external_c4_sparse_indices: bool = False

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        assert alt_streams is not None
        assert len(alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_q = alt_streams[0]
        stream_weights = alt_streams[1]

        stream_q.wait_stream(current_stream)
        stream_weights.wait_stream(current_stream)

        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            q = c4_indexer.compute_q(q_lora, positions=positions)
            q_fp8, q_scale = act_quant(q)
            q_scale_ready = stream_q.record_event()

        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            stream_weights.wait_event(q_scale_ready)
            weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)

        current_stream.wait_stream(stream_q)
        current_stream.wait_stream(stream_weights)

        # 存中间结果供 dump_training hook 使用
        c4_indexer._last_q = q
        c4_indexer._last_weights = weights

        return q_fp8, weights, c4_indexer_kv_cache

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        q = c4_indexer.compute_q(q_lora, positions=positions)
        q_fp8, q_scale = act_quant(q)
        weights = c4_indexer.compute_weights(x, skip_scale=True)
        weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)
        # 存中间结果供 dump_training hook 使用（full-precision q 含 RoPE + Hadamard）
        c4_indexer._last_q = q  # [batch, n_heads * head_dim]
        c4_indexer._last_weights = weights  # [batch, n_heads] (fused: weights_proj * weight_scale * q_scale)
        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )
        return q_fp8, weights, c4_indexer_kv_cache

    def _rs_custom_topk(self, logits, _rmc, _rs_pool, _buf, _k, _TOPK, _B, core_metadata):
        """score-resident custom topk+remap (non-fused path). Mask columns >= resident_count
        to a finite sentinel (validity mask precomputed per-cycle side-band), topk, gather
        physical pool locs via resident_pool_loc_buf, write in-place into the fixed
        c4_sparse_page_indices. Returns the selected locs. See SGLANG_PATHP_RS_NATIVE_TOPK
        for the fused alternative."""
        _valid = getattr(_rmc, "valid_mask_buf", None)
        if _valid is not None and _valid.shape[1] == _buf:
            _masked = torch.where(_valid[:_B], logits, _NEG_LOGIT)
        else:
            _rs_count_1d = _rmc.resident_count_buf[:_B].to(torch.int64)
            _cols = torch.arange(_buf, device=logits.device).unsqueeze(0)
            _masked = torch.where(_cols < _rs_count_1d.unsqueeze(1), logits, _NEG_LOGIT)
        _sc, _jidx = torch.topk(_masked, k=_k, dim=1, largest=True, sorted=False)  # [B,k]
        _sel_loc = torch.gather(_rs_pool, 1, _jidx)                          # [B,k]
        _sel_loc = torch.where(_sc > _NEG_HALF, _sel_loc,
                               torch.full_like(_sel_loc, -1))
        if _k < _TOPK:
            _pad = _sel_loc.new_full((_B, _TOPK - _k), -1)
            _sel_loc = torch.cat([_sel_loc, _pad], dim=1)
        core_metadata.c4_sparse_page_indices.copy_(_sel_loc.to(
            core_metadata.c4_sparse_page_indices.dtype))
        return _sel_loc

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: this runs from MQALayer._forward_prepare,
        # before attn_backend.forward() would trigger the upgrade.
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackend)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        from sglang.srt.layers.attention.deepseek_v4_backend_radix import (
            DSV4AttnMetadataRadix,
        )

        assert isinstance(core_metadata, (PagedCoreMetadata, DSV4AttnMetadataRadix))
        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        if enable_multi_stream:
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)
        assert len(c4_indexer_kv_cache.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132

        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            fn = fp8_paged_mqa_logits_torch
        else:
            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                from sglang.srt.layers.deep_gemm_wrapper.paged_mqa_logits import (
                    fp8_paged_mqa_logits_chunked as fn,
                )
            else:
                from deep_gemm import fp8_paged_mqa_logits as fn

        # ── Step-2: score-resident-only (env SGLANG_PATHP_SCORE_RESIDENT=1). In decode,
        # score ONLY this batch's resident_set (packed into a contiguous scratch by the
        # side-band, see resident_mask_capturer._build_resident_scratch) instead of the
        # full c4 history. The kernel reads the scratch with an IDENTITY page_table; logits
        # become [B, RESIDENT_BUF] → O(buf) FLOP and the full-width mask is unnecessary.
        # Falls back to full-history scoring when: gate off / prefill / scratch not yet
        # built for this batch (e.g. a req's first step before its step-0 finalize → its
        # resident_count is 0 → we keep it on the full path via _score_resident_active).
        from sglang.srt.layers.attention.compressed.resident_mask_capturer import (
            get_resident_mask_capturer,
        )
        _rmc = get_resident_mask_capturer()
        _clid = token_to_kv_pool.layer_mapping[c4_indexer.layer_id].compress_layer_id
        # PAGE-RECALL: scores the LIVE index-K pool with a real page_table (no scratch); its
        # gate is the shared resident_pool_loc_buf being allocated (resident_kv_scratch stays
        # None on this path). Otherwise (chunk-granular score-resident) the gate needs the
        # per-clid scratch present.
        _page_recall = _rmc is not None and getattr(_rmc, "_page_recall", False)
        _score_resident_active = (
            _rmc is not None
            and getattr(_rmc, "_score_resident", False)
            and forward_batch.forward_mode.is_decode()
            and (
                (_page_recall and getattr(_rmc, "resident_pool_loc_buf", None) is not None)
                or (
                    not _page_recall
                    and getattr(_rmc, "resident_kv_scratch", None) is not None
                    and _clid in _rmc.resident_kv_scratch
                )
            )
        )

        _c4sl = indexer_metadata.c4_seq_lens
        if _c4sl.dim() == 1:
            _c4sl = _c4sl.unsqueeze(-1)
        # CAPTURE-FREEZE PROBE (SGLANG_PATHP_CAPTURE_PROBE=1): log _score_resident_active and
        # scratch presence the FIRST time this clid is seen under capture-mode and the FIRST
        # time at runtime (replay), to prove whether the score-resident branch was frozen out
        # at capture. Capture-safe: get_is_capture_mode() is a host flag (no device sync), and
        # we only log once per (clid, capture?) via a set. No-op when env off.
        if os.environ.get("SGLANG_PATHP_CAPTURE_PROBE", "0") == "1":
            try:
                from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                _cap = get_is_capture_mode()
                _seen = getattr(self, "_cap_probe_seen", None)
                if _seen is None:
                    _seen = set(); self._cap_probe_seen = _seen
                _key = (_clid, bool(_cap))
                if _key not in _seen:
                    _seen.add(_key)
                    _has_scratch = (getattr(_rmc, "resident_kv_scratch", None) is not None
                                    and _clid in (_rmc.resident_kv_scratch or {}))
                    print(
                        f"[CAP-PROBE] clid={_clid} capture_mode={_cap} "
                        f"score_resident_active={_score_resident_active} "
                        f"scratch_present={_has_scratch} decode={forward_batch.forward_mode.is_decode()}",
                        flush=True,
                    )
            except Exception as _e:
                print(f"[CAP-PROBE] error: {_e}", flush=True)
        if _score_resident_active:
            _B = q_fp8.shape[0]
            # PAGE-RECALL: score the LIVE index-K pool (c4_indexer_kv_cache, already viewed
            # [n_pages,64,1,132]) with the real page_table holding the K_max selected physical
            # pages — NO scratch packing. Otherwise score the dense per-req scratch with the
            # identity page_table (chunk-granular score-resident path).
            if _page_recall:
                # Path-I SELECTIVE (index-K offload): score the shrunk reserve ONLY for the
                # OFFLOAD layers (those with a real reserve buffer). The retriever TARGET layers
                # keep their full-history GPU pool (index_reserve_buf[_clid] is None) and score
                # it directly — byte-unchanged — because the Level-1 side-band re-selection reads
                # their full history every cycle. For an offload layer, the reserve holds only
                # the K_max recalled pages, addressed by resident_id_page_table_reserve (reserve-
                # block ids the side-band remapped from full-history phys pages); byte-identical
                # to full-pool scoring for the resident set (Phase I-0 proof).
                _eng = getattr(token_to_kv_pool, "_swap_engine_p", None)
                _idx_off = (
                    getattr(_rmc, "_index_k_offload", False)
                    and _eng is not None and getattr(_eng, "index_k_offload", False)
                    and getattr(_rmc, "resident_id_page_table_reserve", None) is not None
                    and getattr(_eng, "index_reserve_buf", None) is not None
                    and _clid < len(_eng.index_reserve_buf)
                    and _eng.index_reserve_buf[_clid] is not None   # per-layer: offload layer only
                )
                if _idx_off:
                    _res_buf = _eng.index_reserve_buf[_clid]        # GPU [recall_blocks, page_bytes] uint8
                    _kv_for_score = _res_buf.view(-1, block_kv, num_heads_kv, head_dim_with_sf)
                else:
                    _kv_for_score = c4_indexer_kv_cache
            else:
                _kv_for_score = _rmc.resident_kv_scratch[_clid].view(
                    -1, block_kv, num_heads_kv, head_dim_with_sf
                )
                _idx_off = False
            _rs_count = _rmc.resident_count_buf[:_B].to(torch.int32).unsqueeze(-1)  # [B,1]
            # Path-I: read the RESERVE-block page_table (points into the shrunk index-K reserve)
            # when index-K offload is active; else the full-history phys-page table (unchanged).
            if _page_recall and _idx_off:
                _rs_pt = _rmc.resident_id_page_table_reserve[:_B]                  # [B,pages] reserve blocks
            else:
                _rs_pt = _rmc.resident_id_page_table[:_B]                          # [B,pages] full-history phys
            _rs_max = _rmc.resident_buf
            # deep_gemm SM-schedule metadata: computed ONCE per cycle SIDE-BAND (resident_count
            # is frozen for `interval` steps), read here from a fixed buffer — NOT recomputed
            # per-step-per-layer (that host-side planning call ×21/step stalled the loop and
            # starved the GPU at high batch). Sliced to [:B] for this batch. None if torch/
            # tilelang/chunk kernel (they ignore it).
            _rs_meta = getattr(_rmc, "resident_dg_meta", None)
            # PROBE: SGLANG_PATHP_RS_META_NONE=1 passes None (kernel self-schedules) to test
            # if the side-band-built schedule_meta is the scratch-fn slowdown vs native.
            if os.environ.get("SGLANG_PATHP_RS_META_NONE", "0") == "1":
                _rs_meta = None
            logits = fn(
                q_fp8,
                _kv_for_score,
                weights,
                _rs_count,
                _rs_pt,
                _rs_meta,
                _rs_max,
                False,
            )
        else:
            logits = fn(
                q_fp8,
                c4_indexer_kv_cache,
                weights,
                _c4sl,
                indexer_metadata.page_table,
                indexer_metadata.deep_gemm_metadata,
                indexer_metadata.max_c4_seq_len,
                False,
            )

        assert indexer_metadata.page_table is core_metadata.page_table

        # ── Phase-1 cuda-graph-compatible two-level Level-2 mask (ALL 21 c4 layers,
        # EVERY step, PURE GPU — runs INSIDE capture). Reads the global static
        # resident_chunk_mask buffer and masks non-resident chunks' logits to -inf
        # BEFORE topk_transform_512, so the native top-512 is confined to the
        # resident_set. The resident_set itself is finalized OUT of graph (side-band /
        # eager cycle step) and scattered into that buffer; see resident_mask_capturer.
        # This replaces the host-syncing score_hook in graph mode (the hook stays for
        # the legacy eager / non-PATHP_CUDAGRAPH path below).
        # (_rmc / _clid computed above the fn() call for the score-resident swap.)
        if _rmc is not None and forward_batch.forward_mode.is_decode():
            # target layers: capture this layer's decode hidden + (once/forward)
            # page_table/c4_seq_lens/positions into fixed buffers (graph-internal copy_)
            # so the side-band Level-1 can read them post-replay without touching the
            # raw backend.forward_metadata stub.
            _rmc.capture_side_band_inputs(
                _clid, x, indexer_metadata.page_table, indexer_metadata.c4_seq_lens,
                core_metadata.positions,
            )
            # Mask is needed ONLY on the full-history path (confine native top-512 to the
            # resident_set). When score-resident is active we already scored ONLY the
            # resident_set → nothing to mask (and logits are [B,buf], not [B,max_blk]).
            # STEP-0 PROBE (SGLANG_PATHP_SKIP_MASK=1): force-skip to measure the mask wall.
            if (
                not _score_resident_active
                and os.environ.get("SGLANG_PATHP_SKIP_MASK", "0") != "1"
            ):
                logits = _rmc.apply_mask_in_graph(
                    logits, indexer_metadata.page_table, indexer_metadata.c4_seq_lens
                )
        elif _rmc is not None and not forward_batch.forward_mode.is_idle():
            # ── REQUEST-BOUNDARY RESET (prefill, out of graph) ──
            # The global resident_chunk_mask + _req_step are keyed by reused pool slots
            # and never cleared on request completion, so a fresh request would inherit
            # the previous occupant's False bits (→ needle chunk masked to -inf) and a
            # stale step counter (→ first finalize delayed up to interval-1 steps). Reset
            # this batch's c4 pool_locs to full residency + zero its cadence here, BEFORE
            # the first decode forward reads the buffer — mirroring the eager hook's
            # "new req = full attention until its own resident_set is finalized". Prefill
            # is never under cuda-graph capture, so host syncs are legal. Idempotent across
            # chunked-prefill chunks (last chunk carries the full c4_seq_len).
            _rmc.reset_req_residency(
                forward_batch.req_pool_indices,
                indexer_metadata.page_table,
                indexer_metadata.c4_seq_lens,
                token_to_kv_pool,
            )

        # --------------- IndexerTracker / Pass-2 mask hook ---------------
        # Legacy eager path: skip during CUDA graph capture — hook does CPU-GPU sync
        # (.item()) which is forbidden inside graph capture. Also skip entirely when
        # the graph-compatible resident-mask path above is active (_rmc is not None).
        if _rmc is None and getattr(c4_indexer, "score_hook", None) is not None:
            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
            if not get_is_capture_mode():
                c4_indexer._last_kv_cache = c4_indexer_kv_cache  # 供 dump_training hook 使用
                c4_indexer._last_page_table = indexer_metadata.page_table
                c4_indexer._last_seq_lens = indexer_metadata.c4_seq_lens
                c4_indexer._last_positions = core_metadata.positions  # decode token positions
                logits = c4_indexer.score_hook(logits, indexer_metadata.c4_seq_lens, forward_batch)
        # -----------------------------------------------------------------

        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer.is_enabled()

        hisparse_coordinator = forward_batch.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : core_metadata.c4_sparse_page_indices.size(0)
            ]

        if _score_resident_active:
            # ── Step-2 topk + remap for score-resident: topk over logits[B,buf] gives
            # COMPACTED column indices j∈[0,buf); the resident chunk at column j has
            # physical pool loc resident_pool_loc_buf[bi,j]. We DON'T use topk's
            # page_table-remapped output (identity pt → scratch positions, wrong). Instead
            # take topk's raw column indices and gather the real physical locs ourselves,
            # writing IN-PLACE into the fixed c4_sparse_page_indices (preserves captured
            # address). Downstream remap_compressed_locs (backend_radix.py) then maps these
            # physical locs → reserve cells, exactly as on the full-history path.
            _B = q_fp8.shape[0]
            _TOPK = core_metadata.c4_sparse_page_indices.size(1)
            _rs_pool = _rmc.resident_pool_loc_buf[:_B]                          # [B,buf]
            _buf = logits.shape[1]
            _k = min(_TOPK, _buf)
            if os.environ.get("SGLANG_PATHP_RS_NATIVE_TOPK", "0") == "1":
                # PROBE/impl: use the FUSED native topk_transform_512 over the scratch
                # logits (identity page_table → out_raw_indices = compacted scratch columns),
                # then map columns→phys loc via resident_pool_loc_buf. Isolates whether the
                # custom 5-op block (below) is the per-step overhead vs the scratch fn().
                _raw = torch.empty((_B, _TOPK), dtype=torch.int32, device=logits.device)
                _dummy_pi = core_metadata.c4_sparse_page_indices
                _rs_count_1d = _rmc.resident_count_buf[:_B].to(torch.int32)
                topk_transform_512(
                    logits, _rs_count_1d, _rmc.resident_id_page_table[:_B],
                    _dummy_pi, _rmc.c4_page_size, _raw,
                )
                _rawl = _raw.to(torch.int64).clamp_(0, _buf - 1)
                if os.environ.get("SGLANG_PATHP_FUSED_REMAP", "0") == "1":
                    # FUSED-REMAP: map top-512 cols straight to RESERVE CELLS via the
                    # side-band-built col_to_cell table (col -> cell, one gather), instead of
                    # col -> phys pool loc here + phys loc -> cell in backend remap_compressed_locs
                    # ×21. The written c4_sparse_page_indices ARE reserve cells → backend skips
                    # remap (deepseek_v4_backend_radix Phase-2 decode). -1 (non-resident / padded)
                    # is preserved as the masked-out sentinel.
                    _col2cell = _rmc.col_to_cell_buf[:_B]                  # [B,buf]
                    _sel_loc = torch.gather(_col2cell, 1, _rawl)
                    _sel_loc = torch.where(_raw >= 0, _sel_loc,
                                           torch.full_like(_sel_loc, -1))
                    core_metadata.c4_sparse_page_indices.copy_(_sel_loc.to(
                        core_metadata.c4_sparse_page_indices.dtype))
                else:
                    _sel_loc = torch.gather(_rs_pool, 1, _rawl)
                    _sel_loc = torch.where(_raw >= 0, _sel_loc, torch.full_like(_sel_loc, -1))
                    core_metadata.c4_sparse_page_indices.copy_(_sel_loc.to(
                        core_metadata.c4_sparse_page_indices.dtype))
            else:
                _sel_loc = self._rs_custom_topk(
                    logits, _rmc, _rs_pool, _buf, _k, _TOPK, _B, core_metadata
                )
        elif envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            topk_transform_512_pytorch_vectorized(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            topk_transform_512_v2(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            topk_transform_512(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.c4_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = token_to_kv_pool.c4_kv_pool.translate_loc_from_compressed_to_hisparse_device(
                    core_metadata.c4_sparse_page_indices
                )
        elif os.environ.get("SGLANG_DECODE_SWAP", "0") == "1" and hasattr(
            token_to_kv_pool.c4_kv_pool, "full_to_hisparse_device_index_mapping"
        ):
            # Decode-swap (Stage 3b): HiSparse pool WITHOUT coordinator. alloc_extend
            # auto-populated the mapping for prefilled tokens; the decode loop's
            # map_last_loc hook maps new tokens. Here we just remap selected
            # compressed locs → device-buffer slots (same as the plain hisparse
            # non-decode branch above).
            core_metadata.c4_sparse_page_indices = (
                token_to_kv_pool.c4_kv_pool.translate_loc_from_compressed_to_hisparse_device(
                    core_metadata.c4_sparse_page_indices
                )
            )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)
