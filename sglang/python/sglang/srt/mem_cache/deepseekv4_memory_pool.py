from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import torch

from sglang.jit_kernel.deepseek_v4 import fused_store_cache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa import index_buf_accessor, index_buf_accessor_v4
from sglang.srt.layers.attention.nsa.index_buf_accessor_v4 import NopeFp8RopeBf16Pack
from sglang.srt.mem_cache.compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import ceil_div

logger = logging.getLogger(__name__)

ONLINE_C128 = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def _index_k_full_layers() -> set:
    """Path-I SELECTIVE index-K offload: the set of c4 COMPRESS-LAYER-IDS whose index-K GPU
    buffer must stay FULL history (never shrunk). These are exactly the retriever's target
    layers — the Level-1 side-band re-selection (resident_mask_capturer._score_paged_layer)
    scores their ENTIRE history every cycle to pick which pages become resident, so their
    full history MUST live on GPU. Every other c4 layer is read only by the in-graph Level-2
    scoring at the K_max recalled pages (served from the SwapEngineP reserve), so it can be
    offloaded. Source = SGLANG_RETRIEVER_INLINE_LAYERS (same env the InlineRetrieverHook /
    ResidentMaskCapturer read to build target_layers; these ARE compress-layer-ids — see
    inline_retriever_hook.target_layers, fed straight to _model_layer_of_compress). Returns
    empty when offload is off (env unset) so nothing changes. NOTE: kept intentionally
    independent of the hook object so the pool (built before the hook) can size buffers."""
    if int(os.environ.get("SGLANG_RETRIEVER_INDEX_K_DEVICE_TOKENS", "0")) <= 0:
        return set()  # offload off → no layer is "special"; pool builds every layer full
    layers_str = os.environ.get("SGLANG_RETRIEVER_INLINE_LAYERS", "10,12,20")
    return {int(x) for x in layers_str.split(",") if x.strip()}


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    # Online c128 keeps a single (max, sum, kv) state per index instead of a
    # 128-slot ring buffer of raw tokens, so ring_size collapses to 1. Online
    # is incompatible with speculative decode for now.
    if compress_ratio == 128 and ONLINE_C128:
        assert not is_speculative, "online c128 does not support MTP"
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


class DeepSeekV4SingleKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        is_swa_pool: Optional[bool] = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        self.scale_pad = 1
        self.quantize_block_size = 64
        self.rope_storage_dtype = torch.bfloat16
        self.k_with_scale_buffer_dtype = torch.int8
        self.is_swa_pool = is_swa_pool
        self._create_buffers()

    @property
    def page_size(self):
        if self.is_swa_pool:
            assert self._page_size == 256, "SWA KV pool page size not correct!"

        return self._page_size

    @page_size.setter
    def page_size(self, value: int):
        self._page_size = value

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        dim_per_token = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            + self.qk_nope_head_dim // self.quantize_block_size
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        bytes_per_token = self.get_bytes_per_token()
        self.kv_cache_total_dim = bytes_per_token
        bytes_per_page_non_padded = self.page_size * bytes_per_token
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

        assert bytes_per_token == 448 + 64 * 2 + 8
        assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=self.store_dtype,
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        index_buf_accessor_v4.SetKAndS.execute(
            pool=self,
            buf=self.kv_buffer[layer_id],
            loc=loc,
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
        )

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use get_key_buffer instead.")


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_from_compressed_to_hisparse_device(
        self, compressed_indices: torch.Tensor
    ):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_from_compressed_to_hisparse_device(
        self, compressed_indices: torch.Tensor
    ):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_from_compressed_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ):
        loc = self.translate_loc_from_compressed_to_hisparse_device(loc)
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        loc = self.translate_loc_from_compressed_to_hisparse_device(loc)
        return super().set_key_buffer_fused(layer_id, loc, cache_k)

    def get_cpu_copy(self, indices):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        device_size: int = 0,
        full_layers: Optional[set] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim
        # Path-I: physical GPU pages = device_size tokens (0 => full logical `size`). MUST be
        # set BEFORE _create_buffer (it reads self.device_size to size the GPU buffer).
        self.device_size = int(device_size)
        # Path-I SELECTIVE: compress-layer-ids that MUST stay full-history on GPU (the
        # retriever's target layers — Level-1 re-selection scores their FULL history every
        # cycle, so they cannot be offloaded). All OTHER c4 layers shrink to device_size.
        # Empty/None => (legacy) every layer shrinks (only correct when device_size==0). The
        # ModelRunner passes the target compress-layer-ids here when index-K offload is on.
        self.full_layers = set(full_layers) if full_layers else set()

        self._create_buffer()

    def _create_buffer(self):
        num_scales_per_token = self.index_head_dim // self.quant_block_size
        page_bytes = self.page_size * self.index_head_dim
        page_bytes += self.page_size * num_scales_per_token * 4
        self.index_page_bytes = page_bytes
        # Path-I (index-K offload): the PHYSICAL GPU buffer can be shrunk below the logical
        # `size` (full history) — but ONLY for the c4 layers that are NOT retriever target
        # layers. SELECTIVE offload: the target layers (self.full_layers, e.g. compress-ids
        # {10,12,20}) keep their FULL-history GPU buffer because the Level-1 side-band
        # re-selection scores their entire history every cycle to PICK the resident pages;
        # shrinking them would make Level-1 read garbage → silently wrong resident set. The
        # other 18 layers are read ONLY by the in-graph Level-2 scoring, which touches just
        # the K_max recalled pages (swapped into a SEPARATE reserve, index_reserve_buf) — so
        # their full history lives in a CPU mirror (SwapEngineP) and their GPU buffer shrinks
        # to device_size. device_size=0 (default) => every layer full (byte-identical to
        # today). Per-layer size: layer in full_layers → full `size`; else → device_size.
        # Set via SGLANG_RETRIEVER_INDEX_K_DEVICE_TOKENS + SGLANG_RETRIEVER_INLINE_LAYERS.
        _shrunk = 0 < getattr(self, "device_size", 0)
        _full_n_pages = (self.size + self.page_size + 1) // self.page_size
        _dev_n_pages = (
            (self.device_size + self.page_size + 1) // self.page_size
            if _shrunk else _full_n_pages
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        (_full_n_pages if (not _shrunk or _lid in self.full_layers)
                         else _dev_n_pages),
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _lid in range(self.layer_num)
                ]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
        )


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


class DeepSeekV4TokenToKVPool(KVCache):

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
    ):
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        c4_logical_size = c128_size * 32

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        self.c4_size = c4_size
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        self.c128_state_pool_size = c128_state_pool_size
        self.state_dtype = state_dtype
        self.compression_ratios = compression_ratios

        assert page_size % swa_page_size == 0

        self.swa_size = swa_size
        self.swa_window_size = swa_page_size
        self.swa_page_size = swa_page_size
        self.scale_pad = 1

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        c4_layer_num = sum(1 for r in compression_ratios if r == 4)
        c128_layer_num = sum(1 for r in compression_ratios if r == 128)
        c4_page_size = page_size // 4
        c128_page_size = page_size // 128
        self.swa_kv_pool = DeepSeekV4SingleKVPool(
            swa_size,
            swa_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            is_swa_pool=True,
        )

        c4_kv_pool_type = DeepSeekV4SingleKVPool
        if enable_hisparse:
            c4_kv_pool_type = HiSparseC4DevicePool
        self.c4_kv_pool = c4_kv_pool_type(
            c4_size,
            c4_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self.c128_kv_pool = DeepSeekV4SingleKVPool(
            c128_size,
            c128_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c128_layer_num,
            device,
            enable_memory_saver,
        )

        self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(
            self.c4_logical_size,
            c4_page_size,
            dtype,
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
            device_size=int(os.environ.get("SGLANG_RETRIEVER_INDEX_K_DEVICE_TOKENS", "0")),
            full_layers=_index_k_full_layers(),
        )

        self._init_compressed_layer_mapping()

        self._init_paged_compress_states(enable_memory_saver)

        self._should_cache_swa = envs.SGLANG_OPT_CACHE_SWA_TRANSLATION.get()

        self._dbg_dump_pool_sizes()

    def _dbg_dump_pool_sizes(self):
        import os

        if os.environ.get("SGLANG_HISPARSE_DBG_POOL_SIZES") != "1":
            return
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = 0
        if rank != 0:
            return

        def sum_bufs(name, bufs):
            if bufs is None:
                return 0
            total = 0
            count = 0
            for b in bufs:
                if b is None:
                    continue
                t = getattr(b, "kv_score", None)
                if t is None:
                    t = b
                try:
                    total += t.element_size() * t.numel()
                except Exception:
                    t = getattr(t, "kv_score_buffer", None)
                    if t is not None and hasattr(t, "kv_score"):
                        total += t.kv_score.element_size() * t.kv_score.numel()
                count += 1
            logger.warning(
                "HSDBG[pool] %-28s #bufs=%3d total=%10.2f MiB",
                name,
                count,
                total / 2**20,
            )
            return total

        total_all = 0
        total_all += sum_bufs("swa_kv_pool", self.swa_kv_pool.kv_buffer)
        total_all += sum_bufs("c4_kv_pool", self.c4_kv_pool.kv_buffer)
        total_all += sum_bufs("c128_kv_pool", self.c128_kv_pool.kv_buffer)
        total_all += sum_bufs(
            "c4_indexer_kv_pool", self.c4_indexer_kv_pool.index_k_with_scale_buffer
        )
        if hasattr(self, "compress_state_pools"):
            c4_state_bufs = []
            c128_state_bufs = []
            for ratio, pool in zip(self.compression_ratios, self.compress_state_pools):
                if pool is None:
                    continue
                if ratio == 4:
                    c4_state_bufs.append(pool.kv_score_buffer.kv_score)
                elif ratio == 128:
                    c128_state_bufs.append(pool.kv_score_buffer.kv_score)
            total_all += sum_bufs("c4_state_pool", c4_state_bufs)
            total_all += sum_bufs("c128_state_pool", c128_state_bufs)
            idx_bufs = []
            for pool in self.indexer_compress_state_pools:
                if pool is None:
                    continue
                idx_bufs.append(pool.kv_score_buffer.kv_score)
            total_all += sum_bufs("c4_indexer_state_pool", idx_bufs)
        logger.warning(
            "HSDBG[pool] %-28s total=%10.2f MiB = %.2f GiB",
            "GRAND_TOTAL",
            total_all / 2**20,
            total_all / 2**30,
        )

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        server_args = get_global_server_args()
        is_speculative = server_args.speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None

        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        # Path-P-B (NIXL c4 direct-to-mirror): on the DECODE server, return the CPU
        # pinned mirror ptrs for the c4 buffers so the disaggregation transfer writes
        # c4 history straight into the host mirror (no GPU landing). indexer/c128 keep
        # their GPU buffers (must stay VRAM-resident). Gated on the swap engine being
        # present with own_reserve (B mode). c4 layers come FIRST so the NIXL register
        # split can identify them by the first c4_layer_num entries.
        _eng = getattr(self, "_swap_engine_p", None)
        _b_mode = _eng is not None and getattr(_eng, "_own_reserve", False)
        # Path-I: index-K also offloaded to a CPU mirror (engine.index_k_offload). Then the
        # indexer group (grp_idx 1) redirects to the index-K mirror ptrs too, so the P->D
        # transfer lands the full-history index-K in DRAM (parallel to c4). Both c4 (grp 0)
        # and indexer (grp 1) become DRAM; c128 (grp 2) stays VRAM. Contiguous DRAM prefix
        # (c4 layers then indexer layers) so the NIXL split (index_dram_layer_num) can mark them.
        _idx_off = _eng is not None and getattr(_eng, "index_k_offload", False)

        # Path-I SELECTIVE: the DRAM (mirror) set is NO LONGER a contiguous prefix — only the
        # OFFLOAD indexer layers redirect to a mirror; the target layers stay VRAM interleaved
        # among them. So we build an explicit per-position DRAM mask (True == this buf's dst is
        # a host mirror). This mask is authoritative (built here where the redirect happens) and
        # is threaded D->P so the NIXL split marks EXACTLY the mirror positions as DRAM,
        # regardless of contiguity. Stored on self for decode.py to read into kv_args.
        _dram_mask: List[bool] = []

        for _grp_idx, bufs in enumerate([
            self.c4_kv_pool.kv_buffer,
            self.c4_indexer_kv_pool.index_k_with_scale_buffer,
            self.c128_kv_pool.kv_buffer,
        ]):
            is_c4 = _grp_idx == 0
            is_indexer = _grp_idx == 1
            for _layer_in_grp, buf in enumerate(bufs):
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                if is_c4 and _b_mode:
                    # redirect c4 dst to the host mirror for this compress-layer
                    m_ptr, m_nbytes, m_item = _eng.mirror_buf_info(_layer_in_grp)
                    data_ptrs.append(m_ptr)
                    data_lens.append(m_nbytes)
                    item_lens.append(m_item)
                    _dram_mask.append(True)
                elif (is_indexer and _idx_off
                      and _eng.index_reserve_buf[_layer_in_grp] is not None):
                    # Path-I: OFFLOAD layer → redirect index-K dst to the index-K host mirror.
                    # (Target/full layers have index_reserve_buf[clid]=None → stay VRAM below.)
                    m_ptr, m_nbytes, m_item = _eng.index_mirror_buf_info(_layer_in_grp)
                    data_ptrs.append(m_ptr)
                    data_lens.append(m_nbytes)
                    item_lens.append(m_item)
                    _dram_mask.append(True)
                else:
                    data_ptrs.append(buf.data_ptr())
                    data_lens.append(buf.nbytes)
                    item_lens.append(buf[0].nbytes)
                    _dram_mask.append(False)

        # publish the mask (decode.py reads it into kv_args.kv_dram_mask; None-safe for pools
        # that never call this / non-offload runs where it is all-False → equivalent to prefix).
        self._kv_dram_mask = _dram_mask
        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for buf in self.swa_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)

        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                if pool is None:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        c4_state_pool_size = self.c4_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        self.compress_state_pools: List[CompressStatePool] = []
        self.indexer_compress_state_pools: List[CompressStatePool] = []

        for ratio in self.compression_ratios:
            overlap = ratio == 4
            compress_state_pool = indexer_compress_state_pool = None
            size = c4_state_pool_size if ratio == 4 else c128_state_pool_size
            ring_size = self.get_ring_size(ratio) if ratio != 0 else 0
            if ratio != 0:
                compress_state_pool = CompressStatePool(
                    size=size,
                    swa_page_size=self.swa_page_size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                    dtype=self.state_dtype,
                    device=self.device,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                    online=(ratio == 128 and ONLINE_C128),
                )

            if ratio == 4:
                indexer_compress_state_pool = CompressStatePool(
                    size=size,
                    swa_page_size=self.swa_page_size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,
                    device=self.device,
                    dtype=self.state_dtype,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                )

            self.compress_state_pools.append(compress_state_pool)
            self.indexer_compress_state_pools.append(indexer_compress_state_pool)

    def _init_compressed_layer_mapping(self):
        c1_cnt, c4_cnt, c128_cnt = 0, 0, 0
        self.layer_mapping: List[DeepSeekV4LayerItem] = []

        for ratio in self.compression_ratios:
            if ratio == 0:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=0,
                        compress_layer_id=c1_cnt,
                    )
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=4,
                        compress_layer_id=c4_cnt,
                        compress_kv_pool=self.c4_kv_pool,
                    )
                )
                c4_cnt += 1
            elif ratio == 128:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=128,
                        compress_layer_id=c128_cnt,
                        compress_kv_pool=self.c128_kv_pool,
                    )
                )
                c128_cnt += 1
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        compress_state_pool = self.compress_state_pools[layer_id]
        assert (
            compress_state_pool is not None
        ), "Only c4/c128 layers have attention states."
        return compress_state_pool

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert (
            indexer_compress_state_pool is not None
        ), "Only c4 layers have indexer states."
        return indexer_compress_state_pool

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.get_key_buffer(layer_id)


    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        # Path-P true dual-pool: when the swap engine owns a SEPARATE reserve buffer,
        # c4 attention must read THAT (recall writes keep-set cells there + remap
        # returns indices into it), NOT the c4_kv_pool landing area. c128 unaffected.
        _eng = getattr(self, "_swap_engine_p", None)
        if (
            _eng is not None
            and getattr(_eng, "_own_reserve", False)
            and compress_kv_pool is self.c4_kv_pool
        ):
            return _eng.get_reserve_buffer(compress_layer_id)
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id, seq_len, page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        self.swa_kv_pool.set_key_buffer(
            layer_id, swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.get_key_buffer(layer_id)

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if self._should_cache_swa:
            if layer_id == 0:
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
            swa_loc = self.cached_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        return self.swa_kv_pool.set_key_buffer_fused(layer_id, swa_loc, cache_k)

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)

