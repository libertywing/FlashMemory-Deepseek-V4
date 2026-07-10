
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseekv4_memory_pool import get_compress_state_ring_size

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class DSv4PoolSizes:

    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


class DSv4MemoryCalculator:

    def __init__(
        self,
        model_config: ModelConfig,
        page_size: int,
        swa_ratio: float,
        is_speculative: bool = False,
        c4_shrink_factor: int = 1,
        credit_offloaded_c4: bool = False,
        swa_max_tokens_override: int = 0,
        max_full_tokens_cap: int = 0,
    ):
        self.qk_nope_head_dim = model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_config.qk_rope_head_dim
        self.indexer_head_dim = model_config.index_head_dim
        self.compression_ratios = model_config.compress_ratios
        self.swa_page_size = model_config.window_size
        self.page_size = page_size
        self.swa_ratio = swa_ratio
        self.is_speculative = is_speculative
        assert c4_shrink_factor >= 1
        self.c4_shrink_factor = c4_shrink_factor
        # Path-P: the c4 classical KV is offloaded to a CPU mirror and the GPU c4
        # pool is pinned tiny (SGLANG_RETRIEVER_C4_DEVICE_TOKENS), so it no longer
        # scales with full_token. Crediting drops the c4 classical term from the
        # per-token budget so full_token (admission ceiling) reflects real GPU use.
        self.credit_offloaded_c4 = credit_offloaded_c4
        # Path-P: SWA KV is window-bounded (slots recycled via _evict_swa /
        # free_swa as the window slides), so the pool only needs
        # max_running_requests * window * margin, NOT swa_ratio * full_token.
        # >0 pins swa_tokens to this fixed value, decoupling it (and the
        # swa-space-indexed c4_state / c128_state pools) from full_token growth.
        self.swa_max_tokens_override = swa_max_tokens_override
        # Path-P: hard upper bound on full_token (admission budget). With swa
        # decoupled, the budget-derived full_token can balloon far past the
        # concurrency target (indexer would eat all GPU); this caps it to a
        # predictable target (e.g. max_running_requests * context_len). 0 = uncapped.
        self.max_full_tokens_cap = max_full_tokens_cap

        self.c4_ring_size = get_compress_state_ring_size(4, self.is_speculative)
        self.c128_ring_size = get_compress_state_ring_size(128, self.is_speculative)

        self.num_layers_total = len(self.compression_ratios)
        self.num_layers_ca4 = sum(1 for r in self.compression_ratios if r == 4)
        self.num_layers_ca128 = sum(1 for r in self.compression_ratios if r == 128)

        self.bytes_per_full_token = self.get_bytes_per_full_token()

    def get_bytes_per_full_token(self) -> float:
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        quant_block_size = 128
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )

        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        state_dtype_size = 4
        c4_state_bytes = 2 * 2 * attn_head_dim * state_dtype_size
        # Online c128 stores (max, sum, kv) per slot (3*head_dim) instead of
        # raw (kv, score) (2*head_dim). Combined with ring_size=1 this still
        # nets a large reduction (~3/256x) but the per-slot bytes go up.
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        c128_state_bytes = (3 if c128_online else 2 * 1) * attn_head_dim * state_dtype_size
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * state_dtype_size

        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        c128_state_ratio = self.c128_ring_size / self.swa_page_size

        c4_frac = 1 / (4 * self.c4_shrink_factor)
        # Path-P credit: c4 classical KV is offloaded to CPU (GPU pool pinned tiny,
        # not scaling with full_token), so it must NOT charge the per-token budget.
        c4_kv_term = 0.0 if self.credit_offloaded_c4 else c4_frac * kv_bytes * self.num_layers_ca4
        # Path-P swa decouple: when swa is pinned to a fixed override, the swa pool
        # and the swa-space-indexed state pools (c4_state / c128_state /
        # c4_indexer_state) no longer scale with full_token; their fixed cost is
        # subtracted from available_bytes in calculate_pool_sizes instead. Zero
        # their per-token terms here so they don't inflate bytes_per_full_token.
        swa_coupled = 0.0 if self.swa_max_tokens_override > 0 else 1.0
        bytes_per_full_token = (
            swa_coupled * self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_kv_term
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            + swa_coupled
            * self.swa_ratio
            * c4_state_ratio
            * c4_state_bytes
            * self.num_layers_ca4
            + swa_coupled
            * self.swa_ratio
            * c128_state_ratio
            * c128_state_bytes
            * self.num_layers_ca128
            + swa_coupled
            * self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )

        return bytes_per_full_token

    def _fixed_swa_state_bytes(self, swa_tokens: int) -> int:
        """Total GPU bytes of the swa_kv pool + swa-space-indexed state pools
        (c4_state, c128_state, c4_indexer_state) for a GIVEN fixed swa_tokens.
        Used when swa is decoupled from full_token: this is a one-time fixed
        cost subtracted from available_bytes before sizing full_token."""
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8
        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        state_dtype_size = 4
        c4_state_bytes = 2 * 2 * attn_head_dim * state_dtype_size
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        c128_state_bytes = (3 if c128_online else 2 * 1) * attn_head_dim * state_dtype_size
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * state_dtype_size
        n_state_slots = swa_tokens // self.swa_page_size
        return int(
            swa_tokens * kv_bytes * self.num_layers_total
            + n_state_slots * self.c4_ring_size * c4_state_bytes * self.num_layers_ca4
            + n_state_slots * self.c128_ring_size * c128_state_bytes * self.num_layers_ca128
            + n_state_slots * self.c4_ring_size * c4_indexer_state_bytes * self.num_layers_ca4
        )

    def _resident_gpu_bytes(self, full_token: int, swa_tokens: int) -> int:
        """Estimate the PHYSICALLY-RESIDENT GPU bytes at the given pool sizes,
        used as an OOM guard. Under Path-P (credit_offloaded_c4), the c4 classical
        KV pool is pinned tiny via SGLANG_RETRIEVER_C4_DEVICE_TOKENS (handled in the
        mixin, not here) so it is excluded; everything else is counted at full size.
        Conservative: counts indexer (full-history scoring, the real wall), c128_kv,
        and the fixed swa+state cost. If credit is OFF this also adds the c4 term."""
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8
        quant_block_size = 128
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )
        resident = self._fixed_swa_state_bytes(swa_tokens)
        resident += (full_token // 4) * indexer_bytes * self.num_layers_ca4
        resident += (full_token // 128) * kv_bytes * self.num_layers_ca128
        if not self.credit_offloaded_c4:
            resident += (
                (full_token // (4 * self.c4_shrink_factor))
                * kv_bytes
                * self.num_layers_ca4
            )
        return int(resident)

    def calculate_pool_sizes(self, available_bytes: int) -> DSv4PoolSizes:
        # Path-P swa decouple: pin swa to a fixed size, subtract its (and the
        # swa-indexed state pools') one-time GPU cost from available_bytes, then
        # size full_token from what remains. bytes_per_full_token already excludes
        # the swa-coupled terms (swa_coupled=0), so the remaining budget funds the
        # full_token-coupled pools (indexer, c128_kv, and credited-away c4).
        if self.swa_max_tokens_override > 0:
            swa_tokens = (
                self.swa_max_tokens_override // self.page_size * self.page_size
            )
            fixed_swa_bytes = self._fixed_swa_state_bytes(swa_tokens)
            budget_for_full = max(1, available_bytes - fixed_swa_bytes)
            full_token = int(budget_for_full / self.bytes_per_full_token)
            full_token = full_token // self.page_size * self.page_size
        else:
            full_token = int(available_bytes / self.bytes_per_full_token)
            full_token = full_token // self.page_size * self.page_size
            swa_tokens = (
                int(full_token * self.swa_ratio) // self.page_size * self.page_size
            )

        # Cap full_token to the concurrency target (predictable ceiling) before the
        # physical OOM guard. Without this, a decoupled-swa budget can balloon the
        # admission budget far past what's wanted, letting indexer consume all GPU.
        if self.max_full_tokens_cap > 0 and full_token > self.max_full_tokens_cap:
            full_token = (
                self.max_full_tokens_cap // self.page_size * self.page_size
            )
        # OOM guard: keep the physically-resident pools within a SAFETY FRACTION of
        # available_bytes (default 0.90), not 100%, to leave physical headroom for
        # activations / allocator fragmentation. The credit/decouple math targets the
        # full budget, so without this margin a large swa-override could push pools to
        # ~100% of available and OOM at decode. Clamp full_token down if over.
        # ONLY active when a budget-reshaping feature is on — otherwise behavior is
        # byte-identical to the original (the _resident_gpu_bytes estimate is coarse
        # and must not perturb the default path).
        _reshaping = (
            self.credit_offloaded_c4
            or self.swa_max_tokens_override > 0
            or self.max_full_tokens_cap > 0
        )
        safety = float(os.environ.get("SGLANG_DSV4_BUDGET_SAFETY_FRAC", "0.90"))
        budget_cap = int(available_bytes * safety)
        resident = self._resident_gpu_bytes(full_token, swa_tokens)
        if _reshaping and resident > budget_cap and full_token > self.page_size:
            scale = budget_cap / resident
            clamped = int(full_token * scale) // self.page_size * self.page_size
            clamped = max(self.page_size, clamped)
            logger.warning(
                f"[DSv4 OOM guard] resident={resident / (1<<30):.2f} GB > "
                f"budget_cap={budget_cap / (1<<30):.2f} GB "
                f"(={safety:.2f} x available {available_bytes / (1<<30):.2f} GB); "
                f"clamping full_token {full_token} -> {clamped} (scale {scale:.3f}). "
                f"Raise GPU mem_fraction or lower SWA/concurrency target."
            )
            full_token = clamped
            if self.swa_max_tokens_override <= 0:
                swa_tokens = (
                    int(full_token * self.swa_ratio)
                    // self.page_size
                    * self.page_size
                )

        pool_sizes = DSv4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            c4_state_pool_size=swa_tokens // self.swa_page_size * self.c4_ring_size,
            c128_state_pool_size=swa_tokens // self.swa_page_size * self.c128_ring_size,
        )

        logger.info(
            f"DSv4 memory calculation: "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"full_token={full_token}, swa_tokens={swa_tokens}, "
            f"credit_c4={self.credit_offloaded_c4}, "
            f"swa_override={self.swa_max_tokens_override}"
        )

        return pool_sizes

    def get_pool_sizes_by_profiling(self, mr: ModelRunner) -> DSv4PoolSizes:
        available_bytes = profile_available_bytes(
            device=mr.device,
            gpu_id=mr.gpu_id,
            total_gpu_memory=mr.total_gpu_memory,
            mem_fraction_static=mr.mem_fraction_static,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        if self.is_speculative:
            draft_layers = 1
            target_layers = self.num_layers_total
            target_ratio = target_layers / (target_layers + draft_layers)
            available_bytes = int(available_bytes * target_ratio)

        return self.calculate_pool_sizes(available_bytes)

    def get_pool_sizes_by_configuration(self, max_total_tokens: int) -> DSv4PoolSizes:
        available_bytes = max_total_tokens * self.bytes_per_full_token
        return self.calculate_pool_sizes(available_bytes)


def profile_available_bytes(
    device: str,
    gpu_id: int,
    total_gpu_memory: float,
    mem_fraction_static: float,
    distributed: bool = False,
    cpu_group=None,
) -> int:
    from sglang.srt.utils.common import get_available_gpu_memory

    available_gpu_memory = get_available_gpu_memory(
        device, gpu_id, distributed=distributed, cpu_group=cpu_group
    )
    rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)

    available_bytes = int(rest_memory * (1 << 30))

    logger.info(
        f"Memory profiling: available_gpu_memory={available_gpu_memory:.2f} GB, "
        f"total_gpu_memory={total_gpu_memory:.2f} GB, "
        f"mem_fraction_static={mem_fraction_static:.2f}, "
        f"rest_memory={rest_memory:.2f} GB"
    )

    return available_bytes
