"""
retriever_hook_impl.py — Retriever Hook 实现
================================================

本文件提供多种 RetrieverHook，用于替换 sglang 原生 FP8 indexer 打分：

  1. MockRetrieverHook            — 随机 top-K（数据流验证）
  2. NoSwapRetrieverHook          — 跳过所有 swap_in（性能上限基准）
  3. TrainedRetrieverHook         — 真实 Retriever (LightningIndexer ckpt)
  4. TrainedEnsembleRetrieverHook — 单 ckpt 内 N 层 ensemble
  5. MultiRetrieverHook           — N 个独立 ckpt, 每请求路由（共享 prefix cache）
  6. WeakBaselineHook             — recency-only / random baselines

接口约定（被 indexer.py forward_c4_indexer 调用）:

    overridden, raw_logical_indices = hook.maybe_override_topk(
        x, forward_batch, indexer_metadata, core_metadata,
        token_to_kv_pool, c4_indexer,
    )

返回值:
  (False, None)                 — 不覆盖，走原生 FP8 路径
  (True, raw_logical_indices)   — 覆盖，HiSparse 路径用 swap_in_selected_pages
                                  raw_logical_indices: [B, top_k] int32, 每个值 ∈ [0, n_chunks)
  (True, None)                  — NoSwap 模式：hook 已直接写入 c4_sparse_page_indices

环境变量启用:
  SGLANG_RETRIEVER_MODE=mock|no_swap|trained|trained_ensemble|multi|recency_only|random|off
  SGLANG_RETRIEVER_CHECKPOINT=/path/to/ckpt.pt   (trained / ensemble 单 ckpt 模式)
  SGLANG_RETRIEVER_CHECKPOINTS=name1:/p1,name2:/p2,...  (multi 模式)
  SGLANG_RETRIEVER_DEFAULT_NAME=name1            (multi 模式 fallback)
  SGLANG_RETRIEVER_WEIGHT_DIR=/path/to/weights/  (trained 模式 fallback)
"""

import json
import logging
import os
import math
import time

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Shared RoPE freqs_cis cache: {(max_position, device_str): tensor}. All trained
# scorers use identical RoPE params, so the (max_pos, ROPE_DIM/2) complex table is
# built once and reused across retrievers/layers (saves startup time + memory).
_FREQS_CIS_CACHE = {}


# ─────────────────────────────────────────────────────────────────────────────
#                           Mock Retriever Hook
# ─────────────────────────────────────────────────────────────────────────────


class MockRetrieverHook:
    """随机选择 top-K logical chunk indices；用于验证 hook + HiSparse swap 数据流。"""

    def __init__(self, top_k: int = 512, retrieval_interval: int = 64):
        self.top_k = top_k
        self.retrieval_interval = retrieval_interval
        self._call_count = 0
        self._override_count = 0
        logger.info(
            f"[RetrieverHook] MockRetrieverHook init: top_k={top_k}, interval={retrieval_interval}"
        )

    def maybe_override_topk(
        self, x, forward_batch, indexer_metadata, core_metadata,
        token_to_kv_pool, c4_indexer,
    ):
        self._call_count += 1

        B = x.shape[0]
        c4_seq_lens = indexer_metadata.c4_seq_lens
        if c4_seq_lens.dim() == 2:
            c4_seq_lens = c4_seq_lens.squeeze(-1)

        device = x.device
        sparse_indices = core_metadata.c4_sparse_page_indices  # [B, top_k]
        top_k = sparse_indices.shape[1]

        raw_logical_indices = torch.zeros(B, top_k, dtype=torch.int32, device=device)
        for i in range(B):
            n_chunks = int(c4_seq_lens[i].item())
            if n_chunks <= 0:
                raw_logical_indices[i] = -1
                continue

            if n_chunks <= top_k:
                raw_logical_indices[i, :n_chunks] = torch.arange(
                    n_chunks, dtype=torch.int32, device=device
                )
                raw_logical_indices[i, n_chunks:] = n_chunks - 1
            else:
                perm = torch.randperm(n_chunks, device=device)[:top_k]
                raw_logical_indices[i] = perm.sort()[0].to(torch.int32)

        self._override_count += 1
        if self._override_count % 100 == 1:
            logger.info(
                f"[MockHook] override #{self._override_count} B={B} avg_chunks={c4_seq_lens.float().mean():.0f}"
            )
        return True, raw_logical_indices


def create_mock_hook(top_k=512, interval=64):
    return MockRetrieverHook(top_k=top_k, retrieval_interval=interval)


# ─────────────────────────────────────────────────────────────────────────────
#                       No-Swap Hook (性能上限基准)
# ─────────────────────────────────────────────────────────────────────────────


class NoSwapRetrieverHook:
    """
    Benchmark mode: 完全跳过 swap_in。
    Prefill 后 HiSparse 已做初始 offload；decode 直接复用 device buffer 中已有的 chunks。
    用于测量 offload 带来的最大可能加速（零 swap 开销）。
    """

    def __init__(self):
        self._call_count = 0
        self.logger = logging.getLogger(__name__)
        self.logger.info("[RetrieverHook] NoSwapRetrieverHook init (skip all swap_in)")

    def maybe_override_topk(self, x, forward_batch, indexer_metadata, core_metadata,
                            token_to_kv_pool, c4_indexer):
        self._call_count += 1

        hisparse_coordinator = forward_batch.hisparse_coordinator
        if hisparse_coordinator is None:
            return False, None

        B = x.shape[0]
        top_k = core_metadata.c4_sparse_page_indices.shape[1]
        layer_id_compressed = token_to_kv_pool.layer_mapping[c4_indexer.layer_id].compress_layer_id
        req_pool_indices = forward_batch.req_pool_indices  # [B] int

        # Vectorized: gather all B requests' device-buffer locations in one op
        # req_device_buffer_token_locs shape: [layers, max_num_reqs, padded_buffer_size]
        # → [B, top_k]
        buf_locs_all = hisparse_coordinator.req_device_buffer_token_locs[
            layer_id_compressed, req_pool_indices.long(), :top_k
        ]
        core_metadata.c4_sparse_page_indices.copy_(
            buf_locs_all.to(core_metadata.c4_sparse_page_indices.dtype)
        )

        if self._call_count % 500 == 1:
            self.logger.info(f"[NoSwapHook] call #{self._call_count} B={B}")

        return True, None  # signals indexer.py to skip swap_in


def create_no_swap_hook():
    return NoSwapRetrieverHook()


# ─────────────────────────────────────────────────────────────────────────────
#                  Trained Retriever Hook (LightningIndexer ckpt)
# ─────────────────────────────────────────────────────────────────────────────


class _TrainedScorer(torch.nn.Module):
    """
    LightningIndexer 推理模块，从 train.py 训出的 ckpt (.pt) 加载。

    训练 vs 推理一致性：
      - wq_a / wq_b / q_norm / weights_proj 都来自 ckpt
      - 不做 FP8 量化（与训练 forward path 一致）
      - 输入: hidden_state [B, 4096], compressed_k [N, 132], positions [B]
      - 输出: logits [B, N]
    """

    N_HEADS = 64
    HEAD_DIM = 128
    ROPE_DIM = 64
    Q_LORA_RANK = 1024
    HIDDEN_DIM = 4096
    ROPE_BASE = 160000.0
    ROPE_FACTOR = 16.0
    ROPE_ORIGINAL_SEQ_LEN = 65536
    ROPE_BETA_FAST = 32.0
    ROPE_BETA_SLOW = 1.0
    RMS_NORM_EPS = 1e-6

    def __init__(self, ckpt_path: str, device: str, max_position: int = None,
                 joint_layer_key: str = None, preloaded_state: dict = None):
        """
        Args:
            ckpt_path: path to .pt (used for logging / fallback load)
            device: cuda or cpu
            max_position: precompute RoPE freqs to this position. If None, read from
                env SGLANG_RETRIEVER_MAX_POSITION (default 1048576 = 1M, DeepSeek-V4
                standard context). The RoPE freqs_cis table must cover the LARGEST
                token position the retriever scores; positions beyond it get clamped
                (RoPE becomes inaccurate). 1M supports full-length DSv4 inputs.
            joint_layer_key: if ckpt is in joint format (keys prefixed with
                'retrievers.l{lid}.'), specify which sub-state to load (e.g. "l10").
                If None and ckpt is joint, raises an error. Ignored for single-layer ckpts.
            preloaded_state: optional already-loaded state dict (torch.load result).
                When the caller loads a joint ckpt once and constructs N per-layer
                scorers from it, passing the shared dict avoids re-reading the
                (~900MB) ckpt from disk N times — a big startup speedup over slow
                shared filesystems.
        """
        super().__init__()
        if max_position is None:
            max_position = int(os.environ.get("SGLANG_RETRIEVER_MAX_POSITION", "1048576"))
        if preloaded_state is not None:
            state = preloaded_state
        else:
            assert os.path.exists(ckpt_path), f"ckpt not found: {ckpt_path}"
            state = torch.load(ckpt_path, map_location=device, weights_only=True)

        # ── Joint-format detection (matches inference.py convention) ────────
        # Joint ckpt keys look like 'retrievers.l10.wq_a.weight' etc., where
        # the 'l10/l12/l20' refers to CSA layer index (NOT transformer layer ID).
        is_joint = any(k.startswith("retrievers.l") for k in state.keys())
        if is_joint:
            assert joint_layer_key is not None, (
                f"Joint ckpt detected but no joint_layer_key passed. "
                f"Available retrievers in ckpt: "
                f"{sorted({k.split('.')[1] for k in state if k.startswith('retrievers.')})}"
            )
            prefix = f"retrievers.{joint_layer_key}."
            sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            assert sub, (
                f"Joint ckpt has no keys with prefix '{prefix}'. "
                f"Available retrievers in ckpt: "
                f"{sorted({k.split('.')[1] for k in state if k.startswith('retrievers.')})}"
            )
            logger.info(f"[_TrainedScorer] joint ckpt: extracted {joint_layer_key} sub-state "
                        f"({len(sub)} keys)")
            state = sub

        self.wq_a = state["wq_a.weight"].to(torch.float32).to(device)              # [1024, 4096]
        self.wq_b = state["wq_b.weight"].to(torch.float32).to(device)              # [N_HEADS*128, 1024]
        self.q_norm_weight = state["q_norm_weight"].to(torch.float32).to(device)   # [1024]
        self.w_proj = state["weights_proj.weight"].to(torch.float32).to(device)    # [N_HEADS, 4096]

        # Auto-detect N_HEADS from wq_b: shape is [N_HEADS * HEAD_DIM, Q_LORA_RANK]
        # P-series (P110/P240/P241) and R601 use N_HEADS=64; R651 uses N_HEADS=128.
        n_heads_detected = self.wq_b.shape[0] // self.HEAD_DIM
        if n_heads_detected != self.N_HEADS:
            logger.info(
                f"[_TrainedScorer] auto-detected N_HEADS={n_heads_detected} "
                f"(class default {self.N_HEADS}) from wq_b shape {tuple(self.wq_b.shape)}"
            )
            self.N_HEADS = n_heads_detected
        self.weight_scale = self.HEAD_DIM ** -0.5 * self.N_HEADS ** -0.5

        # ── Logit calibration offset (post-forward) ─────────────────────────
        # R-series joint ckpts (R601/R651) were optimized for r@K ranking, not
        # sigmoid threshold calibration. The default sigmoid > 0.5 threshold is
        # too strict (only ~1-2% of chunks pass). Adding a constant offset to
        # raw logits shifts the sigmoid up so the threshold catches more chunks.
        # Set via SGLANG_RETRIEVER_LOGIT_OFFSET (default 0.0). Empirically
        # +5.5 brings R601 / R651 close to P-series (sigmoid > 0.5 trained) calibration.
        self.logit_offset = float(os.environ.get("SGLANG_RETRIEVER_LOGIT_OFFSET", "0.0"))
        if self.logit_offset != 0.0:
            logger.info(
                f"[_TrainedScorer] applying logit offset +{self.logit_offset} "
                f"(set via SGLANG_RETRIEVER_LOGIT_OFFSET)"
            )

        self.device = device

        # 预计算 RoPE (max_position controls longest scorable token position; 1M default)
        # Cache the table across scorers: all retrievers/layers share identical RoPE
        # params, so the (1M, 32) table is built once per (max_position, device) and
        # reused — avoids rebuilding a ~270MB table N times at startup.
        cache_key = (max_position, str(device))
        cached = _FREQS_CIS_CACHE.get(cache_key)
        if cached is None:
            cached = self._precompute_freqs_cis(max_position).to(device)
            _FREQS_CIS_CACHE[cache_key] = cached
            logger.info(f"[_TrainedScorer] RoPE max_position={max_position} "
                        f"(freqs_cis table {tuple(cached.shape)}, built+cached)")
        self.freqs_cis = cached  # [max_pos, ROPE_DIM/2] complex64 (shared, read-only)

    @classmethod
    def _yarn_correction_dim(cls, n_rot, d_model, base, max_pos):
        return (d_model * math.log(max_pos / (n_rot * 2 * math.pi))) / (2 * math.log(base))

    def _precompute_freqs_cis(self, seqlen: int) -> torch.Tensor:
        dim = self.ROPE_DIM
        base = self.ROPE_BASE
        factor = self.ROPE_FACTOR
        orig = self.ROPE_ORIGINAL_SEQ_LEN
        low = max(math.floor(self._yarn_correction_dim(self.ROPE_BETA_FAST, dim, base, orig)), 0)
        high = min(math.ceil(self._yarn_correction_dim(self.ROPE_BETA_SLOW, dim, base, orig)), dim // 2 - 1)
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        ramp = torch.zeros(dim // 2)
        for i in range(dim // 2):
            if i < low:
                ramp[i] = 0.0
            elif i >= high:
                ramp[i] = 1.0
            else:
                ramp[i] = (i - low) / max(high - low, 1)
        mixed = freqs * (1 - ramp) + (freqs / factor) * ramp
        t = torch.arange(seqlen, dtype=torch.float32)
        angles = torch.outer(t, mixed)
        return torch.polar(torch.ones_like(angles), angles)

    @torch.no_grad()
    def _apply_rope(self, q, positions):
        """q: [B, H, head_dim], positions: [B] → q after RoPE on last ROPE_DIM dims."""
        head_dim = q.shape[-1]
        q_pass = q[..., : head_dim - self.ROPE_DIM]
        q_rope = q[..., head_dim - self.ROPE_DIM:]
        q_c = torch.view_as_complex(
            q_rope.float().reshape(*q_rope.shape[:-1], self.ROPE_DIM // 2, 2).contiguous()
        )
        # 安全 clamp，防止 positions 越界（freqs_cis 的 max_pos）
        max_pos = self.freqs_cis.shape[0] - 1
        positions_safe = positions.clamp(0, max_pos)
        freqs = self.freqs_cis[positions_safe].unsqueeze(1)  # [B, 1, rope_dim/2]
        q_rot = torch.view_as_real(q_c * freqs).reshape(*q_rope.shape).to(q.dtype)
        return torch.cat([q_pass, q_rot], dim=-1)

    @torch.no_grad()
    def _hadamard(self, x):
        *leading, d = x.shape
        h = x.float()
        s = 1
        while s < d:
            h = h.view(*leading, d // (2 * s), 2, s)
            a, b = h[..., 0, :], h[..., 1, :]
            h = torch.stack([a + b, a - b], dim=-2).view(*leading, d)
            s *= 2
        return h / math.sqrt(d)

    @torch.no_grad()
    def _rmsnorm(self, x):
        x_f = x.float()
        norm = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.RMS_NORM_EPS)
        return x_f / norm * self.q_norm_weight

    @torch.no_grad()
    def compute_q_fp8(self, hidden, positions):
        """Compute the retriever's q-projection (same as forward's Q-side) and quantize to
        fp8 for the paged MQA-logits kernel — so the retriever can score the PAGED index-K
        pool DIRECTLY (via the model's real page_table), with NO _extract_compressed_k gather.

        Returns (q_fp8 [B,1,64,128] float8_e4m3fn, fused_weights [B,64] float32) where
        fused_weights = per_head_w * weight_scale * q_scale (q's act_quant scale folded in,
        exactly as the native indexer does via fused_scale, so the kernel output matches the
        fp32 einsum up to fp8(q) rounding). sigmoid/threshold stay in the caller (post-kernel).
        """
        x = hidden.float()
        B = x.shape[0]
        q_lora = F.linear(x, self.wq_a)
        q_lora = self._rmsnorm(q_lora)
        q = F.linear(q_lora, self.wq_b)
        q = q.view(B, self.N_HEADS, self.HEAD_DIM)
        q = self._apply_rope(q.to(torch.bfloat16), positions.to(torch.int64)).float()
        q = self._hadamard(q)                                            # [B, 64, 128]
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
        # act_quant expects [..., D] with D % 128 == 0; flatten heads into the row dim so
        # each head's 128-dim vector is one quant block (matches native per-head q quant).
        q_flat = q.reshape(B * self.N_HEADS, self.HEAD_DIM).contiguous()
        q_fp8, q_scale = act_quant(q_flat)                               # [B*64,128], [B*64,1]
        q_fp8 = q_fp8.view(B, 1, self.N_HEADS, self.HEAD_DIM)            # kernel wants [B,1,H,D]
        q_scale = q_scale.view(B, self.N_HEADS)                          # [B,64]
        per_head_w = F.linear(x, self.w_proj)                            # [B,64]
        fused_w = per_head_w * self.weight_scale * q_scale               # fold q_scale in
        return q_fp8, fused_w

    @torch.no_grad()
    def forward(self, hidden, k_fp8, k_scale, positions):
        """
        Single-K-shared-across-batch scoring (legacy / unit-test API).

        hidden:    [B, 4096]    bf16/float32
        k_fp8:     [N, 128]     float8_e4m3fn (already viewed)
        k_scale:   [N]          float32
        positions: [B]          int64
        Returns:   logits [B, N] float32
        """
        k_float = k_fp8.float() * k_scale.unsqueeze(-1)  # [N, 128]

        x = hidden.float()
        B = x.shape[0]
        q_lora = F.linear(x, self.wq_a)                                  # [B, 1024]
        q_lora = self._rmsnorm(q_lora)                                   # [B, 1024]
        q = F.linear(q_lora, self.wq_b)                                  # [B, 64*128]
        q = q.view(B, self.N_HEADS, self.HEAD_DIM)                       # [B, 64, 128]
        q = self._apply_rope(q.to(torch.bfloat16), positions.to(torch.int64)).float()
        q = self._hadamard(q)                                            # [B, 64, 128]

        per_head_w = F.linear(x, self.w_proj)                            # [B, 64]
        fused_w = per_head_w * self.weight_scale                          # [B, 64]

        # scores_per_head = relu(k @ q^T)
        scores = F.relu(torch.einsum("bhd,nd->bnh", q, k_float))          # [B, N, 64]
        logits = (scores * fused_w.unsqueeze(1)).sum(-1)                  # [B, N]
        if self.logit_offset != 0.0:
            logits = logits + self.logit_offset
        return logits

    @torch.no_grad()
    def forward_batched(self, hidden, k_fp8, k_scale, positions):
        """
        Per-request K scoring; each query has its own K cache slice.

        hidden:    [B, 4096]      bf16/float32
        k_fp8:     [B, T, 128]    float8_e4m3fn  (per-request K, may have padding)
        k_scale:   [B, T]         float32
        positions: [B]            int64
        Returns:   logits [B, T]  float32  (no masking — caller decides what to mask)
        """
        x = hidden.float()
        B = x.shape[0]

        # Q-side (same path as `forward`, but kept here so the einsum stays
        # batched along the request dim)
        q_lora = F.linear(x, self.wq_a)                                  # [B, 1024]
        q_lora = self._rmsnorm(q_lora)                                   # [B, 1024]
        q = F.linear(q_lora, self.wq_b)                                  # [B, 64*128]
        q = q.view(B, self.N_HEADS, self.HEAD_DIM)                       # [B, 64, 128]
        q = self._apply_rope(q.to(torch.bfloat16), positions.to(torch.int64)).float()
        q = self._hadamard(q)                                            # [B, 64, 128]

        per_head_w = F.linear(x, self.w_proj)                            # [B, 64]
        fused_w = per_head_w * self.weight_scale                          # [B, 64]

        # K-side dequant: [B, T, 128] float32
        k_float = k_fp8.float() * k_scale.unsqueeze(-1)

        # Score: bhd, btd → bth (per-batch dot product)
        scores_per_head = F.relu(torch.einsum("bhd,btd->bth", q, k_float))   # [B, T, 64]
        logits = (scores_per_head * fused_w.unsqueeze(1)).sum(-1)            # [B, T]
        if self.logit_offset != 0.0:
            logits = logits + self.logit_offset
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# (Removed) TrainedRetrieverHook + TrainedEnsembleRetrieverHook — single-ckpt
# variants superseded by MultiRetrieverHook below. Use
#   SGLANG_RETRIEVER_MODE=multi
#   SGLANG_RETRIEVER_CHECKPOINTS=name1:/path1[@layers],name2:/path2[@layers],...
# (a single-name spec is equivalent to the old `trained` / `trained_ensemble`)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#       Multi-Retriever Hook — per-request retriever routing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_retrievers_spec(spec_str: str) -> "tuple[dict[str, str], dict[str, list[int]]]":
    """Parse SGLANG_RETRIEVER_CHECKPOINTS spec.

    Format options:
      * 'name1:/path/a.pt,name2:/path/b.pt'                           (legacy)
      * 'name1:/path/a.pt@10_12_20,name2:/path/b.pt@6_8_10_12_14_16_18_20'
        (with per-retriever target layers)
      * mixed: entries without '@' fall back to global SGLANG_RETRIEVER_LAYERS

    Returns (retrievers_spec, layers_by_name) where layers_by_name only
    contains entries that explicitly specified '@layers' in the spec string.

    Whitespace and trailing commas tolerated. Order preserved (so the first
    entry can be used as the fallback default).
    """
    out: dict[str, str] = {}
    layers: dict[str, list[int]] = {}
    for chunk in spec_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Bad SGLANG_RETRIEVER_CHECKPOINTS entry {chunk!r} — "
                f"expected 'name:/path/to/ckpt.pt[@l1_l2_..]'"
            )
        name, path = chunk.split(":", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Bad SGLANG_RETRIEVER_CHECKPOINTS entry {chunk!r}")
        # Optional '@layers' suffix on path: name:/path.pt@10_12_20
        if "@" in path:
            path, layer_str = path.rsplit("@", 1)
            try:
                ll = sorted({int(x) for x in layer_str.split("_") if x.strip()})
            except ValueError:
                raise ValueError(
                    f"Bad layer list in {chunk!r}: expected '@l1_l2_l3', got {layer_str!r}"
                )
            if not ll:
                raise ValueError(f"Empty layer list in {chunk!r}")
            layers[name] = ll
        out[name] = path
    return out, layers


def _parse_thresh_fallback(spec_str: str) -> "dict[str, float]":
    """Parse SGLANG_RETRIEVER_THRESH_FALLBACK="R935:0.15,R950:0.30" → dict.

    Whitespace and trailing commas tolerated. Bad entries silently dropped.
    """
    out: dict[str, float] = {}
    for chunk in (spec_str or "").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, val = chunk.split(":", 1)
        name = name.strip()
        val = val.strip()
        if not name:
            continue
        try:
            out[name] = float(val)
        except ValueError:
            continue
    return out


class MultiRetrieverHook:
    """
    Multi-retriever ensemble hook with per-request routing.

    Loads N retriever checkpoints (each is a joint multi-layer ckpt) and
    decides per request — at runtime — which retriever's keep_mask to apply.
    The KV cache (and therefore the prefix cache) is fully shared across
    retrievers: only the logit-mask step changes per request, so two
    requests with the same prompt but different retriever_name share prefill.

    Routing
    -------
    Each request can carry::

        extra_body={"custom_params": {"retriever_name": "R921"}}

    in its OpenAI-completion payload. On the server side, we read
    ``forward_batch.sampling_info.custom_params[i]["retriever_name"]`` for
    row i. If absent / unknown, falls back to ``self.default_name``
    (= the first retriever loaded, or the value of
    ``SGLANG_RETRIEVER_DEFAULT_NAME``).

    Compatibility
    -------------
      * SGLANG_RETRIEVER_CHECKPOINTS = "name1:/path1,name2:/path2,..."
        loads multiple retrievers.
      * SGLANG_RETRIEVER_CHECKPOINT (legacy single-ckpt) is honored: if
        SGLANG_RETRIEVER_CHECKPOINTS is unset and CHECKPOINT is set, we load
        a single-name retriever (default name "default") so the rest of the
        plumbing is identical.
      * SGLANG_RETRIEVER_LAYERS / SIGMOID_THRESH / ENSEMBLE_MODE / LAST_KEEP /
        STATS_FILE behave the same as in TrainedEnsembleRetrieverHook.

    Cache layout
    ------------
    Cache is keyed by req_pool_idx (slot id). Each entry stores
    ``retriever_name`` so a slot reused by a different request (possibly
    with a different retriever) is correctly invalidated::

        self._cache[ri] = {
            "step": int,            # last fresh-score call_count
            "c4_seq": int,          # c4 length at last fresh score
            "retriever_name": str,  # which retriever produced the mask
            "keep_mask": bool[c4_seq],
            "partial": {lid: tensor},
            "partial_c4_seq": int,
        }
    """

    LAST_KEEP_TOKENS = int(os.environ.get("SGLANG_RETRIEVER_LAST_KEEP", "2048"))
    # Attention-sink: first N c4 chunks always kept. Each c4 chunk = 4 tokens, so
    # default 128 chunks ≈ 512 tokens — covers system instruction prefix.
    FIRST_KEEP_TOKENS = int(os.environ.get("SGLANG_RETRIEVER_FIRST_KEEP", "128"))
    RETRIEVAL_INTERVAL = int(os.environ.get("SGLANG_RETRIEVER_INTERVAL", "64"))
    SIGMOID_THRESHOLD = float(os.environ.get("SGLANG_RETRIEVER_SIGMOID_THRESH", "0.5"))
    ENSEMBLE_MODE = os.environ.get("SGLANG_RETRIEVER_ENSEMBLE_MODE", "mean").lower()

    # Per-retriever thresh-rate fallback: parsed from
    #   SGLANG_RETRIEVER_THRESH_FALLBACK="R935:0.20,R950:0.40"
    # For each named retriever, once the running cumulative thresh ratio
    # OVER THE EVICTABLE REGION (sum_thresh_in_evictable / sum_evictable
    # across all decode cycles in one request) exceeds the given threshold,
    # the hook sticky-switches that request to FULL ATTENTION
    # (keep_mask = all True) for all subsequent decode steps.
    # Evictable = c4_seq − sink − recent. Tokens forced-kept by sink/recent
    # are excluded from both numerator and denominator so the threshold
    # measures "demand from the freely-evictable region" only.
    # Stats records report n_thresh = c4_seq for triggered cycles, so
    # downstream metrics see thresh_rate = 100% for fallback cases.
    THRESH_FALLBACK_BY_NAME = _parse_thresh_fallback(
        os.environ.get("SGLANG_RETRIEVER_THRESH_FALLBACK", "")
    )

    def __init__(
        self,
        retrievers_spec: "dict[str, str]",
        device: str = "cuda",
        target_layers: list = None,
        default_name: str = None,
        layers_by_name: "dict[str, list[int]]" = None,
    ):
        assert retrievers_spec, "MultiRetrieverHook needs ≥1 retriever in retrievers_spec"
        self._call_count = 0
        if target_layers is None:
            ll = os.environ.get("SGLANG_RETRIEVER_LAYERS", "10,12,20")
            target_layers = sorted({int(x) for x in ll.split(",") if x.strip()})
        assert len(target_layers) >= 1, "Must have at least 1 target layer"

        # ── Per-retriever target_layers (4-way mixed-architecture support) ──
        # If layers_by_name is provided (from '@layers' suffix in CHECKPOINTS),
        # each retriever uses its own subset of layers. Retrievers without an
        # explicit entry fall back to the global target_layers (3-layer default
        # or whatever SGLANG_RETRIEVER_LAYERS specifies).
        layers_by_name = layers_by_name or {}
        self.target_layers_by_name: "dict[str, list[int]]" = {}
        all_layers: set = set()
        for name in retrievers_spec.keys():
            ll = layers_by_name.get(name, target_layers)
            self.target_layers_by_name[name] = sorted(set(ll))
            all_layers.update(self.target_layers_by_name[name])
        # Global "is_target" gate: union of all per-retriever target_layers.
        # The hook short-circuits when the current CSA layer is in NO retriever's
        # target_layers (no work to do for any row).
        self.target_layers = sorted(all_layers)
        self.first_target = self.target_layers[0]
        self.last_target = self.target_layers[-1]
        # Per-retriever first/last for cycle bounds (each retriever runs its own
        # "open partials → finalize keep_mask" cycle on its own target layers).
        self.first_target_by_name = {
            n: ls[0] for n, ls in self.target_layers_by_name.items()
        }
        self.last_target_by_name = {
            n: ls[-1] for n, ls in self.target_layers_by_name.items()
        }

        # name → {csa_layer_id → _TrainedScorer}
        self.scorers: "dict[str, dict[int, _TrainedScorer]]" = {}
        for name, ckpt_path in retrievers_spec.items():
            assert os.path.exists(ckpt_path), (
                f"[MultiRetrieverHook] ckpt for retriever {name!r} not found: {ckpt_path}"
            )
            # Load the (joint) ckpt ONCE per retriever, then build all per-layer
            # scorers from the shared in-memory state. Previously each layer
            # re-read the ~900MB ckpt from disk → slow startup on shared FS.
            shared_state = torch.load(ckpt_path, map_location=device, weights_only=True)
            per_layer: "dict[int, _TrainedScorer]" = {}
            for lid in self.target_layers_by_name[name]:
                joint_layer_key = f"l{lid}"
                try:
                    per_layer[lid] = _TrainedScorer(
                        ckpt_path, device=device, joint_layer_key=joint_layer_key,
                        preloaded_state=shared_state,
                    )
                except AssertionError:
                    # Fall back to single-layer load (no joint_layer_key) — only
                    # makes sense if every target layer uses the same scorer.
                    logger.warning(
                        f"[MultiRetrieverHook] {name}: ckpt is not joint format; "
                        f"using single-layer scorer for all target layers"
                    )
                    per_layer[lid] = _TrainedScorer(
                        ckpt_path, device=device, preloaded_state=shared_state,
                    )
            self.scorers[name] = per_layer

        # Default retriever for requests that don't specify one
        env_default = os.environ.get("SGLANG_RETRIEVER_DEFAULT_NAME", "")
        if default_name and default_name in self.scorers:
            self.default_name = default_name
        elif env_default and env_default in self.scorers:
            self.default_name = env_default
        else:
            # First-loaded retriever (preserves SGLANG_RETRIEVER_CHECKPOINTS order)
            self.default_name = next(iter(self.scorers))

        self.device = device
        self._cache: "dict[int, dict]" = {}
        self._stats_inited = False
        self._is_rank_0 = False
        self._stats_file = None
        # Diagnostics
        self._unknown_names_warned: set = set()
        self._dispatch_log_count = 0
        logger.info(
            f"[RetrieverHook] MultiRetrieverHook init: retrievers={list(self.scorers)}, "
            f"default={self.default_name!r}, target_layers={target_layers}, "
            f"sigmoid_thresh={self.SIGMOID_THRESHOLD}, ensemble_mode={self.ENSEMBLE_MODE}, "
            f"interval={self.RETRIEVAL_INTERVAL}, keep_last={self.LAST_KEEP_TOKENS}, "
            f"first_keep={self.FIRST_KEEP_TOKENS}, "
            f"thresh_fallback={dict(self.THRESH_FALLBACK_BY_NAME)}"
        )

    # ── stats file helper (rank-0 only) ─────────────────────────────────
    def _ensure_stats_file(self):
        if self._stats_inited:
            return
        self._stats_inited = True
        try:
            from sglang.srt.layers.dp_attention import get_attention_tp_rank
            self._is_rank_0 = (get_attention_tp_rank() == 0)
        except Exception:
            self._is_rank_0 = False
        if self._is_rank_0:
            stats_path = os.environ.get("SGLANG_RETRIEVER_STATS_FILE", "/tmp/hook_stats.jsonl")
            try:
                self._stats_file = open(stats_path, "w")
                logger.info(f"[MultiRetrieverHook] stats dump → {stats_path} (rank 0)")
            except Exception as e:
                logger.warning(f"[MultiRetrieverHook] stats file open failed: {e}")
                self._stats_file = None

    # ── route: which retriever for which row ────────────────────────────
    def _resolve_retriever_names(self, forward_batch, B: int) -> "list[str]":
        """Return per-row retriever name (length B), falling back to default
        when missing/unknown. Reads from ``forward_batch.sampling_info.custom_params``.
        Requires the patched sampling_batch_info.py that always populates the
        list (otherwise it will be None and every row falls back)."""
        sampling_info = getattr(forward_batch, "sampling_info", None)
        cp_list = getattr(sampling_info, "custom_params", None) if sampling_info is not None else None
        out: list[str] = [self.default_name] * B
        if cp_list is None:
            return out
        # cp_list length should equal B; defensive in case of mismatch
        n = min(B, len(cp_list))
        for i in range(n):
            cp = cp_list[i]
            if not isinstance(cp, dict):
                continue
            name = cp.get("retriever_name")
            if name is None:
                continue
            if name in self.scorers:
                out[i] = name
            else:
                if name not in self._unknown_names_warned:
                    self._unknown_names_warned.add(name)
                    logger.warning(
                        f"[MultiRetrieverHook] unknown retriever_name={name!r} → "
                        f"falling back to default {self.default_name!r}. Available: {list(self.scorers)}"
                    )
                # leave default
        return out

    def maybe_override_topk(self, *args, **kwargs):
        # Mask mode: never override the native top-k path.
        return False, None

    @torch.no_grad()
    def maybe_mask_logits(
        self, logits, c4_seq_lens, x, forward_batch, indexer_metadata, core_metadata,
        token_to_kv_pool, c4_indexer,
    ):
        self._call_count += 1
        if c4_seq_lens.dim() == 2:
            c4_seq_lens = c4_seq_lens.squeeze(-1)
        B = logits.shape[0]
        max_c4 = logits.shape[1]
        device = logits.device

        compress_layer_id = token_to_kv_pool.layer_mapping[
            c4_indexer.layer_id
        ].compress_layer_id
        # Global "is_target" gate: any retriever cares about this layer?
        is_target = (compress_layer_id in self.target_layers)
        # Legacy aliases (kept for stats dump compat) — meaning "this layer is
        # the global last target across ALL retrievers". Per-row first/last is
        # checked inside the loop using self.{first,last}_target_by_name.
        is_first_target = (compress_layer_id == self.first_target)
        is_last_target = (compress_layer_id == self.last_target)

        c4_seq_lens_cpu = c4_seq_lens.tolist()
        req_pool_indices_cpu = forward_batch.req_pool_indices.tolist()
        row_names = self._resolve_retriever_names(forward_batch, B)

        # ── First-time logging of dispatch decisions ─────────────────────
        if self._dispatch_log_count < 3:
            self._dispatch_log_count += 1
            from collections import Counter
            counts = Counter(row_names)
            logger.info(
                f"[MultiRetrieverHook] dispatch sample call#{self._call_count} "
                f"layer={compress_layer_id} B={B} retriever_counts={dict(counts)}"
            )

        # ── Decide which rows need fresh compute on THIS target layer ────
        # Per-row check: does each row's retriever care about this layer?
        # (Different retrievers can have different target_layers — see
        # SGLANG_RETRIEVER_CHECKPOINTS '@layers' suffix.)
        needs_fresh = [False] * B
        # Cache per-row first/last for downstream use
        is_first_for_row = [False] * B
        is_last_for_row = [False] * B
        if is_target:
            for i, ri in enumerate(req_pool_indices_cpu):
                name_i = row_names[i]
                layers_i = self.target_layers_by_name[name_i]
                if compress_layer_id not in layers_i:
                    continue  # this retriever doesn't target this layer
                is_first_for_row[i] = (compress_layer_id == self.first_target_by_name[name_i])
                is_last_for_row[i] = (compress_layer_id == self.last_target_by_name[name_i])
                entry = self._cache.get(ri)
                # Detect retriever_name change for this slot → invalidate cache.
                if entry is not None and entry.get("retriever_name") != name_i:
                    if is_first_for_row[i]:
                        needs_fresh[i] = True
                    # For middle/last targets we cannot rebuild the partials
                    # mid-cycle, so just wait for first_target to start fresh.
                    continue
                if is_first_for_row[i]:
                    if entry is None:
                        needs_fresh[i] = True
                    elif c4_seq_lens_cpu[i] < entry.get("c4_seq", 0):
                        needs_fresh[i] = True
                    elif (c4_seq_lens_cpu[i] - entry.get("c4_seq", 0)) > 64:
                        needs_fresh[i] = True
                    elif (self._call_count - entry.get("step", -10**9)) >= self.RETRIEVAL_INTERVAL:
                        needs_fresh[i] = True
                else:
                    # Middle/last target: continue cycle iff partial of all
                    # earlier targets exists for this slot (under the same name).
                    partial = (entry or {}).get("partial", {})
                    earlier = [l for l in layers_i if l < compress_layer_id]
                    if partial and all(l in partial for l in earlier):
                        needs_fresh[i] = True

        # ── Run retriever(s) for this target layer ───────────────────────
        if any(needs_fresh):
            page_table = indexer_metadata.page_table
            c4_page_size = indexer_metadata.c4_page_size
            buf = token_to_kv_pool.c4_indexer_kv_pool.get_index_k_with_scale_buffer(
                compress_layer_id
            )
            fp8_seg_bytes = c4_page_size * 128
            scale_seg_bytes = c4_page_size * 4

            positions = core_metadata.positions.to(torch.int64)
            if positions.dim() != 1:
                positions = positions.view(-1)
            if positions.shape[0] != B:
                positions = positions[:B]

            max_n_tokens = int(c4_seq_lens.max().item())
            if max_n_tokens > 0:
                max_pages = (max_n_tokens + c4_page_size - 1) // c4_page_size
                T = max_pages * c4_page_size
                pages_all = page_table[:, :max_pages].long()
                pages_buf = buf[pages_all]
                fp8_seg = pages_buf[..., :fp8_seg_bytes].contiguous()
                k_fp8_full = (
                    fp8_seg
                    .view(B, max_pages, c4_page_size, 128)
                    .view(torch.float8_e4m3fn)
                    .reshape(B, T, 128)
                )
                scale_seg = pages_buf[
                    ..., fp8_seg_bytes:fp8_seg_bytes + scale_seg_bytes
                ].contiguous()
                k_scale_full = scale_seg.view(torch.float32).reshape(B, T)

                # Group fresh rows by retriever name → call each retriever's
                # scorer on its slice only. Each retriever still amortizes
                # fixed K/Q overhead within the group.
                fresh_idxs_by_name: "dict[str, list[int]]" = {}
                for i, fresh in enumerate(needs_fresh):
                    if not fresh:
                        continue
                    fresh_idxs_by_name.setdefault(row_names[i], []).append(i)

                for name, idxs in fresh_idxs_by_name.items():
                    sel = torch.tensor(idxs, dtype=torch.long, device=device)
                    scorer = self.scorers[name][compress_layer_id]
                    sub_logits = scorer.forward_batched(
                        hidden=x[sel],
                        k_fp8=k_fp8_full[sel],
                        k_scale=k_scale_full[sel],
                        positions=positions[sel],
                    )  # [|idxs|, T] float32

                    for k, i in enumerate(idxs):
                        ri = req_pool_indices_cpu[i]
                        cur_n = c4_seq_lens_cpu[i]
                        entry = self._cache.setdefault(ri, {})
                        if is_first_for_row[i]:
                            # Detect new request (vs cadence refresh): brand-new entry,
                            # different retriever, or c4_seq drop / large jump (>64).
                            # In any of those cases, reset cumulative thresh-fallback state.
                            prev_name = entry.get("retriever_name")
                            prev_c4 = entry.get("c4_seq", 0)
                            is_new_request = (
                                prev_name is None
                                or prev_name != name
                                or cur_n < prev_c4
                                or (cur_n - prev_c4) > 64
                            )
                            if is_new_request:
                                entry["cum_thresh"] = 0
                                entry["cum_c4"] = 0
                                entry["fallback_triggered"] = False
                            # New cycle: drop any stale partials AND update
                            # which retriever owns this slot.
                            entry["partial"] = {}
                            entry["partial_c4_seq"] = cur_n
                            entry["retriever_name"] = name
                        partial = entry.setdefault("partial", {})
                        partial[compress_layer_id] = sub_logits[k, :cur_n].clone()

                # Per-row "last_target" finalize: build keep_mask using THIS
                # row's target_layers (each retriever may have a different set).
                for i, fresh in enumerate(needs_fresh):
                    if not fresh or not is_last_for_row[i]:
                        continue
                    ri = req_pool_indices_cpu[i]
                    entry = self._cache[ri]
                    partial = entry.get("partial", {})
                    name_i = row_names[i]
                    layers_i = self.target_layers_by_name[name_i]
                    if not all(l in partial for l in layers_i):
                        continue
                    cur_n = entry.get("partial_c4_seq", c4_seq_lens_cpu[i])
                    sigs = [torch.sigmoid(partial[l].float()) for l in layers_i]
                    sigs_stacked = torch.stack(sigs, dim=0)  # [L_i, cur_n]
                    if self.ENSEMBLE_MODE == "or":
                        thresh_mask = (sigs_stacked > self.SIGMOID_THRESHOLD).any(dim=0)
                    else:
                        ensemble = sigs_stacked.mean(dim=0)
                        thresh_mask = (ensemble > self.SIGMOID_THRESHOLD)
                    arange = torch.arange(cur_n, device=device)
                    last_start = max(0, cur_n - self.LAST_KEEP_TOKENS)
                    recent = (arange >= last_start)
                    # Attention sink: first N c4 chunks (system-instruction prefix)
                    sink = (arange < min(self.FIRST_KEEP_TOKENS, cur_n))
                    keep_mask = thresh_mask | recent | sink

                    # ── Cumulative thresh-rate fallback (per-request, sticky) ──
                    # If the running ratio sum_thresh_in_evictable / sum_evictable
                    # across all decode cycles in this request exceeds the
                    # per-retriever threshold, switch this request to FULL
                    # ATTENTION (keep all c4 chunks) for the current and all
                    # subsequent decode cycles.
                    #
                    # NOTE: ratio is computed over the EVICTABLE region only
                    # (excluding sink + recent), since those tokens are kept
                    # unconditionally and the user-facing semantic is
                    # "what fraction of the freely-evictable region does the
                    # retriever still want to keep". For long contexts where
                    # sink+recent ≪ total this matches the naive ratio; for
                    # short contexts the evictable denominator is much smaller,
                    # which is the intended behavior.
                    evictable_mask = (~recent) & (~sink)
                    n_thresh_evict_this = int(
                        (thresh_mask & evictable_mask).sum().item()
                    )
                    n_evict_this = int(evictable_mask.sum().item())
                    n_thresh_this = int(thresh_mask.sum().item())  # for stats only
                    fb_thresh = self.THRESH_FALLBACK_BY_NAME.get(name_i)
                    fallback_active = bool(entry.get("fallback_triggered", False))
                    if fb_thresh is not None and not fallback_active:
                        entry["cum_thresh"] = (
                            entry.get("cum_thresh", 0) + n_thresh_evict_this
                        )
                        entry["cum_c4"] = entry.get("cum_c4", 0) + n_evict_this
                        if entry["cum_c4"] > 0 and (
                            entry["cum_thresh"] / entry["cum_c4"] > fb_thresh
                        ):
                            entry["fallback_triggered"] = True
                            fallback_active = True
                            logger.info(
                                f"[MultiRetrieverHook] thresh-fallback TRIGGERED "
                                f"slot={ri} name={name_i} cum_evict_ratio="
                                f"{entry['cum_thresh']/entry['cum_c4']:.4f} > {fb_thresh:.4f} "
                                f"(call#{self._call_count}, c4_seq={cur_n})"
                            )

                    if fallback_active:
                        # Full attention: keep every valid c4 chunk
                        keep_mask = torch.ones(cur_n, dtype=torch.bool, device=device)
                        # Stats: report 100% thresh for this cycle
                        n_thresh_for_stats = cur_n
                    else:
                        n_thresh_for_stats = n_thresh_this

                    entry["step"] = self._call_count
                    entry["c4_seq"] = cur_n
                    entry["keep_mask"] = keep_mask.clone()
                    entry["partial"] = {}
                    entry["n_thresh_last"] = n_thresh_for_stats
                    entry["n_recent_last"] = int(recent.sum().item())
                    entry["n_sink_last"] = int(sink.sum().item())
                    entry["n_keep_last"] = int(keep_mask.sum().item())
                    entry["fallback_triggered_last"] = fallback_active

        # ── Apply cached keep_mask to logits ─────────────────────────────
        keep_for_logits = torch.ones(B, max_c4, dtype=torch.bool, device=device)
        had_any_mask = False
        for i, ri in enumerate(req_pool_indices_cpu):
            entry = self._cache.get(ri)
            if entry is None or "keep_mask" not in entry:
                continue
            # Cross-check: cached mask must belong to the row's current retriever
            if entry.get("retriever_name") != row_names[i]:
                # Slot reassigned to a different retriever; skip masking until
                # next first_target rebuild.
                continue
            cur_n = entry["c4_seq"]
            cached_mask = entry["keep_mask"]
            n_copy = min(cur_n, max_c4)
            row_mask = torch.ones(max_c4, dtype=torch.bool, device=device)
            row_mask[:n_copy] = cached_mask[:n_copy]
            keep_for_logits[i] = row_mask
            had_any_mask = True
        if had_any_mask:
            logits.masked_fill_(~keep_for_logits, float("-inf"))

        # ── Stats dump (only on ensemble-build events on last_target) ────
        # Per-row: any row that just hit ITS last_target with fresh compute.
        any_finalized = any(is_last_for_row[i] and needs_fresh[i] for i in range(B))
        if any_finalized:
            self._ensure_stats_file()
            if self._is_rank_0 and self._stats_file is not None:
                try:
                    ts = time.time()
                    for i, fresh in enumerate(needs_fresh):
                        if not fresh or not is_last_for_row[i]:
                            continue
                        ri = req_pool_indices_cpu[i]
                        entry = self._cache.get(ri)
                        if entry is None or "n_keep_last" not in entry:
                            continue
                        rec = {
                            "ts": ts,
                            "req_pool_idx": int(ri),
                            "call": int(self._call_count),
                            "compress_layer_id": int(compress_layer_id),
                            "c4_seq": int(entry["c4_seq"]),
                            "n_thresh": int(entry["n_thresh_last"]),
                            "n_recent": int(entry["n_recent_last"]),
                            "n_sink": int(entry.get("n_sink_last", 0)),
                            "n_keep": int(entry["n_keep_last"]),
                            "max_c4": int(max_c4),
                            "ensemble": True,
                            "ensemble_mode": self.ENSEMBLE_MODE,
                            "first_keep_tokens": self.FIRST_KEEP_TOKENS,
                            "target_layers": self.target_layers_by_name.get(
                                row_names[i], self.target_layers
                            ),
                            "retriever_name": entry.get("retriever_name"),
                            "fallback_triggered": bool(entry.get("fallback_triggered_last", False)),
                            "cum_thresh": int(entry.get("cum_thresh", 0)),
                            "cum_c4": int(entry.get("cum_c4", 0)),
                        }
                        self._stats_file.write(json.dumps(rec) + "\n")
                    self._stats_file.flush()
                except Exception as e:
                    if self._call_count <= 25:
                        logger.warning(f"[MultiRetrieverHook] stats write failed: {e}")

        # ── Periodic GC ──────────────────────────────────────────────────
        if self._call_count % 1000 == 0:
            stale = [
                k for k, v in self._cache.items()
                if self._call_count - v.get("step", 0) > 1000
            ]
            for k in stale:
                del self._cache[k]

        # ── Periodic info log ────────────────────────────────────────────
        if self._call_count <= 25 or self._call_count % 100 == 1:
            n_kept_avg = (keep_for_logits.sum(dim=-1).float().mean().item()
                          if had_any_mask else float(max_c4))
            logger.info(
                f"[MultiRetrieverHook] call#{self._call_count} layer={compress_layer_id} "
                f"is_target={is_target} fresh={sum(needs_fresh)} kept_avg={n_kept_avg:.0f}"
            )

        return logits


def create_multi_retriever_hook(
    retrievers_spec: "dict[str, str]",
    device: str = "cuda",
    target_layers: list = None,
    default_name: str = None,
    layers_by_name: "dict[str, list[int]]" = None,
):
    return MultiRetrieverHook(
        retrievers_spec=retrievers_spec,
        device=device,
        target_layers=target_layers,
        default_name=default_name,
        layers_by_name=layers_by_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
#                          Weak baselines (for ablation)
# ─────────────────────────────────────────────────────────────────────────────


class WeakBaselineHook:
    """
    Mask-mode hook that SKIPS the retriever entirely. Two variants:

    1. variant="recency_only": keep only the last LAST_KEEP_TOKENS c4 chunks.
       Everything earlier gets logit=-inf, so top-K can only pick from the tail.

    2. variant="random": keep last LAST_KEEP_TOKENS PLUS random earlier
       chunks (resampled every RETRIEVAL_INTERVAL decode steps).
       - Default: RANDOM_KEEP=256 fixed extra chunks.
       - With SGLANG_RETRIEVER_RANDOM_RATIO=0.10: keep 10% of total c4 chunks
         randomly (overrides RANDOM_KEEP).
       - SGLANG_RETRIEVER_LAST_KEEP=2048 controls recency window.

    These let us measure how much of the trained retriever's quality comes from
    "the answer is in the last 2K tokens anyway" vs "retrieval finds the right chunks".
    """

    LAST_KEEP_TOKENS = int(os.environ.get("SGLANG_RETRIEVER_LAST_KEEP", "2048"))
    RETRIEVAL_INTERVAL = 64
    RANDOM_KEEP = 256  # for variant="random": # of extra non-recent random chunks (fallback)
    RANDOM_RATIO = None  # if set via env var, use ratio of total c4 chunks instead of fixed count

    def __init__(self, variant: str = "recency_only", device: str = "cuda"):
        assert variant in ("recency_only", "random"), f"unknown variant: {variant}"
        self.variant = variant
        self.device = device
        # Support percentage-based random keep: SGLANG_RETRIEVER_RANDOM_RATIO=0.10 → 10%
        _ratio_str = os.environ.get("SGLANG_RETRIEVER_RANDOM_RATIO", "")
        self.random_ratio = float(_ratio_str) if _ratio_str else None
        if self.random_ratio is not None and self.variant == "random":
            logger.info(
                f"[RetrieverHook] WeakBaselineHook: using random_ratio={self.random_ratio} "
                f"(fixed RANDOM_KEEP={self.RANDOM_KEEP} overridden)"
            )
        self._call_count = 0
        # cache: per req_pool_idx, the keep_mask. For random variant, refresh
        # every RETRIEVAL_INTERVAL steps.
        self._cache: dict[int, dict] = {}
        self._stats_inited = False
        self._is_rank_0 = False
        self._stats_file = None
        # Per-rank random generator for reproducibility; offset by rank to avoid all
        # ranks picking the same random chunks (would be wasteful but not wrong).
        try:
            from sglang.srt.layers.dp_attention import get_attention_tp_rank
            seed = 7777 + get_attention_tp_rank()
        except Exception:
            seed = 7777
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(seed)
        _keep_desc = f"{self.random_ratio*100:.0f}%×c4" if self.random_ratio else str(self.RANDOM_KEEP)
        logger.info(
            f"[RetrieverHook] WeakBaselineHook init (variant={variant}, "
            f"keep_last={self.LAST_KEEP_TOKENS}, random_keep={_keep_desc}, "
            f"interval={self.RETRIEVAL_INTERVAL})"
        )

    def _ensure_stats_file(self):
        if self._stats_inited:
            return
        self._stats_inited = True
        try:
            from sglang.srt.layers.dp_attention import get_attention_tp_rank
            self._is_rank_0 = (get_attention_tp_rank() == 0)
        except Exception:
            self._is_rank_0 = False
        if self._is_rank_0:
            stats_path = os.environ.get("SGLANG_RETRIEVER_STATS_FILE", "/tmp/hook_stats.jsonl")
            try:
                self._stats_file = open(stats_path, "w")
                logger.info(f"[WeakBaselineHook] stats dump → {stats_path} (rank 0)")
            except Exception as e:
                logger.warning(f"[WeakBaselineHook] stats file open failed: {e}")
                self._stats_file = None

    def maybe_override_topk(self, *args, **kwargs):
        return False, None

    @torch.no_grad()
    def maybe_mask_logits(
        self, logits, c4_seq_lens, x, forward_batch, indexer_metadata, core_metadata,
        token_to_kv_pool, c4_indexer,
    ):
        """Apply recency/random keep_mask. Same interface as TrainedRetrieverHook."""
        self._ensure_stats_file()
        self._call_count += 1

        if logits.dim() != 2:
            return logits
        B, max_c4 = logits.shape
        device = logits.device

        if c4_seq_lens.dim() > 1:
            c4_seq_lens = c4_seq_lens.view(-1)
        c4_seq_lens = c4_seq_lens.to(device).long()
        c4_seq_lens_cpu = c4_seq_lens.tolist()
        req_pool_indices_cpu = forward_batch.req_pool_indices.tolist()

        # Determine which rows need a fresh mask refresh
        needs_refresh: list[bool] = []
        for i, ri in enumerate(req_pool_indices_cpu):
            entry = self._cache.get(ri)
            if entry is None:
                needs_refresh.append(True)
            elif c4_seq_lens_cpu[i] < entry["c4_seq"]:
                needs_refresh.append(True)
            elif (c4_seq_lens_cpu[i] - entry["c4_seq"]) > 64:
                needs_refresh.append(True)
            elif (self._call_count - entry["step"]) >= self.RETRIEVAL_INTERVAL:
                needs_refresh.append(True)
            else:
                needs_refresh.append(False)

        # Build keep_for_logits [B, max_c4]
        keep_for_logits = torch.zeros(B, max_c4, dtype=torch.bool, device=device)
        n_keep_arr = []
        n_recent_arr = []
        n_random_arr = []

        for i, ri in enumerate(req_pool_indices_cpu):
            cur_n = c4_seq_lens_cpu[i]
            if needs_refresh[i]:
                # Build fresh row
                row = torch.zeros(max_c4, dtype=torch.bool, device=device)
                # Recency
                last_start = max(0, cur_n - self.LAST_KEEP_TOKENS)
                row[last_start:cur_n] = True
                n_recent = min(cur_n, self.LAST_KEEP_TOKENS)
                # Random extras (only for random variant)
                n_random = 0
                if self.variant == "random" and cur_n > self.LAST_KEEP_TOKENS:
                    if self.random_ratio is not None:
                        # Percentage-based: keep random_ratio of ALL c4 chunks
                        n_extra = max(1, int(self.random_ratio * cur_n))
                    else:
                        n_extra = self.RANDOM_KEEP
                    n_extra = min(n_extra, cur_n - self.LAST_KEEP_TOKENS)
                    # Sample from non-recent (historical) pool
                    pool = torch.arange(0, last_start, device=device)
                    perm = torch.randperm(pool.size(0), generator=self._rng, device=device)
                    rand_idx = pool[perm[:n_extra]]
                    row[rand_idx] = True
                    n_random = n_extra
                self._cache[ri] = {
                    "step": self._call_count,
                    "c4_seq": cur_n,
                    "keep_mask": row[:cur_n].clone(),
                    "n_recent": int(n_recent),
                    "n_random": int(n_random),
                }
                keep_for_logits[i] = row
            else:
                cached = self._cache[ri]
                old_n = cached["c4_seq"]
                row = torch.ones(max_c4, dtype=torch.bool, device=device)
                row[:min(old_n, max_c4)] = cached["keep_mask"][:min(old_n, max_c4)]
                # Tokens decoded since cache → auto-keep
                keep_for_logits[i] = row

            n_keep_arr.append(int(keep_for_logits[i, :cur_n].sum().item()))
            n_recent_arr.append(self._cache[ri]["n_recent"])
            n_random_arr.append(self._cache[ri]["n_random"])

        logits.masked_fill_(~keep_for_logits, float("-inf"))

        # Stats dump (rank 0 only, only on refresh events)
        if self._stats_file is not None:
            import time as _time
            ts = _time.time()
            for i, ri in enumerate(req_pool_indices_cpu):
                if not needs_refresh[i]:
                    continue
                rec = {
                    "ts": ts,
                    "req_pool_idx": int(ri),
                    "call": self._call_count,
                    "compress_layer_id": int(token_to_kv_pool.layer_mapping[
                        c4_indexer.layer_id].compress_layer_id),
                    "c4_seq": int(c4_seq_lens_cpu[i]),
                    "n_thresh": int(n_random_arr[i]),  # reuse field for consistency
                    "n_recent": int(n_recent_arr[i]),
                    "n_keep": int(n_keep_arr[i]),
                    "max_c4": int(max_c4),
                    "variant": self.variant,
                }
                try:
                    self._stats_file.write(json.dumps(rec) + "\n")
                    self._stats_file.flush()
                except Exception:
                    pass

        return logits


def create_weak_hook(variant: str, device: str = "cuda"):
    return WeakBaselineHook(variant=variant, device=device)


# ─────────────────────────────────────────────────────────────────────────────
#                            Auto-init
# ─────────────────────────────────────────────────────────────────────────────


def auto_init_retriever_hook():
    """
    Called from deepseek_v4.py post_load_weights().

    Modes:
      - off / unset:    do nothing
      - mock:           enable MockRetrieverHook
      - no_swap:        enable NoSwapRetrieverHook
      - trained:        enable TrainedRetrieverHook (requires SGLANG_RETRIEVER_CHECKPOINT)
      - recency_only:   enable WeakBaselineHook(variant="recency_only")
      - random:         enable WeakBaselineHook(variant="random")
    """
    mode = os.environ.get("SGLANG_RETRIEVER_MODE", "off").lower()
    if mode in ("off", ""):
        return

    from sglang.srt.layers.attention.compressed.indexer import enable_retriever_hook

    if mode == "no_swap":
        hook = create_no_swap_hook()
        enable_retriever_hook(hook)
        logger.info("[RetrieverHook] AUTO-INIT: no_swap mode")
    elif mode == "mock":
        hook = create_mock_hook(top_k=512, interval=64)
        enable_retriever_hook(hook)
        logger.info("[RetrieverHook] AUTO-INIT: mock mode")
    elif mode in ("recency_only", "random"):
        try:
            hook = create_weak_hook(variant=mode, device="cuda")
            enable_retriever_hook(hook)
            logger.info(f"[RetrieverHook] AUTO-INIT: weak baseline ({mode})")
        except Exception as e:
            logger.exception(f"[RetrieverHook] failed to init weak hook: {e}")
    elif mode in ("trained", "trained_ensemble", "ensemble"):
        # Legacy single-ckpt modes — redirected to multi-retriever mode with one
        # default-name entry. The original TrainedRetrieverHook /
        # TrainedEnsembleRetrieverHook classes were removed; MultiRetrieverHook
        # is a strict superset (single name = single retriever).
        ckpt = os.environ.get("SGLANG_RETRIEVER_CHECKPOINT", "")
        if not ckpt or not os.path.exists(ckpt):
            logger.error(
                f"[RetrieverHook] legacy {mode} mode but "
                f"SGLANG_RETRIEVER_CHECKPOINT={ckpt!r} invalid; skipping"
            )
            return
        logger.warning(
            f"[RetrieverHook] mode={mode!r} is deprecated; redirecting to "
            f"multi-retriever mode with default name 'default'."
        )
        spec = {"default": ckpt}
        layers_by_name = {}  # let MultiRetrieverHook fall through to global target_layers (env SGLANG_RETRIEVER_LAYERS)
        try:
            hook = create_multi_retriever_hook(spec, device="cuda",
                                                layers_by_name=layers_by_name)
            enable_retriever_hook(hook)
            logger.info(f"[RetrieverHook] AUTO-INIT: multi mode (legacy {mode}, ckpt={ckpt})")
        except Exception as e:
            logger.exception(f"[RetrieverHook] failed to init legacy {mode}: {e}")
    elif mode in ("multi", "multi_retriever", "multi_retrievers"):
        # Per-request retriever routing.
        # SGLANG_RETRIEVER_CHECKPOINTS = "name1:/path1,name2:/path2,..."
        # Backwards-compat: if only SGLANG_RETRIEVER_CHECKPOINT is set, we
        # treat it as a single-retriever multi setup (default name "default").
        spec_str = os.environ.get("SGLANG_RETRIEVER_CHECKPOINTS", "").strip()
        single = os.environ.get("SGLANG_RETRIEVER_CHECKPOINT", "").strip()
        if not spec_str and single:
            spec_str = f"default:{single}"
        if not spec_str:
            logger.error(
                "[RetrieverHook] multi mode but neither SGLANG_RETRIEVER_CHECKPOINTS "
                "nor SGLANG_RETRIEVER_CHECKPOINT is set; skipping"
            )
            return
        try:
            spec, layers_by_name = _parse_retrievers_spec(spec_str)
        except Exception as e:
            logger.error(f"[RetrieverHook] failed to parse SGLANG_RETRIEVER_CHECKPOINTS: {e}")
            return
        # Validate every ckpt exists before doing anything heavy.
        missing = [n for n, p in spec.items() if not os.path.exists(p)]
        if missing:
            logger.error(
                f"[RetrieverHook] multi mode: ckpt files missing for {missing}; skipping"
            )
            return
        try:
            hook = create_multi_retriever_hook(spec, device="cuda",
                                                layers_by_name=layers_by_name)
            enable_retriever_hook(hook)
            logger.info(
                f"[RetrieverHook] AUTO-INIT: multi mode "
                f"(retrievers={list(spec)}, default={hook.default_name!r}, "
                f"per_retriever_layers={layers_by_name})"
            )
        except Exception as e:
            logger.exception(f"[RetrieverHook] failed to init multi hook: {e}")
    else:
        logger.warning(f"[RetrieverHook] Unknown mode: {mode}, skipping")
