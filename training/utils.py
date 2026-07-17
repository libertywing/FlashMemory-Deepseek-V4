"""
utils.py — Lightning Indexer 共享工具函数
==========================================
纯 PyTorch 实现，inference.py 和 train.py 均从此处导入，避免重复代码。

包含：
  - precompute_freqs_cis  : YaRN RoPE 频率预计算
  - apply_rope            : 纯 PyTorch 可微 RoPE
  - hadamard_transform    : 归一化 Walsh-Hadamard 变换
"""

import math
import torch


# ── YaRN RoPE ────────────────────────────────────────────────────────────────

def _yarn_find_correction_dim(n_rot: float, d_model: int, base: float, max_pos: int) -> float:
    return (d_model * math.log(max_pos / (n_rot * 2 * math.pi))) / (2 * math.log(base))


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    base: float,
    factor: float,
    original_seq_len: int,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """
    YaRN RoPE 频率预计算。

    Returns:
        freqs_cis: [seqlen, dim//2]  complex64
    """
    low  = max(math.floor(_yarn_find_correction_dim(beta_fast, dim, base, original_seq_len)), 0)
    high = min(math.ceil( _yarn_find_correction_dim(beta_slow, dim, base, original_seq_len)), dim // 2 - 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [dim//2]

    ramp = torch.zeros(dim // 2)
    for i in range(dim // 2):
        if i < low:
            ramp[i] = 0.0
        elif i >= high:
            ramp[i] = 1.0
        else:
            ramp[i] = (i - low) / max(high - low, 1)

    mixed  = freqs * (1 - ramp) + (freqs / factor) * ramp   # [dim//2]
    t      = torch.arange(seqlen, dtype=torch.float32)
    angles = torch.outer(t, mixed)                           # [seqlen, dim//2]
    return torch.polar(torch.ones_like(angles), angles)      # [seqlen, dim//2] complex64


def apply_rope(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    """
    纯 PyTorch RoPE，作用于 q 最后 rope_dim 个维度。
    与 sglang fused_rope 等价，支持 autograd。

    Args:
        q:         [B, n_heads, head_dim]
        freqs_cis: [max_pos, rope_dim//2]  complex64
        positions: [B]  int64
        rope_dim:  施加旋转的维度数（作用于 q[..., -rope_dim:]）

    Returns:
        q after RoPE, same shape as input
    """
    head_dim = q.shape[-1]
    q_pass = q[..., : head_dim - rope_dim]          # [B, H, head_dim-rope_dim]
    q_rope = q[..., head_dim - rope_dim :]           # [B, H, rope_dim]

    q_c = torch.view_as_complex(
        q_rope.float().reshape(*q_rope.shape[:-1], rope_dim // 2, 2).contiguous()
    )  # [B, H, rope_dim//2]

    # Guard: clamp positions into the RoPE table range. Val docs are streamed at
    # eval time and can be longer than the freqs_cis table (e.g. NovelQA val pos
    # 438660 > auto-derived 398012) → freqs_cis[positions] gather OOB → CUDA
    # device-side assert that takes down the whole (DDP) job mid-validation.
    # Clamping to the last entry keeps it alive; YaRN extrapolation already makes
    # the tail an approximation, so a few clamped ultra-long tokens are far better
    # than a crash. Unconditional clamp avoids a per-step host sync (positions.max())
    # in the training hot loop (a no-op when positions are already in range).
    positions = positions.clamp(0, freqs_cis.shape[0] - 1)

    freqs = freqs_cis[positions].unsqueeze(1)        # [B, 1, rope_dim//2]
    q_rot = torch.view_as_real(q_c * freqs).reshape(*q_rope.shape).to(q.dtype)
    return torch.cat([q_pass, q_rot], dim=-1)


# ── Hadamard ──────────────────────────────────────────────────────────────────

def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    归一化 Walsh-Hadamard 变换，作用于最后一个维度（必须是 2 的幂次）。
    与 sglang rotate_activation 等价，支持 autograd。

    x: [..., d] → [..., d]  (normalized by 1/sqrt(d))
    """
    *leading, d = x.shape
    assert d > 0 and (d & (d - 1)) == 0, f"last dim {d} must be a power of 2"
    h = x.float()
    s = 1
    while s < d:
        h = h.view(*leading, d // (2 * s), 2, s)
        a, b = h[..., 0, :], h[..., 1, :]
        h = torch.stack([a + b, a - b], dim=-2).view(*leading, d)
        s *= 2
    return h / math.sqrt(d)
