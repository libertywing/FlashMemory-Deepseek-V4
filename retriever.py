"""
retriever.py — FlashMemory DS-V4 Retriever (standalone reference implementation)
===============================================================================

A self-contained, dependency-light (torch only) PyTorch reference implementation
of the **FlashMemory Retriever** used for sparsifying the DeepSeek-V4
Compressed-Sparse-Attention (CSA) KV cache.

Given the hidden state of a decode token, the retriever predicts which CSA
KV-cache chunks the next tokens will attend to, so that only the top-scoring
chunks need to stay resident on the GPU.

    compressed_k [B, N, 132] uint8  →  dequant  →  k  [B, N, HEAD_DIM]
    hidden [B, 4096]  →  q-proj + RoPE + Hadamard  →  q  [B, N_HEADS, HEAD_DIM]
                         → weights_proj  →  fused_w  [B, N_HEADS]

    score = sigmoid( (relu(k @ q^T) · fused_w).sum(heads) )  ∈ [0, 1]

The shipped checkpoint is a *joint* checkpoint holding three independent CSA
layers (l10 / l12 / l20). At inference time the per-layer sigmoid scores are
ensembled per chunk (cross-layer ``max`` by default, ``mean`` also supported).

This file only depends on ``torch``. The full sglang serving integration
(KV-cache swap, attention-sink, threshold fallback, per-request routing) is
NOT part of this open release because it depends on the internal DeepSeek-V4
CSA framework.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#                       RoPE (YaRN) + Hadamard utilities
#        (copied from the project's utils.py so this release is self-contained)
# ─────────────────────────────────────────────────────────────────────────────


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
    """YaRN RoPE frequency precomputation.

    Returns:
        freqs_cis: [seqlen, dim // 2]  complex64
    """
    low = max(math.floor(_yarn_find_correction_dim(beta_fast, dim, base, original_seq_len)), 0)
    high = min(math.ceil(_yarn_find_correction_dim(beta_slow, dim, base, original_seq_len)), dim // 2 - 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [dim//2]

    ramp = torch.zeros(dim // 2)
    for i in range(dim // 2):
        if i < low:
            ramp[i] = 0.0
        elif i >= high:
            ramp[i] = 1.0
        else:
            ramp[i] = (i - low) / max(high - low, 1)

    mixed = freqs * (1 - ramp) + (freqs / factor) * ramp   # [dim//2]
    t = torch.arange(seqlen, dtype=torch.float32)
    angles = torch.outer(t, mixed)                         # [seqlen, dim//2]
    return torch.polar(torch.ones_like(angles), angles)    # complex64


def apply_rope(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    """Pure-PyTorch RoPE applied to the last ``rope_dim`` dims of ``q``.

    Args:
        q:         [B, n_heads, head_dim]
        freqs_cis: [max_pos, rope_dim // 2]  complex64
        positions: [B]  int64
        rope_dim:  number of trailing dims to rotate (applied to q[..., -rope_dim:])

    Returns:
        q after RoPE, same shape as input.
    """
    head_dim = q.shape[-1]
    q_pass = q[..., : head_dim - rope_dim]
    q_rope = q[..., head_dim - rope_dim:]

    q_c = torch.view_as_complex(
        q_rope.float().reshape(*q_rope.shape[:-1], rope_dim // 2, 2).contiguous()
    )  # [B, H, rope_dim//2]

    # Clamp positions into the RoPE table range. The freqs_cis table covers
    # max_position entries; tokens beyond it get clamped to the last entry
    # (YaRN extrapolation already makes the tail an approximation, so a few
    # clamped ultra-long positions are far better than an out-of-bounds gather).
    positions = positions.clamp(0, freqs_cis.shape[0] - 1)

    freqs = freqs_cis[positions].unsqueeze(1)  # [B, 1, rope_dim//2]
    q_rot = torch.view_as_real(q_c * freqs).reshape(*q_rope.shape).to(q.dtype)
    return torch.cat([q_pass, q_rot], dim=-1)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform over the last dim (must be a power of 2).

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


# ─────────────────────────────────────────────────────────────────────────────
#                         compressed-K dequantization
# ─────────────────────────────────────────────────────────────────────────────


def dequant_compressed_k(compressed_k: torch.Tensor, head_dim: int = 128) -> torch.Tensor:
    """Dequantize compressed CSA keys.

    Each compressed key is ``head_dim + 4`` bytes:
        bytes[:head_dim]      — float8_e4m3 quantized key values (1 byte each)
        bytes[head_dim:+4]    — a single float32 per-chunk scale

    Args:
        compressed_k: [..., head_dim + 4]  uint8
        head_dim:     number of key dims (default 128)

    Returns:
        k: [..., head_dim]  float32   ( = fp8_values * scale )
    """
    assert compressed_k.dtype == torch.uint8, (
        f"compressed_k must be uint8, got {compressed_k.dtype}"
    )
    assert compressed_k.shape[-1] == head_dim + 4, (
        f"compressed_k last dim must be {head_dim + 4}, got {compressed_k.shape[-1]}"
    )

    fp8_bytes = compressed_k[..., :head_dim].contiguous()       # uint8 [..., head_dim]
    k_fp8 = fp8_bytes.view(torch.float8_e4m3fn).float()         # [..., head_dim]

    scale_bytes = compressed_k[..., head_dim:head_dim + 4].contiguous()  # uint8 [..., 4]
    scale = scale_bytes.view(torch.float32)                     # [..., 1]

    return k_fp8 * scale                                        # broadcast → [..., head_dim]


# ─────────────────────────────────────────────────────────────────────────────
#                           per-layer scorer module
# ─────────────────────────────────────────────────────────────────────────────


class _LayerScorer(nn.Module):
    """Holds one CSA layer's retriever weights and computes its logits.

    Weights are stored as (non-trainable) buffers so ``.to(device)`` / ``.half()``
    move them along with the parent module.
    """

    def __init__(
        self,
        wq_a: torch.Tensor,          # [Q_LORA_RANK, 4096]
        wq_b: torch.Tensor,          # [N_HEADS * HEAD_DIM, Q_LORA_RANK]
        q_norm_weight: torch.Tensor, # [Q_LORA_RANK]
        weights_proj: torch.Tensor,  # [N_HEADS, 4096]
        n_heads: int,
        head_dim: int,
        rope_dim: int,
        rms_norm_eps: float,
        weight_scale: float,
    ):
        super().__init__()
        self.register_buffer("wq_a", wq_a.to(torch.float32), persistent=False)
        self.register_buffer("wq_b", wq_b.to(torch.float32), persistent=False)
        self.register_buffer("q_norm_weight", q_norm_weight.to(torch.float32), persistent=False)
        self.register_buffer("weights_proj", weights_proj.to(torch.float32), persistent=False)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.rms_norm_eps = rms_norm_eps
        self.weight_scale = weight_scale

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        norm = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.rms_norm_eps)
        return x_f / norm * self.q_norm_weight

    @torch.no_grad()
    def logits(
        self,
        hidden: torch.Tensor,    # [B, 4096]
        k_float: torch.Tensor,   # [B, N, head_dim]  (already dequantized)
        positions: torch.Tensor, # [B]  int64
        freqs_cis: torch.Tensor, # [max_pos, rope_dim//2]  complex64
    ) -> torch.Tensor:
        """Return raw (pre-sigmoid) logits [B, N] for this layer."""
        x = hidden.float()
        B = x.shape[0]

        # ── Q side ──────────────────────────────────────────────────────────
        q_lora = F.linear(x, self.wq_a)               # [B, Q_LORA_RANK]
        q_lora = self._rmsnorm(q_lora)                # [B, Q_LORA_RANK]
        q = F.linear(q_lora, self.wq_b)               # [B, N_HEADS * HEAD_DIM]
        q = q.view(B, self.n_heads, self.head_dim)    # [B, N_HEADS, HEAD_DIM]
        # RoPE is applied in bf16 then cast back to float32 to match the trained
        # / deployed scoring path exactly.
        q = apply_rope(q.to(torch.bfloat16), freqs_cis, positions.to(torch.int64),
                       rope_dim=self.rope_dim).float()
        q = hadamard_transform(q)                     # [B, N_HEADS, HEAD_DIM]

        per_head_w = F.linear(x, self.weights_proj)   # [B, N_HEADS]
        fused_w = per_head_w * self.weight_scale      # [B, N_HEADS]

        # ── Score: relu(k @ q^T) weighted-sum over heads ────────────────────
        # q: [B, H, D], k_float: [B, N, D] → [B, N, H]
        scores_per_head = F.relu(torch.einsum("bhd,bnd->bnh", q, k_float))  # [B, N, H]
        logits = (scores_per_head * fused_w.unsqueeze(1)).sum(-1)           # [B, N]
        return logits


# ─────────────────────────────────────────────────────────────────────────────
#                          FlashMemoryRetriever
# ─────────────────────────────────────────────────────────────────────────────


class FlashMemoryRetriever(nn.Module):
    """Multi-layer FlashMemory retriever (joint checkpoint).

    Loads a joint checkpoint whose state-dict keys look like
    ``retrievers.l10.wq_a.weight``, builds one ``_LayerScorer`` per CSA layer,
    and scores compressed-K chunks against a decode token's hidden state.

    Typical usage::

        model = FlashMemoryRetriever.from_checkpoint("flashmemory_ds_v4.safetensors",
                                                      device="cuda")
        per_layer = model(hidden_state, compressed_k, positions)  # {"l10": [B,N], ...}
        scores = model.ensemble(hidden_state, compressed_k, positions, mode="max")  # [B,N]
    """

    # RoPE / normalization constants (identical across all CSA layers).
    HEAD_DIM = 128
    ROPE_DIM = 64
    ROPE_BASE = 160000.0
    ROPE_FACTOR = 16.0
    ROPE_ORIGINAL_SEQ_LEN = 65536
    ROPE_BETA_FAST = 32.0
    ROPE_BETA_SLOW = 1.0
    RMS_NORM_EPS = 1e-6

    def __init__(
        self,
        layer_states: "OrderedDict[str, Dict[str, torch.Tensor]]",
        device: Union[str, torch.device] = "cpu",
        max_position: int = 524288,
        head_dim: Optional[int] = None,
    ):
        """
        Args:
            layer_states: ordered mapping ``layer_name -> {"wq_a.weight": ...,
                "wq_b.weight": ..., "q_norm_weight": ..., "weights_proj.weight": ...}``.
                Layer names are arbitrary (e.g. ``"l10"``); ordering is preserved.
            device: device to place the model on.
            max_position: RoPE table length. Must cover the largest token position
                ever scored; positions beyond it are clamped (RoPE becomes an
                approximation). Default 524288; can be raised to 1_048_576 (1M) for
                full-length DeepSeek-V4 contexts.
            head_dim: key/head dimension. Defaults to ``HEAD_DIM`` (128).
        """
        super().__init__()
        assert layer_states, "FlashMemoryRetriever needs at least one layer"
        device = torch.device(device)
        self.head_dim = head_dim if head_dim is not None else self.HEAD_DIM
        self.max_position = max_position
        self.layer_names: List[str] = list(layer_states.keys())

        # Precompute the (shared) YaRN RoPE table once.
        freqs_cis = precompute_freqs_cis(
            dim=self.ROPE_DIM,
            seqlen=max_position,
            base=self.ROPE_BASE,
            factor=self.ROPE_FACTOR,
            original_seq_len=self.ROPE_ORIGINAL_SEQ_LEN,
            beta_fast=self.ROPE_BETA_FAST,
            beta_slow=self.ROPE_BETA_SLOW,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Build one scorer per layer.
        self.scorers = nn.ModuleDict()
        for name, st in layer_states.items():
            wq_b = st["wq_b.weight"]
            n_heads = wq_b.shape[0] // self.head_dim
            weight_scale = self.head_dim ** -0.5 * n_heads ** -0.5
            self.scorers[name] = _LayerScorer(
                wq_a=st["wq_a.weight"],
                wq_b=wq_b,
                q_norm_weight=st["q_norm_weight"],
                weights_proj=st["weights_proj.weight"],
                n_heads=n_heads,
                head_dim=self.head_dim,
                rope_dim=self.ROPE_DIM,
                rms_norm_eps=self.RMS_NORM_EPS,
                weight_scale=weight_scale,
            )

        self.n_heads = next(iter(self.scorers.values())).n_heads
        self.to(device)

    # ── construction helpers ────────────────────────────────────────────────

    @staticmethod
    def _split_joint_state(
        state: Dict[str, torch.Tensor],
        layers: Optional[List[str]] = None,
    ) -> "OrderedDict[str, Dict[str, torch.Tensor]]":
        """Split a joint state-dict (keys ``retrievers.l{ID}.*``) into per-layer dicts."""
        is_joint = any(k.startswith("retrievers.") for k in state.keys())
        if not is_joint:
            raise ValueError(
                "State dict is not in joint 'retrievers.l{ID}.*' format. "
                f"Got keys e.g. {list(state.keys())[:3]}"
            )
        found = sorted({k.split(".")[1] for k in state if k.startswith("retrievers.")})
        use_layers = layers if layers is not None else found
        out: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()
        wanted = ("wq_a.weight", "wq_b.weight", "q_norm_weight", "weights_proj.weight")
        for lname in use_layers:
            prefix = f"retrievers.{lname}."
            sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not sub:
                raise ValueError(
                    f"Layer {lname!r} not found in checkpoint. Available: {found}"
                )
            missing = [w for w in wanted if w not in sub]
            if missing:
                raise ValueError(f"Layer {lname!r} missing weights {missing}")
            out[lname] = {w: sub[w] for w in wanted}
        return out

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: Union[str, torch.device] = "cpu",
        max_position: int = 524288,
        layers: Optional[List[str]] = None,
    ) -> "FlashMemoryRetriever":
        """Load a joint checkpoint and build the retriever.

        Supports both ``.pt`` (``torch.save`` state-dict) and ``.safetensors``
        (HuggingFace convention). Only the learned weights (``wq_a/wq_b/
        q_norm_weight/weights_proj``) are read; the RoPE ``freqs_cis`` table is
        recomputed locally, so a slim ``.safetensors`` loads identically.

        Args:
            ckpt_path: path to the joint checkpoint (``.pt`` or ``.safetensors``).
            device: device to load onto.
            max_position: RoPE table length (see ``__init__``).
            layers: optional subset of layer names (e.g. ``["l10", "l20"]``). If
                None, all layers found in the checkpoint are used.
        """
        if str(ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path, device="cpu")
        else:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        layer_states = cls._split_joint_state(state, layers=layers)
        return cls(layer_states, device=device, max_position=max_position)

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        hidden_state: torch.Tensor,   # [B, 4096]
        compressed_k: torch.Tensor,   # [B, N, head_dim + 4]  uint8
        positions: torch.Tensor,      # [B]  int64
        apply_sigmoid: bool = True,
    ) -> "OrderedDict[str, torch.Tensor]":
        """Score the compressed-K chunks with every CSA layer.

        Args:
            hidden_state: [B, 4096] decode-token hidden states.
            compressed_k: [B, N, head_dim + 4] uint8 compressed keys (shared across
                layers in this reference impl — see note below).
            positions: [B] int64 token positions (for RoPE).
            apply_sigmoid: if True (default) return sigmoid scores ∈ [0, 1];
                if False return raw logits.

        Returns:
            OrderedDict ``{layer_name: scores [B, N]}``.

        Note:
            In the production DeepSeek-V4 CSA system each layer has its *own*
            compressed-K buffer. This reference impl scores all layers against the
            single ``compressed_k`` you pass, which is the right behavior for the
            standalone algorithm demo. If you have per-layer K, call this once per
            layer with that layer's K, or use ``score_layer``.
        """
        device = self.freqs_cis.device
        hidden_state = hidden_state.to(device)
        compressed_k = compressed_k.to(device)
        positions = positions.to(device)

        k_float = dequant_compressed_k(compressed_k, head_dim=self.head_dim)  # [B, N, D]

        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, scorer in self.scorers.items():
            logits = scorer.logits(hidden_state, k_float, positions, self.freqs_cis)
            out[name] = torch.sigmoid(logits) if apply_sigmoid else logits
        return out

    @torch.no_grad()
    def score_layer(
        self,
        layer_name: str,
        hidden_state: torch.Tensor,
        compressed_k: torch.Tensor,
        positions: torch.Tensor,
        apply_sigmoid: bool = True,
    ) -> torch.Tensor:
        """Score a single layer (useful when each layer has its own K)."""
        device = self.freqs_cis.device
        k_float = dequant_compressed_k(compressed_k.to(device), head_dim=self.head_dim)
        logits = self.scorers[layer_name].logits(
            hidden_state.to(device), k_float, positions.to(device), self.freqs_cis
        )
        return torch.sigmoid(logits) if apply_sigmoid else logits

    @torch.no_grad()
    def ensemble(
        self,
        hidden_state: torch.Tensor,
        compressed_k: torch.Tensor,
        positions: torch.Tensor,
        mode: str = "max",
    ) -> torch.Tensor:
        """Cross-layer ensemble of per-chunk sigmoid scores.

        Args:
            mode: ``"max"`` (default) or ``"mean"`` over the per-layer sigmoid
                scores, per chunk.

        Returns:
            scores [B, N]  ∈ [0, 1].
        """
        assert mode in ("max", "mean"), f"unknown ensemble mode: {mode!r}"
        per_layer = self.forward(hidden_state, compressed_k, positions, apply_sigmoid=True)
        stacked = torch.stack(list(per_layer.values()), dim=0)  # [L, B, N]
        if mode == "max":
            return stacked.amax(dim=0)
        return stacked.mean(dim=0)

    @torch.no_grad()
    def select_topk(
        self,
        hidden_state: torch.Tensor,
        compressed_k: torch.Tensor,
        positions: torch.Tensor,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        mode: str = "max",
    ) -> torch.Tensor:
        """Return a boolean keep-mask [B, N] of selected chunks.

        Exactly one of ``top_k`` / ``threshold`` should be given. With ``top_k``
        the top-k highest-scoring chunks per row are kept; with ``threshold`` all
        chunks whose ensembled sigmoid score exceeds the threshold are kept.
        """
        scores = self.ensemble(hidden_state, compressed_k, positions, mode=mode)  # [B, N]
        B, N = scores.shape
        if (top_k is None) == (threshold is None):
            raise ValueError("Provide exactly one of top_k or threshold")
        if threshold is not None:
            return scores > threshold
        k = min(top_k, N)
        keep = torch.zeros(B, N, dtype=torch.bool, device=scores.device)
        idx = scores.topk(k, dim=-1).indices
        keep.scatter_(1, idx, True)
        return keep
