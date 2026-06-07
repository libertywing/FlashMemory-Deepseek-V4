"""
toy_flashmemory_inference.py — Toy sparse-decode loop driven by the FlashMemory Retriever
=========================================================================================

A minimal, torch-only illustration of how the FlashMemory Retriever controls CSA
memory recall during decode. Every 64 steps the retriever scores all N compressed-K
chunks against the current decode hidden state, selects the top-K (or thresholded)
ones to keep, and the rest are masked from attention — exactly as if their KV were
never recalled onto the GPU.

This is NOT a real DeepSeek-V4. The "decoder" is a few toy layers with random
weights. But the retriever, its scoring math, and the decode-time control flow
are all real.

Run::

    python toy_flashmemory_inference.py --ckpt weights/flashmemory_ds_v4.safetensors
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure sibling retriever.py is importable (works from any cwd).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retriever import FlashMemoryRetriever, dequant_compressed_k  # noqa: E402


HIDDEN_DIM = 4096  # fixed: the retriever consumes a [B, 4096] decode hidden state


# ─────────────────────────────────────────────────────────────────────────────
#   Mock CSA KV-cache:  N compressed chunks, each [head_dim + 4] uint8
#   (this is the *indexer's* quantized-K representation that the retriever scores)
# ─────────────────────────────────────────────────────────────────────────────
def make_mock_compressed_k(
    batch: int,
    n_chunks: int,
    head_dim: int = 128,
    device: str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    """Build a valid mock ``compressed_k`` tensor ``[B, N, head_dim + 4]`` uint8.

    This mirrors how the real CSA cache stores a compressed key per chunk:
        bytes[:head_dim]      — float8_e4m3 quantized key values (1 byte each)
        bytes[head_dim:+4]    — one float32 per-chunk dequant scale

    In a real FlashMemory run these bytes are produced during *prefill*, when the
    historical KV is compressed and stored. Here we just sample them randomly —
    the retriever still runs its exact scoring path over them.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    # 1) fp8 key bytes
    k_vals = torch.randn(batch, n_chunks, head_dim, generator=g, device=device) * 0.5
    fp8_bytes = k_vals.to(torch.float8_e4m3fn).view(torch.uint8)          # [B, N, head_dim]

    # 2) float32 per-chunk scale → 4 uint8 bytes
    scale = (0.05 + 0.15 * torch.rand(batch, n_chunks, 1, generator=g, device=device)).float()
    scale_bytes = scale.view(torch.uint8)                                 # [B, N, 4]

    compressed = torch.cat([fp8_bytes, scale_bytes], dim=-1)              # [B, N, head_dim + 4]
    assert compressed.shape[-1] == head_dim + 4
    return compressed.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
#   Toy decoder (random weights).  Only exists to emit a [B,4096] hidden state
#   each step and own a memory cross-attention over N CSA chunks that the
#   retriever's keep-mask sparsifies.
# ─────────────────────────────────────────────────────────────────────────────
def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * norm).to(x.dtype) * weight


class ToyMemoryDecoder(nn.Module):
    """A few layers of toy memory cross-attention + MLP (random weights)."""

    def __init__(
        self,
        n_chunks: int,
        n_layers: int = 2,
        n_heads: int = 8,
        vocab_size: int = 512,
        device: str = "cpu",
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = HIDDEN_DIM
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = self.hidden_dim // n_heads
        self.n_chunks = n_chunks

        # Token embedding (toy; vocab is meaningless).
        self.embed = nn.Embedding(vocab_size, self.hidden_dim)

        # Decoder-space memory bank: one vector per CSA chunk (separate from the
        # retriever's compressed_k — both index the same N chunks).
        self.register_buffer("memory", torch.randn(n_chunks, self.hidden_dim) * 0.02)

        # Per-layer projections + norms.
        self.wq = nn.ModuleList(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(n_layers))
        self.wk = nn.ModuleList(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(n_layers))
        self.wv = nn.ModuleList(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(n_layers))
        self.wo = nn.ModuleList(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(n_layers))
        self.mlp_up = nn.ModuleList(nn.Linear(self.hidden_dim, 2 * self.hidden_dim, bias=False) for _ in range(n_layers))
        self.mlp_down = nn.ModuleList(nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=False) for _ in range(n_layers))
        self.attn_norm = nn.ParameterList(nn.Parameter(torch.ones(self.hidden_dim)) for _ in range(n_layers))
        self.mlp_norm = nn.ParameterList(nn.Parameter(torch.ones(self.hidden_dim)) for _ in range(n_layers))
        self.final_norm = nn.Parameter(torch.ones(self.hidden_dim))
        self.lm_head = nn.Linear(self.hidden_dim, vocab_size, bias=False)

        self.to(device)
        self.eval()

    @torch.no_grad()
    def _memory_attention(self, x: torch.Tensor, layer: int, keep_mask: torch.Tensor | None) -> torch.Tensor:
        """Cross-attention of the current token(s) over the N memory chunks.

        Args:
            x:         [B, hidden] current-token hidden state(s).
            keep_mask: [B, N] bool, True = chunk recalled/kept. ``None`` = keep all
                       (the dense path used during prefill / cold-start).

        Chunks with ``keep_mask == False`` get their attention logit set to
        ``-inf`` → softmax weight 0 → they contribute nothing. THIS is our
        simulation of "the chunk was not recalled onto the GPU".
        """
        B = x.shape[0]
        H, D = self.n_heads, self.head_dim

        q = self.wq[layer](x).view(B, H, 1, D)                              # [B, H, 1, D]
        k = self.wk[layer](self.memory).view(self.n_chunks, H, D).permute(1, 0, 2)  # [H, N, D]
        v = self.wv[layer](self.memory).view(self.n_chunks, H, D).permute(1, 0, 2)  # [H, N, D]

        # [B, H, 1, N] attention logits over the N memory chunks.
        logits = torch.einsum("bhqd,hnd->bhqn", q, k) / math.sqrt(D)
        if keep_mask is not None:
            # Broadcast [B, N] → [B, 1, 1, N] and mask the dropped chunks.
            drop = ~keep_mask.view(B, 1, 1, self.n_chunks)
            logits = logits.masked_fill(drop, float("-inf"))

        attn = torch.softmax(logits, dim=-1)                                # [B, H, 1, N]
        out = torch.einsum("bhqn,hnd->bhqd", attn, v).reshape(B, self.hidden_dim)
        return self.wo[layer](out)

    @torch.no_grad()
    def step(
        self,
        token_ids: torch.Tensor,          # [B] int64
        keep_mask: torch.Tensor | None,   # [B, N] bool, or None for dense
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decode step. Returns (hidden [B, 4096], next-token logits [B, vocab])."""
        x = self.embed(token_ids)                                           # [B, hidden]
        for layer in range(self.n_layers):
            x = x + self._memory_attention(_rmsnorm(x, self.attn_norm[layer]), layer, keep_mask)
            h = _rmsnorm(x, self.mlp_norm[layer])
            x = x + self.mlp_down[layer](F.gelu(self.mlp_up[layer](h)))
        hidden = _rmsnorm(x, self.final_norm)                               # [B, 4096] ← feeds retriever
        return hidden, self.lm_head(hidden)

    @torch.no_grad()
    def prefill(self, prefill_ids: torch.Tensor) -> torch.Tensor:
        """Toy 'prefill': run a short prompt through DENSE memory attention.

        Returns the last token's hidden state, which seeds the very first
        retrieval cycle (the indexer needs a query hidden state to score against).
        Prefill is intentionally dense (keep_mask=None): the model sees the whole
        history before decoding begins.
        """
        hidden = None
        for t in range(prefill_ids.shape[1]):
            hidden, _ = self.step(prefill_ids[:, t], keep_mask=None)
        return hidden                                                       # [B, 4096]


# ─────────────────────────────────────────────────────────────────────────────
#   Retrieval helper: scores → keep-mask (top-K or threshold)
# ─────────────────────────────────────────────────────────────────────────────
def scores_to_keep_mask(
    scores: torch.Tensor,           # [B, N] sigmoid scores ∈ [0, 1]
    select_mode: str,
    top_k: int,
    threshold: float,
) -> torch.Tensor:
    """Turn per-chunk retriever scores into a boolean keep-mask [B, N]."""
    B, N = scores.shape
    if select_mode == "topk":
        k = min(top_k, N)
        keep = torch.zeros(B, N, dtype=torch.bool, device=scores.device)
        idx = scores.topk(k, dim=-1).indices
        keep.scatter_(1, idx, True)
        return keep
    elif select_mode == "threshold":
        return scores > threshold
    raise ValueError(f"unknown select_mode: {select_mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
#                                   main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Toy DeepSeek-V4-FlashMemory sparse-decode loop driven by the FlashMemory Retriever"
    )
    ap.add_argument("--ckpt", required=True, help="path to the FlashMemory DS-V4 joint checkpoint (.pt)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    ap.add_argument("--batch", type=int, default=1, help="number of parallel decode sequences")
    ap.add_argument("--n-chunks", type=int, default=256, help="number of CSA memory chunks (the long history)")
    ap.add_argument("--steps", type=int, default=192, help="number of decode steps to generate")
    ap.add_argument("--retrieval-interval", type=int, default=64,
                    help="run the retriever every N decode steps (FlashMemory default 64)")
    ap.add_argument("--select-mode", default="topk", choices=["topk", "threshold"],
                    help="how to turn scores into a keep-mask")
    ap.add_argument("--top-k", type=int, default=64, help="chunks to recall per cycle (select-mode=topk)")
    ap.add_argument("--threshold", type=float, default=0.5, help="sigmoid keep threshold (select-mode=threshold)")
    ap.add_argument("--ensemble", default="max", choices=["max", "mean"], help="cross-layer ensemble mode")
    ap.add_argument("--max-position", type=int, default=524288, help="RoPE table length")
    ap.add_argument("--n-layers", type=int, default=2, help="toy decoder layers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    B, N = args.batch, args.n_chunks

    # ── 1. Load retriever ──────────────────────────────────────────────────────
    print(f"FlashMemory DS-V4 -- toy sparse-decode loop")
    print(f"[load] {args.ckpt}")
    retriever = FlashMemoryRetriever.from_checkpoint(
        args.ckpt, device=device, max_position=args.max_position
    )
    retriever.eval()
    print(f"[load] layers={retriever.layer_names}  n_heads={retriever.n_heads}  "
          f"head_dim={retriever.head_dim}")

    # ── 2. Build toy decoder + mock CSA memory ─────────────────────────────────
    decoder = ToyMemoryDecoder(n_chunks=N, n_layers=args.n_layers, device=device, seed=args.seed)
    compressed_k = make_mock_compressed_k(B, N, head_dim=retriever.head_dim,
                                          device=device, seed=args.seed)
    print(f"[init] decoder: {args.n_layers} layers, {decoder.n_heads} heads  |  "
          f"CSA memory: {N} chunks [{retriever.head_dim + 4}] uint8")

    # ── 3. Prefill ─────────────────────────────────────────────────────────────
    prefill_len = 8
    prefill_ids = torch.randint(0, 512, (B, prefill_len), device=device)
    last_hidden = decoder.prefill(prefill_ids)
    base_pos = prefill_len
    last_pos = torch.full((B,), prefill_len - 1, dtype=torch.int64, device=device)

    sel_desc = (f"top-K={args.top_k}" if args.select_mode == "topk"
                else f"sigmoid>{args.threshold}")
    print(f"\n[decode] {args.steps} steps, retriever every {args.retrieval_interval} steps "
          f"({args.select_mode} [{sel_desc}], ensemble={args.ensemble})")
    print("-" * 60)

    # ── 4. Decode loop ──────────────────────────────────────────────────────────
    keep_mask = None
    token = decoder.embed.weight.new_zeros(B, dtype=torch.int64)
    keep_ratios: list[float] = []
    cycle = 0

    for t in range(args.steps):
        abs_pos = base_pos + t

        if t % args.retrieval_interval == 0:
            scores = retriever.ensemble(last_hidden, compressed_k, last_pos, mode=args.ensemble)
            keep_mask = scores_to_keep_mask(scores, args.select_mode, args.top_k, args.threshold)

            n_keep = keep_mask.sum(-1)
            ratio = (n_keep.float() / N)
            keep_ratios.extend(ratio.tolist())
            w_lo = abs_pos
            w_hi = min(abs_pos + args.retrieval_interval, base_pos + args.steps) - 1

            print(f"[cycle {cycle:>2}] pos {w_lo:>5}..{w_hi:<5}  |  "
                  f"keep {fmt_ratio(ratio, B)} ({int(n_keep[0])}/{N})  |  "
                  f"score mean={scores.mean():.4f} max={scores.max():.4f}")
            cycle += 1

        hidden, logits = decoder.step(token, keep_mask)
        token = logits.argmax(-1)
        last_hidden = hidden
        last_pos = torch.full((B,), abs_pos, dtype=torch.int64, device=device)

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    avg_keep = sum(keep_ratios) / max(len(keep_ratios), 1)
    print("-" * 60)
    print(f"[done] {args.steps} tokens, {cycle} cycles, "
          f"avg keep/cycle: {avg_keep:.1%}  =>  ~{1 - avg_keep:.0%} CSA KV dropped")
    print(f"[note] Dropped chunks are masked to -inf in attention (= KV not recalled to GPU).  "
          f"Production swap engine not included in this release.")


def fmt_ratio(t: torch.Tensor, B: int) -> str:
    vals = t.tolist()
    return f"{vals[0]:.1%}" if B == 1 else "[" + ", ".join(f"{v:.1%}" for v in vals) + "]"


if __name__ == "__main__":
    main()
