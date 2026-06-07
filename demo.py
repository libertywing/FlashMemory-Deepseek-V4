"""
demo.py — minimal standalone demo for the FlashMemory DS-V4 Retriever
=====================================================================

Builds random mock inputs, loads the FlashMemory DS-V4 joint checkpoint, runs
a forward pass, and prints per-chunk scores plus a top-K selection summary.

Run::

    python demo.py --ckpt weights/flashmemory_ds_v4.safetensors

Runs on CPU by default; pass ``--device cuda`` to use a GPU.
"""

from __future__ import annotations

import argparse

import torch

from retriever import FlashMemoryRetriever, dequant_compressed_k


def make_mock_compressed_k(
    batch: int,
    n_chunks: int,
    head_dim: int = 128,
    device: str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    """Construct a valid mock ``compressed_k`` tensor [B, N, head_dim + 4] uint8.

    Layout per chunk: ``head_dim`` float8_e4m3 bytes followed by one float32 scale
    (4 bytes). We build it the same way the real CSA cache stores it:
      1. sample random key vectors, cast to float8_e4m3, view as uint8;
      2. sample a small positive per-chunk scale, view its float32 as 4 uint8 bytes;
      3. concatenate along the last dim.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    # 1) fp8 key bytes
    k_vals = torch.randn(batch, n_chunks, head_dim, generator=g, device=device) * 0.5
    k_fp8 = k_vals.to(torch.float8_e4m3fn)
    fp8_bytes = k_fp8.view(torch.uint8)                       # [B, N, head_dim]

    # 2) float32 per-chunk scale → 4 bytes
    scale = (0.05 + 0.15 * torch.rand(batch, n_chunks, 1, generator=g, device=device)).float()
    scale_bytes = scale.view(torch.uint8)                     # [B, N, 4]

    compressed = torch.cat([fp8_bytes, scale_bytes], dim=-1)  # [B, N, head_dim + 4]
    assert compressed.shape[-1] == head_dim + 4
    return compressed.contiguous()


def main():
    ap = argparse.ArgumentParser(description="FlashMemory DS-V4 Retriever demo")
    ap.add_argument("--ckpt", required=True, help="path to joint checkpoint (.pt)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    ap.add_argument("--batch", type=int, default=2, help="number of decode tokens")
    ap.add_argument("--n-chunks", type=int, default=64, help="number of compressed-K chunks")
    ap.add_argument("--max-position", type=int, default=524288,
                    help="RoPE table length (raise to 1048576 for 1M context)")
    ap.add_argument("--top-k", type=int, default=16, help="top-K chunks to select")
    ap.add_argument("--threshold", type=float, default=0.5, help="sigmoid keep threshold")
    ap.add_argument("--ensemble", default="max", choices=["max", "mean"],
                    help="cross-layer ensemble mode")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    print(f"[demo] loading checkpoint: {args.ckpt}")
    model = FlashMemoryRetriever.from_checkpoint(
        args.ckpt, device=device, max_position=args.max_position
    )
    model.eval()
    print(f"[demo] loaded layers={model.layer_names}  n_heads={model.n_heads}  "
          f"head_dim={model.head_dim}  max_position={model.max_position}")

    # ── Mock inputs ─────────────────────────────────────────────────────────
    B, N = args.batch, args.n_chunks
    hidden = torch.randn(B, 4096, device=device, dtype=torch.float32)
    compressed_k = make_mock_compressed_k(B, N, head_dim=model.head_dim,
                                          device=device, seed=args.seed)
    # token positions for each decode token (arbitrary; here spaced out)
    positions = torch.arange(B, device=device, dtype=torch.int64) * 1000 + 4096

    print(f"\n[demo] mock inputs: hidden={tuple(hidden.shape)} "
          f"compressed_k={tuple(compressed_k.shape)} ({compressed_k.dtype}) "
          f"positions={positions.tolist()}")

    # sanity: show dequant works
    k_float = dequant_compressed_k(compressed_k, head_dim=model.head_dim)
    print(f"[demo] dequantized K: shape={tuple(k_float.shape)} "
          f"mean={k_float.mean().item():+.4f} std={k_float.std().item():.4f}")

    # ── Per-layer scores ──────────────────────────────────────────────────────
    per_layer = model(hidden, compressed_k, positions, apply_sigmoid=True)
    print("\n[demo] per-layer sigmoid score stats (over all chunks):")
    for name, s in per_layer.items():
        print(f"    {name}: min={s.min().item():.4f} mean={s.mean().item():.4f} "
              f"max={s.max().item():.4f}")

    # ── Cross-layer ensemble ──────────────────────────────────────────────────
    scores = model.ensemble(hidden, compressed_k, positions, mode=args.ensemble)  # [B, N]
    print(f"\n[demo] ensembled ({args.ensemble}) per-chunk scores [B={B}, N={N}]:")
    for b in range(B):
        row = scores[b]
        preview = ", ".join(f"{v:.3f}" for v in row[:12].tolist())
        print(f"    row {b}: [{preview}{', ...' if N > 12 else ''}]")

    # ── Selection: threshold ──────────────────────────────────────────────────
    keep_thr = model.select_topk(hidden, compressed_k, positions,
                                 threshold=args.threshold, mode=args.ensemble)
    print(f"\n[demo] threshold selection (sigmoid > {args.threshold}):")
    for b in range(B):
        n_keep = int(keep_thr[b].sum().item())
        print(f"    row {b}: keep {n_keep}/{N} chunks  (keep ratio {n_keep / N:.1%})")

    # ── Selection: top-K ──────────────────────────────────────────────────────
    keep_topk = model.select_topk(hidden, compressed_k, positions,
                                  top_k=args.top_k, mode=args.ensemble)
    print(f"\n[demo] top-K selection (k={args.top_k}):")
    for b in range(B):
        idx = keep_topk[b].nonzero(as_tuple=True)[0].tolist()
        print(f"    row {b}: kept chunk indices = {idx}")

    print("\n[demo] done. ✅  forward + scoring + selection all ran.")


if __name__ == "__main__":
    main()
