"""
mine_hard_negs.py — Static hard-neg mining via mean-ensemble baseline scoring.

策略:
  1. 加载 3 个 baseline ckpts (P110+P246+P247) 当 mean-ensemble scorer
  2. 对每个 doc, 对每个 decode token t (sample_interval=4):
     - 3 个 layer 各自 forward → score [n_blocks]
     - mean 聚合 → 1 个 score
  3. 跨 t 聚合: 每个 block 取 max score across all sampled t (代表 doc-level hard 程度)
  4. 在 non-golden 池内取 top-K → per-doc hard neg pool
  5. 保存 dict {doc_path: np.array([K_top_idxs])} → .pkl

Output: hard_neg_pool.pkl (≈ 6 MB for K=300, 5364 docs)

Usage:
  python3 mine_hard_negs.py \
    --ckpt-l10 experiments/expP_110_R10_lr1e4_wt_combfull_l10/ckpts/ckpt_best_recall_k.pt \
    --ckpt-l12 experiments/expP_246_R11_pw_l12_bs512_bf16/ckpts/ckpt_best_recall_k.pt \
    --ckpt-l20 experiments/expP_247_R11_pw_l20_lr2e4_bf16/ckpts/ckpt_best_recall_k.pt \
    --data-config combined_v2 --top-k 300 --sample-interval 4 \
    --output ./hard_neg_pool.pkl --device cuda:0
"""
import os
import sys
import argparse
import pickle
import glob
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import LightningIndexerTrainable, DATA_CONFIGS


def load_ckpt(path: str, csa_layer_idx: int, max_position: int, device: str):
    """Load a single LightningIndexerTrainable from .pt file."""
    model = LightningIndexerTrainable(
        csa_layer_idx=csa_layer_idx, max_position=max_position, n_heads=64,
    ).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    state = {k: v for k, v in state.items() if k != "freqs_cis"}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def mine_doc(models: dict, pkl_path: str, top_k: int,
             sample_interval: int, batch_size: int, device: str) -> np.ndarray:
    """
    Score a single doc, return top-K non-golden block indices.

    Args:
        models: {10: model_l10, 12: model_l12, 20: model_l20}
    Returns:
        top_k_idx: np.array([K]) int32 — global block indices in non-golden pool
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    n_blocks = d["compk_layer_10"].shape[0]
    n_decode = d["hidden_layer_10"].shape[0]

    # golden_blocks
    gb = d.get("golden_blocks")
    if gb is None:
        # 没 golden_blocks → 全 block 都可选, hard pool = top-K across all
        golden_set = np.array([], dtype=np.int32)
    else:
        if hasattr(gb, "numpy"): gb = gb.numpy()
        golden_set = np.unique(gb.astype(np.int32))

    token_indices = list(range(0, n_decode, sample_interval))
    if len(token_indices) == 0:
        token_indices = [0]

    # max_score[b] = max over (sampled t, ensemble layers) of score(t, b)
    max_score = np.full(n_blocks, -np.inf, dtype=np.float32)

    for lid in [10, 12, 20]:
        model = models[lid]
        hidden_all = d[f"hidden_layer_{lid}"]
        compk_all = d[f"compk_layer_{lid}"]
        positions_all = d[f"positions_layer_{lid}"]
        compk_np = compk_all.numpy() if hasattr(compk_all, "numpy") else np.array(compk_all)
        compk_gpu = torch.from_numpy(compk_np).to(device)

        # batch over tokens
        for batch_start in range(0, len(token_indices), batch_size):
            tidxs = token_indices[batch_start : batch_start + batch_size]
            B = len(tidxs)
            hidden_batch = torch.stack([
                torch.as_tensor(hidden_all[t], dtype=torch.float32) for t in tidxs
            ]).to(device)
            pos_batch = torch.stack([
                torch.as_tensor(int(positions_all[t]), dtype=torch.int64) for t in tidxs
            ]).to(device)
            compk_batch = compk_gpu.unsqueeze(0).expand(B, -1, -1)
            lei = torch.zeros(B, dtype=torch.long, device=device)
            scores = model(hidden_batch, compk_batch, pos_batch, layer_embed_idx=lei)  # [B, n_blocks]
            scores = scores.cpu().numpy()
            # update max across t and layers
            np.maximum(max_score, scores.max(axis=0), out=max_score)

    # Mask golden blocks (they're not eligible for hard neg)
    if len(golden_set) > 0:
        # only golden_set members within range
        valid_golden = golden_set[golden_set < n_blocks]
        max_score[valid_golden] = -np.inf

    # Top-K
    k_actual = min(top_k, int(np.sum(np.isfinite(max_score))))
    if k_actual <= 0:
        return np.array([], dtype=np.int32)
    top_idx = np.argpartition(max_score, -k_actual)[-k_actual:]
    # sort by score desc for cleanliness
    top_idx = top_idx[np.argsort(-max_score[top_idx])]
    return top_idx.astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-l10", required=True)
    parser.add_argument("--ckpt-l12", required=True)
    parser.add_argument("--ckpt-l20", required=True)
    parser.add_argument("--data-config", default="combined_v2",
                        choices=list(DATA_CONFIGS.keys()))
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--sample-interval", type=int, default=4,
                        help="Score every Nth decode token")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="./hard_neg_pool.pkl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-position", type=int, default=200000,
                        help="RoPE max position; combined_v2 docs go up to ~131k")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N docs (for testing)")
    args = parser.parse_args()

    print(f"Loading 3 baseline ckpts on {args.device}...")
    models = {
        10: load_ckpt(args.ckpt_l10, csa_layer_idx=10, max_position=args.max_position, device=args.device),
        12: load_ckpt(args.ckpt_l12, csa_layer_idx=12, max_position=args.max_position, device=args.device),
        20: load_ckpt(args.ckpt_l20, csa_layer_idx=20, max_position=args.max_position, device=args.device),
    }
    print(f"Models loaded.")

    # Collect all train docs from data config
    cfg = DATA_CONFIGS[args.data_config]
    all_specs = cfg["train"] + cfg["val"]   # also mine on val (for completeness, val won't be used at train time)
    all_pkl_paths = []
    for spec in all_specs:
        data_dir = spec["data_dir"]
        doc_ids = spec.get("doc_ids")
        pkl_paths = sorted(glob.glob(os.path.join(data_dir, "doc_*.pkl")))
        if doc_ids is not None:
            allowed = {f"doc_{i:05d}.pkl" for i in doc_ids}
            pkl_paths = [p for p in pkl_paths if os.path.basename(p) in allowed]
        all_pkl_paths.extend(pkl_paths)

    if args.limit:
        all_pkl_paths = all_pkl_paths[:args.limit]
    print(f"Mining hard negs for {len(all_pkl_paths)} docs...")

    pool: dict = {}
    t0 = time.time()
    for i, pkl_path in enumerate(all_pkl_paths):
        try:
            top_idx = mine_doc(models, pkl_path, args.top_k,
                               args.sample_interval, args.batch_size, args.device)
            pool[pkl_path] = top_idx
        except Exception as e:
            print(f"  [{i}/{len(all_pkl_paths)}] FAILED on {pkl_path}: {e}")
            pool[pkl_path] = np.array([], dtype=np.int32)

        if (i + 1) % 50 == 0 or (i + 1) == len(all_pkl_paths):
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(all_pkl_paths) - i - 1)
            avg_k = np.mean([len(v) for v in pool.values()])
            print(f"  [{i+1}/{len(all_pkl_paths)}]  avg_k={avg_k:.0f}  "
                  f"elapsed={dt:.0f}s  ETA={eta:.0f}s")

    print(f"\nSaving pool to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(args.output) / 1024**2
    print(f"Done. Pool size: {size_mb:.1f} MB")
    print(f"Coverage: {sum(1 for v in pool.values() if len(v) > 0)}/{len(pool)} docs have non-empty hard pool")


if __name__ == "__main__":
    main()
