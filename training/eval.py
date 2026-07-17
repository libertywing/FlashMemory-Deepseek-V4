"""
eval.py — 独立评估脚本（全集候选模式）

功能：
  1. 加载指定 checkpoint（ckpt_best_f1.pt 或任意 .pt 文件）
  2. 在指定 split（默认 test）上计算真实场景下的指标：
       - recall@K（K=128, 256, 512, 1024）：用 hidden_state 对文档所有 ~7900 个 block
         打分，top-K 覆盖正例的比例  ← 这是真实推理的衡量标准
       - F1（threshold=0.5）：同样在全集候选上计算
  3. 支持批量扫描整个 experiments/ 目录，对每个实验的最优 checkpoint 统一评估
  4. 支持多层（--layer-ids）和多数据目录（--data-config）

⚠️  注意：之前的 eval.py 在「正负各半的 sampled 子集」上评估，positive rate=50%，
    与真实推理（positive rate≈2-3%，候选集≈7900 blocks）相差一个量级，
    导致 F1 虚高（~0.95）、recall@1024 恒为 1.0。
    本脚本改为对文档全集 block 评估。

用法（单个 checkpoint，旧数据）：
  python eval.py \\
      --ckpt experiments/expP_17_baseline_best/ckpts/ckpt_best_f1.pt \\
      --layer 20 --n-heads 64 \\
      --split test

用法（多层 + 层条件化，新数据配置）：
  python eval.py \\
      --ckpt experiments/expP_XX_config_c/ckpts/ckpt_best_f1.pt \\
      --layer-ids 10,12,20 --n-heads 64 --layer-embed \\
      --data-config new_only

用法（批量扫描所有实验）：
  python eval.py --scan-dir experiments --layer 20 --split test

用法（同时评估 val + test）：
  python eval.py --ckpt <path> --layer 20 --n-heads 64 --split val test
"""

import argparse
import os
import sys
import glob
import pickle
import numpy as np
import torch
from pathlib import Path

from train import LightningIndexerTrainable, DATA_CONFIGS, LAYER_EMBED_MAP
try:
    from train import SPLIT_DOC_IDS  # legacy --data-dir mode only; removed from train.py
except ImportError:
    SPLIT_DOC_IDS = {}  # data-config mode (default) does not need it


# ─────────────────────────────────────────────────────────────────────────────
# 核心评估：单目录、支持多层 union
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_full_spec(model, data_dir, doc_ids, layer_ids,
                       label_interval, device,
                       sample_interval=1,
                       topk_list=(128, 256, 512, 1024),
                       batch_size=256):
    """
    对单个目录内的文档做全集评测（多层 union 模式）。

    对每个 decode token：
      1. 对 layer_ids 中每一层用 hidden_state 对该文档所有 n_blocks 个 block 打分
      2. 取各层分数的 element-wise max（union top-K 语义）
      3. 用真实 label_pointers/label_indices 确定正例集合
      4. 计算 recall@K 和 F1（threshold=0.5）

    Returns:
        tp, fp, fn, recallk_sum (dict k→float), recallk_count (int)
    """
    tp = fp = fn = 0
    recallk_sum   = {k: 0.0 for k in topk_list}
    recallk_count = 0

    pkl_paths = sorted(glob.glob(os.path.join(data_dir, "doc_*.pkl")))
    if doc_ids is not None:
        allowed = {f"doc_{i:05d}.pkl" for i in doc_ids}
        pkl_paths = [p for p in pkl_paths if os.path.basename(p) in allowed]

    print(f"    [{data_dir}]  {len(pkl_paths)} docs ...")

    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        ref_lid       = layer_ids[0]
        n_decode      = data[f"hidden_layer_{ref_lid}"].shape[0]
        n_blocks      = data[f"compk_layer_{ref_lid}"].shape[0]
        label_ptrs    = data["label_pointers"].numpy()
        label_idxs    = data["label_indices"].numpy()

        token_indices = list(range(0, n_decode, sample_interval))
        n_tokens      = len(token_indices)

        # agg_scores[i, b] = max score across all layers for token i, block b
        agg_scores = np.zeros((n_tokens, n_blocks), dtype=np.float32)

        for lid in layer_ids:
            hidden_all    = data[f"hidden_layer_{lid}"]
            compk_all     = data[f"compk_layer_{lid}"]
            positions_all = data[f"positions_layer_{lid}"]

            compk_np  = compk_all.numpy() if hasattr(compk_all, "numpy") else np.array(compk_all)
            compk_gpu = torch.from_numpy(compk_np).to(device)   # [n_blocks, 132]

            lid_embed_val = LAYER_EMBED_MAP.get(lid, 0)

            for batch_start in range(0, n_tokens, batch_size):
                batch_tidxs = token_indices[batch_start : batch_start + batch_size]
                B = len(batch_tidxs)

                hidden_batch = torch.stack([
                    torch.as_tensor(hidden_all[t], dtype=torch.float32)
                    for t in batch_tidxs
                ]).to(device)   # [B, 4096]

                pos_batch = torch.stack([
                    torch.as_tensor(int(positions_all[t]), dtype=torch.int64)
                    for t in batch_tidxs
                ]).to(device)   # [B]

                compk_batch = compk_gpu.unsqueeze(0).expand(B, -1, -1)   # [B, n_blocks, 132]
                lei_batch   = torch.full((B,), lid_embed_val, dtype=torch.long, device=device)

                scores = model(
                    hidden_batch, compk_batch, pos_batch, layer_embed_idx=lei_batch
                ).cpu().numpy()   # [B, n_blocks]

                # union via element-wise max across layers
                agg_scores[batch_start : batch_start + B] = np.maximum(
                    agg_scores[batch_start : batch_start + B], scores
                )

        # ── 计算 F1 和 recall@K（向量化）────────────────────────────────────
        for bi, t in enumerate(token_indices):
            t_end    = min(t + label_interval, n_decode)
            pos_idxs = np.unique(label_idxs[label_ptrs[t] : label_ptrs[t_end]])
            pos_idxs = pos_idxs[pos_idxs < n_blocks]  # guard: data gen off-by-one in some docs

            if len(pos_idxs) == 0:
                continue

            score_t = agg_scores[bi]   # [n_blocks]
            pred_t  = score_t >= 0.5

            # F1
            lbl_mask = np.zeros(n_blocks, dtype=bool)
            lbl_mask[pos_idxs] = True
            tp += int(( pred_t &  lbl_mask).sum())
            fp += int(( pred_t & ~lbl_mask).sum())
            fn += int((~pred_t &  lbl_mask).sum())

            # recall@K（argpartition 取 top-max_k，内部排序后复用）
            n_pos      = len(pos_idxs)
            max_k      = min(max(topk_list), n_blocks)
            # argpartition O(n) 取 top-max_k, 再对 max_k 个元素排序 O(k·log k)
            top_idx    = np.argpartition(-score_t, max_k - 1)[:max_k]
            top_sorted = top_idx[np.argsort(-score_t[top_idx])]
            for k in topk_list:
                actual_k = min(k, n_blocks)
                topk_idx = top_sorted[:actual_k]
                hit      = int(np.isin(topk_idx, pos_idxs).sum())
                recallk_sum[k] += hit / n_pos

            recallk_count += 1

    return tp, fp, fn, recallk_sum, recallk_count


@torch.no_grad()
def evaluate_full_multi(model, specs, layer_ids, label_interval, device,
                        sample_interval=1, topk_list=(128, 256, 512, 1024),
                        batch_size=256):
    """
    多目录聚合评测。对 specs 列表中每个 {"data_dir", "doc_ids"} 调用
    evaluate_full_spec，累积 tp/fp/fn 和 recall@K 后统一计算指标。

    Returns:
        dict with keys: precision, recall, f1, n_samples,
                        recall@128, recall@256, recall@512, recall@1024
    """
    model.eval()
    total_tp = total_fp = total_fn = 0
    total_recallk_sum   = {k: 0.0 for k in topk_list}
    total_recallk_count = 0

    for spec in specs:
        t, f, n, rk_sum, rk_cnt = evaluate_full_spec(
            model          = model,
            data_dir       = spec["data_dir"],
            doc_ids        = spec.get("doc_ids"),
            layer_ids      = layer_ids,
            label_interval = label_interval,
            device         = device,
            sample_interval= sample_interval,
            topk_list      = topk_list,
            batch_size     = batch_size,
        )
        total_tp += t
        total_fp += f
        total_fn += n
        for k in topk_list:
            total_recallk_sum[k] += rk_sum[k]
        total_recallk_count += rk_cnt

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    result = dict(
        precision = precision,
        recall    = recall,
        f1        = f1,
        n_samples = total_recallk_count,
    )
    for k in topk_list:
        result[f"recall@{k}"] = (
            total_recallk_sum[k] / total_recallk_count
            if total_recallk_count > 0 else 0.0
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble 评测：3 个独立 retriever (per-layer) + score-level 聚合
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_full_spec_ensemble(models_per_layer, data_dir, doc_ids,
                                label_interval, device,
                                sample_interval=1,
                                topk_list=(128, 256, 512, 1024),
                                batch_size=256):
    """
    Ensemble 单目录评测：每个 layer 用各自的 model 打分，3 种聚合 + per-layer 单独。

    Args:
        models_per_layer: dict {layer_id: model} —— 3 个独立 retriever

    Returns:
        results dict, key ∈ {"mean", "max", "min", "L{layer}"} for each layer,
        each value = (tp, fp, fn, recallk_sum, recallk_count)
    """
    layers = sorted(models_per_layer.keys())
    agg_keys = ["mean", "max", "min"] + [f"L{lid}" for lid in layers]

    # 累积 stats: per agg key
    stats = {k: dict(tp=0, fp=0, fn=0,
                     rk_sum={kk: 0.0 for kk in topk_list},
                     rk_cnt=0) for k in agg_keys}

    pkl_paths = sorted(glob.glob(os.path.join(data_dir, "doc_*.pkl")))
    if doc_ids is not None:
        allowed = {f"doc_{i:05d}.pkl" for i in doc_ids}
        pkl_paths = [p for p in pkl_paths if os.path.basename(p) in allowed]

    print(f"    [{data_dir}]  {len(pkl_paths)} docs ...")

    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        ref_lid       = layers[0]
        n_decode      = data[f"hidden_layer_{ref_lid}"].shape[0]
        n_blocks      = data[f"compk_layer_{ref_lid}"].shape[0]
        label_ptrs    = data["label_pointers"].numpy()
        label_idxs    = data["label_indices"].numpy()

        token_indices = list(range(0, n_decode, sample_interval))
        n_tokens      = len(token_indices)

        # per_layer_scores[i, b] for each layer
        per_layer_scores = {lid: np.zeros((n_tokens, n_blocks), dtype=np.float32)
                            for lid in layers}

        for lid in layers:
            model         = models_per_layer[lid]
            hidden_all    = data[f"hidden_layer_{lid}"]
            compk_all     = data[f"compk_layer_{lid}"]
            positions_all = data[f"positions_layer_{lid}"]

            compk_np  = compk_all.numpy() if hasattr(compk_all, "numpy") else np.array(compk_all)
            compk_gpu = torch.from_numpy(compk_np).to(device)

            for batch_start in range(0, n_tokens, batch_size):
                batch_tidxs = token_indices[batch_start : batch_start + batch_size]
                B = len(batch_tidxs)

                hidden_batch = torch.stack([
                    torch.as_tensor(hidden_all[t], dtype=torch.float32)
                    for t in batch_tidxs
                ]).to(device)
                pos_batch = torch.stack([
                    torch.as_tensor(int(positions_all[t]), dtype=torch.int64)
                    for t in batch_tidxs
                ]).to(device)
                compk_batch = compk_gpu.unsqueeze(0).expand(B, -1, -1)

                # Each model is single-layer: layer_embed_idx not needed
                scores = model(hidden_batch, compk_batch, pos_batch).cpu().numpy()
                per_layer_scores[lid][batch_start : batch_start + B] = scores

        # ── Aggregations: stack into [n_layers, n_tokens, n_blocks] ─────────
        score_stack = np.stack([per_layer_scores[lid] for lid in layers], axis=0)
        agg_scores_dict = {
            "mean": score_stack.mean(axis=0),
            "max":  score_stack.max(axis=0),
            "min":  score_stack.min(axis=0),
        }
        for lid in layers:
            agg_scores_dict[f"L{lid}"] = per_layer_scores[lid]

        # ── Compute metrics for each agg ─────────────────────────────────────
        for bi, t in enumerate(token_indices):
            t_end    = min(t + label_interval, n_decode)
            pos_idxs = np.unique(label_idxs[label_ptrs[t] : label_ptrs[t_end]])
            pos_idxs = pos_idxs[pos_idxs < n_blocks]  # guard: data gen off-by-one in some docs

            if len(pos_idxs) == 0:
                continue

            n_pos    = len(pos_idxs)
            lbl_mask = np.zeros(n_blocks, dtype=bool)
            lbl_mask[pos_idxs] = True

            for agg_key, agg_scores in agg_scores_dict.items():
                score_t = agg_scores[bi]
                pred_t  = score_t >= 0.5

                stats[agg_key]["tp"] += int(( pred_t &  lbl_mask).sum())
                stats[agg_key]["fp"] += int(( pred_t & ~lbl_mask).sum())
                stats[agg_key]["fn"] += int((~pred_t &  lbl_mask).sum())

                max_k      = min(max(topk_list), n_blocks)
                top_idx    = np.argpartition(-score_t, max_k - 1)[:max_k]
                top_sorted = top_idx[np.argsort(-score_t[top_idx])]
                for k in topk_list:
                    actual_k = min(k, n_blocks)
                    topk_idx = top_sorted[:actual_k]
                    hit      = int(np.isin(topk_idx, pos_idxs).sum())
                    stats[agg_key]["rk_sum"][k] += hit / n_pos

                stats[agg_key]["rk_cnt"] += 1

    return stats


@torch.no_grad()
def evaluate_full_multi_ensemble(models_per_layer, specs, label_interval, device,
                                 sample_interval=1, topk_list=(128, 256, 512, 1024),
                                 batch_size=256):
    """
    Ensemble 多目录评测。聚合 specs 列表 → 返回每种 agg 的 metrics。
    """
    for m in models_per_layer.values():
        m.eval()

    layers = sorted(models_per_layer.keys())
    agg_keys = ["mean", "max", "min"] + [f"L{lid}" for lid in layers]
    total = {k: dict(tp=0, fp=0, fn=0,
                     rk_sum={kk: 0.0 for kk in topk_list},
                     rk_cnt=0) for k in agg_keys}

    for spec in specs:
        spec_stats = evaluate_full_spec_ensemble(
            models_per_layer = models_per_layer,
            data_dir         = spec["data_dir"],
            doc_ids          = spec.get("doc_ids"),
            label_interval   = label_interval,
            device           = device,
            sample_interval  = sample_interval,
            topk_list        = topk_list,
            batch_size       = batch_size,
        )
        for agg_key in agg_keys:
            total[agg_key]["tp"] += spec_stats[agg_key]["tp"]
            total[agg_key]["fp"] += spec_stats[agg_key]["fp"]
            total[agg_key]["fn"] += spec_stats[agg_key]["fn"]
            for k in topk_list:
                total[agg_key]["rk_sum"][k] += spec_stats[agg_key]["rk_sum"][k]
            total[agg_key]["rk_cnt"] += spec_stats[agg_key]["rk_cnt"]

    # Reduce to final metrics per agg
    results = {}
    for agg_key in agg_keys:
        s = total[agg_key]
        precision = s["tp"] / (s["tp"] + s["fp"] + 1e-8)
        recall    = s["tp"] / (s["tp"] + s["fn"] + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        r = dict(precision=precision, recall=recall, f1=f1, n_samples=s["rk_cnt"])
        for k in topk_list:
            r[f"recall@{k}"] = (s["rk_sum"][k] / s["rk_cnt"]) if s["rk_cnt"] > 0 else 0.0
        results[agg_key] = r
    return results


def eval_ensemble(ckpt_paths, layers, n_heads,
                  splits, data_dir, data_config,
                  label_interval, sample_interval, device, topk_list, batch_size):
    """
    Ensemble 评测入口：3 个 ckpt + 3 个 layer，输出 mean/max/min + per-layer 单独。

    Args:
        ckpt_paths: list[str], 长度=len(layers)
        layers:     list[int], 如 [10, 12, 20]
    """
    assert len(ckpt_paths) == len(layers), \
        f"ckpts ({len(ckpt_paths)}) ≠ layers ({len(layers)})"

    print(f"\n{'='*70}")
    print(f"ENSEMBLE EVAL")
    for lid, cp in zip(layers, ckpt_paths):
        print(f"  Layer {lid}: {cp}")
    print(f"  N_HEADS={n_heads}")
    mode = f"DATA_CONFIG={data_config}" if data_config else f"DATA_DIR={data_dir}"
    print(f"  Mode    : FULL-SET (all blocks)  {mode}")

    # Load 3 models
    models_per_layer = {}
    for lid, cp in zip(layers, ckpt_paths):
        models_per_layer[lid] = load_model(cp, lid, n_heads, device, layer_embed=False)
    print(f"  Loaded {len(models_per_layer)} retriever models")

    all_results = {}
    for split in splits:
        if data_config is not None:
            if split not in DATA_CONFIGS[data_config]:
                print(f"  [SKIP] split='{split}' not in DATA_CONFIGS['{data_config}']")
                continue
            specs = DATA_CONFIGS[data_config][split]
        else:
            doc_ids = SPLIT_DOC_IDS.get(split)
            specs   = [{"data_dir": data_dir, "doc_ids": doc_ids}]

        n_docs = sum(len(s.get("doc_ids") or []) for s in specs)
        print(f"\n  [{split.upper()}]  {len(specs)} dir(s), ~{n_docs} docs total")

        res = evaluate_full_multi_ensemble(
            models_per_layer = models_per_layer,
            specs            = specs,
            label_interval   = label_interval,
            device           = device,
            sample_interval  = sample_interval,
            topk_list        = topk_list,
            batch_size       = batch_size,
        )
        all_results[split] = res

        # ── Print results table ──────────────────────────────────────────────
        print(f"\n  ── {split.upper()} Results ──────────────────────────────────────────")
        agg_keys_print = [f"L{lid}" for lid in sorted(layers)] + ["mean", "max", "min"]
        header = f"{'Aggregation':<8}  {'P':>6}  {'R':>6}  {'F1':>6}"
        for k in topk_list:
            header += f"  {f'r@{k}':>7}"
        print(f"    {header}")
        print(f"    {'-'*len(header)}")
        for ak in agg_keys_print:
            r = res[ak]
            line = f"{ak:<8}  {r['precision']:>6.4f}  {r['recall']:>6.4f}  {r['f1']:>6.4f}"
            for k in topk_list:
                line += f"  {r[f'recall@{k}']:>7.4f}"
            print(f"    {line}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 向后兼容：保留旧接口（单层、单目录）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_full(model, data_dir, csa_layer_idx, doc_ids,
                  label_interval, device,
                  sample_interval=1,
                  topk_list=(128, 256, 512, 1024),
                  batch_size=256):
    """
    旧接口兼容包装：单层、单目录。内部调用 evaluate_full_multi。
    """
    return evaluate_full_multi(
        model          = model,
        specs          = [{"data_dir": data_dir, "doc_ids": doc_ids}],
        layer_ids      = [csa_layer_idx],
        label_interval = label_interval,
        device         = device,
        sample_interval= sample_interval,
        topk_list      = topk_list,
        batch_size     = batch_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, layer, n_heads, device, layer_embed=False):
    """
    从 checkpoint 加载模型。

    Args:
        layer:       primary CSA layer index (用于初始化 LightningIndexerTrainable)
        n_heads:     query head 数量（默认 64）
        layer_embed: 是否启用层条件化 embedding（Config C，--layer-embed）
    """
    state = torch.load(ckpt_path, map_location=device)
    # 从 checkpoint 中推断 max_position，避免 freqs_cis 尺寸不匹配
    # （训练时若数据中有超长序列，max_position > 默认的 131072）
    max_position = state["freqs_cis"].shape[0] if "freqs_cis" in state else 131072
    # 从 wq_a.weight shape [q_lora_rank, 4096] 自动推断 rank
    q_lora_rank = state["wq_a.weight"].shape[0] if "wq_a.weight" in state else None
    # 从 wq_b.weight shape [n_heads*head_dim, q_lora_rank] 自动推断 n_heads
    if "wq_b.weight" in state:
        wq_b_out = state["wq_b.weight"].shape[0]
        n_heads_inferred = wq_b_out // 128  # head_dim=128
        if n_heads_inferred != n_heads:
            print(f"  [auto] override n_heads {n_heads} → {n_heads_inferred} (from wq_b shape)")
            n_heads = n_heads_inferred
    model = LightningIndexerTrainable(
        csa_layer_idx=layer,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        max_position=max_position,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Single-checkpoint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_one(ckpt_path, layer, layer_ids, n_heads, layer_embed,
             splits, data_dir, data_config,
             label_interval, sample_interval, device, topk_list, batch_size):
    """
    评估单个 checkpoint。

    数据来源优先级：
      1. data_config（--data-config）→ 使用 DATA_CONFIGS[data_config][split]
      2. data_dir（--data-dir）+ split → 使用 SPLIT_DOC_IDS[split] 的旧模式

    层模式：
      - layer_ids 非空（--layer-ids）→ 多层 union 评估
      - 否则 → 单层（layer）评估
    """
    if not layer_ids:
        layer_ids = [layer]

    print(f"\n{'='*70}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Layer(s)   : {layer_ids}   N_HEADS={n_heads}   layer_embed={layer_embed}")
    mode = f"DATA_CONFIG={data_config}" if data_config else f"DATA_DIR={data_dir}"
    print(f"Mode       : FULL-SET (all blocks per doc)  {mode}")

    model = load_model(ckpt_path, layer, n_heads, device, layer_embed=layer_embed)
    results = {}

    for split in splits:
        # ── 确定 specs 列表 ──────────────────────────────────────────────────
        if data_config is not None:
            if split not in DATA_CONFIGS[data_config]:
                print(f"  [SKIP] split='{split}' not in DATA_CONFIGS['{data_config}']")
                continue
            specs = DATA_CONFIGS[data_config][split]
        else:
            doc_ids = SPLIT_DOC_IDS.get(split)
            specs   = [{"data_dir": data_dir, "doc_ids": doc_ids}]

        n_docs = sum(len(s.get("doc_ids") or []) for s in specs)
        print(f"\n  [{split.upper()}]  {len(specs)} dir(s), ~{n_docs} docs total")
        print(f"    Evaluating on ALL blocks (full-set mode) ...")

        res = evaluate_full_multi(
            model          = model,
            specs          = specs,
            layer_ids      = layer_ids,
            label_interval = label_interval,
            device         = device,
            sample_interval= sample_interval,
            topk_list      = topk_list,
            batch_size     = batch_size,
        )
        results[split] = res

        print(f"    precision  = {res['precision']:.4f}")
        print(f"    recall     = {res['recall']:.4f}")
        print(f"    F1         = {res['f1']:.4f}")
        print(f"    n_samples  = {res['n_samples']}")
        for k in topk_list:
            print(f"    recall@{k:<5} = {res[f'recall@{k}']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scan-all-experiments mode
# ─────────────────────────────────────────────────────────────────────────────

def infer_n_heads(exp_dir):
    name = os.path.basename(exp_dir)
    if "nh128" in name:
        return 128
    return 64


def infer_layer_embed(exp_dir):
    name = os.path.basename(exp_dir)
    return "layerembed" in name or "layer_embed" in name or "config_c" in name.lower()


def scan_experiments(scan_dir, layer, layer_ids, n_heads, layer_embed,
                     splits, data_dir, data_config,
                     label_interval, sample_interval, device, topk_list, batch_size,
                     ckpt_name="ckpt_best_f1.pt"):
    pattern    = os.path.join(scan_dir, "expP_*", "ckpts", ckpt_name)
    ckpt_paths = sorted(glob.glob(pattern))

    if not ckpt_paths:
        print(f"[scan] No checkpoints found matching: {pattern}")
        sys.exit(1)

    print(f"[scan] Found {len(ckpt_paths)} checkpoints in {scan_dir}")

    summary_rows = []
    for ckpt_path in ckpt_paths:
        exp_dir  = Path(ckpt_path).parent.parent
        _n_heads = n_heads if n_heads else infer_n_heads(str(exp_dir))
        _le      = layer_embed or infer_layer_embed(str(exp_dir))
        exp_name = exp_dir.name

        try:
            res_dict = eval_one(
                ckpt_path   = ckpt_path,
                layer       = layer,
                layer_ids   = layer_ids,
                n_heads     = _n_heads,
                layer_embed = _le,
                splits      = splits,
                data_dir    = data_dir,
                data_config = data_config,
                label_interval  = label_interval,
                sample_interval = sample_interval,
                device      = device,
                topk_list   = topk_list,
                batch_size  = batch_size,
            )
            for split, res in res_dict.items():
                row = {"exp": exp_name, "split": split, "n_heads": _n_heads,
                       "layer_embed": _le}
                row.update(res)
                summary_rows.append(row)
        except Exception as e:
            print(f"  [ERROR] {exp_name}: {e}")

    if summary_rows:
        print(f"\n{'='*70}")
        print("SUMMARY  (full-set eval, all blocks as candidates)")
        print(f"{'='*70}")
        header = (f"{'Experiment':<40} {'split':<6} {'nh':>4} "
                  f"{'F1':>6} {'r@128':>7} {'r@512':>7} {'r@1024':>7}")
        print(header)
        print("-" * 75)
        for row in sorted(summary_rows, key=lambda r: -r.get("recall@128", 0)):
            print(
                f"{row['exp']:<40} {row['split']:<6} {row['n_heads']:>4} "
                f"{row['f1']:>6.4f} "
                f"{row.get('recall@128', 0):>7.4f} "
                f"{row.get('recall@512', 0):>7.4f} "
                f"{row.get('recall@1024', 0):>7.4f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lightning Indexer Retriever — 全集候选评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ckpt",      type=str, help="单个 checkpoint 路径")
    mode.add_argument("--scan-dir",  type=str, help="扫描目录，批量评估 expP_*/ckpts/ckpt_best_f1.pt")
    mode.add_argument("--ensemble-ckpts", type=str,
                      help="Ensemble 模式：3 个 ckpt 路径，逗号分隔。"
                           "需配合 --ensemble-layers 使用，每层一个 ckpt。")

    # ── Ensemble 配置（仅 --ensemble-ckpts 模式生效） ──────────────────────
    parser.add_argument("--ensemble-layers", type=str, default="10,12,20",
                        help="Ensemble 模式下每个 ckpt 对应的 CSA layer，逗号分隔")

    # ── 模型架构 ──────────────────────────────────────────────────────────────
    parser.add_argument("--layer",    type=int, default=20,
                        help="primary CSA layer index（单层模式 或 多层模式的 primary layer）")
    parser.add_argument("--layer-ids", type=str, default=None,
                        help="多层模式：逗号分隔的 CSA layer 列表，如 10,12,20 "
                             "（覆盖 --layer）")
    parser.add_argument("--n-heads",  type=int, default=64)
    parser.add_argument("--layer-embed", action="store_true",
                        help="启用层条件化 embedding（Config C，需与训练时一致）")

    # ── 数据来源 ──────────────────────────────────────────────────────────────
    data_src = parser.add_mutually_exclusive_group()
    data_src.add_argument("--data-config", type=str, default=None,
                          choices=list(DATA_CONFIGS.keys()),
                          help="使用 DATA_CONFIGS 中预定义的数据配置（推荐）")
    data_src.add_argument("--data-dir",    type=str, default="./data",
                          help="旧模式：单个 pkl 目录（与 --split 搭配使用）")

    # ── 评估参数 ──────────────────────────────────────────────────────────────
    parser.add_argument("--split",           nargs="+", default=["test"],
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--label-interval",  type=int,  default=64)
    parser.add_argument("--sample-interval", type=int,  default=1,
                        help="每隔几个 token 评估一次（1=全部，慢但准确；4=快速粗估）")
    parser.add_argument("--batch-size",      type=int,  default=256,
                        help="hidden_state 批大小（不影响结果，只影响速度）")
    parser.add_argument("--topk",            nargs="+", type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument("--ckpt-name",       default="ckpt_best_f1.pt")
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # ── 解析 layer_ids ─────────────────────────────────────────────────────
    if args.layer_ids is not None:
        layer_ids = [int(x.strip()) for x in args.layer_ids.split(",")]
    else:
        layer_ids = []   # 空列表 → eval_one 内退化为 [args.layer]

    print(f"Device          : {args.device}")
    print(f"Split           : {args.split}")
    print(f"Top-K           : {args.topk}")
    print(f"Sample-interval : {args.sample_interval}")
    if layer_ids:
        print(f"Layer IDs       : {layer_ids}  (multi-layer union mode)")
    else:
        print(f"Layer           : {args.layer}  (single-layer mode)")
    if args.layer_embed:
        print(f"Layer embed     : ENABLED (Config C)")
    if args.data_config:
        print(f"Data config     : {args.data_config}")
    print(f"⚠️  Full-set mode: scoring ALL blocks in each document")

    if args.ckpt:
        eval_one(
            ckpt_path       = args.ckpt,
            layer           = args.layer,
            layer_ids       = layer_ids,
            n_heads         = args.n_heads,
            layer_embed     = args.layer_embed,
            splits          = args.split,
            data_dir        = args.data_dir,
            data_config     = args.data_config,
            label_interval  = args.label_interval,
            sample_interval = args.sample_interval,
            device          = args.device,
            topk_list       = args.topk,
            batch_size      = args.batch_size,
        )
    elif args.ensemble_ckpts:
        ckpt_paths = [s.strip() for s in args.ensemble_ckpts.split(",")]
        ens_layers = [int(s.strip()) for s in args.ensemble_layers.split(",")]
        eval_ensemble(
            ckpt_paths      = ckpt_paths,
            layers          = ens_layers,
            n_heads         = args.n_heads,
            splits          = args.split,
            data_dir        = args.data_dir,
            data_config     = args.data_config,
            label_interval  = args.label_interval,
            sample_interval = args.sample_interval,
            device          = args.device,
            topk_list       = args.topk,
            batch_size      = args.batch_size,
        )
    else:
        scan_experiments(
            scan_dir        = args.scan_dir,
            layer           = args.layer,
            layer_ids       = layer_ids,
            n_heads         = args.n_heads,
            layer_embed     = args.layer_embed,
            splits          = args.split,
            data_dir        = args.data_dir,
            data_config     = args.data_config,
            label_interval  = args.label_interval,
            sample_interval = args.sample_interval,
            device          = args.device,
            topk_list       = args.topk,
            batch_size      = args.batch_size,
            ckpt_name       = args.ckpt_name,
        )


if __name__ == "__main__":
    main()
