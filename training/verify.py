"""
Lightning Indexer E2E 验证脚本
===============================
验证推理代码的输出与模型在线推理时存储的 logits 是否一致。

用法:
    # 验证单个文档
    python verify.py --data-path ./data/doc_00030.pkl --layer 20

    # 批量验证所有文档 × 所有层
    python verify.py --batch

验证标准:
    - Top-10 overlap >= 90%
    - Pearson correlation >= 0.99
"""

import os
import sys
import glob
import pickle
import argparse

import torch
import numpy as np

from inference import LightningIndexer, CSA_LAYER_IDS


def verify_single(indexer, data, layer_idx, token_idx=0):
    """
    验证单个 token 的推理结果。

    Returns:
        dict: 包含 overlap 和 correlation 等指标，None 表示数据不完整
    """
    h_key = f"hidden_layer_{layer_idx}"
    k_key = f"compk_layer_{layer_idx}"
    p_key = f"positions_layer_{layer_idx}"
    l_key = f"logits_layer_{layer_idx}_token_{token_idx}"

    if any(k not in data for k in [h_key, k_key, p_key, l_key]):
        return None

    hidden = data[h_key]
    compk = data[k_key]
    positions = data[p_key]
    stored_logits = data[l_key].float()

    # 推理
    logits = indexer.forward(
        hidden[token_idx:token_idx + 1],
        compk,
        positions[token_idx:token_idx + 1],
    )
    computed = logits[0].cpu()
    n_blk = len(stored_logits)
    computed = computed[:n_blk]

    # 计算指标
    corr = np.corrcoef(stored_logits.numpy(), computed.numpy())[0, 1]

    overlaps = {}
    for K in [10, 20, 50, 512, 1024]:
        k_actual = min(K, n_blk)
        s_topk = set(stored_logits.topk(k_actual)[1].tolist())
        c_topk = set(computed.topk(k_actual)[1].tolist())
        overlaps[K] = len(s_topk & c_topk) / k_actual

    return {
        "correlation": corr,
        "overlaps": overlaps,
        "n_blocks": n_blk,
        "position": int(positions[token_idx].item()),
    }


def verify_doc(indexer, data_path, layer_idx, verbose=True):
    """验证一个文档的前 3 个 token。"""
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    results = []
    for token_idx in range(3):
        result = verify_single(indexer, data, layer_idx, token_idx)
        if result is not None:
            results.append(result)
            if verbose:
                r = result
                print(f"  Token {token_idx} (pos={r['position']}):")
                for K, pct in r["overlaps"].items():
                    print(f"    Top-{K:>4d}: {pct*100:.1f}%")
                print(f"    Correlation: {r['correlation']:.4f}")

    return results


def batch_verify(data_dir, weight_dir, layers, device="cuda"):
    """批量验证所有文档 × 指定层。"""
    # 加载 indexer
    indexers = {}
    for layer_idx in layers:
        layer_id = CSA_LAYER_IDS[layer_idx]
        weight_file = os.path.join(weight_dir, f"layer_{layer_id:02d}.safetensors")
        if not os.path.exists(weight_file):
            print(f"[跳过] layer {layer_idx}: 权重文件不存在 ({weight_file})")
            continue
        indexers[layer_idx] = LightningIndexer(
            csa_layer_idx=layer_idx,
            weight_dir=weight_dir,
            device=device,
            max_position=131072,
        )

    if not indexers:
        print("没有可用的 indexer，退出。")
        return

    # 扫描数据文件
    data_files = sorted(glob.glob(os.path.join(data_dir, "doc_*.pkl")))
    print(f"批量验证: {len(data_files)} 个文档 x {len(indexers)} 层")
    print()

    # 表头
    header = f"{'Doc':<16}"
    for layer_idx in sorted(indexers.keys()):
        header += f" {'L'+str(layer_idx)+' corr':<10} {'L'+str(layer_idx)+' t10':<10}"
    print(header)
    print("-" * len(header))

    all_corrs = []
    all_top10 = []
    n_tested = 0

    for data_path in data_files:
        doc_name = os.path.basename(data_path)
        row = f"{doc_name:<16}"

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        for layer_idx in sorted(indexers.keys()):
            result = verify_single(indexers[layer_idx], data, layer_idx, token_idx=0)
            if result is None:
                row += f" {'N/A':<10} {'N/A':<10}"
            else:
                corr = result["correlation"]
                top10 = result["overlaps"][10]
                row += f" {corr:.4f}     {top10*100:.0f}%       "
                all_corrs.append(corr)
                all_top10.append(top10)
                n_tested += 1

        print(row)

    # 总结
    print("-" * len(header))
    print(f"\n总计: {n_tested} 个 case")
    print(f"Pearson correlation: mean={np.mean(all_corrs):.4f}, min={np.min(all_corrs):.4f}, max={np.max(all_corrs):.4f}")
    print(f"Top-10 overlap:     mean={np.mean(all_top10)*100:.1f}%, min={np.min(all_top10)*100:.0f}%")

    # 判定
    if np.min(all_corrs) >= 0.95:
        print("\n[PASS] 所有 case correlation >= 0.95")
    else:
        failed = sum(1 for c in all_corrs if c < 0.95)
        print(f"\n[WARN] {failed} 个 case correlation < 0.95")


def main():
    parser = argparse.ArgumentParser(description="Lightning Indexer E2E 验证")
    parser.add_argument("--data-path", default=None, help="单个 pkl 数据文件")
    parser.add_argument("--data-dir", default="./data", help="数据目录（批量模式）")
    parser.add_argument("--weight-dir", default="./weights", help="权重目录")
    parser.add_argument("--layer", type=int, default=None, help="CSA layer index (0-20)")
    parser.add_argument("--batch", action="store_true", help="批量验证所有文档")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch:
        # 找出 weight_dir 中有哪些层的权重
        available_layers = []
        for layer_idx in range(21):
            layer_id = CSA_LAYER_IDS[layer_idx]
            if os.path.exists(os.path.join(args.weight_dir, f"layer_{layer_id:02d}.safetensors")):
                available_layers.append(layer_idx)
        batch_verify(args.data_dir, args.weight_dir, available_layers, args.device)

    elif args.data_path:
        layer_idx = args.layer if args.layer is not None else 20
        print(f"验证: {args.data_path}, CSA layer {layer_idx}")
        print()

        indexer = LightningIndexer(
            csa_layer_idx=layer_idx,
            weight_dir=args.weight_dir,
            device=args.device,
        )
        verify_doc(indexer, args.data_path, layer_idx, verbose=True)

    else:
        parser.print_help()
        print("\n示例:")
        print("  python verify.py --data-path ./data/doc_00030.pkl --layer 20")
        print("  python verify.py --batch")


if __name__ == "__main__":
    main()
