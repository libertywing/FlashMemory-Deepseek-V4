"""
Lightning Indexer 离线推理模块
==============================
给定 hidden_state 和 compressed_k，输出 logits（每个 block 的相关性分数）。

用法:
  # 使用训练好的 checkpoint（推荐）
  python inference.py --data-path ./data/doc_00030.pkl --layer 10 --ckpt checkpoints/best_model.pt

  # 使用原始预训练权重
  python inference.py --data-path ./data/doc_00030.pkl --layer 20

  # 指定 top-K
  python inference.py --data-path ./data/doc_00030.pkl --layer 10 --topk 1024 --ckpt checkpoints/best_model.pt

关键参数:
  - RoPE base = 160000 (compress_rope_theta, CSA 层专用，不是普通的 10000)
  - 权重 dequant 保持 float32（匹配硬件 FP32 累加器）
  - weight_scale = HEAD_DIM^{-0.5} * N_HEADS^{-0.5}
"""

import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from utils import precompute_freqs_cis, apply_rope, hadamard_transform


# DeepSeek-V4 的 21 个 CSA layer 对应的 transformer layer ID
CSA_LAYER_IDS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]


def act_quant_fp8(x: torch.Tensor, block_size: int = 128):
    """
    块级 FP8 量化（与 sglang act_quant 等价）。

    Args:
        x:          [N, K]  float，K 必须是 block_size 的倍数
        block_size: 量化块大小（默认 128，即每个 attention head 单独量化）

    Returns:
        x_fp8:  [N, K]  float8_e4m3fn
        scale:  [N, K // block_size]  float32，每块的量化尺度
    """
    N, K = x.shape
    n_blocks = K // block_size
    x_blocks = x.view(N, n_blocks, block_size)                      # [N, n_blocks, block_size]
    scale = x_blocks.abs().amax(dim=-1).clamp(min=1e-12) / 448.0   # [N, n_blocks]  (448 = FP8 e4m3 max)
    x_q = (x_blocks / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_q.view(N, K), scale


class LightningIndexer:
    """
    Lightning Indexer 离线推理。纯 PyTorch 实现，无 sglang 依赖。

    输入：hidden state, compressed K
    输出：logits of each CSA block

    前向流程:
        hidden -> wq_a -> RMSNorm -> wq_b -> RoPE(pos, base=160000) -> Hadamard
               -> FP8 quant -> q_fp8
        hidden -> weights_proj -> per_head_w
        fused_w = per_head_w * weight_scale * q_scale

        k_fp8, k_scale = decompress(compressed_k)
        logits = ReLU(k_fp8 @ q_fp8^T) * fused_w, summed over heads, * k_scale
    """

    # 模型常量
    N_HEADS               = 64
    HEAD_DIM              = 128
    ROPE_DIM              = 64
    Q_LORA_RANK           = 1024
    HIDDEN_DIM            = 4096
    ROPE_BASE             = 160000     # compress_rope_theta（CSA 层专用，不是普通的 10000）
    ROPE_FACTOR           = 16
    ROPE_ORIGINAL_SEQ_LEN = 65536
    ROPE_BETA_FAST        = 32
    ROPE_BETA_SLOW        = 1
    RMS_NORM_EPS          = 1e-6

    def __init__(self, csa_layer_idx: int, weight_dir: str = "./weights",
                 device: str = "cuda", max_position: int = 131072,
                 ckpt_path: str = None):
        """
        Args:
            csa_layer_idx: CSA layer index (0-20)
            weight_dir:    已提取的 indexer 权重目录，包含 layer_XX.safetensors 文件
            device:        "cuda" 或 "cpu"
            max_position:  预计算 RoPE 的最大位置（默认 131072）
            ckpt_path:     训练好的 checkpoint 路径（.pt 文件），优先于 weight_dir
        """
        self.device = device
        self.csa_layer_idx = csa_layer_idx

        # weight_scale = HEAD_DIM^{-0.5} * N_HEADS^{-0.5}
        self.weight_scale = self.HEAD_DIM ** -0.5 * self.N_HEADS ** -0.5

        # 加载权重
        if ckpt_path:
            self._load_checkpoint(ckpt_path)
        else:
            self._load_weights(weight_dir)

        # 预计算 RoPE 频率
        self.freqs_cis = precompute_freqs_cis(
            dim=self.ROPE_DIM,
            seqlen=max_position,
            original_seq_len=self.ROPE_ORIGINAL_SEQ_LEN,
            base=self.ROPE_BASE,
            factor=self.ROPE_FACTOR,
            beta_fast=self.ROPE_BETA_FAST,
            beta_slow=self.ROPE_BETA_SLOW,
        ).to(device)

    def _load_weights(self, weight_dir: str) -> None:
        """从已提取的 dequant float32 权重文件加载（原始预训练权重）。"""
        layer_id    = CSA_LAYER_IDS[self.csa_layer_idx]
        weight_file = os.path.join(weight_dir, f"layer_{layer_id:02d}.safetensors")
        assert os.path.exists(weight_file), f"权重文件不存在: {weight_file}"

        with safe_open(weight_file, framework="pt") as sf:
            self.wq_a         = sf.get_tensor("wq_a.weight").to(torch.float32).to(self.device)
            self.wq_b         = sf.get_tensor("indexer.wq_b.weight").to(torch.float32).to(self.device)
            self.q_norm_weight = sf.get_tensor("q_norm.weight").to(torch.float32).to(self.device)
            self.w_proj       = sf.get_tensor("indexer.weights_proj.weight").to(torch.float32).to(self.device)

    def _load_checkpoint(self, ckpt_path: str) -> None:
        """从训练好的 checkpoint (.pt) 加载微调后的权重。

        支持两种格式：
          1. Single-layer (legacy): keys = wq_a.weight / wq_b.weight / q_norm_weight / weights_proj.weight
          2. Joint multi-layer (R13+): keys = retrievers.l{lid}.{wq_a,wq_b,q_norm_weight,weights_proj}.weight
             — 自动按 self.csa_layer_idx 提取对应 layer 的 sub-state
        """
        assert os.path.exists(ckpt_path), f"Checkpoint 不存在: {ckpt_path}"
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        # Detect joint format (any key starts with "retrievers.l")
        is_joint = any(k.startswith("retrievers.l") for k in state.keys())
        if is_joint:
            # csa_layer_idx is the index INTO CSA_LAYER_IDS list, not the layer ID itself.
            # Map back: idx 4 → layer 10, idx 5 → layer 12, idx 9 → layer 20.
            layer_id = CSA_LAYER_IDS[self.csa_layer_idx]
            prefix   = f"retrievers.l{layer_id}."
            sub      = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            assert sub, (
                f"Joint ckpt has no keys with prefix '{prefix}'. "
                f"Available retrievers in ckpt: "
                f"{sorted({k.split('.')[1] for k in state if k.startswith('retrievers.')})}"
            )
            print(f"  [joint ckpt] extracted layer {layer_id} sub-state ({len(sub)} keys)")
            state = sub

        self.wq_a          = state["wq_a.weight"].to(torch.float32).to(self.device)
        self.wq_b          = state["wq_b.weight"].to(torch.float32).to(self.device)
        self.q_norm_weight = state["q_norm_weight"].to(torch.float32).to(self.device)
        self.w_proj        = state["weights_proj.weight"].to(torch.float32).to(self.device)
        self._from_checkpoint = True

        # ── 自动 infer 架构参数 (支持 N_HEADS=128, Q_LORA_RANK=2048 等变体) ──
        inferred_q_rank = self.wq_a.shape[0]                       # [Q, 4096]
        inferred_heads  = self.wq_b.shape[0] // self.HEAD_DIM      # [N*128, Q]
        if inferred_heads != self.N_HEADS or inferred_q_rank != self.Q_LORA_RANK:
            print(f"  [arch override] N_HEADS: {self.N_HEADS} → {inferred_heads}, "
                  f"Q_LORA_RANK: {self.Q_LORA_RANK} → {inferred_q_rank}")
            self.N_HEADS      = inferred_heads
            self.Q_LORA_RANK  = inferred_q_rank
            # weight_scale 依赖 N_HEADS,需要重算
            self.weight_scale = self.HEAD_DIM ** -0.5 * self.N_HEADS ** -0.5

        # 验证形状一致性
        assert self.q_norm_weight.shape[0] == self.Q_LORA_RANK, (
            f"q_norm_weight rank {self.q_norm_weight.shape[0]} != wq_a rank {self.Q_LORA_RANK}")
        assert self.w_proj.shape[0] == self.N_HEADS, (
            f"weights_proj heads {self.w_proj.shape[0]} != wq_b heads {self.N_HEADS}")

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        x_f  = x.float()
        norm = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.RMS_NORM_EPS)
        return x_f / norm * self.q_norm_weight

    def forward(
        self,
        hidden_state: torch.Tensor,   # [batch, 4096]  bf16/float32
        compressed_k: torch.Tensor,   # [n_blocks, 132] uint8
        positions:    torch.Tensor,   # [batch]  int64
    ) -> torch.Tensor:
        """
        Returns:
            logits: [batch, n_blocks] float32, 每个 block 的相关性分数
                    从 checkpoint 加载时不带 FP8 量化（匹配训练），返回 raw logits
                    从 safetensors 加载时带 FP8 量化（匹配原始推理路径）
        """
        if compressed_k.dim() == 3:
            compressed_k = compressed_k.squeeze(1)

        # ── 解码 compressed K ──────────────────────────────────────────────
        # [n_blocks, 132] → fp8 values [n_blocks, 128]  +  scale [n_blocks]
        k_fp8   = compressed_k[:, :128].contiguous().view(torch.float8_e4m3fn).float().to(self.device)
        k_scale = compressed_k[:, 128:132].contiguous().view(torch.float32).squeeze(-1).to(self.device)

        # ── Query 路径 ─────────────────────────────────────────────────────
        B = hidden_state.shape[0]
        x = hidden_state.to(torch.float32).to(self.device)        # [B, 4096]

        q_lora = F.linear(x, self.wq_a)                           # [B, 1024]
        q_lora = self._rmsnorm(q_lora)                             # [B, 1024]
        q      = F.linear(q_lora, self.wq_b)                      # [B, 8192]
        q      = q.view(B, self.N_HEADS, self.HEAD_DIM)            # [B, 64, 128]

        q = apply_rope(q.to(torch.bfloat16),
                       self.freqs_cis,
                       positions.to(torch.int64).to(self.device),
                       self.ROPE_DIM).float()                      # [B, 64, 128]
        q = hadamard_transform(q)                                  # [B, 64, 128]

        # ── Weight 路径 ────────────────────────────────────────────────────
        per_head_w = F.linear(x, self.w_proj)                      # [B, 64]

        if getattr(self, '_from_checkpoint', False):
            # 训练模式：不做 FP8 量化，匹配 LightningIndexerTrainable.forward
            # k 已 dequant: k_fp8 * k_scale → k_float
            k_float = k_fp8 * k_scale.unsqueeze(-1)                # [n_blocks, 128]
            fused_w = per_head_w * self.weight_scale               # [B, 64]

            # scores_per_head = relu(k @ q^T)
            scores = F.relu(torch.einsum("bhd,nd->bnh", q, k_float))  # [B, n_blocks, 64]
            logits = (scores * fused_w.unsqueeze(1)).sum(-1)           # [B, n_blocks]
        else:
            # 原始推理模式：带 FP8 量化
            q_flat         = q.view(B, self.N_HEADS * self.HEAD_DIM)   # [B, 8192]
            q_fp8, q_scale = act_quant_fp8(q_flat)                     # [B, 8192], [B, 64]
            q_fp8          = q_fp8.view(B, self.N_HEADS, self.HEAD_DIM).float()
            fused_w        = per_head_w * self.weight_scale * q_scale   # [B, 64]

            scores = F.relu(torch.einsum("bhd,nd->bnh", q_fp8, k_fp8))  # [B, n_blocks, 64]
            logits = (scores * fused_w.unsqueeze(1)).sum(-1)             # [B, n_blocks]
            logits = logits * k_scale.unsqueeze(0)                       # [B, n_blocks]

        return logits


# ── 统计：第一个 decode token 召回的「prompt 早段」chunk 占比 ──────────────────────

def early_prompt_recall_stat(
    logits_first_token: torch.Tensor,   # [n_blocks] 第一个 decode token 的 logits
    n_prompt: int,                       # prompt 总 chunk 数（= 第一个 decode token 可见的 block 数）
    topk: int,
    tail_chunks: int = 2000,
) -> dict:
    """
    计算第一个 decode token 召回的 prompt chunk 里，落在「prompt 后 tail_chunks 个 chunk 之前」
    （即 chunk index < n_prompt - tail_chunks）的数量，再除以 prompt 总 chunk 数。

    Args:
        logits_first_token: [n_blocks] float, forward() 输出的第 0 行
        n_prompt:           prompt 总 chunk 数（block 按顺序排列，prompt 在前）
        topk:               top-K 召回数量
        tail_chunks:        prompt 末尾保留区的 chunk 数（默认 2000，约对应后 8k token）

    Returns:
        dict: n_prompt / n_recalled_prompt / boundary / n_before_tail / ratio
    """
    # 只在 prompt 范围内取 top-K（第一个 decode token 只能看到 prompt blocks）
    prompt_logits = logits_first_token[:n_prompt]
    k = min(topk, n_prompt)
    recalled = prompt_logits.topk(k, dim=0)[1]            # [k] prompt chunk indices

    boundary = max(n_prompt - tail_chunks, 0)             # 后 tail_chunks 之前的分界点
    n_before = int((recalled < boundary).sum().item())    # 召回的、落在早段的 chunk 数
    ratio    = n_before / n_prompt if n_prompt > 0 else 0.0

    return {
        "n_prompt":          n_prompt,
        "n_recalled_prompt": k,
        "boundary":          boundary,
        "n_before_tail":     n_before,
        "ratio":             ratio,
    }


# ── 命令行入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import pickle
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Lightning Indexer 推理")
    parser.add_argument("--weight-dir", default="./weights", help="原始权重目录")
    parser.add_argument("--ckpt",       default=None,        help="训练好的 checkpoint (.pt 文件)")
    parser.add_argument("--data-path",  default=None,        help="输入数据 pkl 文件（单题）")
    parser.add_argument("--data-dir",   default=None,        help="输入数据目录（批量，处理目录下所有 *.pkl）")
    parser.add_argument("--layer",      type=int, required=True, help="CSA layer index (0-20)")
    parser.add_argument("--topk",       type=int, default=1024,  help="输出 top-K 个 block indices")
    parser.add_argument("--tail-chunks", type=int, default=2000,
                        help="prompt 末尾保留区 chunk 数（默认 2000 ≈ 后 8k token）")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    # ── 收集待处理的 pkl 文件 ──────────────────────────────────────────────────
    assert args.data_path or args.data_dir, "必须指定 --data-path 或 --data-dir"
    if args.data_dir:
        pkl_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pkl")))
        assert pkl_files, f"目录下没有 *.pkl: {args.data_dir}"
    else:
        pkl_files = [args.data_path]
    single = len(pkl_files) == 1

    print(f"Lightning Indexer 推理")
    print(f"  权重: {args.ckpt or args.weight_dir}")
    print(f"  CSA layer: {args.layer} (transformer layer {CSA_LAYER_IDS[args.layer]})")
    print(f"  数据: {args.data_dir or args.data_path}  ({len(pkl_files)} 题)")
    print(f"  top-K: {args.topk},  tail-chunks: {args.tail_chunks}")

    hkey, ckey, pkey = (f"hidden_layer_{args.layer}",
                        f"compk_layer_{args.layer}",
                        f"positions_layer_{args.layer}")
    stored_key = f"logits_layer_{args.layer}_token_0"

    # ── 预扫描全局最大 position，一次性构建 indexer（freqs_cis 预计算到足够长）──────
    global_max_pos = 0
    for fp in pkl_files:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        global_max_pos = max(global_max_pos, int(d[pkey].max().item()))

    indexer = LightningIndexer(
        csa_layer_idx=args.layer,
        weight_dir=args.weight_dir,
        device=args.device,
        max_position=global_max_pos + 1,
        ckpt_path=args.ckpt,
    )

    # ── 逐题推理 + 统计 ────────────────────────────────────────────────────────
    ratios, skipped = [], 0
    for fp in pkl_files:
        with open(fp, "rb") as f:
            data = pickle.load(f)

        hidden_state = data[hkey]
        compressed_k = data[ckey]
        positions    = data[pkey]

        # 只需第一个 decode token
        logits = indexer.forward(hidden_state[:1], compressed_k, positions[:1])  # [1, n_blocks]
        logits_first = logits[0].cpu()

        # prompt 总 chunk 数：第一个 decode token 可见的 block 数 = stored logits 长度
        if stored_key in data:
            n_prompt = len(data[stored_key])
        else:
            n_prompt = logits_first.shape[0]   # 回退：把全部 block 当 prompt（含 decode 段，偏大）

        stat = early_prompt_recall_stat(logits_first, n_prompt, args.topk, args.tail_chunks)
        ratios.append(stat["ratio"])

        if single:
            topk_values, topk_indices = logits.topk(args.topk, dim=1)
            print(f"\n  hidden_state: {hidden_state.shape}")
            print(f"  compressed_k: {compressed_k.shape}")
            print(f"  positions range=[{positions.min()}, {positions.max()}]")
            print(f"  logits shape: {logits.shape}, range=[{logits.min():.4f}, {logits.max():.4f}]")
            print(f"\n  Top-{args.topk} (第 1 个 token): {topk_indices[0, :10].tolist()} ...")

            # 如果有 stored logits，做对比验证
            if stored_key in data:
                stored   = data[stored_key].float()
                computed = logits_first[:len(stored)]
                corr     = np.corrcoef(stored.numpy(), computed.numpy())[0, 1]
                k_actual = min(args.topk, len(stored))
                overlap  = len(set(stored.topk(k_actual)[1].tolist())
                               & set(computed.topk(k_actual)[1].tolist()))
                print(f"\n  验证 (vs stored logits):")
                print(f"    Top-{k_actual} overlap: {overlap}/{k_actual} ({overlap/k_actual*100:.1f}%)")
                print(f"    Pearson correlation: {corr:.4f}")

            print(f"\n  ── 早段召回统计 (第 1 个 decode token) ──")
            print(f"    prompt 总 chunk 数 n_prompt          : {stat['n_prompt']}")
            print(f"    后段保留区分界 (n_prompt - {args.tail_chunks})    : index < {stat['boundary']}")
            print(f"    召回的 prompt chunk 数 (top-{args.topk})    : {stat['n_recalled_prompt']}")
            print(f"    其中落在早段(后2k之前)的 chunk 数     : {stat['n_before_tail']}")
            print(f"    ratio = 早段召回数 / prompt 总 chunk 数 : {stat['ratio']:.6f}")
        else:
            print(f"  [{os.path.basename(fp)}] n_prompt={stat['n_prompt']:6d}  "
                  f"recalled={stat['n_recalled_prompt']:5d}  "
                  f"before_tail={stat['n_before_tail']:5d}  ratio={stat['ratio']:.6f}")

    # ── 汇总平均 ──────────────────────────────────────────────────────────────
    if ratios:
        mean_ratio = sum(ratios) / len(ratios)
        print(f"\n{'='*70}")
        print(f"  平均 ratio (早段召回数 / prompt 总 chunk 数), 共 {len(ratios)} 题")
        print(f"    mean = {mean_ratio:.6f}")
        print(f"    min  = {min(ratios):.6f}   max = {max(ratios):.6f}")
        print(f"{'='*70}")
