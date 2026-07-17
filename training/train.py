"""
train.py — Lightning Indexer 训练脚本
======================================
模型架构与 inference.py 的 LightningIndexer 一致，但有以下 3 处修改：

  1. 改为 nn.Module，权重可训练
  2. 跳过 FP8 量化（act_quant 不可微），用 float32 直接做矩阵乘
  3. 评分流程：
       sigmoid(k @ q.T)  →  加权求和  →  sigmoid  →  BCE loss
     （原来 relu → 无后 sigmoid → 无 BCE）

用法：
  # 单层训练
  python train.py --layer 10 --data-config combined_full --weight-dir ./weights \
      --epochs 30 --batch-size 512 --lr 5e-5 --neg-ratio 3 --weighted-loss \
      --val-fullset --num-workers 0

  # 8-layer joint 训练（R950 系列）
  python train.py --joint-layers 6,8,10,12,14,16,18,20 \
      --data-config combined_8layers --weight-dir ./weights \
      --epochs 5 --batch-size 256 --lr 1e-4 --neg-ratio 3 --focal-loss \
      --val-fullset --val-every-steps 1000 --patience-vals 3 \
      --num-workers 0 --output-dir experiments/<exp_name>/ckpts

数据管理：
  - DATA_CONFIGS 字典管理各数据配置的 train/val/test doc index
  - 新增数据时只需在 DATA_CONFIGS 的 specs 中追加 {"data_dir": ..., "doc_ids": [...]}
"""

import os
import glob as _glob
import pickle as _pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from safetensors import safe_open

from torch.utils.data import DataLoader
from dataloader import (
    build_dataloader,
    build_combined_dataloader,
    build_joint_combined_dataloader,
    LAYER_EMBED_MAP,
)
from utils import precompute_freqs_cis, apply_rope, hadamard_transform

# CSA_LAYER_IDS inlined to avoid importing inference.py (keeps train.py self-contained)
CSA_LAYER_IDS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]


# ── 可训练模型 ───────────────────────────────────────────────────────────────

class LightningIndexerTrainable(nn.Module):
    """
    与 LightningIndexer 相同的架构，改为 nn.Module 并支持训练。

    主要差异（对比 inference.py）：
      - wq_a / wq_b / weights_proj 改为 nn.Linear（可训练）
      - q_norm_weight 改为 nn.Parameter
      - 不做 FP8 量化（act_quant），fused_w 去掉 q_scale 项
      - per-head 打分用 sigmoid 替换 relu
      - 最终 score_per_block 再过一次 sigmoid，输出 [0,1]
      - n_heads 可配置（默认 64，可设为 128 等）
    """

    # Fixed architecture constants (class-level)
    HEAD_DIM             = 128
    ROPE_DIM             = 64
    Q_LORA_RANK_DEFAULT  = 1024   # 预训练权重的 rank；可通过 __init__ 覆盖
    HIDDEN_DIM           = 4096
    ROPE_BASE            = 160000
    ROPE_FACTOR          = 16
    ROPE_ORIGINAL_SEQ_LEN = 65536
    ROPE_BETA_FAST       = 32
    ROPE_BETA_SLOW       = 1
    RMS_NORM_EPS         = 1e-6

    def __init__(
        self,
        csa_layer_idx: int,
        max_position: int = 131072,
        n_heads: int = 64,
        q_lora_rank: int = None,     # 可调整 Q LoRA rank (默认 1024 = 预训练 rank)
    ):
        super().__init__()
        self.csa_layer_idx = csa_layer_idx
        self.N_HEADS       = n_heads                                          # instance var
        self.Q_LORA_RANK   = q_lora_rank if q_lora_rank is not None else self.Q_LORA_RANK_DEFAULT
        self.weight_scale  = self.HEAD_DIM ** -0.5 * self.N_HEADS ** -0.5

        # ── 可训练参数 ──────────────────────────────────────────────────────
        self.wq_a        = nn.Linear(self.HIDDEN_DIM,  self.Q_LORA_RANK,               bias=False)
        self.wq_b        = nn.Linear(self.Q_LORA_RANK, self.N_HEADS * self.HEAD_DIM,   bias=False)
        self.q_norm_weight = nn.Parameter(torch.ones(self.Q_LORA_RANK))
        self.weights_proj  = nn.Linear(self.HIDDEN_DIM, self.N_HEADS,                  bias=False)

        # ── RoPE 频率（非参数，仅 buffer）──────────────────────────────────
        freqs = precompute_freqs_cis(
            dim=self.ROPE_DIM,
            seqlen=max_position,
            original_seq_len=self.ROPE_ORIGINAL_SEQ_LEN,
            base=self.ROPE_BASE,
            factor=self.ROPE_FACTOR,
            beta_fast=self.ROPE_BETA_FAST,
            beta_slow=self.ROPE_BETA_SLOW,
        )
        self.register_buffer("freqs_cis", freqs)   # [max_position, rope_dim//2] complex

    # ── 初始化 ──────────────────────────────────────────────────────────────

    def load_pretrained(self, weight_dir: str) -> None:
        """从 .safetensors 文件加载预训练权重（用于 fine-tuning）。
        支持 N_HEADS > 64: 前 64 head 从 pretrained 加载, 后续 head 保持 kaiming_uniform 随机初始化.
        支持 Q_LORA_RANK > 1024: 前 1024 维从 pretrained 加载, 其余维度保持随机/默认初始化.
        """
        layer_id    = CSA_LAYER_IDS[self.csa_layer_idx]
        weight_file = os.path.join(weight_dir, f"layer_{layer_id:02d}.safetensors")
        assert os.path.exists(weight_file), f"权重文件不存在: {weight_file}"

        device = self.wq_a.weight.device   # 保持与模型一致的 device
        with safe_open(weight_file, framework="pt") as sf:
            wq_a_pre = sf.get_tensor("wq_a.weight").to(torch.float32).to(device)             # [1024, 4096]
            qn_pre   = sf.get_tensor("q_norm.weight").to(torch.float32).to(device)           # [1024]
            wq_b_pre = sf.get_tensor("indexer.wq_b.weight").to(torch.float32).to(device)        # [64*128, 1024] = [8192, 1024]
            wp_pre   = sf.get_tensor("indexer.weights_proj.weight").to(torch.float32).to(device) # [64, 4096]

        # ── wq_a / q_norm: rank 维度 ─────────────────────────────────────
        if self.Q_LORA_RANK == 1024:
            self.wq_a.weight.data    = wq_a_pre
            self.q_norm_weight.data  = qn_pre
        elif self.Q_LORA_RANK > 1024:
            # 前 1024 维加载 pretrained，剩余维度保持随机初始化
            self.wq_a.weight.data[:1024, :]  = wq_a_pre
            self.q_norm_weight.data[:1024]   = qn_pre
            # 剩余 q_norm 维度已在 __init__ 中初始化为 ones (默认 RMSNorm 中性值)
            print(f"  Q_LORA_RANK={self.Q_LORA_RANK}: loaded pretrained 1024 dims, "
                  f"kept {self.Q_LORA_RANK-1024} new dims at random/ones init")
        else:
            raise ValueError(
                f"Q_LORA_RANK={self.Q_LORA_RANK} < 1024 not supported (pretrained has rank 1024)"
            )

        # ── wq_b / weights_proj: head 维度 ────────────────────────────────
        if self.N_HEADS == 64 and self.Q_LORA_RANK == 1024:
            self.wq_b.weight.data         = wq_b_pre
            self.weights_proj.weight.data = wp_pre
        else:
            # 任意一边升维: 前 64 head × 前 1024 rank 加载 pretrained, 其余随机
            n_pretrained_rows = 64 * self.HEAD_DIM    # wq_b: 前 8192 行
            self.wq_b.weight.data[:n_pretrained_rows, :1024] = wq_b_pre
            self.weights_proj.weight.data[:64, :]            = wp_pre
            if self.N_HEADS > 64:
                print(f"  N_HEADS={self.N_HEADS}: loaded pretrained 64 heads, "
                      f"kept {self.N_HEADS-64} new heads at random init")

        print(f"Loaded pretrained weights from {weight_file}")

    # ── 辅助 ────────────────────────────────────────────────────────────────

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        x_f  = x.float()
        norm = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.RMS_NORM_EPS)
        return (x_f / norm) * self.q_norm_weight

    # ── 前向 ────────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_state:    torch.Tensor,              # [B, 4096]  bf16 / float32
        selected_compk:  torch.Tensor,              # [B, N, 132] uint8  (N = 2*n_pos, 含 padding)
        positions:       torch.Tensor,              # [B]  int64
        layer_embed_idx: torch.Tensor = None,       # unused, kept for call-site compat
        return_logits:   bool = False,              # True → 返回 raw logits (pairwise loss 需要)
    ) -> torch.Tensor:
        """
        Returns:
            If return_logits=False: scores [B, N] ∈ [0, 1] (sigmoid output, for BCE)
            If return_logits=True:  logits [B, N] ∈ (-∞, +∞) (raw, for pairwise loss)
            (padding 位置的分数无意义，训练时由 mask 过滤)
        """
        B = hidden_state.shape[0]
        x = hidden_state.to(torch.float32)   # [B, 4096]

        # ── Query 路径 ───────────────────────────────────────────────────
        q_lora = self.wq_a(x)                                            # [B, 1024]
        q_lora = self._rmsnorm(q_lora)                                   # [B, 1024]
        q      = self.wq_b(q_lora)                                       # [B, 8192]
        q      = q.view(B, self.N_HEADS, self.HEAD_DIM)                  # [B, 64, 128]
        q      = apply_rope(q, self.freqs_cis, positions, self.ROPE_DIM) # [B, 64, 128]
        q      = hadamard_transform(q)                                   # [B, 64, 128]
        # 训练时不做 FP8 量化（act_quant 不可微），直接用 float32

        # ── Weight 路径（无 q_scale，因为跳过了 FP8）────────────────────
        per_head_w = self.weights_proj(x)               # [B, 64]
        fused_w    = per_head_w * self.weight_scale      # [B, 64]

        # ── 解码 compressed K 并反量化 ─────────────────────────────────
        # selected_compk: [B, N, 132] uint8
        #   前 128 bytes → FP8 values，后 4 bytes → float32 scale
        k_fp8  = selected_compk[..., :128].contiguous().view(
            torch.float8_e4m3fn).float()                # [B, N, 128]
        k_scale = selected_compk[..., 128:132].contiguous().view(
            torch.float32).squeeze(-1)                  # [B, N]

        # 关键：反量化 k（k_fp8 * k_scale），与 inference 的 relu(k_fp8@q_fp8)*k_scale 等价
        k_dequant = k_fp8 * k_scale.unsqueeze(-1)       # [B, N, 128]  实际 float key

        # ── 评分（全向量化，对齐 inference 的 relu 激活）────────────────
        scores_per_head = F.relu(
            torch.einsum("bnd,bhd->bnh", k_dequant, q.float())
        )                                               # [B, N, 64]

        # 加权求和 → [B, N]，sigmoid 映射到 [0,1] 用于 BCE loss
        score  = (scores_per_head * fused_w.unsqueeze(1)).sum(dim=-1)  # [B, N]

        if return_logits:
            return score                                   # [B, N] raw logits
        return torch.sigmoid(score)                        # [B, N] ∈ [0,1]


# ── 数据配置 —————————————————————————————————————————————————————————————————
# 每个配置包含 train/val/test 的 specs 列表，每个 spec 格式：
#   {"data_dir": str, "doc_ids": list or None}
# doc_ids 使用与文件名对应的整数编号（0-based 或 1-based 均可）。
# 新增数据时只需在对应配置的 specs 列表中追加一个 {"data_dir": ..., "doc_ids": [...]}。
# ─────────────────────────────────────────────────────────────────────────────

# 旧数据目录（1-based 文档编号，doc_00001.pkl ... doc_00101.pkl）
_OLD_DATA_DIR = "./data"

# 新数据根目录 — 本地盘副本（/dockerdata，读取 5 GB/s vs CephFS 17 MB/s）
# 原始路径: /apdcephfs_fsgm/share_303846923/user/qifanzhang/long_context_project/ds_compress/offline_retriever_data
# 本地副本: /dockerdata/retriever_data/
_NEW_BASE = "/dockerdata/retriever_data"

# ── 各新数据集 doc index 分片（0-based 文档编号）─────────────────────────────
# single_turn_mrcr: 80 docs (doc_00000 – doc_00079)
_MRCR_TRAIN = list(range(0, 56))    # 56 docs (~70%)
_MRCR_VAL   = list(range(56, 68))   # 12 docs (~15%)
_MRCR_TEST  = list(range(68, 80))   # 12 docs (~15%)
_MRCR_VAL_SMALL = [56, 57]          # 2 docs (1/5 of val, 加速评测)

# niah_last8k_64_128: 100 docs (doc_00000 – doc_00099)
_NIAH1_TRAIN = list(range(0, 70))   # 70 docs (~70%)
_NIAH1_VAL   = list(range(70, 85))  # 15 docs (~15%)
_NIAH1_TEST  = list(range(85, 100)) # 15 docs (~15%)
_NIAH1_VAL_SMALL = [70, 71, 72]     # 3 docs (1/5 of val)

# niah_64k-128k: 100 docs (doc_00000 – doc_00099)
_NIAH2_TRAIN = list(range(0, 70))
_NIAH2_VAL   = list(range(70, 85))
_NIAH2_TEST  = list(range(85, 100))
_NIAH2_VAL_SMALL = [70, 71, 72]     # 3 docs (1/5 of val)

# creative_writing: 100 docs (doc_00000 – doc_00099)
_CW_TRAIN = list(range(0, 70))
_CW_VAL   = list(range(70, 85))
_CW_TEST  = list(range(85, 100))
_CW_VAL_SMALL = [70, 71, 72]        # 3 docs (1/5 of val)

# ── 旧数据 doc index 分片（1-based 文档编号）─────────────────────────────────
_OLD_TRAIN = list(range(1, 71))     # docs 001–070  (70 docs, ~69%)
_OLD_VAL   = list(range(71, 86))    # docs 071–085  (15 docs, ~15%)
_OLD_TEST  = list(range(86, 102))   # docs 086–101  (16 docs, ~16%)
_OLD_VAL_SMALL = [71, 72, 73]       # 3 docs (1/5 of val)

# ── under128 新增大量数据（全部用于训练，不分 val/test）──────────────────────
_UNDER128_BASE = f"{_NEW_BASE}/under128"
_U128_NIAH_ALL     = list(range(0, 2800))    # 2800 docs — ALL for training
_U128_SINGLE_ALL   = list(range(0, 1473))    # 1473 docs — ALL for training

# ── R12 v2 数据（unfiltered，含噪声标签）─────────────────────────────────────
# 路径: /dockerdata/retriever_data_v2_backup/{niah, single_turn, creative_writing}
# 数据规模: niah 2780 + single_turn 2321 + creative_writing 1604 = 6705 docs
# 划分: 训练 80%, val 10%, test 10%（按 doc_id 顺序，固定划分保证可重复）
_V2_UNFILTERED_BASE = "/dockerdata/retriever_data_v2_backup"   # labels 未过滤 (含原始噪声标签)
_V2_NIAH_TRAIN  = list(range(0, 2224))      # 80% (2224 docs)
_V2_NIAH_VAL    = list(range(2224, 2502))   # 10% (278)
_V2_NIAH_TEST   = list(range(2502, 2780))   # 10% (278)
_V2_ST_TRAIN    = list(range(0, 1857))      # 80% (1857)
_V2_ST_VAL      = list(range(1857, 2089))   # 10% (232)
_V2_ST_TEST     = list(range(2089, 2321))   # 10% (232)
_V2_CW_TRAIN    = list(range(0, 1283))      # 80% (1283)
_V2_CW_VAL      = list(range(1283, 1444))   # 10% (161)
_V2_CW_TEST     = list(range(1444, 1604))   # 10% (160)

# ── R15 8-layer 新数据 (2026-05-30 更新) ─────────────────────────────────────
# 路径: /dockerdata/retriever_data_8layers/{niah, longbench, creative_writing, single_turn}
# 包含 8 个 CSA layer (6, 8, 10, 12, 14, 16, 18, 20) hidden + compk;
# 标签格式: label_indices/label_scores/label_pointers (sparse 三元组, 与 v2_unfiltered 一致, 噪声未清洗)
# 新增 longbench 数据集 (1765 docs) — 此前没有的子集
# 注意: 本地 creative_writing/ 是源 creative_writing_backup/ 的复制 (2195 docs);
# 同事正在源端把 creative_writing 重组为 copyrightprotected + publicdomain, 我们用 backup 稳定版
_8L_BASE      = "/dockerdata/retriever_data_8layers"
_8L_NIAH_TRAIN = list(range(0, 1300))   # niah: 1625 docs (80/10/10)
_8L_NIAH_VAL   = list(range(1300, 1462))
_8L_NIAH_TEST  = list(range(1462, 1625))
_8L_LB_TRAIN   = list(range(0, 1412))   # longbench: 1765 docs
_8L_LB_VAL     = list(range(1412, 1588))
_8L_LB_TEST    = list(range(1588, 1765))
_8L_CW_TRAIN   = list(range(0, 1756))   # creative_writing (= backup): 2195 docs
_8L_CW_VAL     = list(range(1756, 1975))
_8L_CW_TEST    = list(range(1975, 2195))
_8L_ST_TRAIN   = list(range(0, 1771))   # single_turn: 2214 docs (idx 0–2243, 有空缺, loader 跳过)
_8L_ST_VAL     = list(range(1771, 1995))
_8L_ST_TEST    = list(range(1995, 2244))

# ── 全量 10 子目录新增的 6 个 (2026-06-04) — 每个 80/10/10 ───────────────────
# 新增: copyrightprotected, publicdomain, NovelQA, multi_doc_long_qa_{32k_128k,128k_256k,256k_512k}
# 全部含 hidden/compk/positions for L6..L20 + sparse labels (与上面 4 个同格式)。
# 切分用 max_idx+1 区间 (部分子目录文件编号有空缺, loader 按 doc_{i:05d}.pkl 白名单匹配, 缺失自动跳过)。
_8L_CP_TRAIN   = list(range(0, 446))    # copyrightprotected: 557 docs (idx 0–556)
_8L_CP_VAL     = list(range(446, 502))
_8L_CP_TEST    = list(range(502, 557))
_8L_PD_TRAIN   = list(range(0, 580))    # publicdomain: 724 docs (idx 0–723)
_8L_PD_VAL     = list(range(580, 652))
_8L_PD_TEST    = list(range(652, 724))
_8L_NV_TRAIN   = list(range(0, 320))    # NovelQA: 400 docs (idx 0–399)
_8L_NV_VAL     = list(range(320, 360))
_8L_NV_TEST    = list(range(360, 400))
_8L_MDA_TRAIN  = list(range(0, 2240))   # multi_doc_long_qa_32k_128k: 2800 docs (idx 0–2799)
_8L_MDA_VAL    = list(range(2240, 2520))
_8L_MDA_TEST   = list(range(2520, 2800))
_8L_MDB_TRAIN  = list(range(0, 402))    # multi_doc_long_qa_128k_256k: 500 docs (idx 0–502, 有空缺)
_8L_MDB_VAL    = list(range(402, 452))
_8L_MDB_TEST   = list(range(452, 503))
_8L_MDC_TRAIN  = list(range(0, 363))    # multi_doc_long_qa_256k_512k: 454 docs (idx 0–453)
_8L_MDC_VAL    = list(range(363, 408))
_8L_MDC_TEST   = list(range(408, 454))

DATA_CONFIGS = {
    # ── smoke: 冒烟/示例配置 —— 指向 data_generation 生成的目录 (改成你自己的绝对路径)。
    #    doc_ids 用 doc_XXXXX.pkl 里的整数编号。示例: train 用 doc 1,2, val 用 doc 1。
    "smoke": {
        "train": [{"data_dir": "/ABSOLUTE/path/to/generated_data", "doc_ids": [1, 2]}],
        "val":   [{"data_dir": "/ABSOLUTE/path/to/generated_data", "doc_ids": [1]}],
        "val_small": [{"data_dir": "/ABSOLUTE/path/to/generated_data", "doc_ids": [1]}],
        "test":  [{"data_dir": "/ABSOLUTE/path/to/generated_data", "doc_ids": [1]}],
    },
    # ── legacy: 旧数据单目录（向后兼容）─────────────────────────────────────
    # 等价于旧版 --data-dir ./data --split train/val/test
    "legacy": {
        "train": [{"data_dir": _OLD_DATA_DIR, "doc_ids": _OLD_TRAIN}],
        "val":   [{"data_dir": _OLD_DATA_DIR, "doc_ids": _OLD_VAL}],
        "val_small": [{"data_dir": _OLD_DATA_DIR, "doc_ids": _OLD_VAL_SMALL}],
        "test":  [{"data_dir": _OLD_DATA_DIR, "doc_ids": _OLD_TEST}],
    },

    # ── new_only: 仅使用 4 个新数据集──────────────────────────────────────
    "new_only": {
        "train": [
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_TRAIN},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_TRAIN},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_TRAIN},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_TRAIN},
        ],
        "val": [
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_VAL},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_VAL},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_VAL},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_VAL},
        ],
        "val_small": [
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_VAL_SMALL},
        ],
        "test": [
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_TEST},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_TEST},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_TEST},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_TEST},
        ],
    },

    # ── combined: 旧数据 + 全部新数据──────────────────────────────────────
    "combined": {
        "train": [
            {"data_dir": _OLD_DATA_DIR,                      "doc_ids": _OLD_TRAIN},
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_TRAIN},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_TRAIN},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_TRAIN},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_TRAIN},
        ],
        "val": [
            {"data_dir": _OLD_DATA_DIR,                      "doc_ids": _OLD_VAL},
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_VAL},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_VAL},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_VAL},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_VAL},
        ],
        "val_small": [
            {"data_dir": _OLD_DATA_DIR,                      "doc_ids": _OLD_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_VAL_SMALL},
        ],
        "test": [
            {"data_dir": _OLD_DATA_DIR,                      "doc_ids": _OLD_TEST},
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_TEST},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_TEST},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_TEST},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_TEST},
        ],
    },

    # ── combined_full: combined + under128 大量新数据（训练 13.8×）────────────
    # Val/Test 保持不变（与 combined 相同），新增数据只进入训练集
    "combined_full": {
        "train": [
            {"data_dir": _OLD_DATA_DIR,                      "doc_ids": _OLD_TRAIN},
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_TRAIN},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_TRAIN},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_TRAIN},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_TRAIN},
            # ── under128 新增数据 (4273 docs) ──────────────────────────────
            {"data_dir": f"{_UNDER128_BASE}/niah",           "doc_ids": _U128_NIAH_ALL},
            {"data_dir": f"{_UNDER128_BASE}/single_turn",    "doc_ids": _U128_SINGLE_ALL},
        ],
        "val": [
            {"data_dir": _OLD_DATA_DIR,                      "doc_ids": _OLD_VAL},
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_VAL},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_VAL},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_VAL},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_VAL},
        ],
        "val_small": [
            {"data_dir": _OLD_DATA_DIR,                      "doc_ids": _OLD_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_VAL_SMALL},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_VAL_SMALL},
        ],
        "test": [
            {"data_dir": _OLD_DATA_DIR,                      "doc_ids": _OLD_TEST},
            {"data_dir": f"{_NEW_BASE}/single_turn_mrcr",   "doc_ids": _MRCR_TEST},
            {"data_dir": f"{_NEW_BASE}/niah_last8k_64_128", "doc_ids": _NIAH1_TEST},
            {"data_dir": f"{_NEW_BASE}/niah_64k-128k",      "doc_ids": _NIAH2_TEST},
            {"data_dir": f"{_NEW_BASE}/creative_writing",   "doc_ids": _CW_TEST},
        ],
    },

    # ── combined_v2_unfiltered: v2 docs, labels 未过滤 (含原始噪声) ──────────
    # 数据源: /dockerdata/retriever_data_v2_backup/{niah, single_turn, creative_writing}
    # 用途: R930+ 实验 — 含噪声标签起 implicit smoothing 作用，logit 分布更校准
    "combined_v2_unfiltered": {
        "train": [
            {"data_dir": f"{_V2_UNFILTERED_BASE}/niah",             "doc_ids": _V2_NIAH_TRAIN},
            {"data_dir": f"{_V2_UNFILTERED_BASE}/single_turn",      "doc_ids": _V2_ST_TRAIN},
            {"data_dir": f"{_V2_UNFILTERED_BASE}/creative_writing", "doc_ids": _V2_CW_TRAIN},
        ],
        "val": [
            {"data_dir": f"{_V2_UNFILTERED_BASE}/niah",             "doc_ids": _V2_NIAH_VAL},
            {"data_dir": f"{_V2_UNFILTERED_BASE}/single_turn",      "doc_ids": _V2_ST_VAL},
            {"data_dir": f"{_V2_UNFILTERED_BASE}/creative_writing", "doc_ids": _V2_CW_VAL},
        ],
        "test": [
            {"data_dir": f"{_V2_UNFILTERED_BASE}/niah",             "doc_ids": _V2_NIAH_TEST},
            {"data_dir": f"{_V2_UNFILTERED_BASE}/single_turn",      "doc_ids": _V2_ST_TEST},
            {"data_dir": f"{_V2_UNFILTERED_BASE}/creative_writing", "doc_ids": _V2_CW_TEST},
        ],
    },

    # ── combined_8layers (R15, 2026-05-30): 8 个 CSA layer 的 hidden + compk ──
    # 数据源: /dockerdata/retriever_data_8layers/{niah, longbench, creative_writing, single_turn}
    # 包含 layers 6/8/10/12/14/16/18/20 (旧版只有 10/12/20)
    # 新增 longbench 子集 (1765 docs)
    # 标签格式: sparse 三元组 (label_indices/scores/pointers), 噪声未清洗 (像 v2_unfiltered)
    # 用途: 8-layer joint training, 配合 --joint-layers 6,8,10,12,14,16,18,20
    "combined_8layers": {
        "train": [
            {"data_dir": f"{_8L_BASE}/niah",             "doc_ids": _8L_NIAH_TRAIN},
            {"data_dir": f"{_8L_BASE}/longbench",        "doc_ids": _8L_LB_TRAIN},
            {"data_dir": f"{_8L_BASE}/creative_writing", "doc_ids": _8L_CW_TRAIN},
            {"data_dir": f"{_8L_BASE}/single_turn",      "doc_ids": _8L_ST_TRAIN},
        ],
        "val": [
            {"data_dir": f"{_8L_BASE}/niah",             "doc_ids": _8L_NIAH_VAL},
            {"data_dir": f"{_8L_BASE}/longbench",        "doc_ids": _8L_LB_VAL},
            {"data_dir": f"{_8L_BASE}/creative_writing", "doc_ids": _8L_CW_VAL},
            {"data_dir": f"{_8L_BASE}/single_turn",      "doc_ids": _8L_ST_VAL},
        ],
        "test": [
            {"data_dir": f"{_8L_BASE}/niah",             "doc_ids": _8L_NIAH_TEST},
            {"data_dir": f"{_8L_BASE}/longbench",        "doc_ids": _8L_LB_TEST},
            {"data_dir": f"{_8L_BASE}/creative_writing", "doc_ids": _8L_CW_TEST},
            {"data_dir": f"{_8L_BASE}/single_turn",      "doc_ids": _8L_ST_TEST},
        ],
    },

    # ── combined_8layers_full: 全量 10 子目录 (2026-06-04) ───────────────────
    # 原 combined_8layers 的 4 个 + 新增 6 个 (copyrightprotected/publicdomain/
    # NovelQA/multi_doc_long_qa_{32k_128k,128k_256k,256k_512k})。全部含 L6..L20
    # hidden/compk + sparse labels。每子目录 80/10/10。用于 R930_v1/R932_v1/R935_v1。
    "combined_8layers_full": {
        "train": [
            {"data_dir": f"{_8L_BASE}/niah",                        "doc_ids": _8L_NIAH_TRAIN},
            {"data_dir": f"{_8L_BASE}/longbench",                   "doc_ids": _8L_LB_TRAIN},
            {"data_dir": f"{_8L_BASE}/creative_writing",            "doc_ids": _8L_CW_TRAIN},
            {"data_dir": f"{_8L_BASE}/single_turn",                 "doc_ids": _8L_ST_TRAIN},
            {"data_dir": f"{_8L_BASE}/copyrightprotected",          "doc_ids": _8L_CP_TRAIN},
            {"data_dir": f"{_8L_BASE}/publicdomain",                "doc_ids": _8L_PD_TRAIN},
            {"data_dir": f"{_8L_BASE}/NovelQA",                     "doc_ids": _8L_NV_TRAIN},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_32k_128k",  "doc_ids": _8L_MDA_TRAIN},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_128k_256k", "doc_ids": _8L_MDB_TRAIN},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_256k_512k", "doc_ids": _8L_MDC_TRAIN},
        ],
        "val": [
            {"data_dir": f"{_8L_BASE}/niah",                        "doc_ids": _8L_NIAH_VAL},
            {"data_dir": f"{_8L_BASE}/longbench",                   "doc_ids": _8L_LB_VAL},
            {"data_dir": f"{_8L_BASE}/creative_writing",            "doc_ids": _8L_CW_VAL},
            {"data_dir": f"{_8L_BASE}/single_turn",                 "doc_ids": _8L_ST_VAL},
            {"data_dir": f"{_8L_BASE}/copyrightprotected",          "doc_ids": _8L_CP_VAL},
            {"data_dir": f"{_8L_BASE}/publicdomain",                "doc_ids": _8L_PD_VAL},
            {"data_dir": f"{_8L_BASE}/NovelQA",                     "doc_ids": _8L_NV_VAL},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_32k_128k",  "doc_ids": _8L_MDA_VAL},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_128k_256k", "doc_ids": _8L_MDB_VAL},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_256k_512k", "doc_ids": _8L_MDC_VAL},
        ],
        "test": [
            {"data_dir": f"{_8L_BASE}/niah",                        "doc_ids": _8L_NIAH_TEST},
            {"data_dir": f"{_8L_BASE}/longbench",                   "doc_ids": _8L_LB_TEST},
            {"data_dir": f"{_8L_BASE}/creative_writing",            "doc_ids": _8L_CW_TEST},
            {"data_dir": f"{_8L_BASE}/single_turn",                 "doc_ids": _8L_ST_TEST},
            {"data_dir": f"{_8L_BASE}/copyrightprotected",          "doc_ids": _8L_CP_TEST},
            {"data_dir": f"{_8L_BASE}/publicdomain",                "doc_ids": _8L_PD_TEST},
            {"data_dir": f"{_8L_BASE}/NovelQA",                     "doc_ids": _8L_NV_TEST},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_32k_128k",  "doc_ids": _8L_MDA_TEST},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_128k_256k", "doc_ids": _8L_MDB_TEST},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_256k_512k", "doc_ids": _8L_MDC_TEST},
        ],
    },

    # ── smoke: 极小数据子集, 用于秒级验证启动路径 (DDP init / step-0 val / 死锁 / OOB) ──
    # 不是为了训练质量, 只为快速 debug。用 longbench(短 doc, 快) + multi_doc_long_qa_128k_256k
    # (够长 max~231809<524288, 触发 RoPE-OOB + 长文档 val 路径)。这两个目录在所有机器都有
    # (不用 NovelQA/256k_512k, 因为部分机器如 M3 没同步这俩, 见 combined_8layers_m3)。
    # 每个目录只取个位数~几十 doc, 加载 ~秒级 vs 全量 ~10min。
    # 用法: --data-config smoke --val-every-steps 20 --epochs 1 (配合 smoke_test.sh)
    "smoke": {
        "train": [
            {"data_dir": f"{_8L_BASE}/longbench",                   "doc_ids": list(range(0, 40))},
            {"data_dir": f"{_8L_BASE}/niah",                        "doc_ids": list(range(0, 16))},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_128k_256k", "doc_ids": list(range(0, 6))},
        ],
        "val": [
            {"data_dir": f"{_8L_BASE}/longbench",                   "doc_ids": list(range(1412, 1420))},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_128k_256k", "doc_ids": list(range(402, 406))},
        ],
        "test": [
            {"data_dir": f"{_8L_BASE}/longbench", "doc_ids": list(range(1588, 1592))},
        ],
    },

    # ── combined_8layers_m3: M3(6卡机)专用全量配置 — 去掉 M3 未同步的 2 个目录 ──
    # M3 本地只有 8/10 目录 (缺 NovelQA + multi_doc_long_qa_256k_512k)。其余与
    # combined_8layers_full 完全一致, 8 目录每个 80/10/10。其它机器仍用 _full。
    "combined_8layers_m3": {
        "train": [
            {"data_dir": f"{_8L_BASE}/niah",                        "doc_ids": _8L_NIAH_TRAIN},
            {"data_dir": f"{_8L_BASE}/longbench",                   "doc_ids": _8L_LB_TRAIN},
            {"data_dir": f"{_8L_BASE}/creative_writing",            "doc_ids": _8L_CW_TRAIN},
            {"data_dir": f"{_8L_BASE}/single_turn",                 "doc_ids": _8L_ST_TRAIN},
            {"data_dir": f"{_8L_BASE}/copyrightprotected",          "doc_ids": _8L_CP_TRAIN},
            {"data_dir": f"{_8L_BASE}/publicdomain",                "doc_ids": _8L_PD_TRAIN},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_32k_128k",  "doc_ids": _8L_MDA_TRAIN},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_128k_256k", "doc_ids": _8L_MDB_TRAIN},
        ],
        "val": [
            {"data_dir": f"{_8L_BASE}/niah",                        "doc_ids": _8L_NIAH_VAL},
            {"data_dir": f"{_8L_BASE}/longbench",                   "doc_ids": _8L_LB_VAL},
            {"data_dir": f"{_8L_BASE}/creative_writing",            "doc_ids": _8L_CW_VAL},
            {"data_dir": f"{_8L_BASE}/single_turn",                 "doc_ids": _8L_ST_VAL},
            {"data_dir": f"{_8L_BASE}/copyrightprotected",          "doc_ids": _8L_CP_VAL},
            {"data_dir": f"{_8L_BASE}/publicdomain",                "doc_ids": _8L_PD_VAL},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_32k_128k",  "doc_ids": _8L_MDA_VAL},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_128k_256k", "doc_ids": _8L_MDB_VAL},
        ],
        "test": [
            {"data_dir": f"{_8L_BASE}/niah",                        "doc_ids": _8L_NIAH_TEST},
            {"data_dir": f"{_8L_BASE}/longbench",                   "doc_ids": _8L_LB_TEST},
            {"data_dir": f"{_8L_BASE}/creative_writing",            "doc_ids": _8L_CW_TEST},
            {"data_dir": f"{_8L_BASE}/single_turn",                 "doc_ids": _8L_ST_TEST},
            {"data_dir": f"{_8L_BASE}/copyrightprotected",          "doc_ids": _8L_CP_TEST},
            {"data_dir": f"{_8L_BASE}/publicdomain",                "doc_ids": _8L_PD_TEST},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_32k_128k",  "doc_ids": _8L_MDA_TEST},
            {"data_dir": f"{_8L_BASE}/multi_doc_long_qa_128k_256k", "doc_ids": _8L_MDB_TEST},
        ],
    },
}


# ── Joint multi-layer retriever (R13) ────────────────────────────────────────
class JointLayerRetriever(nn.Module):
    """
    R13: 包含多个独立 LightningIndexerTrainable 的 ModuleDict，
    各层共享 labels/positions, 但用各自的 hidden_state + compk + 独立权重。

    用于 score-level ensemble 训练：训练时同步 forward 3 层，
    val 时直接计算 mean/max/min 聚合，可以 early-stop on ensemble metric。
    """
    def __init__(self, layer_ids: list, n_heads: int = 64,
                 max_position: int = 131072,
                 q_lora_rank: int = None):
        super().__init__()
        self.layer_ids = list(layer_ids)
        self.retrievers = nn.ModuleDict({
            f"l{lid}": LightningIndexerTrainable(
                csa_layer_idx=lid, n_heads=n_heads,
                max_position=max_position,
                q_lora_rank=q_lora_rank,
            ) for lid in layer_ids
        })

    def load_pretrained(self, weight_dir: str):
        """每个 sub-retriever 加载自己层的预训练权重。"""
        for lid in self.layer_ids:
            self.retrievers[f"l{lid}"].load_pretrained(weight_dir)

    def forward(self, hidden: dict, compk: dict, positions: torch.Tensor,
                return_logits: bool = False) -> dict:
        """
        Args:
            hidden: dict {lid: [B, 4096]}
            compk:  dict {lid: [B, N, 132]}
            positions: [B] int64 (shared across layers)
        Returns:
            dict {lid: [B, N]} of scores (sigmoid) or logits (raw)
        """
        out = {}
        for lid in self.layer_ids:
            out[lid] = self.retrievers[f"l{lid}"](
                hidden[lid], compk[lid], positions,
                layer_embed_idx=None, return_logits=return_logits,
            )
        return out


def _eval_full_dir(model, data_dir, doc_ids, label_interval, device,
                   layer_ids, sample_interval=4, batch_size=256, recall_k=512):
    """
    内部辅助：对单个目录做全集评测，返回 (tp, fp, fn, recall_k_hit, recall_k_total)。
    由 evaluate_full_val 调用，支持多目录累积。
    """
    tp = fp = fn = 0
    rk_hit = rk_total = 0  # recall@K numerator/denominator

    pkl_paths = sorted(_glob.glob(os.path.join(data_dir, "doc_*.pkl")))
    if doc_ids is not None:
        allowed = {f"doc_{i:05d}.pkl" for i in doc_ids}
        pkl_paths = [p for p in pkl_paths if os.path.basename(p) in allowed]

    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as f:
            data = _pickle.load(f)

        label_ptrs = data["label_pointers"].numpy()
        label_idxs = data["label_indices"].numpy()

        ref_lid  = layer_ids[0]
        n_decode = data[f"hidden_layer_{ref_lid}"].shape[0]
        n_blocks = data[f"compk_layer_{ref_lid}"].shape[0]

        token_indices = list(range(0, n_decode, sample_interval))
        n_tokens      = len(token_indices)

        # agg_scores[i, b] = max score across all layers for token i, block b
        agg_scores = np.zeros((n_tokens, n_blocks), dtype=np.float32)

        for lid in layer_ids:
            hidden_all    = data[f"hidden_layer_{lid}"]
            compk_all     = data[f"compk_layer_{lid}"]
            positions_all = data[f"positions_layer_{lid}"]

            compk_np  = compk_all.numpy() if hasattr(compk_all, "numpy") else np.array(compk_all)
            compk_gpu = torch.from_numpy(compk_np).to(device)

            lid_embed_val = LAYER_EMBED_MAP.get(lid, 0)

            for batch_start in range(0, n_tokens, batch_size):
                batch_tidxs = token_indices[batch_start : batch_start + batch_size]
                B = len(batch_tidxs)

                hidden_batch = torch.stack([
                    torch.as_tensor(hidden_all[t], dtype=torch.float32) for t in batch_tidxs
                ]).to(device)
                pos_batch = torch.stack([
                    torch.as_tensor(int(positions_all[t]), dtype=torch.int64) for t in batch_tidxs
                ]).to(device)
                compk_batch = compk_gpu.unsqueeze(0).expand(B, -1, -1)
                lei_batch   = torch.full((B,), lid_embed_val, dtype=torch.long, device=device)

                scores = model(
                    hidden_batch, compk_batch, pos_batch, layer_embed_idx=lei_batch
                ).cpu().numpy()
                # union via element-wise max
                agg_scores[batch_start : batch_start + B] = np.maximum(
                    agg_scores[batch_start : batch_start + B], scores
                )

        # F1 with threshold = 0.5 on aggregated scores
        for bi, t in enumerate(token_indices):
            t_end    = min(t + label_interval, n_decode)
            pos_idxs = np.unique(label_idxs[label_ptrs[t] : label_ptrs[t_end]])
            if len(pos_idxs) == 0:
                continue
            # --- F1 (threshold=0.5) ---
            pred_t   = agg_scores[bi] >= 0.5
            lbl_mask = np.zeros(n_blocks, dtype=bool)
            lbl_mask[pos_idxs] = True
            tp += int(( pred_t &  lbl_mask).sum())
            fp += int(( pred_t & ~lbl_mask).sum())
            fn += int((~pred_t &  lbl_mask).sum())
            # --- recall@K ---
            if n_blocks > recall_k:
                top_k_idx = np.argpartition(agg_scores[bi], -recall_k)[-recall_k:]
            else:
                top_k_idx = np.arange(n_blocks)
            rk_hit   += int(np.isin(pos_idxs, top_k_idx).sum())
            rk_total += len(pos_idxs)

    return tp, fp, fn, rk_hit, rk_total


@torch.no_grad()
def evaluate_full_val(model, data_dir=None, doc_ids=None, label_interval=64, device="cuda",
                      layer_ids=None, csa_layer_idx=None,
                      sample_interval=4, batch_size=256,
                      data_specs=None):
    """
    Full-set val evaluation: score ALL blocks per document.
    Gives unbiased F1 for checkpoint selection (no max_pos truncation).

    单层模式：layer_ids=None → 使用 csa_layer_idx（兼容旧调用）
    多层 union 模式：layer_ids=[10,12,20] → 对每个 token 取各层 max score，
                     模拟推理时多层 union top-K 的行为。
    多目录模式：data_specs=[{"data_dir":..., "doc_ids":...}, ...] → 累积 tp/fp/fn

    Args:
        model:           LightningIndexerTrainable (eval mode)
        data_dir:        pkl directory（单目录模式；data_specs 优先）
        doc_ids:         doc ID list（单目录模式）
        label_interval:  look-ahead window for positive labels
        device:          'cuda' or 'cpu'
        layer_ids:       list of CSA layer indices, e.g. [10, 12, 20]
        csa_layer_idx:   single layer (legacy; used when layer_ids is None)
        sample_interval: evaluate every N-th decode token (default 4 for speed)
        batch_size:      hidden-state batch size
        data_specs:      list of {"data_dir": str, "doc_ids": list} — 多目录模式

    Returns:
        (precision, recall, f1, recall_at_k)
    """
    if layer_ids is None:
        assert csa_layer_idx is not None, "Must provide layer_ids or csa_layer_idx"
        layer_ids = [csa_layer_idx]

    # 统一到 data_specs 形式
    if data_specs is None:
        assert data_dir is not None, "Must provide data_dir or data_specs"
        data_specs = [{"data_dir": data_dir, "doc_ids": doc_ids}]

    model.eval()
    total_tp = total_fp = total_fn = 0
    total_rk_hit = total_rk_total = 0

    for spec in data_specs:
        t, f, n, rk_h, rk_t = _eval_full_dir(
            model, spec["data_dir"], spec.get("doc_ids"),
            label_interval, device, layer_ids, sample_interval, batch_size
        )
        total_tp += t
        total_fp += f
        total_fn += n
        total_rk_hit   += rk_h
        total_rk_total += rk_t

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    recall_at_k = total_rk_hit / (total_rk_total + 1e-8)
    return precision, recall, f1, recall_at_k


# ── 训练循环 ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _eval_joint_full_dir(joint_model, data_dir, doc_ids, label_interval, device,
                         layer_ids, sample_interval=4, batch_size=256, recall_k=512):
    """
    R13 joint val: 单目录全集评测，对每个 (token, block) 跨层取 mean ensemble score。
    返回 (per_layer_stats, ensemble_stats)。

    per_layer_stats: dict {lid: (tp, fp, fn, rk_hit, rk_total)}
    ensemble_stats: (tp, fp, fn, rk_hit, rk_total) using mean across layers
    """
    per_layer = {lid: [0, 0, 0, 0, 0] for lid in layer_ids}
    ens = [0, 0, 0, 0, 0]   # tp, fp, fn, rk_hit, rk_total

    pkl_paths = sorted(_glob.glob(os.path.join(data_dir, "doc_*.pkl")))
    if doc_ids is not None:
        allowed = {f"doc_{i:05d}.pkl" for i in doc_ids}
        pkl_paths = [p for p in pkl_paths if os.path.basename(p) in allowed]

    for pkl_path in pkl_paths:
        try:
            with open(pkl_path, "rb") as f:
                data = _pickle.load(f)
        except Exception as e:
            print(f"  [val skip] corrupt pkl {os.path.basename(pkl_path)}: {type(e).__name__}")
            continue
        label_ptrs = data["label_pointers"].numpy()
        label_idxs = data["label_indices"].numpy()

        ref_lid = layer_ids[0]
        n_decode = data[f"hidden_layer_{ref_lid}"].shape[0]
        n_blocks = data[f"compk_layer_{ref_lid}"].shape[0]
        token_indices = list(range(0, n_decode, sample_interval))
        n_tokens = len(token_indices)

        # per-layer scores [n_layers, n_tokens, n_blocks]
        all_scores = {lid: np.zeros((n_tokens, n_blocks), dtype=np.float32)
                      for lid in layer_ids}

        for lid in layer_ids:
            sub_model    = joint_model.retrievers[f"l{lid}"]
            hidden_all   = data[f"hidden_layer_{lid}"]
            compk_all    = data[f"compk_layer_{lid}"]
            positions_all= data[f"positions_layer_{lid}"]
            compk_np = compk_all.numpy() if hasattr(compk_all, "numpy") else np.array(compk_all)
            compk_gpu = torch.from_numpy(compk_np).to(device)

            for batch_start in range(0, n_tokens, batch_size):
                batch_tidxs = token_indices[batch_start : batch_start + batch_size]
                B = len(batch_tidxs)
                hidden_batch = torch.stack([
                    torch.as_tensor(hidden_all[t], dtype=torch.float32) for t in batch_tidxs
                ]).to(device)
                pos_batch = torch.stack([
                    torch.as_tensor(int(positions_all[t]), dtype=torch.int64) for t in batch_tidxs
                ]).to(device)
                compk_batch = compk_gpu.unsqueeze(0).expand(B, -1, -1)
                lei_batch = torch.zeros(B, dtype=torch.long, device=device)
                scores = sub_model(
                    hidden_batch, compk_batch, pos_batch, layer_embed_idx=lei_batch
                ).cpu().numpy()
                all_scores[lid][batch_start : batch_start + B] = scores

        # mean ensemble across layers
        ens_scores = np.mean(np.stack([all_scores[lid] for lid in layer_ids], axis=0), axis=0)

        for bi, t in enumerate(token_indices):
            t_end = min(t + label_interval, n_decode)
            pos_idxs = np.unique(label_idxs[label_ptrs[t] : label_ptrs[t_end]])
            if len(pos_idxs) == 0:
                continue
            lbl_mask = np.zeros(n_blocks, dtype=bool); lbl_mask[pos_idxs] = True

            # per-layer
            for lid in layer_ids:
                pred_t = all_scores[lid][bi] >= 0.5
                per_layer[lid][0] += int(( pred_t &  lbl_mask).sum())
                per_layer[lid][1] += int(( pred_t & ~lbl_mask).sum())
                per_layer[lid][2] += int((~pred_t &  lbl_mask).sum())
                if n_blocks > recall_k:
                    top_k_idx = np.argpartition(all_scores[lid][bi], -recall_k)[-recall_k:]
                else:
                    top_k_idx = np.arange(n_blocks)
                per_layer[lid][3] += int(np.isin(pos_idxs, top_k_idx).sum())
                per_layer[lid][4] += len(pos_idxs)

            # ensemble (mean)
            pred_e = ens_scores[bi] >= 0.5
            ens[0] += int(( pred_e &  lbl_mask).sum())
            ens[1] += int(( pred_e & ~lbl_mask).sum())
            ens[2] += int((~pred_e &  lbl_mask).sum())
            if n_blocks > recall_k:
                top_k_idx = np.argpartition(ens_scores[bi], -recall_k)[-recall_k:]
            else:
                top_k_idx = np.arange(n_blocks)
            ens[3] += int(np.isin(pos_idxs, top_k_idx).sum())
            ens[4] += len(pos_idxs)

    return per_layer, ens


@torch.no_grad()
def evaluate_full_val_joint(joint_model, data_specs, label_interval, device,
                            layer_ids, sample_interval=4, batch_size=256):
    """
    R13: joint multi-layer val. 返回 dict:
      {
        'per_layer': {lid: (prec, rec, f1, rk)},
        'ensemble':  (prec, rec, f1, rk),
      }
    """
    joint_model.eval()
    per_layer_acc = {lid: [0, 0, 0, 0, 0] for lid in layer_ids}
    ens_acc = [0, 0, 0, 0, 0]

    for spec in data_specs:
        pl, en = _eval_joint_full_dir(
            joint_model, spec["data_dir"], spec.get("doc_ids"),
            label_interval, device, layer_ids, sample_interval, batch_size,
        )
        for lid in layer_ids:
            for i in range(5):
                per_layer_acc[lid][i] += pl[lid][i]
        for i in range(5):
            ens_acc[i] += en[i]

    def _stats(t):
        tp, fp, fn, rk_h, rk_t = t
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        rk   = rk_h / (rk_t + 1e-8)
        return prec, rec, f1, rk

    return {
        "per_layer": {lid: _stats(per_layer_acc[lid]) for lid in layer_ids},
        "ensemble":  _stats(ens_acc),
    }


def _scan_max_position_in_specs(specs, sample_stride: int = 1) -> int:
    """Scan val/test specs' pkls for the largest `positions_layer_*` value.

    The RoPE freqs_cis table must cover BOTH train and val positions. Val docs are
    streamed separately at eval time (not via the train DataLoader), so a val
    position larger than the table → freqs_cis[positions] gather OOB → CUDA
    device-side assert mid-validation that kills the whole DDP job. This was the
    root cause of the synchronized 3-machine crash (NovelQA val pos 438660 >
    train-only auto-derived 398012).

    Only invoked on the `--max-position 0` auto-derive fallback. Corrupt pkls are
    skipped (consistent with the eval loop). Returns 0 if nothing scannable.
    """
    gmax = 0
    for spec in specs:
        data_dir = spec["data_dir"]
        doc_ids = spec.get("doc_ids")
        pkl_paths = sorted(_glob.glob(os.path.join(data_dir, "doc_*.pkl")))
        if doc_ids is not None:
            allowed = {f"doc_{i:05d}.pkl" for i in doc_ids}
            pkl_paths = [p for p in pkl_paths if os.path.basename(p) in allowed]
        for pkl_path in pkl_paths[::sample_stride]:
            try:
                with open(pkl_path, "rb") as f:
                    data = _pickle.load(f)
            except Exception:
                continue
            pos = None
            for k in data:
                if k.startswith("positions_layer_"):
                    pos = data[k]
                    break
            if pos is None:
                continue
            m = int(np.asarray(pos).max())
            if m > gmax:
                gmax = m
    return gmax


def _train_joint(args):
    """
    R13: 联合多层训练 — 3 个独立 retriever 同时训练，共享 labels/positions，
    各层独立 BCE loss，sum 后 backward。Val 时对每层独立评测 + mean ensemble。
    Checkpoint 按层独立保存（兼容 eval.py 单层加载）。

    DDP 支持 (auto-detected from torchrun env vars RANK / LOCAL_RANK / WORLD_SIZE):
      torchrun --nproc_per_node 2 train.py --joint-layers 10,12,20 ...

    DDP 注意:
      - --batch-size 是 per-rank batch (global = bs × world_size)
      - val/ckpt/print 只在 rank 0 执行
      - find_unused_parameters=True (pairwise loss 偶尔会 skip 某层)
      - DistributedSampler: 每个 epoch 调 set_epoch(epoch) 保证 shuffle 不同
    """
    # ── DDP setup ───────────────────────────────────────────────────────────
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not dist.is_initialized():
            # Raise the NCCL collective timeout well above the 10-min default.
            # rank0 runs full-val alone (over ultra-long combined_8layers_full
            # docs) while ranks 1..N-1 wait at the post-val dist.broadcast/barrier;
            # a slow val must not trip the watchdog and kill the job.
            import datetime as _dt
            dist.init_process_group(
                backend="nccl", timeout=_dt.timedelta(minutes=60)
            )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_main = (rank == 0)
        if is_main:
            print(f"[DDP] world_size={world_size} rank={rank} local_rank={local_rank} device={device}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = args.device
        is_main = True

    def log(msg):
        """Print only on rank 0."""
        if is_main:
            print(msg)

    joint_layer_ids = [int(x.strip()) for x in args.joint_layers.split(",")]
    log(f"R13 Joint training: layer_ids={joint_layer_ids}")

    # ── 数据 ────────────────────────────────────────────────────────────────
    assert args.data_config is not None, "Joint training requires --data-config (e.g., combined_v2)"
    if args.data_config not in DATA_CONFIGS:
        raise ValueError(f"Unknown --data-config '{args.data_config}'")
    cfg = DATA_CONFIGS[args.data_config]
    train_specs = cfg["train"]
    val_specs   = cfg["val_small"] if args.val_small else cfg["val"]
    log(f"Data config: '{args.data_config}'  "
        f"({len(train_specs)} train dirs, {len(val_specs)} val dirs"
        f"{', val_small=True' if args.val_small else ''})")

    # ── Parse bucket config ──────────────────────────────────────────────────
    bucket_config = None
    if args.bucketed and args.bucket_config:
        bucket_config = []
        for part in args.bucket_config.split(","):
            max_s, bs_s = part.strip().split(":")
            max_blocks = float("inf") if max_s.lower() == "inf" else int(max_s)
            bucket_config.append((max_blocks, int(bs_s)))

    # Build dataloader. For DDP, need a sampler we can set_epoch on.
    if is_ddp:
        # First build dataset only, then wrap with DistributedSampler
        from dataloader import (
            CombinedJointLayerDataset, joint_collate_fn,
            LengthBucketedBatchSampler,
        )
        dataset = CombinedJointLayerDataset(
            specs=train_specs, layer_ids=joint_layer_ids,
            sample_interval=args.sample_interval,
            label_interval=args.label_interval,
            max_pos=args.max_pos,
            seed=args.seed,
            neg_ratio=args.neg_ratio,
            weighted_loss=args.weighted_loss,
            cache_size=args.cache_size,
        )
        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        if args.bucketed:
            batch_sampler = LengthBucketedBatchSampler(
                sample_n_blocks=dataset.sample_n_blocks,
                batch_size=args.batch_size,
                shuffle=True, seed=args.seed, drop_last=True,
                rank=rank, world_size=world_size,
            )
            loader = DataLoader(
                dataset, batch_sampler=batch_sampler,
                collate_fn=joint_collate_fn(joint_layer_ids),
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
            train_sampler = batch_sampler  # for set_epoch()
        else:
            loader = DataLoader(
                dataset, batch_size=args.batch_size, sampler=train_sampler,
                shuffle=False,
                collate_fn=joint_collate_fn(joint_layer_ids),
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                drop_last=True,
            )
    else:
        train_sampler = None
        loader = build_joint_combined_dataloader(
            specs=train_specs, layer_ids=joint_layer_ids,
            batch_size=args.batch_size,
            sample_interval=args.sample_interval,
            label_interval=args.label_interval,
            max_pos=args.max_pos,
            shuffle=True, num_workers=args.num_workers,
            seed=args.seed,
            neg_ratio=args.neg_ratio,
            weighted_loss=args.weighted_loss,
            cache_size=args.cache_size,
            bucketed=args.bucketed,
            bucket_config=bucket_config,
        )
        if args.bucketed:
            train_sampler = loader.batch_sampler  # for set_epoch()

    # ── 推算 max_position ───────────────────────────────────────────────────
    # RoPE freqs_cis table must cover the LARGEST position in BOTH train and val
    # (val docs are streamed separately at eval time; an OOB position there → CUDA
    # device-side assert during val). combined_8layers_full has ultra-long docs
    # (NovelQA/multi_doc up to ~460k positions). Use --max-position to set the table
    # size explicitly; otherwise derive from train data with a safe floor.
    if args.max_position and args.max_position > 0:
        max_position = args.max_position
    else:
        train_max = max(
            int(doc["positions"].max().item()) for doc in loader.dataset.docs
        )
        # Also scan val (and test) specs — they are streamed at eval time, not via
        # the train loader, so a val position > table size → CUDA assert in val.
        val_max  = _scan_max_position_in_specs(val_specs)
        test_max = _scan_max_position_in_specs(cfg.get("test", []))
        max_pos_in_data = max(train_max, val_max, test_max)
        max_position = max(max_pos_in_data + 1, 131072)
        log(f"  auto max_position scan: train={train_max} val={val_max} test={test_max}")
    log(f"RoPE max_position = {max_position}")

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = JointLayerRetriever(
        layer_ids=joint_layer_ids,
        n_heads=args.n_heads,
        max_position=max_position,
        q_lora_rank=args.q_lora_rank,
    ).to(device)

    if args.resume_from:
        # resume_from 期望是 joint state dict (full ModuleDict)
        state_dict = torch.load(args.resume_from, map_location=device, weights_only=True)
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("freqs_cis")}
        model.load_state_dict(state_dict, strict=False)
        log(f"Resumed joint model from {args.resume_from}")
    elif not args.no_pretrain and args.weight_dir:
        model.load_pretrained(args.weight_dir)
    else:
        log("Joint training from scratch (random initialization).")

    # ── DDP wrap ────────────────────────────────────────────────────────────
    # find_unused_parameters=True because pairwise loss can skip a layer when no pairs;
    # also some sub-retrievers' RoPE freqs_cis is buffer (not param) but we set as
    # buffer in original LightningIndexerTrainable, so should be fine.
    if is_ddp:
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
    # Helper to access underlying module (for save/val) regardless of DDP wrap
    def _inner(m):
        return m.module if isinstance(m, DDP) else m

    # ── 优化器 ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = len(loader) * args.epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── 训练状态 ────────────────────────────────────────────────────────────
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    state = dict(
        best_ens_f1     = 0.0,
        best_ens_rk     = 0.0,
        best_per_layer_f1 = {lid: 0.0 for lid in joint_layer_ids},
        best_per_layer_rk = {lid: 0.0 for lid in joint_layer_ids},
        no_improve_vals = 0,
        early_stop      = False,
    )
    patience_vals = args.patience_vals

    def _save_layer_ckpt(lid: str, kind: str, tag: str):
        """保存单层 retriever 权重到独立文件，便于 eval.py 单层加载。Rank 0 only."""
        if not is_main:
            return
        sub_state = _inner(model).retrievers[f"l{lid}"].state_dict()
        path = os.path.join(args.output_dir, f"ckpt_l{lid}_{kind}.pt")
        torch.save(sub_state, path)
        if tag:
            hist = os.path.join(args.output_dir, f"ckpt_l{lid}_{kind}_{tag}.pt")
            torch.save(sub_state, hist)

    def _save_joint_ckpt(kind: str, tag: str):
        """保存 joint 模型完整权重。Rank 0 only."""
        if not is_main:
            return
        path = os.path.join(args.output_dir, f"ckpt_joint_{kind}.pt")
        torch.save(_inner(model).state_dict(), path)
        if tag:
            hist = os.path.join(args.output_dir, f"ckpt_joint_{kind}_{tag}.pt")
            torch.save(_inner(model).state_dict(), hist)

    def _run_val_and_save(epoch, step):
        """Run joint val, save best ckpts (per-layer + joint), update state.
        DDP: only rank 0 runs val (single-GPU val func), then broadcasts state.early_stop."""
        if not args.val_fullset:
            log("WARNING: joint training without --val-fullset, skipping val")
            return None, None

        if is_main:
            result = evaluate_full_val_joint(
                joint_model     = _inner(model),
                data_specs      = val_specs,
                label_interval  = args.label_interval,
                device          = device,
                layer_ids       = joint_layer_ids,
                sample_interval = 4,
                batch_size      = args.batch_size,
            )
            ens_prec, ens_rec, ens_f1, ens_rk = result["ensemble"]
            tag_str = f"ep{epoch}" if step is None else f"ep{epoch}_s{step}"
            print(f"    [Joint Val @{tag_str}]  "
                  f"ENS prec={ens_prec:.4f} rec={ens_rec:.4f} f1={ens_f1:.4f} r@512={ens_rk:.4f}")
            for lid in joint_layer_ids:
                p, r, f, rk = result["per_layer"][lid]
                print(f"        L{lid}:  prec={p:.4f} rec={r:.4f} f1={f:.4f} r@512={rk:.4f}")

            improved_anything = False
            if ens_f1 > state["best_ens_f1"]:
                state["best_ens_f1"] = ens_f1
                improved_anything = True
                for lid in joint_layer_ids: _save_layer_ckpt(lid, "best_ens_f1", tag_str)
                _save_joint_ckpt("best_ens_f1", tag_str)
                print(f"    New best ensemble F1={ens_f1:.4f} ({tag_str})")
            if ens_rk > state["best_ens_rk"]:
                state["best_ens_rk"] = ens_rk
                improved_anything = True
                for lid in joint_layer_ids: _save_layer_ckpt(lid, "best_ens_rk", tag_str)
                _save_joint_ckpt("best_ens_rk", tag_str)
                print(f"    New best ensemble r@512={ens_rk:.4f} ({tag_str})")
            for lid in joint_layer_ids:
                p, r, f, rk = result["per_layer"][lid]
                if f > state["best_per_layer_f1"][lid]:
                    state["best_per_layer_f1"][lid] = f
                    _save_layer_ckpt(lid, "best_f1", tag_str)
                if rk > state["best_per_layer_rk"][lid]:
                    state["best_per_layer_rk"][lid] = rk
                    _save_layer_ckpt(lid, "best_recall_k", tag_str)

            if improved_anything:
                state["no_improve_vals"] = 0
            else:
                state["no_improve_vals"] += 1
                if patience_vals > 0:
                    print(f"    No ensemble improvement: {state['no_improve_vals']}/{patience_vals}")
                if patience_vals > 0 and state["no_improve_vals"] >= patience_vals:
                    print(f"\n[Early stop] No ensemble improvement for {patience_vals} vals.")
                    state["early_stop"] = True

            ret = (ens_f1, ens_rk)
        else:
            ret = (None, None)
        # Sync early_stop flag across all ranks via broadcast on a tensor
        if is_ddp:
            es_tensor = torch.tensor(
                [1 if state["early_stop"] else 0], dtype=torch.int64, device=device
            )
            dist.broadcast(es_tensor, src=0)
            state["early_stop"] = bool(es_tensor.item())
            dist.barrier()
        model.train()
        return ret

    # ── Pre-training validation (step 0) — catch max_position OOB early ─────
    # NOTE: ALL ranks must enter _run_val_and_save. Internally only rank0 runs the
    # val forward (if is_main), but the function ends with a collective
    # dist.broadcast + dist.barrier that EVERY rank must participate in. Gating the
    # call with `if rank == 0` (the old bug) left ranks 1..N-1 skipping straight
    # into the training loop's first all_reduce while rank0 waited at the
    # broadcast → mismatched collectives → NCCL watchdog timeout (~10 min) → the
    # whole 8-GPU job dies before step 1. This mirrors the in-loop call at line
    # ~1296 which (correctly) lets all ranks in.
    if args.val_fullset:
        log("Running pre-training validation (step 0 sanity check) ...")
        _run_val_and_save(epoch=0, step=0)

    # ── 训练 loop ───────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        # Ensure sampler shuffles differently each epoch (DDP + bucketed non-DDP)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            hidden    = {lid: batch["hidden"][lid].to(device) for lid in joint_layer_ids}
            compk     = {lid: batch["compk"][lid].to(device)  for lid in joint_layer_ids}
            positions = batch["positions"].to(device)
            labels    = batch["labels"].to(device)
            mask      = batch["mask"].to(device)
            weights   = batch["weights"].to(device)

            amp_ctx = (torch.cuda.amp.autocast(dtype=torch.bfloat16)
                       if args.bf16 else nullcontext())
            with amp_ctx:
                if args.pairwise:
                    logits_dict = model(hidden, compk, positions, return_logits=True)
                else:
                    scores_dict = model(hidden, compk, positions, return_logits=False)

            # ── per-layer loss (sum) ─────────────────────────────────────────
            total_loss = 0.0
            n_layers_used = 0
            for lid in joint_layer_ids:
                if args.pairwise:
                    logits = logits_dict[lid].float()
                    B_cur, N_cur = logits.shape
                    chunk_size = 64
                    pos_mask_2d = (labels == 1) & mask
                    neg_mask_2d = (labels == 0) & mask
                    loss_sum = torch.tensor(0.0, device=logits.device)
                    count_sum = torch.tensor(0.0, device=logits.device)
                    for cs in range(0, B_cur, chunk_size):
                        ce = min(cs + chunk_size, B_cur)
                        logits_c = logits[cs:ce]
                        diff_c = logits_c.unsqueeze(2) - logits_c.unsqueeze(1)
                        pair_mask_c = pos_mask_2d[cs:ce].unsqueeze(2) & neg_mask_2d[cs:ce].unsqueeze(1)
                        if args.pairwise_loss == "bpr":
                            elem_c = -F.logsigmoid(diff_c)
                        else:
                            elem_c = F.relu(args.margin - diff_c)
                        pair_mask_c_f = pair_mask_c.float()
                        loss_sum = loss_sum + (elem_c * pair_mask_c_f).sum()
                        count_sum = count_sum + pair_mask_c_f.sum()
                    # DDP-safe: clamp denominator instead of skipping (avoids hang
                    # when one rank has 0 pairs while another doesn't)
                    loss_l = loss_sum / count_sum.clamp(min=1.0)
                else:
                    scores = scores_dict[lid].float()
                    # Label smoothing: y=1 → 1-eps, y=0 → eps (限制 logit 极端值)
                    eps = args.label_smoothing
                    if eps > 0:
                        t_smooth = labels[mask] * (1 - 2*eps) + eps
                    else:
                        t_smooth = labels[mask]
                    if args.focal_loss:
                        p = scores[mask]; t = t_smooth; w = weights[mask]
                        p_t = p * t + (1 - p) * (1 - t)
                        focal_w = (1 - p_t) ** 2
                        bce = F.binary_cross_entropy(p, t, reduction='none')
                        loss_l = (bce * focal_w * w).mean()
                    else:
                        loss_l = F.binary_cross_entropy(scores[mask], t_smooth,
                                                       weight=weights[mask])
                total_loss = total_loss + loss_l
                n_layers_used += 1

            # n_layers_used always == len(joint_layer_ids) now (no skip)
            loss = total_loss / n_layers_used

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % args.log_interval == 0 and is_main:
                n_pos = labels[mask].sum().item()
                n_neg = mask.sum().item() - n_pos
                lr_now = scheduler.get_last_lr()[0]
                print(f"[joint epoch {epoch:3d} | step {step:5d}/{len(loader)}]  "
                      f"loss={loss.item():.4f}  pos={int(n_pos)}  neg={int(n_neg)}  "
                      f"lr={lr_now:.2e}")

            if (args.val_every_steps > 0 and step > 0
                    and step % args.val_every_steps == 0):
                _run_val_and_save(epoch, step)
                if state["early_stop"]:
                    break

        if state["early_stop"]:
            break

        avg_loss = epoch_loss / max(len(loader), 1)
        log(f"\n>>> [Joint] Epoch {epoch} done.  avg_loss={avg_loss:.4f}")
        _run_val_and_save(epoch, None)
        log("")

        # 每 epoch 保存 joint state for resume (rank 0 only)
        if is_main:
            ckpt_path = os.path.join(args.output_dir, f"ckpt_joint_epoch{epoch:03d}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": _inner(model).state_dict(),
                "optimizer":   optimizer.state_dict(),
                "avg_loss":    avg_loss,
                "best_ens_f1": state["best_ens_f1"],
                "best_ens_rk": state["best_ens_rk"],
                "args":        vars(args),
            }, ckpt_path)
            print(f"Joint checkpoint saved → {ckpt_path}")
        if is_ddp:
            dist.barrier()

        if state["early_stop"]:
            break

    log(f"\n[Joint] Training complete. "
        f"best_ens_f1={state['best_ens_f1']:.4f}  best_ens_r@512={state['best_ens_rk']:.4f}")
    for lid in joint_layer_ids:
        log(f"    L{lid}: best_f1={state['best_per_layer_f1'][lid]:.4f}  "
            f"best_r@512={state['best_per_layer_rk'][lid]:.4f}")

    if is_ddp:
        dist.destroy_process_group()


def train(args):
    # ── R13: joint multi-layer training dispatch ───────────────────────────
    if args.joint_layers is not None:
        if args.layer is not None:
            raise ValueError("--joint-layers is mutually exclusive with --layer")
        return _train_joint(args)

    device = args.device

    # ── 单层模式 ──────────────────────────────────────────────────────────────
    if args.layer is None:
        raise ValueError("Must specify --layer")
    primary_layer = args.layer

    # ── 数据配置 ──────────────────────────────────────────────────────────────
    if args.data_config is None:
        raise ValueError("Must specify --data-config")
    if args.data_config not in DATA_CONFIGS:
        raise ValueError(
            f"Unknown --data-config '{args.data_config}'. "
            f"Available: {list(DATA_CONFIGS.keys())}"
        )
    cfg         = DATA_CONFIGS[args.data_config]
    train_specs = cfg["train"]
    val_specs   = cfg["val_small"] if args.val_small else cfg["val"]
    print(f"Data config: '{args.data_config}'  "
          f"({len(train_specs)} train dirs, {len(val_specs)} val dirs"
          f"{', val_small=True' if args.val_small else ''})")

    loader = build_combined_dataloader(
        specs=train_specs,
        csa_layer_idx=primary_layer,
        batch_size=args.batch_size,
        sample_interval=args.sample_interval,
        label_interval=args.label_interval,
        max_pos=args.max_pos,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        neg_ratio=args.neg_ratio,
        weighted_loss=args.weighted_loss,
    )
    if not args.val_fullset:
        print("WARNING: --val-fullset not set → no val eval will be run. "
              "Recommend adding --val-fullset for proper checkpoint selection.")

    # ── 推算 RoPE 所需的 max_position ──────────────────────────────────────
    # Must cover BOTH train and val/test (val docs streamed at eval time; an OOB
    # position there → CUDA device-side assert during val).
    train_max = max(
        int(doc["positions"].max().item()) for doc in loader.dataset.docs
    )
    val_max  = _scan_max_position_in_specs(val_specs)
    test_max = _scan_max_position_in_specs(cfg.get("test", []))
    max_pos_in_data = max(train_max, val_max, test_max)
    max_position = max(max_pos_in_data + 1, 131072)
    print(f"RoPE max_position = {max_position} "
          f"(train={train_max} val={val_max} test={test_max})")

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = LightningIndexerTrainable(
        csa_layer_idx=primary_layer,
        max_position=max_position,
        n_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
    ).to(device)

    if args.resume_from:
        # 从指定 checkpoint 加载模型权重（用于两阶段训练: Stage 1 → Stage 2）
        state = torch.load(args.resume_from, map_location=device, weights_only=True)
        # 跳过 freqs_cis (RoPE buffer, 非学习参数, 大小可能因 max_position 不同而不匹配)
        state = {k: v for k, v in state.items() if k != "freqs_cis"}
        model.load_state_dict(state, strict=False)
        print(f"Resumed model weights from {args.resume_from}")
    elif not args.no_pretrain and args.weight_dir:
        model.load_pretrained(args.weight_dir)
    else:
        print("Training from scratch (random initialization).")

    # ── 优化器 & 学习率调度 ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = len(loader) * args.epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── 训练 ────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss       = float("inf")
    state           = dict(
        best_val_f1     = 0.0,
        best_recall_k   = 0.0,
        no_improve_cnt  = 0,        # patience by epoch (legacy)
        no_improve_vals = 0,        # patience by val-call (per-step)
        early_stop      = False,
    )
    patience        = args.patience
    patience_vals   = args.patience_vals

    def _run_val_and_save(epoch, step):
        """Run val, save best ckpts, update state. Returns (val_f1, val_rk)."""
        val_f1 = None
        val_rk = None
        if args.val_fullset:
            prec, rec, val_f1, val_rk = evaluate_full_val(
                model           = model,
                data_specs      = val_specs,
                label_interval  = args.label_interval,
                device          = device,
                layer_ids       = [primary_layer],
                sample_interval = 4,
                batch_size      = args.batch_size,
            )
            tag = f"ep{epoch} step{step}" if step is not None else f"ep{epoch} end"
            print(
                f"    [Val-full (layer={primary_layer}) @{tag}]  "
                f"precision={prec:.4f}  recall={rec:.4f}  val_f1={val_f1:.4f}  "
                f"recall@512={val_rk:.4f}"
            )
            model.train()

        improved_anything = False
        # ── best val_f1 ──────────────────────────────────────────────────────
        if val_f1 is not None and val_f1 > state["best_val_f1"]:
            state["best_val_f1"] = val_f1
            improved_anything = True
            best_f1_path = os.path.join(args.output_dir, "ckpt_best_f1.pt")
            torch.save(model.state_dict(), best_f1_path)
            tag_str = f"ep{epoch}" if step is None else f"ep{epoch}_s{step}"
            hist_path = os.path.join(args.output_dir, f"ckpt_best_f1_{tag_str}.pt")
            torch.save(model.state_dict(), hist_path)
            print(f"    New best val F1={state['best_val_f1']:.4f} → {best_f1_path} ({tag_str})")

        # ── best recall@K ────────────────────────────────────────────────────
        if val_rk is not None and val_rk > state["best_recall_k"]:
            state["best_recall_k"] = val_rk
            improved_anything = True
            best_rk_path = os.path.join(args.output_dir, "ckpt_best_recall_k.pt")
            torch.save(model.state_dict(), best_rk_path)
            tag_str = f"ep{epoch}" if step is None else f"ep{epoch}_s{step}"
            hist_rk_path = os.path.join(args.output_dir, f"ckpt_best_rk_{tag_str}.pt")
            torch.save(model.state_dict(), hist_rk_path)
            print(f"    New best recall@512={state['best_recall_k']:.4f} → {best_rk_path} ({tag_str})")

        # ── Patience tracking ────────────────────────────────────────────────
        if val_f1 is not None or val_rk is not None:
            if improved_anything:
                state["no_improve_vals"] = 0
            else:
                state["no_improve_vals"] += 1
                if patience_vals > 0:
                    print(f"    No val improvement: {state['no_improve_vals']}/{patience_vals}")

            # Early stop by val-count (per-step granularity)
            if patience_vals > 0 and state["no_improve_vals"] >= patience_vals:
                print(
                    f"\n[Early stop] No val improvement for {patience_vals} consecutive vals. "
                    f"Best val_f1={state['best_val_f1']:.4f}, "
                    f"best_recall@K={state['best_recall_k']:.4f}."
                )
                state["early_stop"] = True

        return val_f1, val_rk

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            hidden    = batch["hidden_state"].to(device)    # [B, 4096]
            positions = batch["positions"].to(device)       # [B]
            compk     = batch["selected_compk"].to(device)  # [B, N, 132]
            labels    = batch["labels"].to(device)           # [B, N]
            mask      = batch["mask"].to(device)             # [B, N]
            weights   = batch["weights"].to(device)          # [B, N]
            lei       = batch["layer_embed_idx"].to(device)  # [B]

            # ── Loss 计算 ──────────────────────────────────────────────────
            # bf16 autocast: ONLY model forward (matmul/Linear/einsum 用 bf16)
            # Loss computation (BCE/logsigmoid/focal) MUST be in fp32:
            #   - F.binary_cross_entropy 在 autocast 下会 raise RuntimeError
            #   - logsigmoid/focal 数值敏感，fp32 更稳
            amp_ctx = (torch.cuda.amp.autocast(dtype=torch.bfloat16)
                       if args.bf16 else nullcontext())
            with amp_ctx:
                if args.pairwise:
                    logits = model(hidden, compk, positions, layer_embed_idx=lei,
                                   return_logits=True)           # [B, N] bf16 if --bf16
                else:
                    scores = model(hidden, compk, positions, layer_embed_idx=lei)

            if args.pairwise:
                # 关键: 退出 autocast 后 cast 到 fp32, logsigmoid/focal 数值稳定
                logits = logits.float()
                B_cur, N_cur = logits.shape

                # 分块处理 batch 维度避免 OOM (B*N*N 可能 ~2GB)
                chunk_size = 64
                pos_mask_2d = (labels == 1) & mask              # [B, N]
                neg_mask_2d = (labels == 0) & mask              # [B, N]

                loss_sum = torch.tensor(0.0, device=logits.device)
                count_sum = torch.tensor(0.0, device=logits.device)
                for cs in range(0, B_cur, chunk_size):
                    ce = min(cs + chunk_size, B_cur)
                    logits_c = logits[cs:ce]
                    diff_c = logits_c.unsqueeze(2) - logits_c.unsqueeze(1)  # [chunk, N, N]
                    pair_mask_c = pos_mask_2d[cs:ce].unsqueeze(2) & \
                                  neg_mask_2d[cs:ce].unsqueeze(1)            # [chunk, N, N]
                    if args.pairwise_loss == "bpr":
                        elem_c = -F.logsigmoid(diff_c)                       # [chunk, N, N]
                    else:
                        elem_c = F.relu(args.margin - diff_c)                # [chunk, N, N]
                    pair_mask_c_f = pair_mask_c.float()
                    loss_sum = loss_sum + (elem_c * pair_mask_c_f).sum()
                    count_sum = count_sum + pair_mask_c_f.sum()

                if count_sum.item() < 1.0:
                    # 极端情况: batch 中无有效 pair → 跳过
                    continue
                loss = loss_sum / count_sum
            else:
                # Pointwise loss (BCE / Focal) — fp32 only, 已在 autocast 外
                scores = scores.float()
                if args.focal_loss:
                    # Focal Loss: -(1-p_t)^gamma * log(p_t), gamma=2
                    p = scores[mask]
                    t = labels[mask]
                    w = weights[mask]
                    # Label smoothing
                    eps = args.label_smoothing
                    if eps > 0:
                        t = t * (1 - 2*eps) + eps
                    p_t = p * t + (1 - p) * (1 - t)
                    focal_weight = (1 - p_t) ** 2  # gamma=2
                    bce = F.binary_cross_entropy(p, t, reduction='none')
                    loss = (bce * focal_weight * w).mean()
                else:
                    eps = args.label_smoothing
                    t_smooth = labels[mask] * (1 - 2*eps) + eps if eps > 0 else labels[mask]
                    loss = F.binary_cross_entropy(scores[mask], t_smooth,
                                                 weight=weights[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % args.log_interval == 0:
                n_pos  = labels[mask].sum().item()
                n_neg  = mask.sum().item() - n_pos
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[epoch {epoch:3d} | step {step:5d}/{len(loader)}]  "
                    f"loss={loss.item():.4f}  "
                    f"pos={int(n_pos)}  neg={int(n_neg)}  "
                    f"lr={lr_now:.2e}"
                )

            # ── Per-step validation (mid-epoch) ────────────────────────────
            if (args.val_every_steps > 0 and step > 0
                    and step % args.val_every_steps == 0):
                _run_val_and_save(epoch, step)
                if state["early_stop"]:
                    break

        if state["early_stop"]:
            break

        avg_loss = epoch_loss / len(loader)
        print(f"\n>>> Epoch {epoch} done.  avg_loss={avg_loss:.4f}")

        # ── Val F1 evaluation (end of epoch) ────────────────────────────────
        prev_best_f1 = state["best_val_f1"]
        val_f1, val_rk = _run_val_and_save(epoch, None)

        # legacy patience (by epoch) — track separately for backward compat
        if val_f1 is not None and val_f1 > prev_best_f1:
            state["no_improve_cnt"] = 0
        elif val_f1 is not None:
            state["no_improve_cnt"] += 1
            if patience > 0:
                print(f"    No improvement for {state['no_improve_cnt']}/{patience} epochs.")
        print()

        # ── 保存 epoch checkpoint (full state) ──────────────────────────────
        ckpt_path = os.path.join(args.output_dir, f"ckpt_epoch{epoch:03d}.pt")
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "avg_loss":     avg_loss,
            "best_val_f1":  state["best_val_f1"],
            "args":         vars(args),
        }, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "ckpt_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model (loss) → {best_path}")

        # ── Early stopping (epoch-based, legacy) ────────────────────────────
        if patience > 0 and state["no_improve_cnt"] >= patience:
            print(f"\n[Early stop] val_f1 did not improve for {patience} epochs. "
                  f"Best val_f1={state['best_val_f1']:.4f} at a previous epoch.")
            break

        # Per-step early stop also exits outer loop
        if state["early_stop"]:
            break

    print(
        f"\nTraining complete. Best avg_loss={best_loss:.4f}  "
        f"best_val_f1={state['best_val_f1']:.4f}  best_recall@512={state['best_recall_k']:.4f}"
    )


# ── 命令行入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lightning Indexer 训练")

    # 数据 & 权重
    parser.add_argument("--data-config",     default=None,
                        choices=list(DATA_CONFIGS.keys()),
                        help=(
                            "多目录数据配置。"
                            f"可选: {list(DATA_CONFIGS.keys())}。"
                            "legacy=旧数据, new_only=4个新数据集, combined=旧+新"
                        ))
    parser.add_argument("--weight-dir",      default="./weights", help="预训练权重目录（safetensors）")
    parser.add_argument("--output-dir",      default="./checkpoints", help="checkpoint 保存目录")
    parser.add_argument("--cache-size",      type=int, default=4096,
                        help="joint 数据集 LRU 缓存的文档数；设为 >= train doc 数则全量常驻内存 "
                             "(避免每 batch 重读 pkl 造成的磁盘抖动)")
    parser.add_argument("--bucketed",        action="store_true",
                        help="启用 length-bucketed batch sampler：按文档长度分桶，短文档大 batch、"
                             "长文档小 batch，避免超长文档的 padding 撑爆内存")
    parser.add_argument("--bucket-config",   type=str, default=None,
                        help="自定义 bucket 配置，格式: max_blocks1:bs1,max_blocks2:bs2,... "
                             "(如 16000:256,32000:128,64000:64,inf:32)。默认使用 DEFAULT_BUCKET_CONFIG")
    parser.add_argument("--max-position",    type=int, default=0,
                        help="RoPE freqs_cis 表大小 (覆盖 train+val 的最大 position)。0=从 train 数据"
                             "自动推算 (floor 131072)；含超长文档 (combined_8layers_full, 最大 ~460k) "
                             "时务必显式设大值如 524288, 否则 val 阶段 position 越界 → CUDA assert")
    parser.add_argument("--no-pretrain",     action="store_true", help="不加载预训练权重，从随机初始化训练")
    parser.add_argument("--val-fullset",     action="store_true",
                        help="用全集候选（~7900+ blocks）替代 dataloader 评测 val F1，"
                             "支持多目录；推荐搭配 --data-config 使用")
    parser.add_argument("--val-small",       action="store_true",
                        help="使用 1/5 val 子集加速评测（11 docs vs 57 docs）；"
                             "需搭配 --val-fullset 使用")
    parser.add_argument("--layer",           type=int, default=None,
                        help="单层 CSA layer index (10/12/20)")
    parser.add_argument("--joint-layers",    type=str, default=None,
                        help="R13: 同时训练多层 retriever (各层独立权重, 共享 labels), "
                             "格式 '10,12,20'. 训练时 score-level mean ensemble 为 val 信号. "
                             "与 --layer 互斥.")

    # 模型架构
    parser.add_argument("--n-heads",         type=int, default=64,
                        help="query/key head 数量（默认 64；128 需配合 --no-pretrain）")
    parser.add_argument("--q-lora-rank",     type=int, default=None,
                        help="Q LoRA rank（默认 None=1024，与预训练一致；可设 2048 等扩容）")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing eps: y=1 → 1-eps, y=0 → eps. 限制 logit 极端值,"
                             "防止过度极化。典型值 0.05 或 0.1。默认 0=关闭。")

    # 数据采样
    parser.add_argument("--sample-interval", type=int, default=1,
                        help="每隔几个 token 产生一个训练样本（默认 1 = 全部）")
    parser.add_argument("--label-interval",  type=int, default=64,
                        help="正例标签窗口大小（默认 64）")
    parser.add_argument("--max-pos",         type=int, default=512,
                        help="单个样本最大正例数，超出则随机子采样（默认 512）")

    # 训练超参
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--patience",        type=int,   default=5,
                        help="early stopping patience (epochs without val_f1 improvement); 0=disabled")
    parser.add_argument("--val-every-steps", type=int,   default=0,
                        help="额外每 N steps 做一次 val（mid-epoch）；0=只在 epoch 末做")
    parser.add_argument("--patience-vals",   type=int,   default=0,
                        help="early stop after N consecutive vals (val_f1 AND val_rk both没提升); 0=disabled。"
                             "适合搭配 --val-every-steps 使用，控制 step 级早停。")
    parser.add_argument("--bf16",            action="store_true",
                        help="启用 bf16 混合精度 (autocast) — 仅 forward+loss，"
                             "backward/optimizer/eval 仍 fp32。BCE/sigmoid/RMSNorm 自动保 fp32 数值稳定。")
    parser.add_argument("--batch-size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--wd",              type=float, default=1e-2, help="weight decay")
    parser.add_argument("--neg-ratio",       type=int,   default=1,
                        help="负例/正例比例（默认 1=等量；设为 3 则负例 3 倍于正例）")
    parser.add_argument("--weighted-loss",   action="store_true",
                        help="使用 label_scores 加权 BCE loss（高分 chunk 权重更大）")
    parser.add_argument("--focal-loss",      action="store_true",
                        help="使用 Focal Loss 替代标准 BCE（gamma=2, 聚焦困难样本）")
    parser.add_argument("--pairwise",        action="store_true",
                        help="使用 pairwise ranking loss 替代 pointwise BCE")
    parser.add_argument("--pairwise-loss",   default="bpr", choices=["bpr", "margin"],
                        help="pairwise loss 类型: bpr (-log σ(s_pos-s_neg)) 或 margin (hinge)")
    parser.add_argument("--margin",          type=float, default=1.0,
                        help="margin ranking loss 的 margin 值（仅 --pairwise --pairwise-loss margin 时生效）")
    parser.add_argument("--resume-from",     default=None,
                        help="从指定 checkpoint (.pt) 加载模型权重继续训练（不加载 optimizer）")
    parser.add_argument("--log-interval",    type=int,   default=50,   help="每隔多少步打印一次 loss")
    parser.add_argument("--seed",            type=int,   default=42)

    # 系统
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--num-workers",     type=int,   default=0)

    args = parser.parse_args()
    print(vars(args))
    train(args)


if __name__ == "__main__":
    main()
