"""
IndexerDataset — 训练数据加载器
================================
每个样本对应一个 decode token，包含：
  - hidden_state [4096]  bf16    — 当前 token 的 hidden state
  - position     int64           — 当前 token 的绝对位置
  - selected_compk [2*n_pos, 132] uint8 — 正例+负例的 compressed K
  - labels       [2*n_pos]       float32 — 1 for 正例，0 for 负例
  - layer_embed_idx  int64       — CSA 层的 embedding 下标 (10→0, 12→1, 20→2)

正例定义：token t 的正例 = label_indices[t : t + label_interval) 的并集 (union)
负例定义：从不在正例集合中的 blocks 随机采样，数量与正例相同

两个 interval 参数：
  sample_interval  — 每隔几步取一个 token 作为训练样本
  label_interval   — 正例标签窗口大小（默认 64）

内存管理（懒加载 + LRU 缓存）：
  __init__ 只加载轻量字段（label_pointers / label_indices / positions），
  大字段（hidden / compk）按需从磁盘加载，由 LRU 缓存控制内存上限。
  cache_size 控制最多在内存中保留多少个文档的重字段；
  设置为超过文档总数时行为与旧版全量加载等价（无额外 I/O）。
"""

import os
import glob
import pickle
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# CSA layer index → layer embedding index（用于层条件化 retriever）
LAYER_EMBED_MAP = {10: 0, 12: 1, 20: 2}


class IndexerDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        csa_layer_idx: int,
        sample_interval: int = 1,
        label_interval: int = 64,
        max_pos: int = 512,      # 最多取多少个正例（超过则随机子采样）
        seed: int = 42,
        doc_ids: list = None,    # 若指定，只加载对应编号的文档（支持 0-based 和 1-based）
        cache_size: int = 4096,  # LRU 缓存容量（文档数）；设为大于训练集 doc 数即等价于全量加载
        neg_ratio: int = 1,      # 负例/正例比例（默认 1:1；设为 3 则 3:1）
        weighted_loss: bool = False,  # 是否返回 label_scores 用于加权 BCE
    ):
        """
        Args:
            data_dir:        pkl 文件所在目录（doc_00001.pkl ... 或 doc_00000.pkl ...）
            csa_layer_idx:   CSA 层索引（10/12/20），用于确定从 pkl 里读哪一层
            sample_interval: 步进间隔，每隔多少个 token 产生一个训练样本
            label_interval:  正例窗口大小，token t 的正例 = [t, t+label_interval) 的 label 并集
            max_pos:         单个样本最大正例数，超出则随机采样
            seed:            随机种子（用于负例采样）
            doc_ids:         若为 None 则加载全部文档；否则只加载指定编号的文档
            cache_size:      LRU 缓存最多保留多少个文档的重字段（hidden/compk）。
                             设为 len(doc_ids) 或更大时等价于旧版全量加载。
        """
        self.label_interval = label_interval
        self.max_pos = max_pos
        self.neg_ratio = neg_ratio
        self.weighted_loss = weighted_loss
        self.rng = np.random.default_rng(seed)
        self.csa_layer_idx = csa_layer_idx
        self._cache_size = cache_size
        self._cache: OrderedDict = OrderedDict()  # doc_id → {"hidden": tensor, "compk": tensor}

        layer_id = csa_layer_idx

        # ── 轻量加载：读取所有文档的小字段，重字段（hidden/compk）立即释放 ─────
        self.doc_meta = []   # list of dict; 永驻内存（字段均为小数组）
        pkl_paths = sorted(glob.glob(os.path.join(data_dir, "doc_*.pkl")))

        if doc_ids is not None:
            allowed = {f"doc_{i:05d}.pkl" for i in doc_ids}
            pkl_paths = [p for p in pkl_paths if os.path.basename(p) in allowed]

        assert len(pkl_paths) > 0, f"在 {data_dir} 下未找到任何 doc_*.pkl 文件"

        split_desc = f"doc_ids={doc_ids[0]}–{doc_ids[-1]}" if doc_ids is not None else "all"
        print(f"Loading {len(pkl_paths)} documents (CSA layer {csa_layer_idx}, split={split_desc}) ...")

        for pkl_path in pkl_paths:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            # 只保留小字段；positions 用于 train.py 推算 max_position（保留向后兼容）
            meta_entry = {
                "pkl_path":       pkl_path,
                "positions":      data[f"positions_layer_{layer_id}"],   # [n_decode] int64
                "label_pointers": data["label_pointers"].numpy(),         # [n_decode + 1]
                "label_indices":  data["label_indices"].numpy(),          # [total_selected]
            }
            # label_scores 仅在 weighted_loss 模式下保留（节约内存）
            if weighted_loss and "label_scores" in data:
                meta_entry["label_scores"] = data["label_scores"].numpy()  # [total_selected]
            self.doc_meta.append(meta_entry)
            del data  # ← 显式释放 hidden/compk 等重字段，GC 可立即回收

        # ── 构建 per-doc block 频率（用于 trivial block 降权）────────────────
        # block_freq[doc_id][block_id] = 出现在多少比例的 token 中 (0~1)
        # 频率 > 0.8 的 block 视为 "trivial"，正例权重降为 0.5
        self._block_freq = []
        for meta in self.doc_meta:
            ptrs = meta["label_pointers"]
            idxs = meta["label_indices"]
            n_decode = ptrs.shape[0] - 1
            if n_decode == 0:
                self._block_freq.append({})
                continue
            freq = {}
            for t in range(n_decode):
                for b in idxs[ptrs[t]:ptrs[t+1]]:
                    freq[int(b)] = freq.get(int(b), 0) + 1
            # Normalize to ratio
            self._block_freq.append({b: c / n_decode for b, c in freq.items()})

        # ── 构建样本索引 (doc_id, token_idx) ─────────────────────────────────
        # label_pointers.shape[0] - 1 == n_decode（等价于旧版 hidden.shape[0]）
        self.samples = []
        for doc_id, meta in enumerate(self.doc_meta):
            n_decode = meta["label_pointers"].shape[0] - 1
            ptrs = meta["label_pointers"]
            idxs = meta["label_indices"]

            for t in range(0, n_decode, sample_interval):
                t_end = min(t + label_interval, n_decode)
                flat = idxs[ptrs[t]:ptrs[t_end]]
                if len(flat) > 0:
                    self.samples.append((doc_id, t))

        print(f"  → {len(self.samples)} training samples "
              f"(sample_interval={sample_interval}, label_interval={label_interval})")

    # ── 向后兼容接口 ──────────────────────────────────────────────────────────
    @property
    def docs(self):
        """
        返回 doc_meta 列表，保持与旧版 self.docs 相同的访问接口。
        train.py 中唯一用途：doc["positions"].max().item() 推算 max_position。
        """
        return self.doc_meta

    # ── LRU 缓存管理 ──────────────────────────────────────────────────────────
    def _get_heavy(self, doc_id: int) -> dict:
        """
        返回 doc_id 对应的重字段（hidden / compk）。
        命中时 O(1)；未命中时从磁盘加载并插入缓存，超容量时逐出最久未用条目。
        num_workers > 0 时每个 worker 进程拥有独立缓存，天然线程安全。
        """
        if doc_id in self._cache:
            self._cache.move_to_end(doc_id)
            return self._cache[doc_id]

        # 超容量：逐出 LRU（最久未用）
        while len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

        # 从磁盘加载
        pkl_path = self.doc_meta[doc_id]["pkl_path"]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        layer_id = self.csa_layer_idx
        heavy = {
            "hidden": data[f"hidden_layer_{layer_id}"],   # [n_decode, 4096] bf16
            "compk":  data[f"compk_layer_{layer_id}"],    # [n_blocks, 132] uint8
        }
        self._cache[doc_id] = heavy
        return heavy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        doc_id, t = self.samples[idx]
        meta  = self.doc_meta[doc_id]
        heavy = self._get_heavy(doc_id)

        hidden   = heavy["hidden"][t]       # [4096] bf16
        compk    = heavy["compk"]           # [n_blocks, 132] uint8
        n_blocks = compk.shape[0]
        ptrs     = meta["label_pointers"]
        idxs     = meta["label_indices"]
        n_decode = ptrs.shape[0] - 1       # 等价于旧版 doc["hidden"].shape[0]

        # ── 正例：[t, t+label_interval) 窗口内 label 的并集 ─────────────────
        t_end   = min(t + self.label_interval, n_decode)
        pos_all = np.unique(idxs[ptrs[t]:ptrs[t_end]])  # all true positives
        # 过滤越界 block 索引（部分 under128 数据 label_indices > compk.shape[0]）
        pos_all = pos_all[pos_all < n_blocks]

        if len(pos_all) == 0:
            # 全部 label 越界 — 随机选一个 block 当伪正例，避免空 batch
            pos_all = np.array([self.rng.integers(n_blocks)])

        if len(pos_all) > self.max_pos:
            pos_idx = self.rng.choice(pos_all, self.max_pos, replace=False)
        else:
            pos_idx = pos_all.copy()
        pos_idx = pos_idx.tolist()
        n_pos = len(pos_idx)

        # ── 负例：从正例之外随机采样 neg_ratio 倍的 blocks ─────────────────
        # 排除所有真实正例（pos_all），防止真实正例混入负例池导致 label noise
        neg_pool = np.setdiff1d(np.arange(n_blocks), pos_all)
        n_neg = min(n_pos * self.neg_ratio, len(neg_pool))

        neg_idx = self.rng.choice(neg_pool, n_neg, replace=False).tolist()
        n_neg = len(neg_idx)

        # ── 组合返回 ────────────────────────────────────────────────────────
        all_idx        = pos_idx + neg_idx
        selected_compk = compk[all_idx]   # [n_pos + n_neg, 132] uint8

        labels = torch.cat([
            torch.ones(n_pos,  dtype=torch.float32),
            torch.zeros(n_neg, dtype=torch.float32),
        ])  # [n_pos + n_neg]

        # ── 正例权重计算 ───────────────────────────────────────────────────
        # 基础权重：1.0；weighted_loss 时用 label_scores 归一化
        pos_weights = np.ones(n_pos, dtype=np.float32)
        if self.weighted_loss:
            scores_all = meta["label_scores"]
            pos_set = {pb: pi for pi, pb in enumerate(pos_idx)}
            pos_scores_raw = np.zeros(n_pos, dtype=np.float32)
            for tt in range(t, t_end):
                chunk_idxs = idxs[ptrs[tt]:ptrs[tt+1]]
                chunk_scores = scores_all[ptrs[tt]:ptrs[tt+1]]
                for ci, cs in zip(chunk_idxs.tolist(), chunk_scores.tolist()):
                    if ci in pos_set:
                        idx = pos_set[ci]
                        if cs > pos_scores_raw[idx]:
                            pos_scores_raw[idx] = cs
            # normalize to [0.5, 2.0] range: score=3→0.5, score=21→2.0
            pos_weights = 0.5 + 1.5 * (pos_scores_raw - 3.0) / 18.0

        # trivial block 降权（始终生效）：频率>0.8 的 block 权重降为 0.5
        block_freq = self._block_freq[doc_id]
        for pi, pb in enumerate(pos_idx):
            if block_freq.get(pb, 0) > 0.8:
                pos_weights[pi] = 0.5

        weights = torch.cat([
            torch.from_numpy(pos_weights),
            torch.ones(n_neg, dtype=torch.float32),
        ])

        return {
            "hidden_state":    hidden,                                              # [4096]
            "position":        meta["positions"][t].clone().detach().to(torch.int64),  # scalar
            "selected_compk":  selected_compk,                                     # [n_pos+n_neg, 132]
            "labels":          labels,                                             # [n_pos+n_neg]
            "weights":         weights,                                            # [n_pos+n_neg]
            "layer_embed_idx": torch.tensor(
                LAYER_EMBED_MAP.get(self.csa_layer_idx, 0), dtype=torch.long
            ),  # scalar: 0/1/2 for layer 10/12/20
        }


def collate_fn(batch: list) -> dict:
    """
    将一个 batch 内长度不等的 selected_compk / labels 用 0 填充至最大长度，
    并返回一个 mask 供 loss 计算时过滤填充位置。
    """
    hidden    = torch.stack([b["hidden_state"] for b in batch])   # [B, 4096]
    positions = torch.stack([b["position"]     for b in batch])   # [B]
    layer_embed_idx = torch.stack([b["layer_embed_idx"] for b in batch])  # [B]

    max_n = max(b["selected_compk"].shape[0] for b in batch)
    B     = len(batch)

    selected_compk = torch.zeros(B, max_n, 132, dtype=torch.uint8)
    labels         = torch.zeros(B, max_n,      dtype=torch.float32)
    weights        = torch.zeros(B, max_n,      dtype=torch.float32)
    mask           = torch.zeros(B, max_n,      dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["selected_compk"].shape[0]
        selected_compk[i, :n] = b["selected_compk"]
        labels[i, :n]         = b["labels"]
        weights[i, :n]        = b["weights"]
        mask[i, :n]           = True

    return {
        "hidden_state":    hidden,           # [B, 4096]
        "positions":       positions,        # [B]
        "selected_compk":  selected_compk,   # [B, max_n, 132]
        "labels":          labels,           # [B, max_n]
        "weights":         weights,          # [B, max_n]  loss 权重
        "mask":            mask,             # [B, max_n]  True = valid
        "layer_embed_idx": layer_embed_idx,  # [B]  0/1/2
    }


def build_dataloader(
    data_dir: str,
    csa_layer_idx: int,
    batch_size: int = 8,
    sample_interval: int = 1,
    label_interval: int = 64,
    max_pos: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    doc_ids: list = None,
    cache_size: int = 4096,
) -> DataLoader:
    dataset = IndexerDataset(
        data_dir=data_dir,
        csa_layer_idx=csa_layer_idx,
        sample_interval=sample_interval,
        label_interval=label_interval,
        max_pos=max_pos,
        seed=seed,
        doc_ids=doc_ids,
        cache_size=cache_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Combined multi-directory datasets
# ─────────────────────────────────────────────────────────────────────────────

class CombinedIndexerDataset(Dataset):
    """
    合并多个目录的 IndexerDataset，用于多数据源联合训练（单层模式）。

    Args:
        specs:           list of dicts, each with keys:
                           - "data_dir": str  — pkl 文件目录
                           - "doc_ids":  list or None  — 要加载的文档编号列表
        csa_layer_idx:   CSA 层索引（10/12/20）
        sample_interval: 步进间隔
        label_interval:  正例窗口大小
        max_pos:         单个样本最大正例数
        seed:            随机种子（各子数据集 seed+i 以保证独立性）
        cache_size:      每个子 IndexerDataset 的 LRU 缓存容量

    用法示例：
        specs = [
            {"data_dir": "./data",      "doc_ids": list(range(1, 71))},
            {"data_dir": "/path/mrcr",  "doc_ids": list(range(0, 56))},
        ]
        dataset = CombinedIndexerDataset(specs, csa_layer_idx=20)
    """
    def __init__(
        self,
        specs: list,
        csa_layer_idx: int,
        sample_interval: int = 1,
        label_interval: int = 64,
        max_pos: int = 512,
        seed: int = 42,
        cache_size: int = 4096,
        neg_ratio: int = 1,
        weighted_loss: bool = False,
    ):
        self.sub_datasets = []
        for i, spec in enumerate(specs):
            ds = IndexerDataset(
                data_dir=spec["data_dir"],
                csa_layer_idx=csa_layer_idx,
                sample_interval=sample_interval,
                label_interval=label_interval,
                max_pos=max_pos,
                seed=seed + i,
                doc_ids=spec.get("doc_ids"),
                cache_size=cache_size,
                neg_ratio=neg_ratio,
                weighted_loss=weighted_loss,
            )
            self.sub_datasets.append(ds)

        # 展平索引
        self.index_map = [
            (i, j)
            for i, ds in enumerate(self.sub_datasets)
            for j in range(len(ds))
        ]

        total = len(self.index_map)
        print(
            f"CombinedIndexerDataset: {len(specs)} dirs, "
            f"csa_layer={csa_layer_idx}, total={total} samples"
        )

    @property
    def docs(self):
        """返回所有子数据集的 docs 列表（用于 max_position 推算）"""
        all_docs = []
        for ds in self.sub_datasets:
            all_docs.extend(ds.docs)
        return all_docs

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict:
        sub_i, sample_j = self.index_map[idx]
        return self.sub_datasets[sub_i][sample_j]


def build_combined_dataloader(
    specs: list,
    csa_layer_idx: int,
    batch_size: int = 8,
    sample_interval: int = 1,
    label_interval: int = 64,
    max_pos: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    cache_size: int = 4096,
    neg_ratio: int = 1,
    weighted_loss: bool = False,
) -> DataLoader:
    """
    构建多目录单层训练 DataLoader。

    Args:
        specs: list of {"data_dir": str, "doc_ids": list or None}
        其余参数同 build_dataloader
    """
    dataset = CombinedIndexerDataset(
        specs=specs,
        csa_layer_idx=csa_layer_idx,
        sample_interval=sample_interval,
        label_interval=label_interval,
        max_pos=max_pos,
        seed=seed,
        cache_size=cache_size,
        neg_ratio=neg_ratio,
        weighted_loss=weighted_loss,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Joint multi-layer training: each sample carries N layers' hidden+compk,
# with SHARED labels. Used by R950-R952 joint training.
# ─────────────────────────────────────────────────────────────────────────────

class JointLayerIndexerDataset(Dataset):
    """
    单目录 joint 多层数据集：每个 sample 同时返回 3 层的 hidden + compk，
    labels/positions/golden_blocks 共享（一份）。

    与 IndexerDataset 区别:
      - heavy 缓存存 3 层的 hidden + compk
      - __getitem__ 返回 dict: hidden_l{lid}, compk_l{lid} for each lid in layer_ids,
        以及 shared labels/mask/weights/position
      - pos_idx/neg_idx 选择一次（基于共享 labels）, 各层共用同一组 block IDs
    """
    def __init__(
        self,
        data_dir: str,
        layer_ids: list,         # e.g. [10, 12, 20]
        sample_interval: int = 1,
        label_interval: int = 64,
        max_pos: int = 512,
        seed: int = 42,
        doc_ids: list = None,
        cache_size: int = 4096,
        neg_ratio: int = 1,
        weighted_loss: bool = False,
    ):
        self.layer_ids       = list(layer_ids)
        self.label_interval  = label_interval
        self.max_pos         = max_pos
        self.neg_ratio       = neg_ratio
        self.weighted_loss   = weighted_loss
        self.rng             = np.random.default_rng(seed)
        self._cache_size     = cache_size
        self._cache: OrderedDict = OrderedDict()

        # Light fields per doc
        self.doc_meta = []
        pkl_paths = sorted(glob.glob(os.path.join(data_dir, "doc_*.pkl")))
        if doc_ids is not None:
            allowed = {f"doc_{i:05d}.pkl" for i in doc_ids}
            pkl_paths = [p for p in pkl_paths if os.path.basename(p) in allowed]
        assert len(pkl_paths) > 0, f"在 {data_dir} 下未找到任何 doc_*.pkl 文件"

        split_desc = f"doc_ids={doc_ids[0]}–{doc_ids[-1]}" if doc_ids is not None else "all"
        print(f"Loading {len(pkl_paths)} documents (joint layers={layer_ids}, split={split_desc}) ...")

        for pkl_path in pkl_paths:
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"  [skip] corrupt/unreadable pkl {os.path.basename(pkl_path)}: {type(e).__name__}: {str(e)[:60]}")
                continue
            ref_lid = layer_ids[0]
            meta_entry = {
                "pkl_path":       pkl_path,
                # positions are shared across layers (verified)
                "positions":      data[f"positions_layer_{ref_lid}"],
                "label_pointers": data["label_pointers"].numpy(),
                "label_indices":  data["label_indices"].numpy(),
                "n_blocks":       data[f"compk_layer_{ref_lid}"].shape[0],
            }
            if weighted_loss and "label_scores" in data:
                meta_entry["label_scores"] = data["label_scores"].numpy()
            self.doc_meta.append(meta_entry)
            del data

        # block frequency for trivial downweight (per doc)
        self._block_freq = []
        for meta in self.doc_meta:
            ptrs = meta["label_pointers"]
            idxs = meta["label_indices"]
            n_decode = ptrs.shape[0] - 1
            if n_decode == 0:
                self._block_freq.append({}); continue
            freq = {}
            for t in range(n_decode):
                for b in idxs[ptrs[t]:ptrs[t+1]]:
                    freq[int(b)] = freq.get(int(b), 0) + 1
            self._block_freq.append({b: c / n_decode for b, c in freq.items()})

        # Build (doc_id, t) sample index
        self.samples = []
        for doc_id, meta in enumerate(self.doc_meta):
            n_decode = meta["label_pointers"].shape[0] - 1
            ptrs = meta["label_pointers"]
            idxs = meta["label_indices"]
            for t in range(0, n_decode, sample_interval):
                t_end = min(t + label_interval, n_decode)
                flat = idxs[ptrs[t]:ptrs[t_end]]
                if len(flat) > 0:
                    self.samples.append((doc_id, t))
        print(f"  → {len(self.samples)} training samples "
              f"(sample_interval={sample_interval}, label_interval={label_interval})")

    @property
    def docs(self):
        return self.doc_meta

    def _get_heavy(self, doc_id: int) -> dict:
        if doc_id in self._cache:
            self._cache.move_to_end(doc_id)
            return self._cache[doc_id]
        while len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        pkl_path = self.doc_meta[doc_id]["pkl_path"]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        heavy = {}
        for lid in self.layer_ids:
            heavy[f"hidden_l{lid}"] = data[f"hidden_layer_{lid}"]
            heavy[f"compk_l{lid}"]  = data[f"compk_layer_{lid}"]
        self._cache[doc_id] = heavy
        return heavy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        doc_id, t = self.samples[idx]
        meta  = self.doc_meta[doc_id]
        heavy = self._get_heavy(doc_id)

        ref_lid  = self.layer_ids[0]
        n_blocks = heavy[f"compk_l{ref_lid}"].shape[0]
        ptrs     = meta["label_pointers"]
        idxs     = meta["label_indices"]
        n_decode = ptrs.shape[0] - 1

        # ── Positives (shared across layers) ─────────────────────────────────
        t_end   = min(t + self.label_interval, n_decode)
        pos_all = np.unique(idxs[ptrs[t]:ptrs[t_end]])
        pos_all = pos_all[pos_all < n_blocks]
        if len(pos_all) == 0:
            pos_all = np.array([self.rng.integers(n_blocks)])
        if len(pos_all) > self.max_pos:
            pos_idx = self.rng.choice(pos_all, self.max_pos, replace=False).tolist()
        else:
            pos_idx = pos_all.tolist()
        n_pos = len(pos_idx)

        # ── Negatives (shared selection) ─────────────────────────────────────
        neg_pool = np.setdiff1d(np.arange(n_blocks), pos_all)
        n_neg = min(n_pos * self.neg_ratio, len(neg_pool))

        neg_idx = self.rng.choice(neg_pool, n_neg, replace=False).tolist()
        n_neg = len(neg_idx)
        all_idx = pos_idx + neg_idx

        # Per-layer compk (same indices, different content)
        out = {}
        for lid in self.layer_ids:
            out[f"hidden_l{lid}"] = heavy[f"hidden_l{lid}"][t]                     # [4096]
            out[f"compk_l{lid}"]  = heavy[f"compk_l{lid}"][all_idx]                # [n_pos+n_neg, 132]

        # Shared scalars/labels
        out["position"] = meta["positions"][t].clone().detach().to(torch.int64)
        labels = torch.cat([
            torch.ones(n_pos,  dtype=torch.float32),
            torch.zeros(n_neg, dtype=torch.float32),
        ])
        # Weights: same logic as IndexerDataset
        pos_weights = np.ones(n_pos, dtype=np.float32)
        if self.weighted_loss:
            scores_all = meta["label_scores"]
            pos_set = {pb: pi for pi, pb in enumerate(pos_idx)}
            pos_scores_raw = np.zeros(n_pos, dtype=np.float32)
            for tt in range(t, t_end):
                chunk_idxs = idxs[ptrs[tt]:ptrs[tt+1]]
                chunk_scores = scores_all[ptrs[tt]:ptrs[tt+1]]
                for ci, cs in zip(chunk_idxs.tolist(), chunk_scores.tolist()):
                    if ci in pos_set:
                        ipos = pos_set[ci]
                        if cs > pos_scores_raw[ipos]:
                            pos_scores_raw[ipos] = cs
            pos_weights = 0.5 + 1.5 * (pos_scores_raw - 3.0) / 18.0
        block_freq = self._block_freq[doc_id]
        for pi, pb in enumerate(pos_idx):
            if block_freq.get(pb, 0) > 0.8:
                pos_weights[pi] = 0.5
        weights = torch.cat([
            torch.from_numpy(pos_weights),
            torch.ones(n_neg, dtype=torch.float32),
        ])
        out["labels"]  = labels
        out["weights"] = weights
        return out


def joint_collate_fn(layer_ids: list):
    """Returns a collate_fn for joint multi-layer batches."""
    def _collate(batch: list) -> dict:
        hidden = {lid: torch.stack([b[f"hidden_l{lid}"] for b in batch]) for lid in layer_ids}
        positions = torch.stack([b["position"] for b in batch])
        max_n = max(b[f"compk_l{layer_ids[0]}"].shape[0] for b in batch)
        B = len(batch)

        compk = {lid: torch.zeros(B, max_n, 132, dtype=torch.uint8) for lid in layer_ids}
        labels  = torch.zeros(B, max_n, dtype=torch.float32)
        weights = torch.zeros(B, max_n, dtype=torch.float32)
        mask    = torch.zeros(B, max_n, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b[f"compk_l{layer_ids[0]}"].shape[0]
            # Invariant: all layers of the same sample share the same block count
            # (CSA chunks are token-aligned, layer-independent). Padding length is
            # derived from layer_ids[0] only, so a cross-layer mismatch would
            # silently zero-pad / truncate other layers. Assert instead of corrupt.
            for lid in layer_ids:
                assert b[f"compk_l{lid}"].shape[0] == n, (
                    f"joint_collate cross-layer n_blocks mismatch: "
                    f"{[(l, b[f'compk_l{l}'].shape[0]) for l in layer_ids]}"
                )
            for lid in layer_ids:
                compk[lid][i, :n] = b[f"compk_l{lid}"]
            labels[i, :n]  = b["labels"]
            weights[i, :n] = b["weights"]
            mask[i, :n]    = True
        return {
            "hidden":    hidden,    # dict {lid: [B, 4096]}
            "positions": positions, # [B]
            "compk":     compk,     # dict {lid: [B, max_n, 132]}
            "labels":    labels,    # [B, max_n]
            "weights":   weights,   # [B, max_n]
            "mask":      mask,      # [B, max_n]
        }
    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Length-bucketed Batch Sampler (2026-06-04)
#   Groups samples by document length → variable batch_size per bucket.
#   Eliminates the massive padding cost when ultra-long docs (>256K tokens)
#   share a batch with short docs.
# ─────────────────────────────────────────────────────────────────────────────

# (max_c4_blocks, batch_size) — c4 block = 4 tokens
DEFAULT_BUCKET_CONFIG = [
    (16000,  256),   #  < 64K tokens
    (32000,  128),   #  64K–128K tokens
    (64000,   64),   # 128K–256K tokens
    (float("inf"), 32),   # 256K+ tokens
]


class LengthBucketedBatchSampler:
    """Sort samples by document length, then create **fixed-size** batches.
    Samples within a batch have similar lengths → minimal collate padding.
    All batches have identical size → NCCL-safe for DDP.

    DDP: all ranks sort the same global data (deterministic), then each rank
    takes every ``world_size``-th batch."""

    def __init__(self, sample_n_blocks, batch_size=256,
                 shuffle=True, seed=42, drop_last=True,
                 rank=0, world_size=1):
        self.sample_n_blocks = sample_n_blocks
        self.batch_size      = batch_size  # FIXED — NCCL requires uniform shapes
        self.shuffle         = shuffle
        self.seed            = seed
        self.drop_last       = drop_last
        self.rank            = rank
        self.world_size      = world_size
        self._epoch          = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        # 1. Sort ALL samples by doc n_blocks → similar-length neighbours
        indices = sorted(range(len(self.sample_n_blocks)),
                         key=lambda i: self.sample_n_blocks[i])

        # 2. Create fixed-size batches (every batch = batch_size)
        all_batches = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                all_batches.append(batch)

        # 3. Shuffle batch order (not within-batch — that breaks length grouping)
        if self.shuffle:
            rng.shuffle(all_batches)

        # 4. DDP: each rank takes every world_size-th batch
        #    Truncate to multiple of world_size for equal counts
        n_full = (len(all_batches) // self.world_size) * self.world_size
        all_batches = all_batches[:n_full]
        my_batches = all_batches[self.rank::self.world_size]
        yield from my_batches

    def __len__(self):
        n_total = len(self.sample_n_blocks)
        if n_total == 0:
            return 0
        total_batches = n_total // self.batch_size
        return max(1, total_batches // self.world_size)


class CombinedJointLayerDataset(Dataset):
    """合并多目录 joint 多层数据集（用于 multi-data-config 训练）"""
    def __init__(self, specs: list, layer_ids: list, **kwargs):
        self.sub_datasets = []
        for i, spec in enumerate(specs):
            seed_i = kwargs.pop("seed", 42) + i if i == 0 else kwargs.get("seed", 42) + i
            sub_kwargs = dict(kwargs)
            sub_kwargs["seed"] = seed_i
            ds = JointLayerIndexerDataset(
                data_dir=spec["data_dir"], layer_ids=layer_ids,
                doc_ids=spec.get("doc_ids"), **sub_kwargs,
            )
            self.sub_datasets.append(ds)
        self.layer_ids = list(layer_ids)
        self.index_map = [
            (i, j) for i, ds in enumerate(self.sub_datasets) for j in range(len(ds))
        ]
        self._sample_n_blocks = None  # lazy
        total = len(self.index_map)
        print(f"CombinedJointLayerDataset: {len(specs)} dirs, layers={layer_ids}, total={total} samples")

    @property
    def docs(self):
        all_docs = []
        for ds in self.sub_datasets:
            all_docs.extend(ds.docs)
        return all_docs

    @property
    def sample_n_blocks(self):
        """Lazily build per-sample n_blocks list (for bucketed sampler)."""
        if self._sample_n_blocks is None:
            self._sample_n_blocks = []
            for sub_i, sample_j in self.index_map:
                ds = self.sub_datasets[sub_i]
                doc_id, _t = ds.samples[sample_j]
                self._sample_n_blocks.append(ds.doc_meta[doc_id]["n_blocks"])
        return self._sample_n_blocks

    def __len__(self): return len(self.index_map)
    def __getitem__(self, idx: int) -> dict:
        sub_i, sample_j = self.index_map[idx]
        return self.sub_datasets[sub_i][sample_j]


def build_joint_combined_dataloader(
    specs: list, layer_ids: list,
    batch_size: int = 8, sample_interval: int = 1, label_interval: int = 64,
    max_pos: int = 512, shuffle: bool = True, num_workers: int = 0,
    seed: int = 42, cache_size: int = 4096,
    neg_ratio: int = 1, weighted_loss: bool = False,
    sampler=None,   # DDP: pass DistributedSampler; if not None, shuffle param is ignored
    bucketed: bool = False,  # length-bucketed variable batch_size
    bucket_config: list = None,
) -> DataLoader:
    """构建 joint 多层多目录 DataLoader (R950+).

    DDP usage:
        sampler = DistributedSampler(dataset, ...) — set sampler= to enable
        Note: when sampler is set, DataLoader shuffle MUST be False.

    Bucketed mode (--bucketed):
        Sorts samples by doc length, then creates fixed-size batches.
        Samples within a batch have similar lengths → minimal padding.
        All batches have identical size → NCCL-safe for DDP.
    """
    dataset = CombinedJointLayerDataset(
        specs=specs, layer_ids=layer_ids,
        sample_interval=sample_interval, label_interval=label_interval,
        max_pos=max_pos, seed=seed, cache_size=cache_size,
        neg_ratio=neg_ratio, weighted_loss=weighted_loss,
    )
    if bucketed:
        batch_sampler = LengthBucketedBatchSampler(
            sample_n_blocks=dataset.sample_n_blocks,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
            rank=(sampler.rank if sampler is not None else 0),
            world_size=(sampler.num_replicas if sampler is not None else 1),
        )
        return DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=joint_collate_fn(layer_ids),
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    if sampler is not None:
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
            collate_fn=joint_collate_fn(layer_ids),
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=True,   # DDP: keep batch sizes equal
        )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=joint_collate_fn(layer_ids),
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
