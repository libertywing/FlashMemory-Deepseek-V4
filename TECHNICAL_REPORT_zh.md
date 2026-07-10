# 技术报告 — DeepSeek-V4 Memory-Indexer 推理

*两级 Memory-Indexer (two-level Memory-Indexer) retriever 推理系统的实现级文档。*

---

## 目录

1. [概述](#1-概述)
2. [架构背景](#2-架构背景)
3. [两级 Memory-Indexer 召回算法](#3-两级-memory-indexer-召回算法)
4. [推理数据流](#4-推理数据流)
5. [各项优化](#5-各项优化)
6. [实测结果](#6-实测结果)
7. [正确性验证](#7-正确性验证)
8. [局限与未来工作](#8-局限与未来工作)

---

## 1. 概述

DeepSeek-V4 (DSv4) 用 **CSA (Compressed Sparse Attention, 压缩稀疏注意力)** 服务长上下文。每个 decode token, 在全部 21 个 `c4` (compress_ratio = 4) 层上, 模型原生的 **Lightning Indexer** 会给全历史 (`ceil(context / 4)` 个 chunk) 里的*每一个* chunk 打分, 选出 top-512 做稀疏注意力。两项成本随上下文长度增长, 在 1M token 时占主导:

- **(A) 算力** — indexer 打分是 `O(全历史)`。在 1M 上下文下这是一次巨大的、逐 token、逐层的全量扫描。
- **(B) GPU 显存** — 全历史的 `c4` classical KV 必须物理常驻 GPU, 注意力才能读到它。由于 MLA KV 是**按 TP rank 复制 (replicated)** 而非分片 (sharded) 的 (见 §2), 加卡**并不能**增加 KV 容量, 所以单请求 KV 占用直接给并发设了一个硬顶。

我们的方案同时打击这两项, 且是**两条独立的路径**:

1. **省算** — 用一个训练好的 **Memory Indexer** (我们的 retriever) 替换原生的全历史打分。它每 64 个 decode step 跑一次, 产出一个小的**常驻集 (resident set)** (约占历史 ~10%, 实际是恒定的 6144 chunk)。之后 decode 只对常驻集打分, 而非全历史。这使 1M 下整模型 per-token FLOPs 从 118.9 GFLOP 降到 30.2 GFLOP (**0.25×**, 即省 75%)。
2. **省显存 → 拉并发** — 把未被召回的 chunk 的 KV 物理 **offload** 到 CPU pinned mirror, GPU 只留召回页在一个缩小的 "reserve" 里 (**Path-P**); 再额外把 18 个非-target 层的打分 key 池也 offload (**Path-I**)。单请求 GPU 占用骤降, 于是同样的 8×H20 能容纳数倍的并发长上下文请求。

**结果摘要 (8×H20, TP8, 跨机 PD, 2026-07):** 在 256K / 512K / 1M 上, 并发分别提升 **1.3× / 2.4× / 3.6×**, 稳态聚合吞吐分别提升 **1.05× / 1.49× / 2.60×**。上下文越长, 优势越大。所有优化都是 **env-gated (环境变量门控)** 的 —— 门控关闭时, server 与原生 DSv4 **字节一致 (byte-identical)**, 是零风险回退。

有两堵诚实的墙仍在 (§6, §8): `batch > 64` 的 in-graph CUDA crash 把 256K/512K 卡在并发 60, CPU-mirror 的**主机内存 (host-RAM)** 墙把 1M 卡在并发 40。1M 这堵墙是主机内存, 不是 GPU。

---

## 2. 架构背景

### 2.1 DSv4 维度 (来自 `config.json`, 事实基准)

| 字段 | 值 |
|---|---|
| hidden size | 4096 |
| 层数 | 共 43 = **3 dense** + **21 c4** (compress_ratio 4) + **20 c128** (compress_ratio 128) |
| MoE | 256 experts, top-6 (+1 shared), moe_inter 2048 |
| MLA heads | `num_attention_heads` = 64, `num_key_value_heads` = **1** (单个 MLA latent KV head) |
| `q_lora_rank` | 1024 |
| Indexer | `index_n_heads` = 64, `index_head_dim` = 128, `index_topk` = **512** |
| vocab | 129280 |
| paging | `page_size` = 256; `c4_page_size` = 64。**1 c4-chunk = 4 token; 1 c4-page = 64 chunk = 256 token。** |

### 2.2 CSA 与原生 Lightning Indexer

21 个 `c4` 层用 CSA。每个历史 chunk 压缩 4 个 token。对每个 decode token, 在每个 c4 层, 原生 Lightning Indexer 对历史里*每一个*压缩 chunk 打分, 保留该层稀疏注意力的 **top-512**。每层背后有两个池:

- **index-K** (`c4_indexer_kv_pool`) — Lightning-Indexer 的*打分 key*, **132 B/chunk × 21 层**。单 chunk 便宜, 但每步都要对全历史打分。
- **c4 classical KV** (`c4_kv_pool`) — top-512 选出后被读的*注意力* KV, **584 B/chunk**。这是贵的 KV。

### 2.3 并发物理事实: MLA KV 是复制而非分片的

这是并发最重要的物理约束。在张量并行下, `get_num_kv_heads(tp=8) == 1` —— MLA latent KV 是**在每个 TP rank 上复制 (replicated)**, 而非在它们之间切分。`nvidia-smi` 已证实: 8 张卡显存占用完全一致。后果:

- **加卡并不能倍增 KV 容量。** 每个 rank 都扛着整份 KV, 所以无论 TP 度数如何, 单请求 KV 占用都一样。这正是 baseline 无法靠加卡拉并发的原因。
- **权重确实分片。** 实测: TP8 = 21.7 GB/卡, TP4 = 39.8 GB/卡。卡越少 → 单卡权重越多 → 留给 KV 的空间越少 → 并发越低 (这就是 §6 里 TP4 估算中 baseline 进一步下降而 ours 不变的原因)。

### 2.4 页 (page) 与块 (chunk) 粒度

一个 c4 **chunk** = 4 token; 一个 c4 **page** = 64 chunk = 256 token。召回可以按 chunk 粒度或 page 粒度做。page 粒度 (§5.2) 让 page→block 映射表比 chunk 粒度表**小 64×**, 这对于避免 reserve 内的驱逐风暴 (eviction storm) 很关键。

---

## 3. 两级 Memory-Indexer 召回算法

> 这是**权威 (authoritative)** 算法。偏离它就是 bug。这里有**两个不同的 indexer, 分属两级**; 绝不能混为一谈。

### 3.1 Level-1 = Memory Indexer (我们训练的 retriever)

- **它是什么:** 我们训练的 retriever —— **R930 joint** checkpoint, 跨 **3 层** (compress-layer-id **10, 12, 20**)。
- **何时运行:** **每 64 个 decode step 一次** (retrieval interval), 在每个 cycle 边界。
- **打什么分:** **全历史**的压缩 K (compressed-K)。
- **如何 ensemble:** 3 层各自对每个 chunk 出一个 sigmoid 分数; 用**跨层取 max (cross-layer max)** ensemble (默认 `or` 模式) → 每 chunk 一个分数。
- **如何选择:** **阈值 `sigmoid > 0.5`** (等价于 `logit > 0`), **不是 top-512**。(top-K 只是 release-demo 的 API 示例; 部署用阈值。) 阈值自适应地保留大约 **~10% 的历史** —— 这就是 ~90% KV 节省的来源。
- **产出什么:** 一个全局的 **`resident_set`** —— 未来 64 步要常驻 GPU 的关键 chunk —— 外加一个 **recency-tail / sink** 回退, 无条件包含最新的 chunk (以及少量头部 / sink)。

### 3.2 Level-2 = 原生 Lightning Indexer (DSv4 自带)

- 在**全部 21 个 c4 层、每个 token** 上运行 (与原生 DSv4 一致, 未改)。
- 每层每步给它的 CSA chunk 打分并选 **top-512** 做稀疏注意力 —— **但 top-512 只能从 `resident_set` 里选。** 非常驻 chunk 被 mask 成 `-inf`; 它们物理上根本不在 GPU 上。
- **全部 21 层都受 `resident_set` 约束, 包括那 3 个 target 层。** 那 3 个 target 层*自己*的原生 Level-2 indexer 也只从 `resident_set` 里选 top-512。Level-1 *设置* `resident_set` 和 Level-2 *选 512* 是**两个独立操作**, 只是恰好在这 3 层上同处一地; 它们不是同一步。

### 3.3 单个 64-step cycle 的时序

在每个 cycle 边界, Memory Indexer 给全历史打分, 为整个窗口固定常驻集。窗口内**零 CPU→GPU swap** —— swap 只发生在 cycle 边界。

```text
# ── cycle 边界 (每 64 个 decode step 一次) ────────────────────────────────
Level-1  Memory Indexer (3 层 ensemble, 跨层取 max)
           score  = sigmoid( retriever(全历史压缩K) )                 # 全历史
           keep   = { chunk : score > 0.5 }  ∪  recency_tail  ∪  sink  # 阈值, 不是 top-512
           resident_set = keep                                        # 约历史的 ~10% ≈ 6144 chunk
         swap  resident_set → GPU reserve                             # 本 cycle 唯一的 swap
         resident_set 在接下来 64 步内保持不变

# ── 64-step 窗口内 (每个 decode step, 每个 c4 层) ─────────────────────────
for step in 1..64:
    for layer in 全部 21 个 c4 层:                                    # Level-2, 原生 Lightning Indexer
        logits = native_indexer_score(chunks)                         # 只在 resident_set 上打分
        logits[chunk ∉ resident_set] = -inf                           # 非常驻的被 mask 掉
        top512 = topk(logits, 512)                                    # top-512 ⊂ resident_set
        attention(top512)                                             # 从 GPU reserve 读
    # 这里没有 CPU↔GPU swap —— 窗口需要的一切都已常驻
```

### 3.4 为什么 n→n+1 延迟无害

用在 cycle *n+1* 的常驻集是从刚结束的 cycle *n* 算出的 (滞后一个 cycle)。这无害, 因为:

1. retriever 是**被训练来预测未来 64 步的**, 所以相邻窗口重叠 **63/64** —— 下一步要的 chunk 绝大部分就是现在要的 chunk。
2. **recency-tail / sink** 回退无条件包含最新的 chunk, 恰好覆盖滞后一个 cycle 的预测可能漏掉的区域。

---

## 4. 推理数据流

单个长上下文请求的端到端路径, 以及每项优化插入的位置。

```text
prompt
  │
  ▼
┌──────────────────────── PREFILL (PD 下的 P server) ──────────────────────┐
│ 算全 prompt 的 KV (c4 classical KV, index-K, MLA latent, SWA, ...)         │
└───────────────────────────────────────────────────────────────────────────┘
  │  NIXL transfer  (§5.5)
  │  c4 classical KV + 已 offload 的 index-K 直传 D 的 CPU mirror
  │  (VRAM → DRAM, 不落到 D 的 GPU); target 层的 index-K → D 的 VRAM
  │  逐位置的 `kv_dram_mask` 告诉 NIXL 哪些 buffer 是 DRAM
  ▼
┌──────────────────────── DECODE (PD 下的 D server) ───────────────────────┐
│                                                                            │
│  全历史活在 CPU pinned mirror (c4 KV + 18 层 index-K)                       │
│  GPU 只留: 缩小的 c4 "reserve" + 缩小的 index-K 池 + 3 层 target index-K    │
│                                                                            │
│  ── cycle 边界 (每 64 步) ────────────────────────────────────────────     │
│    Level-1 Memory Indexer 给全历史打分 (3 个 target 层)          ◄── §5.4  │
│      → resident_set (页召回: top-K_max=96 最密页)               ◄── §5.2  │
│      → 把 resident_set 的页 swap 进 GPU reserve                  ◄── §5.1  │
│    (这些 host-syncing 工作跑在 side-band / 后台线程)             ◄── §5.3  │
│                                                                            │
│  ── 64 步中的每一步, 在被 capture 的 CUDA graph 内 ──────────────          │
│    读固定的 resident_chunk_mask buffer                          ◄── §5.3  │
│    Level-2 原生 indexer: mask 非常驻 → -inf → top-512                       │
│    把 top-512 remap 到 reserve cell (chunk_cell_lut)            ◄── §5.1  │
│    稀疏注意力, 全程在 GPU reserve 内                                        │
│    没有 CPU↔GPU swap                                                        │
└───────────────────────────────────────────────────────────────────────────┘
  │
  ▼
生成 token
```

非 PD (单机) 模式把 P 和 D 合到一个 server; NIXL transfer 步骤消失, prefill/decode 共享同一 GPU。**score-masking** 验证模式 (见 §7 与 inference README) 把*全量* KV 留在 GPU 上, 只做 Level-1/Level-2 的*选择* (把非选中 mask 成 `-inf`) —— 它改变选择但不改变显存驻留, 因此把 retriever 准确率与 offload 机制隔离开来。

---

## 5. 各项优化

下面每项优化都是 **env-gated** 的。**门控关闭 = 字节一致 baseline** —— 不设 env 就是原生 DSv4 行为。这条零风险回退原则贯穿整个系统。

### 5.1 Path-P — c4 classical KV offload

- **Env:** `SGLANG_DECODE_SWAP_P=1`
- **机制:** 未被召回的 chunk 的 c4 classical KV (584 B/chunk —— *贵的*注意力 KV) 被 offload 到一个持有*全*历史的 **CPU pinned mirror**。一个缩小的 GPU **"reserve"** 只装召回页。`chunk_cell_lut` 把逻辑 chunk 位置 → 它的 reserve cell, 于是 in-graph 注意力无需知道全局历史布局就能寻址 reserve。
- **为什么是杠杆:** c4 classical KV 是长上下文下**单请求最大的 GPU 成本**。offload 它把单请求 GPU c4 KV 从 ~1.6 GB (512K 时) 降到 ~0。
- **收益:** 单请求 GPU c4 KV → ~0; 这是 §6 并发提升的首要使能项。

### 5.2 Page-recall — 页粒度的两级召回

- **Env:** `SGLANG_PATHP_PAGE_RECALL=1`, `PAGE_KMAX=96`
- **机制:** Level-1 选出 top **`K_max = 96`** 个*最密的页* (一个页的分数 = 其 chunk 中 `sigmoid > 阈值` 的数量), 外加强制的 recency-tail / sink 页。召回随后在 **page-block 粒度**而非逐 chunk 上进行。
- **为什么是杠杆:** page→block 表比 chunk 粒度表**小 64×** (1 页 = 64 chunk), 从而在常驻集于 cycle 之间移动时避免 reserve 里的**驱逐风暴 (eviction storm)**。
- **收益:** `resident_set = 96 × 64 = 6144 chunk`, 一个与上下文长度**无关的常数** —— 这就是 §6 里固定 FLOP 预算的那个 6144。

### 5.3 cuda-graph 兼容 — ~50× 吞吐杠杆

- **Env:** `SGLANG_PATHP_CUDAGRAPH=1`; 配合 `SGLANG_PATHP_SCORE_RESIDENT=1`
- **机制:** 所有 **host-syncing** 工作 (Level-1 打分、召回、decode-store) 被移**出 forward capture**, 放到 side-band (`on_forward_end` / 后台线程)。被 capture 的 graph 内只剩**纯 GPU 操作**: 读固定的 `resident_chunk_mask` buffer → 把非常驻 logits mask 成 `-inf` → 原生 top-512 → remap → 注意力。`SGLANG_PATHP_SCORE_RESIDENT=1` 让 decode 只对**常驻集**打分 (而非全历史), 走*真实*的 page pool 和*真实*的 page_table (**零 gather**)。
- **为什么是杠杆:** 没有 cuda-graph 兼容时, eager Python hook 每 fire 要 ~**100 ms**; forward 里的 host sync 本身就会完全禁止 graph capture。把它们移到 side-band 让 decode 跑在 **graph 速度**上。
- **收益:** 相对 eager hook ~**50× 吞吐**。这是*单项*最大的吞吐杠杆。

### 5.4 Path-I — 选择性 index-K offload

- **Env:** `SGLANG_PATHP_INDEX_K_OFFLOAD=1`, `INDEX_K_DEVICE_TOKENS=1572864`
- **背景:** 仅 **1M 高并发**时需要。当 Path-P offload 掉 c4 classical KV 后, index-K 打分池 (`c4_indexer_kv_pool`, Lightning-Indexer 的打分 key, 132 B/chunk × 21 层) 成为**新的 GPU 墙**。
- **关键洞察 —— index-K 有两个消费者:**
  - **Level-1** 的 side-band 重选给**全历史**打分, 但**只针对 3 个 target 层 (10, 12, 20)**。
  - **Level-2** 的 in-graph 跑**全部 21 层**, 但只读 `K_max = 96` 个召回页。
- **机制:** 因此**保留**那 3 个 target 层的 index-K **全量在 GPU** (Level-1 要给它们打分; 字节不变), 把其余 **18 层** offload (Level-2 只在召回页处读它们) 到 CPU mirror + reserve。被 offload 层上的打分是**位一致 (bit-identical)** 的 (byte-test 已证 —— §7)。若把全部 21 层统一 offload, 会让 Level-1 对 target 层读到**垃圾** → **静默错误**; 选择性不是可选项。
- **收益:** index-K 池 **42.5 GB → 9.6 GB @ 1M**, 省 **~33 GB/卡**。

### 5.5 PD 分离 (PD disaggregation)

- **机制:** 分离的 **P** (prefill) 和 **D** (decode) server。P prefill 长 prompt; **NIXL** 把 KV 传给 D。关键在于, c4 KV 和已 offload 的 index-K 是**直传到 D 的 CPU mirror (DRAM)** 的 —— **VRAM → DRAM**, *不*落到 D 的 GPU。逐位置的 **`kv_dram_mask`** 告诉 NIXL transfer 哪些 buffer 是 DRAM; 在选择性 index-K offload 下这个 mask 是**非连续 (non-contiguous)** 的, 因为 target 层留在 VRAM。D 随后执行全部 offload / 召回。
- **测量用 barrier:** `SGLANG_PD_DECODE_BARRIER=N` 攒住 `N` 个已传输请求, 一起放行做 **lockstep decode**。这是测量**稳态 N-并发**吞吐的正确工具 (否则请求错峰到达, 聚合数字不是干净的稳态值)。
- **为什么是杠杆:** 它让 D 成为一个显存精简的 decode 引擎 (它的 GPU 从不必持有传来的长上下文 KV), 而 P 吸收一次性的 prefill 成本。这是让 §5 所有 offload 复合成 §6 并发数字的拓扑。

### 5.6 支撑优化: admission / SWA / online-compress

这些重塑 admission 预算和次级池, 让释放出的 GPU 显存真正转化为被放行的并发。

| Env | 作用 |
|---|---|
| `SGLANG_PD_CREDIT_OFFLOADED_C4=1` | admission 预算把已 offload 的 c4 记为 **0** → 放行更高并发 (否则预算仍会为不在 GPU 的 c4 预留)。 |
| `SGLANG_PD_SWA_WINDOW_PREALLOC=1` + `SGLANG_DSV4_SWA_MAX_TOKENS=131072` | 把 SWA 池与上下文长度解耦 —— 按窗口预分配, SWA 大小跟并发走, 不跟上下文走。 |
| `SGLANG_OPT_USE_ONLINE_COMPRESS=1` | 把 `c128_state` 从 **25 GB → 486 MB**。 |

没有这些, Path-P / Path-I 释放的显存不会转化为更多被放行的请求 —— admission 会计仍会为不再驻留的 KV 预留。

---

## 6. 实测结果

### 6.1 并发与稳态聚合吞吐

8×H20, TP8, 跨机 PD, 2026-07。"conc" = 被放行的最大并发长上下文请求数; "tput" = 稳态聚合 decode 吞吐 (tok/s), 用 decode barrier 测。

| context | baseline conc / tput | ours conc / tput | conc× | tput× |
|---|---|---|---|---|
| 256K | 47 / 1584 | 60 / 1663 | **1.3×** | **1.05×** |
| 512K | 25 / 1028 | 60 / 1535 | **2.4×** | **1.49×** |
| 1M | 11 / 455 | 40 / 1183 | **3.6×** | **2.60×** |

**上限成因 (诚实交代):**

- **256K / 512K 卡在 60** —— `batch > 64` 的 **in-graph CUDA crash**。不是显存限制, 是 graph/batch-size bug。
- **1M 卡在 40** —— **CPU-mirror 主机内存 (host-RAM) 墙** (非 GPU)。并发 60 时, 仅 c4 mirror 就是 **188.8 GB/卡 × 8 = 1510 GB**; 加上 index-K (**292 GB**) 超过 **2265 GB** host → host-OOM。并发 40 能塞进 ~**1200 GB**。**1M 这堵墙是主机内存, 不是 GPU。**

### 6.2 Per-decode-token FLOPs (整模型, 来自真实权重形状)

per-token 的**常数**部分 —— MoE + MLA 投影 + compressor + indexer q-proj + LM head —— baseline 和 ours **完全一致**: **26.6 GFLOP**。只有 indexer 的**全历史打分**不同: baseline `O(context)`, ours 是对 6144 个常驻 chunk 的**常数 2.13 GFLOP**。

| context | baseline 整模型 | ours 整模型 | ours / baseline | 省下 |
|---|---|---|---|---|
| 256K | 50.8 GFLOP | 30.2 GFLOP | **0.59×** | 40.6% |
| 512K | 73.5 GFLOP | 30.2 GFLOP | **0.41×** | 58.9% |
| 1M | 118.9 GFLOP | 30.2 GFLOP | **0.25×** | 74.6% |

> **重要口径。** **不要**把孤立的 indexer-scoring 比值 (看起来会是 11× / 21× / 43×) 当成整模型比值来报 —— 那是早先的错误。正确的**整模型**比值是 **0.59× / 0.41× / 0.25×**。**attend 本身是完全一样的** (双方都做 top-512); ours 省的是**打分 (scoring)**, 不是 attend。

### 6.3 TP4 (4 卡 D) 估算

用 4 卡 D server, baseline 并发再降 **~25%** (39.8 GB/卡 的权重吃掉更多单卡预算, 且 KV 复制 → 少卡不带来 KV 缓解), 而 **ours 不变** (它的 GPU 占用已经极小)。估算吞吐比值:

| context | 估算 tput× (TP4) |
|---|---|
| 256K | ~1.4× |
| 512K | ~2.0× |
| 1M | ~3.5× |

**上下文越长, 优势越大** —— 打分节省和 KV-offload 节省都随历史长度 scale。

---

## 7. 正确性验证

### 7.1 字节级一致性测试

- **Path-I reserve-packed 打分与全量打分位一致。** 选择性 index-K offload 用 **byte-test** 验证过: 从 reserve-packed (CPU-mirror 支撑) 的池给那 18 个 offload 层打分, 结果与从全量 on-GPU 池打分**位一致 (bit-identical)**。正是这一点使得在保留 3 个 target 层在 GPU (供 Level-1) 的同时 offload 那些层是安全的。
- **gate-on vs gate-off needle 一致性。** 在 needle 任务上把完整 offload/召回路径 (gate-on) 与纯 baseline (gate-off) 对比, **5 / 7** 的预测**完全相同**。**另外 2** 处差异是 **PD 浮点非确定性 (floating-point nondeterminism)** —— server 对**自己** (baseline vs baseline 跨 run) 也表现出同样的非确定性, 所以那**不是**我们路径引入的 bug, 而是 PD 固有的浮点非确定性。

### 7.2 env-gate 零风险回退原则

每项优化都在一个 env 门控之后, 且 **门控关闭 = 与原生 DSv4 字节一致**。这是结构性可验证的: 不设任何 `SGLANG_PATHP_*` / `SGLANG_DECODE_SWAP_P` / index-K-offload env 时, 那些改动显存驻留或 graph capture 的代码路径根本不会进入, 于是 server 跑的是未改动的 baseline。实际意义: (a) 任何优化都能独立关闭做 A/B 隔离; (b) 生产部署可以通过 unset env 瞬间回退到已知良好的 baseline, **无需改代码、无准确率风险**。

---

## 8. 局限与未来工作

墙都诚实交代; 没有一堵藏在头条数字后面。

1. **`batch > 64` in-graph CUDA crash** —— 把 256K 和 512K 卡在并发 **60**。这是 graph/batch-size crash, 不是根本的显存限制; 256K/512K 时 GPU 仍有余量。修掉它能让 256K/512K 越过 60 (两者当前都是 GPU-余量-受限, 非 offload-受限)。
2. **1M 主机内存 mirror 墙** —— 把 1M 卡在并发 **40**。CPU pinned mirror 为每个 in-flight 请求持有全历史 c4 KV (+ 已 offload 的 index-K); 并发 60 时超过 2265 GB host (1510 GB c4 + 292 GB index > host)。要在 1M 越过 40, 需要一个**有界 / 更小的 mirror** (例如带上限或分层的 CPU mirror, 或把最冷的历史 spill 到磁盘), 而非每请求一份全历史 CPU 副本。**这堵墙是主机内存, 不是 GPU** —— GPU 侧 offload 已经成功了, 瓶颈搬到了 DRAM。

两堵墙都是*当前*前沿, 不是架构死胡同: (1) 是待修的工程 bug, (2) 是内存层级的设计选择 (给 mirror 设界), 用适度的 re-fetch 成本换取在 1M 越过并发 40。

---

*本报告所有数字、机制、env 变量均取自实测 run 与已实现代码; 中文版内容与英文版 `TECHNICAL_REPORT_en.md` 完全一致。部署 / 运行说明见 `inference/README.md`。*
