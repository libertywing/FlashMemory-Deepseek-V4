# Technical Report — DeepSeek-V4 Memory-Indexer Inference

*Implementation-level documentation of the two-level Memory-Indexer retriever inference system.*

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture background](#2-architecture-background)
3. [Two-level Memory-Indexer recall algorithm](#3-two-level-memory-indexer-recall-algorithm)
4. [Inference data flow](#4-inference-data-flow)
5. [Optimizations](#5-optimizations)
6. [Measured results](#6-measured-results)

---

## 1. Overview

DeepSeek-V4 (DSv4) serves long context with **CSA — Compressed Sparse Attention**. On every decode token, at each of the 21 `c4` (compress_ratio = 4) layers, the model's native **Lightning Indexer** scores *every* chunk in the full history (`ceil(context / 4)` chunks) and selects the top-512 for sparse attention. Two costs scale with context length and dominate at 1M tokens:

- **(A) Compute** — the indexer scoring is `O(full history)`. At 1M context this is a large, per-token, per-layer scan.
- **(B) GPU memory** — the `c4` classical KV for *all* history must physically reside on GPU for the attention to read it. Because MLA KV is **replicated** per TP rank (not sharded — see §2), adding GPU cards does **not** add KV capacity, so the per-request KV footprint sets a hard ceiling on concurrency.

Our solution attacks both, and does so as **two independent prongs**:

1. **Save compute** — replace the native full-history scoring with a trained **Memory Indexer** (our retriever) that runs once every 64 decode steps and produces a small **resident set** (~10 % of history, = `KMAX × 64` chunks: 6144 / 12288 / 24576 for 256K / 512K / 1M). Decode then scores *only* the resident set, not the full history. This drops whole-model per-token FLOPs at 1M from 118.9 GFLOP to 35.4 GFLOP (**0.30×**, i.e. 70 % saved).
2. **Save GPU memory → raise concurrency** — physically **offload** the KV of non-recalled chunks to a CPU pinned mirror and keep only the recalled pages in a shrunk GPU "reserve" (**Path-P**), then additionally offload the scoring-key pool of the 18 non-target layers (**Path-I**). Per-request GPU footprint drops sharply, so the same 8×H20 admits several times more concurrent long-context requests.

**Summary of results (8×H20, TP8, cross-machine PD, 2026-07):** concurrency rises **1.6× / 2.4× / 2.7×** and steady-state aggregate throughput rises **1.7× / 2.0× / 2.8×** at 256K / 512K / 1M respectively. The advantage grows with context length. Every optimization is **env-gated** — with the gate off, the server is **byte-identical** to stock DSv4, giving a zero-risk fallback.

Two honest walls remain (§6, §8): cuda-graph pre-allocation caps 256K/512K concurrency at batch≈80 (256K reaches ~76, 512K's sweet spot is ~60), and the CPU-mirror **host-RAM** wall caps 1M at concurrency 40. The 1M wall is host memory, not GPU.

---

## 2. Architecture background

### 2.1 DSv4 dimensions (from `config.json`, ground truth)

| Field | Value |
|---|---|
| hidden size | 4096 |
| layers | 43 total = **3 dense** + **21 c4** (compress_ratio 4) + **20 c128** (compress_ratio 128) |
| MoE | 256 experts, top-6 (+1 shared), moe_inter 2048 |
| MLA heads | `num_attention_heads` = 64, `num_key_value_heads` = **1** (single MLA latent KV head) |
| `q_lora_rank` | 1024 |
| Indexer | `index_n_heads` = 64, `index_head_dim` = 128, `index_topk` = **512** |
| vocab | 129280 |
| paging | `page_size` = 256; `c4_page_size` = 64. **1 c4-chunk = 4 tokens; 1 c4-page = 64 chunks = 256 tokens.** |

### 2.2 CSA and the native Lightning Indexer

The 21 `c4` layers use Compressed Sparse Attention. Each history chunk compresses 4 tokens. For every decode token, at each c4 layer, the native Lightning Indexer computes a score against *every* compressed chunk in history and keeps the **top-512** for that layer's sparse attention. Two pools back this per layer:

- **index-K** (`c4_indexer_kv_pool`) — the Lightning-Indexer *scoring keys*, **132 B/chunk × 21 layers**. Cheap per chunk, but scored over the full history every step.
- **c4 classical KV** (`c4_kv_pool`) — the *attention* KV read after top-512 selection, **584 B/chunk**. This is the expensive KV.

### 2.3 The concurrency-physics fact: MLA KV is replicated, not sharded

This is the single most important physical constraint for concurrency. Under tensor parallelism, `get_num_kv_heads(tp=8) == 1` — the MLA latent KV is **replicated on every TP rank**, not split across them. This was confirmed by `nvidia-smi`: all 8 cards show identical memory usage. Consequences:

- **Adding cards does NOT multiply KV capacity.** Each rank carries the whole KV, so per-request KV footprint is the same regardless of TP degree. This is precisely why the baseline cannot scale concurrency by adding cards.
- **Weights DO shard.** Measured: TP8 = 21.7 GB/card, TP4 = 39.8 GB/card. Fewer cards → more weight per card → less room left for KV → lower concurrency (this is why the TP4 estimate in §6 has the baseline dropping further while ours holds).

### 2.4 Page vs chunk granularity

A c4 **chunk** = 4 tokens; a c4 **page** = 64 chunks = 256 tokens. Recall can be done at chunk granularity or page granularity. Page granularity (§5.2) makes the page→block mapping table **64× smaller** than a chunk-granular table, which matters for avoiding an eviction storm inside the reserve.

---

## 3. Two-level Memory-Indexer recall algorithm

> This is the **authoritative** algorithm. Deviating from it is a bug. There are **two distinct indexers at two levels**; they must not be conflated.

### 3.1 Level-1 = Memory Indexer (our trained retriever)

- **What it is:** our trained retriever — the **R930 joint** checkpoint, spanning **3 layers** (compress-layer-ids **10, 12, 20**).
- **When it runs:** **once every 64 decode steps** (the retrieval interval), at each cycle boundary.
- **What it scores:** the **full history's** compressed-K.
- **How it ensembles:** the 3 layers each emit a sigmoid score per chunk; they are ensembled by **cross-layer max** (default `or` mode) → one per-chunk score.
- **How it selects:** **threshold `sigmoid > 0.5`** (equivalently `logit > 0`), **NOT top-512**. (Top-K is only the release-demo API example; deployment uses the threshold.) The threshold adaptively keeps roughly **~10 % of history** — this is the origin of the ~90 % KV saving.
- **What it produces:** one global **`resident_set`** — the critical chunks to keep resident on GPU for the next 64 steps — plus a **recency-tail / sink** fallback that always includes the newest chunks (and a small head/sink).

### 3.2 Level-2 = native Lightning Indexer (DSv4's own)

- Runs on **all 21 c4 layers, every token** (unchanged from stock DSv4).
- Each layer/step scores and picks its **top-512** CSA chunks for sparse attention — **but the top-512 can only be chosen from the `resident_set`.** Non-resident chunks are masked to `-inf`; they do not physically exist on GPU.
- **All 21 layers are constrained by `resident_set`, including the 3 target layers.** The 3 target layers' *own* native Level-2 indexer also picks its top-512 only from `resident_set`. Level-1 *setting* `resident_set` and Level-2 *picking-512* are **independent operations** that merely co-locate on those 3 layers; they are not the same step.

### 3.3 Timing of one 64-step cycle

At each cycle boundary the Memory Indexer scores the full history and fixes the resident set for the whole window. Within the window there is **zero CPU→GPU swap** — swap happens only at cycle boundaries.

```text
# ── cycle boundary (once every 64 decode steps) ───────────────────────────
Level-1  Memory Indexer (3-layer ensemble, cross-layer max)
           score  = sigmoid( retriever(full_history_compressed_K) )   # full history
           keep   = { chunk : score > 0.5 }  ∪  recency_tail  ∪  sink  # threshold, NOT top-512
           resident_set = keep                                        # ~10% of history ≈ 6144 chunks
         swap  resident_set → GPU reserve                             # the ONLY swap in this cycle
         resident_set stays FIXED for the next 64 steps

# ── inside the 64-step window (every decode step, every c4 layer) ─────────
for step in 1..64:
    for layer in all 21 c4 layers:                                    # Level-2, native Lightning Indexer
        logits = native_indexer_score(chunks)                         # scored over resident_set only
        logits[chunk ∉ resident_set] = -inf                           # non-resident masked out
        top512 = topk(logits, 512)                                    # top-512 ⊂ resident_set
        attention(top512)                                             # read from GPU reserve
    # NO CPU↔GPU swap here — everything the window needs is already resident
```

### 3.4 Why the n→n+1 latency is harmless

The resident set applied to cycle *n+1* is computed from the just-completed cycle *n* (one cycle stale). This is harmless because:

1. The retriever is **trained to predict the next 64 steps**, so successive windows overlap by **63/64** — the chunks needed next are overwhelmingly the chunks needed now.
2. The **recency-tail / sink** fallback unconditionally includes the newest chunks, covering exactly the region the one-cycle-stale prediction could miss.

---

## 4. Inference data flow

End-to-end path of a single long-context request, and where each optimization plugs in.

```text
prompt
  │
  ▼
┌──────────────────────── PREFILL (P server under PD) ─────────────────────┐
│ compute full-prompt KV (c4 classical KV, index-K, MLA latent, SWA, ...)   │
└───────────────────────────────────────────────────────────────────────────┘
  │  NIXL transfer  (§5.5)
  │  c4 classical KV + offloaded index-K sent DIRECTLY to D's CPU mirror
  │  (VRAM → DRAM, does NOT land on D's GPU); target-layer index-K → D VRAM
  │  a per-position `kv_dram_mask` tells NIXL which buffers are DRAM
  ▼
┌──────────────────────── DECODE (D server under PD) ──────────────────────┐
│                                                                            │
│  full history lives in the CPU pinned mirror (c4 KV + 18-layer index-K)    │
│  GPU holds: shrunk c4 "reserve" + shrunk index-K pool + 3-layer index-K    │
│                                                                            │
│  ── cycle boundary (every 64 steps) ─────────────────────────────────     │
│    Level-1 Memory Indexer scores full history (3 target layers)  ◄── §5.4  │
│      → resident_set (page-recall: top-K_max=96 densest pages)    ◄── §5.2  │
│      → swap resident_set pages into GPU reserve                  ◄── §5.1  │
│    (this host-syncing work runs in a side-band / bg thread)      ◄── §5.3  │
│                                                                            │
│  ── each of the 64 steps, inside the captured CUDA graph ───────────       │
│    read fixed resident_chunk_mask buffer                         ◄── §5.3  │
│    Level-2 native indexer: mask non-resident → -inf → top-512               │
│    remap top-512 → reserve cells (chunk_cell_lut)                ◄── §5.1  │
│    sparse attention, entirely inside GPU reserve                            │
│    NO CPU↔GPU swap                                                          │
└───────────────────────────────────────────────────────────────────────────┘
  │
  ▼
generated tokens
```

---

## 5. Optimizations

Every optimization below is **env-gated**. **Gate-off = byte-identical baseline** — not setting the env yields stock DSv4 behavior. This is the zero-risk fallback principle that runs through the whole system.

### 5.1 Path-P — c4 classical KV offload

- **Env:** `SGLANG_DECODE_SWAP_P=1`
- **Mechanism:** the c4 classical KV (584 B/chunk — the *expensive* attention KV) for non-recalled chunks is offloaded to a **CPU pinned mirror** holding the *full* history. A shrunk GPU **"reserve"** holds only the recalled pages. A `chunk_cell_lut` maps a logical chunk location → its reserve cell, so the in-graph attention can address the reserve without knowing global history layout.
- **Why it's the lever:** the c4 classical KV is the **biggest per-request GPU cost** at long context. Offloading it drops per-request GPU c4 KV from ~1.6 GB (at 512K) to ~0.
- **Payoff:** per-request GPU c4 KV → ~0; this is the primary enabler of the concurrency gains in §6.

### 5.2 Page-recall — two-level recall at page granularity

- **Env:** `SGLANG_PATHP_PAGE_RECALL=1`, `PAGE_KMAX=96`
- **Mechanism:** Level-1 selects the top **`K_max = 96`** *densest pages* (a page's score = the number of its chunks with `sigmoid > threshold`), plus forced recency-tail / sink pages. Recall then operates at **page-block granularity** rather than per-chunk.
- **Why it's the lever:** a page→block table is **64× smaller** than a chunk-granular table (1 page = 64 chunks), which avoids an **eviction storm** in the reserve when the resident set shifts between cycles.
- **Payoff:** `resident_set = 96 × 64 = 6144 chunks`, a **constant** independent of context length — this is the 6144 that fixes the FLOP budget in §6.

### 5.3 cuda-graph compatibility — the ~50× throughput lever

- **Env:** `SGLANG_PATHP_CUDAGRAPH=1`; with `SGLANG_PATHP_SCORE_RESIDENT=1`
- **Mechanism:** all **host-syncing** work (Level-1 scoring, recall, decode-store) is moved **out of the forward capture** into a side-band (`on_forward_end` / a background thread). Inside the captured graph only **pure-GPU ops** remain: read a fixed `resident_chunk_mask` buffer → mask non-resident logits to `-inf` → native top-512 → remap → attention. `SGLANG_PATHP_SCORE_RESIDENT=1` makes decode score **only the resident set** (not the full history) via the *real* page pool and *real* page_table (**zero gather**).
- **Why it's the lever:** without cuda-graph compatibility, the eager Python hook costs ~**100 ms per fire**; the host syncs inside the forward would forbid graph capture entirely. Moving them to the side-band lets decode run at **graph speed**.
- **Payoff:** ~**50× throughput** vs the eager hook. This is *the* single largest throughput lever.

### 5.4 Path-I — selective index-K offload

- **Env:** `SGLANG_PATHP_INDEX_K_OFFLOAD=1`, `INDEX_K_DEVICE_TOKENS=1572864`
- **Context:** needed only for **1M high concurrency**. After Path-P offloads the c4 classical KV, the index-K scoring pool (`c4_indexer_kv_pool`, the Lightning-Indexer scoring keys, 132 B/chunk × 21 layers) becomes the **new GPU wall**.
- **Key insight — index-K has two consumers:**
  - **Level-1** side-band re-selection scores the **full history** but **only for the 3 target layers (10, 12, 20)**.
  - **Level-2** in-graph runs **all 21 layers** but only reads the `K_max = 96` recalled pages.
- **Mechanism:** therefore **keep** the 3 target layers' index-K **full on GPU** (Level-1 scores them; byte-unchanged) and **offload** the other **18 layers** (Level-2 reads them only at recalled pages) to a CPU mirror + reserve. Scoring on the offloaded layers is **bit-identical** (proven by a byte-test — §7). Uniformly offloading all 21 layers would make Level-1 read **garbage** for the target layers → **silently wrong**; selectivity is not optional.
- **Payoff:** index-K pool **42.5 GB → 9.6 GB @ 1M**, saving **~33 GB/card**.

### 5.5 PD disaggregation

- **Mechanism:** separate **P** (prefill) and **D** (decode) servers. P prefills the long prompt; **NIXL** transfers the KV to D. Crucially, the c4 KV and the offloaded index-K are transferred **directly to D's CPU mirror (DRAM)** — **VRAM → DRAM**, *not* landing on D's GPU. A per-position **`kv_dram_mask`** tells the NIXL transfer which buffers are DRAM; this mask is **non-contiguous** under selective index-K offload because the target layers stay in VRAM. D then performs all the offload / recall.
- **Why it's the lever:** it lets D be a memory-lean decode engine (its GPU never has to hold the transferred long-context KV) while P absorbs the one-shot prefill cost. It is the topology in which all the §5 offloads compound into the §6 concurrency numbers.

### 5.6 Supporting optimizations: admission / SWA / online-compress

These reshape the admission budget and secondary pools so the freed GPU memory actually converts into admitted concurrency.

| Env | Effect |
|---|---|
| `SGLANG_PD_CREDIT_OFFLOADED_C4=1` | Credits offloaded c4 as **0** in the admission budget → admits higher concurrency (the budget otherwise still reserves for c4 that no longer lives on GPU). |
| `SGLANG_PD_SWA_WINDOW_PREALLOC=1` + `SGLANG_DSV4_SWA_MAX_TOKENS=131072` | Decouples the SWA pool from context length — preallocates by window, so SWA size tracks concurrency, not context. |
| `SGLANG_OPT_USE_ONLINE_COMPRESS=1` | Collapses `c128_state` from **25 GB → 486 MB**. |

Without these, the memory freed by Path-P / Path-I would not translate into more admitted requests — the admission accountant would still reserve for KV that is no longer resident.

---

## 6. Measured results

### 6.1 Concurrency and steady-state aggregate throughput

8×H20, TP8, cross-machine PD, 2026-07. "conc" = max concurrent long-context requests admitted; "tput" = steady-state aggregate decode throughput (tok/s).

| context | KMAX | baseline conc / tput | ours conc / tput | conc× | tput× |
|---|---|---|---|---|---|
| 256K | 96 | 47 / 1584 | 76 / 2759 | **1.6×** | **1.7×** |
| 512K | 192 | 25 / 1028 | 60 / 2008 | **2.4×** | **2.0×** |
| 1M | 384 | 11 / 455 | 30 / 1266 | **2.7×** | **2.8×** |

> KMAX (densest pages recalled per request) scales with context to hold ~10 % recall coverage:
> 256K→96 pages (6144 chunks) / 512K→192 (12288) / 1M→384 (24576). Larger KMAX = fuller recall
> coverage (higher quality) but moves more pages per recall (lower throughput). At the smaller
> KMAX=96, 1M reaches ~1537 tok/s (concurrency 40) at the cost of shallower recall coverage.

**Upper-bound causes (honest):**

- **256K / 512K** — bounded by **cuda-graph pre-allocation**: above batch ≈ 80 the overlap
  scheduler's future-map / recall pre-allocation goes out of bounds (illegal memory access).
  256K runs stably to running ≈ 76 (peak ~2759 tok/s); 512K's sweet spot is conc ≈ 60
  (peak ~2008 tok/s, higher risks a bimodal recall collapse). Not a memory limit.
- **1M capped at 40** — the **CPU-mirror host-RAM wall** (not GPU). 60 × 1M CPU mirrors
  (~1.7 TB pinned) exceed the **2265 GB** host → host-OOM (OOM-killer kills a scheduler).
  Concurrency 40 fits in ~1.2 TB. **The 1M wall is host memory, not GPU.**

### 6.2 Per-decode-token FLOPs (whole model, from real weight shapes)

The per-token **constant** work — MoE + MLA projections + compressors + indexer q-proj + LM head — is **identical** for baseline and ours: **26.6 GFLOP**. Only the indexer **full-history scoring** differs: baseline scores `O(context)` (all history chunks), ours scores only the **resident set** = `KMAX × 64` chunks (6144 / 12288 / 24576 at 256K / 512K / 1M, ~10 % of history). Scoring FLOP ≈ `resident_chunks × 21 layers × 64 heads × 128 dim × 2` (relu(k·q)).

| context | KMAX | baseline whole-model | ours whole-model | ours / baseline | saved |
|---|---|---|---|---|---|
| 256K | 96 | 50.8 GFLOP | 28.8 GFLOP | **0.57×** | 43.3 % |
| 512K | 192 | 73.5 GFLOP | 31.0 GFLOP | **0.42×** | 57.8 % |
| 1M | 384 | 118.9 GFLOP | 35.4 GFLOP | **0.30×** | 70.2 % |

> **Important framing.** Do **not** report the isolated indexer-scoring ratio (~10–11× at every context, since KMAX scales with history) as if it were the whole-model ratio — that was an earlier mistake. The correct **whole-model** ratio is **0.57× / 0.42× / 0.30×**. The **attend itself is identical** (both do top-512); ours saves the **scoring**, not the attend. Ours scoring rises linearly with KMAX (2.2 / 4.4 / 8.8 GFLOP), but since the resident set stays ~10 % of history, the longer the context the more is saved.
