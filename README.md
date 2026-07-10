# FlashMemory-DeepSeek-V4

Memory-Indexer retriever for **DeepSeek-V4 Compressed-Sparse-Attention (CSA)** KV-cache.
Replaces the native Lightning Indexer with a lightweight trained retriever — instead of
scoring the full history every decode step, it predicts which ~10–15% of chunks the next
64 tokens will attend to. Only the selected chunks stay on GPU; the rest are offloaded to
CPU. Downstream evaluation matches or beats the full-attention baseline while **raising
concurrency 3.6× at 1M context**.

**[Model weights on Hugging Face](https://huggingface.co/libertywing/FlashMemory-Deepseek-V4)**

---

## Project structure

```
FlashMemory-Deepseek-V4/
├── sglang/                     # Full modified sglang source (editable install)
├── start_server.sh             # Mode A: single-machine score-masking
├── high_concurrency/           # Mode B: PD-disaggregated high-concurrency
│   ├── launch_decode.sh        #   D (decode) server — all offload/recall here
│   ├── launch_prefill.sh       #   P (prefill) server
│   └── launch_router.sh        #   router (mini load-balancer)
├── TECHNICAL_REPORT_en.md      # English technical report
├── TECHNICAL_REPORT_zh.md      # Chinese technical report
└── requirements.txt
```

---

## Quick start (Mode A — single-machine score-masking)

The simplest way to run: a single sglang server with the retriever hook. All KV stays
on GPU; the retriever masks non-selected chunks to `-inf` each decode cycle.

```bash
pip install -e sglang/python
# Also requires sgl_kernel: pip install sgl_kernel==0.3.21

# Download checkpoints from HuggingFace into checkpoints/
MODEL=/path/to/ds_fp8 bash start_server.sh   # default TP=4, port 30000
```

Look for `[_TrainedScorer] extracted l10/l12/l20` and `[InlineRetriever] resident#N`
in the logs to confirm the retriever is active.

Configurable via environment variables (see `start_server.sh` header):
`MODEL` / `CKPT` / `LAYERS` / `PORT` / `TP` / `THRESH` / `INTERVAL` /
`FIRST_KEEP` / `LAST_KEEP` / `ENSEMBLE_MODE`.

---

## Production (Mode B — PD-disaggregated high-concurrency)

Three servers: **P** (prefill), **D** (decode — all offload/recall), **router**
(mini load-balancer). Clients connect only to the router at `http://<router>:31503/v1/chat/completions`.

Key optimizations (all env-gated; gate-off = exact baseline DeepSeek-V4 behavior):

| Optimization | Env variable | What it does |
|---|---|---|
| Path-P (c4 KV offload) | `SGLANG_DECODE_SWAP_P` | c4 classical KV → CPU mirror + GPU reserve |
| Path-I (index-K offload) | `SGLANG_PATHP_INDEX_K_OFFLOAD` | Offload non-target 18 layers' index-K to CPU |
| Two-level recall | `SGLANG_PATHP_SCORE_RESIDENT` | Decode only scores resident set (not full history) |
| Page-recall | `SGLANG_PATHP_PAGE_RECALL` | Page-granular recall (64× smaller page→block table) |
| Cuda-graph | `SGLANG_PATHP_CUDAGRAPH` | Two-level recall inside cuda-graph (~50× faster) |
| Async recall | `SGLANG_PATHP_ASYNC_RECALL` | Recall on background thread, overlaps next 64 steps |
| Fused remap | `SGLANG_PATHP_FUSED_REMAP` | Top-512 columns map directly to reserve cells |

**Startup** (order: D first, then P, then router):

```bash
cd high_concurrency

# 1) D (decode) — on D machine GPU 0–7. 512K config example:
TGT_CONC=60 TGT_CTX=524288 CTX_LEN=1100000 bash launch_decode.sh

# 2) P (prefill) — on P machine GPU 0–7:
CTX_LEN=1100000 SWA_RATIO=0.1 HOST=<P_IP> bash launch_prefill.sh

# 3) router — any machine that can reach P + D:
PREFILL_IP=<P_IP> DECODE_IP=<D_IP> bash launch_router.sh
```

**Reference performance (8×H20 TP8, cross-machine PD, vs baseline standard DSv4):**

| Context | Ours concurrent / throughput | Baseline concurrent / throughput | Gain concurrent | Gain throughput |
|---------|------------------------------|----------------------------------|-----------------|-----------------|
| 256K | 60 / ~1663 tok/s | 47 / ~1584 tok/s | 1.3× | 1.05× |
| 512K | 60 / ~1535 tok/s | 25 / ~1028 tok/s | 2.4× | 1.49× |
| 1M | 40 / ~1183 tok/s | 11 / ~455 tok/s | 3.6× | 2.60× |

Longer context → larger offload advantage (baseline c4 KV per-request grows faster).

---

## Retriever architecture

Per CSA layer, scores are computed as:

```
hidden [B, 4096]
    → wq_a        (4096 → Q_LORA_RANK)
    → RMSNorm     (q_norm_weight, eps=1e-6)
    → wq_b        (Q_LORA_RANK → N_HEADS * HEAD_DIM)
    → reshape     [B, N_HEADS, HEAD_DIM]
    → RoPE        (YaRN, last ROPE_DIM=64 dims, base=160000)
    → Hadamard    (normalized Walsh-Hadamard)
    → q           [B, N_HEADS, HEAD_DIM]

hidden [B, 4096]
    → weights_proj (4096 → N_HEADS)
    → × weight_scale  (= HEAD_DIM^-0.5 * N_HEADS^-0.5)
    → fused_w     [B, N_HEADS]

compressed_k [B, N, HEAD_DIM + 4] (uint8)
    → bytes[:HEAD_DIM]  viewed as float8_e4m3 → dequant
    → × bytes[HEAD_DIM:]  viewed as float32   → k [B, N, HEAD_DIM]

score = sigmoid( sum_heads( relu(k @ q^T) * fused_w ) )   in [0, 1]
```

### Joint checkpoint + ensemble

The checkpoint holds **three independent CSA layers** (`l10`, `l12`, `l20`),
each with its own weights. At inference time per-layer sigmoid scores are
**ensembled per chunk** — `max` (union, default) or `mean` — to produce a
single keep/drop decision.

### Hyperparameters

| Param | Value |
|-------|-------|
| `N_HEADS` | 128 |
| `HEAD_DIM` | 128 |
| `Q_LORA_RANK` | 2048 |
| `ROPE_DIM` | 64 (last 64 dims) |
| `ROPE_BASE` | 160000 (YaRN) |
| `ROPE_FACTOR` | 16 |
| `ROPE_ORIGINAL_SEQ_LEN` | 65536 |
| `ROPE_BETA_FAST` | 32 |
| `ROPE_BETA_SLOW` | 1 |
| `RMS_NORM_EPS` | 1e-6 |

### `compressed_k` format

Each chunk = `HEAD_DIM + 4 = 132` `uint8` bytes:

| Bytes | Type | Meaning |
|-------|------|---------|
| `[:128]` | `float8_e4m3` | Quantized key values |
| `[128:132]` | `float32` | Per-chunk dequant scale |

Dequant: `fp8_values.view(float8_e4m3).float() * scale`.

---

## Two-level recall algorithm

```
 ┌──────────┐  compress & store    ┌────────────────────────────┐
 │ PREFILL  │  historical K/V      │  CSA KV-cache (the memory) │
 │ (dense   │ ───────────────────► │  N compressed chunks,      │
 │  attn)   │                      │  each = [132] uint8 fp8-K  │
 └────┬─────┘                      └──────────────┬─────────────┘
      │ last hidden state                         │
      ▼                                           │
 ┌────────────── LEVEL 1 (Memory Indexer, every 64 steps) ──────┼──────────────┐
 │ 3 target layers (l10/l12/l20):                                  │              │
 │   ensemble sigmoid scores → threshold + recent + sink          │              │
 │   → resident_set (bool[N], ~10–15% of chunks)                  │              │
 └────────────────────────────────────────────────────────┬───────┘              │
                                                          │                      │
 ┌────────────── LEVEL 2 (Lightning Indexer, every step, all 21 c4 layers) ──┐  │
 │ native top-512 confined to resident_set (non-resident logits → -inf)      │  │
 │ → attend only to recalled chunks                                          │  │
 └───────────────────────────────────────────────────────────────────────────┘  │
                                                                                │
 ┌────────── OFFLOAD (Path-P + Path-I) ──────────────────────────────────────┐  │
 │ c4 classical KV + non-target index-K → CPU mirror                         │  │
 │ GPU holds only recalled pages (resident_set)                              │  │
 └───────────────────────────────────────────────────────────────────────────┘  │
```

1. **Level 1 (Memory Indexer, every 64 steps).** The trained retriever scores the
   full history at the 3 target layers, ensembles per-layer scores, thresholds,
   and keeps recency tail + attention sink → produces a single global `resident_set`.
2. **Level 2 (native Lightning Indexer, every step, all 21 c4 layers).** The
   native indexer's top-512 is confined to the `resident_set` by masking non-resident
   chunk logits to `-inf`. This runs at full throughput inside cuda-graph.
3. **Offload (Path-P + Path-I).** Chunks not in `resident_set` are offloaded to
   CPU mirror. Recalled pages are swapped into a GPU reserve when the resident_set
   changes at cycle boundaries.

---

## Downstream evaluation

FlashMemory DS-V4 matches or beats the full-attention baseline on reasoning-heavy
long-context tasks while keeping only **~10–15% of CSA KV cache** on-device:

| Task | Context | vs. Full-Attn | KV Saved |
|------|---------|:---:|------|
| RULER (64k–512k) | 64K–512K | −1 ~ +2 pp | ~80–90% |
| LongMemEval-s | 125K | ±1 pp | ~86% |
| LongMemEval-m | 500K | ±1 pp | ~91% |
| LongBench V2 | 46K–493K | +1 ~ +2 pp | ~73–90% |
| MRCR (needle) | 274K | needs fallback | ~86% |

Precise needle-retrieval tasks (MRCR) require an additional **threshold-fallback**
in the serving layer — this is supported in the sglang integration.

---

## Checkpoints

Trained retriever weights are on Hugging Face:
**[libertywing/FlashMemory-Deepseek-V4](https://huggingface.co/libertywing/FlashMemory-Deepseek-V4)**

Download `checkpoints/` into the repo root. The default checkpoint is
`top3_R930_joint.pt` (3-layer joint, CSA layers 10/12/20). Other options:
`top1_R932` / `top2_R935` with joint + per-layer variants.

---

## Core source files

| File | Purpose |
|------|---------|
| `sglang/python/sglang/srt/layers/attention/compressed/retriever_hook_impl.py` | `_TrainedScorer` + `MultiRetrieverHook` (scoring + mask) |
| `sglang/python/sglang/srt/layers/attention/compressed/inline_retriever_hook.py` | Mode A eager inline retriever hook |
| `sglang/python/sglang/srt/layers/attention/compressed/resident_mask_capturer.py` | Mode B cuda-graph two-level recall (Level 1 side-band + Level 2 in-graph mask) |
| `sglang/python/sglang/srt/layers/attention/compressed/swap_engine_p.py` | Path-P / Path-I KV offload engine (CPU mirror + LRU recall) |
| `sglang/python/sglang/srt/models/deepseek_v4.py` | `_make_score_hook` — wires retriever into DSv4 decode |
| `start_server.sh` | Mode A launch script |
| `high_concurrency/launch_*.sh` | Mode B PD-disaggregation launch scripts |

---

## License

MIT

---

## Citation

If you use FlashMemory in your research, please cite:

```bibtex
@article{wang2026flashmemory,
  title   = {FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention},
  author  = {Yan Wang and Qifan Zhang and Jiachen Yu and Tian Liang and Dongyang Ma and
             Xiang Hu and Zibo Lin and Chunyang Li and Zhichao Wang and Jia Li and
             Yujiu Yang and Haitao Mi and Dong Yu},
  year    = {2026},
  journal = {arXiv preprint arXiv:2606.09079},
  url     = {https://huggingface.co/papers/2606.09079},
}
```
