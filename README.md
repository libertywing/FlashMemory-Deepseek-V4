# FlashMemory DS-V4 Retriever

A lightweight retriever that sparsifies **DeepSeek-V4 Compressed-Sparse-Attention (CSA) KV-cache**.

Given the hidden state of a decode token, the retriever predicts which CSA
KV-cache chunks the next ~64 tokens will attend to. Only the top-scoring chunks
stay resident on the GPU; the rest can be offloaded to CPU/disk. In downstream
evaluation it matches or beats the full-attention baseline while keeping **~10–15%
of the KV cache** on-device.

**[Model weights on Hugging Face](https://huggingface.co/libertywing/FlashMemory-Deepseek-V4)**

---

## Quick start

```bash
pip install torch safetensors

# Demo with mock inputs
python demo.py --ckpt weights/flashmemory_ds_v4.safetensors

# Toy sparse-decode loop
python toy_flashmemory_inference.py --ckpt weights/flashmemory_ds_v4.safetensors
```

## Usage

```python
from retriever import FlashMemoryRetriever

model = FlashMemoryRetriever.from_checkpoint(
    "weights/flashmemory_ds_v4.safetensors", device="cuda"
)

# hidden:     [B, 4096] decode-token hidden state
# comp_k:     [B, N, 132] uint8 compressed CSA keys
# positions:  [B] int64 token positions

# Per-layer sigmoid scores: {"l10": [B,N], "l12": [B,N], "l20": [B,N]}
per_layer = model(hidden, comp_k, positions)

# Cross-layer ensemble (mode="max" or "mean")
scores = model.ensemble(hidden, comp_k, positions, mode="max")       # [B, N]

# Boolean keep mask
keep = model.select_topk(hidden, comp_k, positions, top_k=512)       # top-K
keep = model.select_topk(hidden, comp_k, positions, threshold=0.5)   # threshold
```

### `compressed_k` format

Each chunk = `HEAD_DIM + 4 = 132` `uint8` bytes:

| Bytes | Type | Meaning |
|-------|------|---------|
| `[:128]` | `float8_e4m3` | Quantized key values |
| `[128:132]` | `float32` | Per-chunk dequant scale |

Dequant: `fp8_values.view(float8_e4m3).float() * scale`.  
See `make_mock_compressed_k()` in `demo.py`.

---

## Architecture

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

---

## Toy inference reference (`toy_flashmemory_inference.py`)

A self-contained illustration of how the retriever drives memory recall during
decode — the actual control flow used inside DeepSeek-V4-FlashMemory.

### Inference flow

```
 ┌──────────┐  compress & store    ┌────────────────────────────┐
 │ PREFILL  │  historical K/V      │  CSA KV-cache (the memory) │
 │ (dense   │ ───────────────────► │  N compressed chunks,      │
 │  attn)   │                      │  each = [132] uint8 fp8-K  │
 └────┬─────┘                      └──────────────┬─────────────┘
      │ last hidden state                         │ scored every 64 steps
      ▼                                           │
 ┌──────────────────────── DECODE LOOP ──────────┼──────────────────────────┐
 │ for each decode step t:                       │                          │
 │   hidden = toy_decoder.step(token, keep_mask)  │  (sparse memory attn)   │
 │                                               │                          │
 │   every RETRIEVAL_INTERVAL (= 64) steps:      ▼                          │
 │     scores[N]   = retriever.ensemble(hidden, compressed_k, pos)          │
 │     keep_mask[N] = top-K (or sigmoid>thresh) of scores                   │
 │     -> unselected chunks masked to -inf in next 64 steps                 │
 └──────────────────────────────────────────────────────────────────────────┘
```

1. **Prefill (dense).** Short prompt runs through dense memory attention. Its
   last hidden state seeds the first retrieval cycle.
2. **Decode loop.** Toy decoder produces a `[B, 4096]` hidden state each step.
3. **Retrieval cycle (every 64 steps).** The real `FlashMemoryRetriever` scores
   all `N` compressed-K chunks, ensembles per-layer scores, selects keep chunks.
4. **Sparse attention.** Unselected chunks' attention logits are set to `-inf`.

### What this simulates

- **This toy does NOT** perform real CPU↔GPU KV-cache transfer. The swap engine
  is internal FlashMemory infrastructure and is not included.
- **We simulate memory recall by masking attention logits** to `-inf`. A masked
  chunk contributes nothing to attention — the same effect as not loading its KV.
- The purpose is to make the **decode-time control flow concrete**.

### What it is / is NOT

| IS | IS NOT |
|----|--------|
| Minimal torch-only illustration of memory recall | A runnable DeepSeek-V4 |
| Uses the real retriever weights & scoring math | Production KV swap engine |
| Pedagogical: shows the control flow | Meaningful text generation |

The production version depends on the internal sglang + DeepSeek-V4 CSA framework
(native FP8 indexer, real compressed KV-cache, attention-sink, threshold fallback,
per-request routing, actual KV swap) and **cannot be released**.

---

## Downstream evaluation

FlashMemory DS-V4 beats or ties the full-attention baseline on reasoning-heavy
long-context tasks while keeping only **~10–15% of CSA KV cache** on-device:

| Task | Context | vs. Full-Attn | KV Saved |
|------|---------|:---:|------|
| RULER (64k–512k) | 64K–512K | −1 ~ +2 pp | ~80–90% |
| LongMemEval-s | 125K | ±1 pp | ~86% |
| LongMemEval-m | 500K | ±1 pp | ~91% |
| LongBench V2 | 46K–493K | +1 ~ +2 pp | ~73–90% |
| MRCR (needle) | 274K | needs fallback | ~86% |

Precise needle-retrieval tasks (MRCR) require an additional **threshold-fallback**
in the serving layer — this is not part of the standalone release.

---

## Files

| File | Purpose |
|------|---------|
| `retriever.py` | `FlashMemoryRetriever` model + RoPE/Hadamard + FP8 dequant |
| `demo.py` | Minimal demo with mock inputs |
| `toy_flashmemory_inference.py` | Toy sparse-decode loop |
| `weights/flashmemory_ds_v4.safetensors` | Trained weights (~510 MB, on Hugging Face) |
| `requirements.txt` | Dependencies |

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
