---
language: en
license: mit
tags:
- deepseek-v4
- retrieval
- kv-cache
- sparse-attention
- long-context
- flashmemory
datasets:
- ruler
- longmemeval
- longbench-v2
- mrcr
---

# FlashMemory DS-V4 Retriever

A lightweight retriever that sparsifies **DeepSeek-V4 CSA KV-cache**. Given a
decode-token hidden state, it predicts which compressed-K chunks the next
~64 tokens will attend to — keeping only those on GPU, offloading the rest.

In downstream evaluation it matches or beats full-attention baseline on
reasoning-heavy long-context tasks (**RULER, LongMemEval, LongBench V2**)
while reducing KV-cache usage by **~85–90%**. Precise needle-retrieval tasks
require an additional threshold-fallback mechanism (not in this release).

## Quick start

```bash
pip install torch safetensors
python demo.py --ckpt weights/flashmemory_ds_v4.safetensors
```

## Usage

```python
from retriever import FlashMemoryRetriever

model = FlashMemoryRetriever.from_checkpoint(
    "weights/flashmemory_ds_v4.safetensors", device="cuda"
)

# hidden: [B, 4096] decode hidden state
# compressed_k: [B, N, 132] uint8 CSA keys
# positions: [B] int64 token positions

scores = model.ensemble(hidden, compressed_k, positions, mode="max")        # [B, N]
keep   = model.select_topk(hidden, compressed_k, positions, top_k=512)      # boolean mask
```

**`compressed_k` format:** each chunk = 128 bytes `float8_e4m3` values + 4 bytes `float32` scale. See `make_mock_compressed_k()` in `demo.py`.

## Architecture

3-layer joint model (`l10`, `l12`, `l20`), 128 heads, 2048 LoRA rank. Per-layer
sigmoid scores are ensembled (`max` or `mean`) per chunk.

```
hidden [B,4096] → q-proj → RoPE(YaRN) → Hadamard → q [B,128,128]
               → weights_proj → fused_w [B,128]
compressed_k    → FP8 dequant → k [B,N,128]

score = sigmoid( Σ( relu(k @ qᵀ) · fused_w ) )  ∈ [0,1]
```

## Toy inference reference

`toy_flashmemory_inference.py` illustrates how the retriever drives memory
recall during decode: every 64 steps it re-scores all chunks, and unselected
ones are masked from attention (equivalent to "not recalled to GPU").

```bash
python toy_flashmemory_inference.py --ckpt weights/flashmemory_ds_v4.safetensors
```

> The decoder is a few toy layers with random weights — it is **not** a real
> DeepSeek-V4. The retriever, scoring math, and decode-time control flow are real.

## Files

| File | Purpose |
|------|---------|
| `retriever.py` | `FlashMemoryRetriever` model (torch-only, self-contained) |
| `demo.py` | minimal demo with mock inputs |
| `toy_flashmemory_inference.py` | toy sparse-decode loop |
| `weights/flashmemory_ds_v4.safetensors` | trained weights (~510 MB) |

## License

MIT
