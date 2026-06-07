---
language: en
license: mit
tags:
- deepseek-v4
- retrieval
- kv-cache
- sparse-attention
- compress-sparse-attention
- long-context
- flashmemory
datasets:
- ruler
- longmemeval
- longbench-v2
- mrcr
---

# FlashMemory DS-V4 Retriever

A standalone, dependency-light reference implementation of the **FlashMemory DS-V4
Retriever** — a lightweight retriever that sparsifies the **DeepSeek-V4
Compressed-Sparse-Attention (CSA)** KV cache.

Given the hidden state of a decode token, the retriever predicts which CSA
KV-cache chunks (compressed keys) the upcoming tokens will attend to, so that
only the **top-scoring chunks** need to stay resident on the GPU and the rest can
be offloaded to CPU / disk. This recovers most of the quality of full attention
on long-context tasks while keeping a small fraction of the KV cache on-device.

This release contains the **algorithm + weights + a minimal, runnable PyTorch
demo**. It depends only on `torch` (plus `numpy` / `safetensors` for convenience).

> **Scope note.** The full sglang serving integration — KV-cache swap-in/out,
> attention-sink, threshold fallback, per-request retriever routing — is **not**
> included here, because it is tightly coupled to the internal DeepSeek-V4 CSA
> framework and cannot run outside it. This repository provides the retriever
> **algorithm reference implementation and trained weights only.**

---

## Model architecture

The retriever scores each compressed-K chunk against the decode token's hidden
state. For a single CSA layer:

```
hidden [B, 4096]
    → wq_a        (4096 → Q_LORA_RANK)
    → RMSNorm     (q_norm_weight, eps=1e-6)
    → wq_b        (Q_LORA_RANK → N_HEADS * HEAD_DIM)
    → reshape     [B, N_HEADS, HEAD_DIM]
    → RoPE        (YaRN, applied to the last ROPE_DIM=64 dims, base=160000)
    → Hadamard    (normalized Walsh-Hadamard transform)
    → q           [B, N_HEADS, HEAD_DIM]

hidden [B, 4096]
    → weights_proj (4096 → N_HEADS)
    → × weight_scale          (= HEAD_DIM^-0.5 * N_HEADS^-0.5)
    → fused_w     [B, N_HEADS]

compressed_k [B, N, HEAD_DIM + 4] (uint8)
    → bytes[:HEAD_DIM]  viewed as float8_e4m3   → dequantize
    → bytes[HEAD_DIM:]  viewed as float32        → per-chunk scale
    → k           [B, N, HEAD_DIM]

score_per_head = relu( einsum('bnd,bhd->bnh', k, q) )           # [B, N, N_HEADS]
logit          = (score_per_head * fused_w[:, None, :]).sum(-1) # [B, N]
score          = sigmoid(logit)  ∈ [0, 1]                       # [B, N]
```

**Hyperparameters (FlashMemory DS-V4):** `Q_LORA_RANK = 2048`, `N_HEADS = 128`,
`HEAD_DIM = 128`, `ROPE_DIM = 64`, `ROPE_BASE = 160000`, `ROPE_FACTOR = 16`,
`ROPE_ORIGINAL_SEQ_LEN = 65536`, `ROPE_BETA_FAST = 32`, `ROPE_BETA_SLOW = 1`,
`RMS_NORM_EPS = 1e-6`.

### Joint multi-layer checkpoint + ensemble

FlashMemory DS-V4 is a **joint checkpoint** holding three independent CSA layers
(`l10`, `l12`, `l20`), each with its own weights. At inference time the per-layer
sigmoid scores are **ensembled per chunk** — cross-layer `max` (default) or
`mean` — to produce a single keep/drop decision per chunk.

---

## What is FlashMemory DS-V4?

FlashMemory DS-V4 is part of the latest retraining generation of these retrievers. In the
project's downstream evaluation it stays close to the full-attention baseline on
long-context tasks (e.g. RULER, LongMemEval, LongBench V2) while keeping only a
small fraction of the CSA KV cache on-device (≈90% KV reduction in the deployment
sweet spot for reasoning-heavy long-context tasks). Precise-needle retrieval
tasks need an extra threshold-fallback mechanism in the serving layer (not part
of this standalone release).

---

## Installation

```bash
pip install -r requirements.txt
```

Only `torch` is strictly required to run the model and demo. `float8_e4m3`
tensor support requires a reasonably recent PyTorch (≥ 2.1).

---

## Running the demo

```bash
python demo.py --ckpt weights/flashmemory_ds_v4.safetensors
```

The demo builds **random mock inputs** (a batch of decode-token hidden states, a
set of `uint8` compressed-K chunks, and token positions), loads the FlashMemory DS-V4
checkpoint, runs the forward pass, prints the per-layer and ensembled per-chunk
scores, and demonstrates both **threshold** and **top-K** chunk selection.

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batch` | `2` | number of decode tokens |
| `--n-chunks` | `64` | number of compressed-K chunks |
| `--top-k` | `16` | top-K chunks to select |
| `--threshold` | `0.5` | sigmoid keep threshold |
| `--ensemble` | `max` | cross-layer ensemble mode (`max` / `mean`) |
| `--max-position` | `524288` | RoPE table length (raise to `1048576` for 1M context) |

Example output (CPU, default args):

```
[demo] loaded layers=['l10', 'l12', 'l20']  n_heads=128  head_dim=128  max_position=524288
[demo] per-layer sigmoid score stats (over all chunks):
    l10: min=0.4474 mean=0.5021 max=0.6416
    ...
[demo] threshold selection (sigmoid > 0.5):
    row 0: keep 64/64 chunks  (keep ratio 100.0%)
    row 1: keep 49/64 chunks  (keep ratio 76.6%)
[demo] done. ✅  forward + scoring + selection all ran.
```

> The scores above come from **random mock K**, so they cluster near 0.5 — they
> are only meaningful on real CSA keys. The demo's purpose is to verify the
> load → forward → selection path end-to-end.

---

## Using the model in your own code

```python
import torch
from retriever import FlashMemoryRetriever

model = FlashMemoryRetriever.from_checkpoint(
    "weights/flashmemory_ds_v4.safetensors", device="cuda", max_position=524288
)

hidden       = torch.randn(B, 4096, device="cuda")          # decode-token hidden states
compressed_k = ...                                          # [B, N, 132] uint8 CSA keys
positions    = torch.arange(B, device="cuda")               # int64 token positions

# Per-layer sigmoid scores: {"l10": [B, N], "l12": [B, N], "l20": [B, N]}
per_layer = model(hidden, compressed_k, positions)

# Cross-layer ensembled per-chunk scores [B, N] ∈ [0, 1]
scores = model.ensemble(hidden, compressed_k, positions, mode="max")

# Boolean keep-mask [B, N] for the chunks to keep on-device
keep = model.select_topk(hidden, compressed_k, positions, top_k=512)        # top-K
keep = model.select_topk(hidden, compressed_k, positions, threshold=0.5)    # threshold
```

**`compressed_k` format.** Each chunk is `HEAD_DIM + 4 = 132` `uint8` bytes:
the first `128` bytes are the `float8_e4m3` quantized key values, the last `4`
bytes are a single `float32` per-chunk scale. Dequantization is
`fp8_values.view(float8_e4m3).float() * scale`. See `make_mock_compressed_k` in
`demo.py` for how to construct a valid tensor.

---

## Weights

**Download:** [Hugging Face](https://huggingface.co/<HF_REPO>) — `flashmemory_ds_v4.safetensors` (≈510 MB).

```bash
huggingface-cli download <HF_REPO> flashmemory_ds_v4.safetensors --local-dir ./weights
python demo.py --ckpt ./weights/flashmemory_ds_v4.safetensors
```

`from_checkpoint` accepts either a `.pt` (`torch.save` state-dict) or a
`.safetensors` file. The released `.safetensors` is the **slim** form: it stores
only the four learned tensors per layer
(`wq_a.weight`, `wq_b.weight`, `q_norm_weight`, `weights_proj.weight` for
`l10` / `l12` / `l20`) and **omits the `freqs_cis` RoPE table** (≈400 MB), which
is recomputed at load time from `max_position`. Loading the slim `.safetensors`
is bit-for-bit identical to loading the full `.pt` (verified by output match).

---

## Files

| File | Purpose |
|------|---------|
| `retriever.py` | `FlashMemoryRetriever` model + RoPE/Hadamard utils + FP8 dequant (torch-only, self-contained) |
| `demo.py` | minimal runnable demo with mock inputs |
| `toy_flashmemory_inference.py` | toy DeepSeek-V4-FlashMemory sparse-decode loop showing **how the retriever drives memory recall at inference time** (see below) |
| `requirements.txt` | `torch`, `safetensors`, `numpy` |
| `LICENSE` | MIT |

---

## Toy FlashMemory inference reference (`toy_flashmemory_inference.py`)

`demo.py` shows a single `hidden → scores` call. `toy_flashmemory_inference.py`
is the **next step up**: a tiny, fully-runnable illustration of *how the Lightning
Indexer Retriever is used inside a DeepSeek-V4-FlashMemory style sparse-decode
loop* to drive "memory recall".

It is intentionally small and pedagogical. It depends only on `torch` and the
sibling `retriever.py`, and it **reuses the real FlashMemory DS-V4 retriever verbatim** — none
of the scoring math is re-implemented.

### The inference flow it demonstrates

```
 ┌──────────┐  compress & store   ┌────────────────────────────┐
 │ PREFILL  │  historical K/V     │  CSA KV-cache (the memory) │
 │ (dense   │ ──────────────────► │  N compressed chunks,      │
 │  attn)   │                     │  each = [132] uint8 fp8-K  │
 └────┬─────┘                     └──────────────┬─────────────┘
      │ last hidden state                        │ scored every 64 steps
      ▼                                          │
 ┌──────────────────────── DECODE LOOP ─────────┼──────────────────────────┐
 │ for each decode step t:                       │                          │
 │   hidden = toy_decoder.step(token, keep_mask) │  (sparse memory attn)   │
 │                                               │                          │
 │   every RETRIEVAL_INTERVAL (= 64) steps:      ▼                          │
 │     scores[N]   = retriever.ensemble(hidden, compressed_k, pos)          │
 │     keep_mask[N] = top-K  (or  sigmoid > threshold)  of scores           │
 │     → chunks NOT kept are masked to -inf in the next 64 decode steps     │
 │       of memory attention  (== "not recalled onto the GPU")             │
 └──────────────────────────────────────────────────────────────────────────┘
```

1. **Prefill (dense).** A short prompt is run through dense memory attention. Its
   last hidden state seeds the first retrieval cycle (the indexer needs a query
   hidden state to score against). In a real run, prefill is also where the
   historical KV is compressed into the `[N, 132]` `uint8` CSA chunks.
2. **Decode loop.** Every step the toy decoder produces a `[B, 4096]` hidden state
   and attends over the `N` memory chunks.
3. **Retrieval cycle (every 64 steps).** The real `FlashMemoryRetriever` scores all
   `N` compressed-K chunks against the current decode hidden state, ensembles the
   per-layer (`l10`/`l12`/`l20`) sigmoid scores, and selects the chunks to keep —
   either **top-K** or **sigmoid > threshold**. This predicts which chunks the
   *next ~64 tokens* will attend to.
4. **Sparse attention.** For the next 64 steps, chunks **not** selected have their
   memory-attention logits set to `-inf`, so they contribute nothing.

### What the masking simulates (important)

* This toy does **not** perform any real CPU↔GPU KV-cache transfer. The swap-in /
  swap-out machinery is part of the internal FlashMemory engineering and is **not**
  included in this release.
* We **simulate memory recall by masking the FlashMemory Retriever's per-chunk
  decisions**: a chunk the retriever did not select gets its attention logit set
  to `-inf`. This is equivalent to *"that chunk's KV was never recalled onto the
  GPU, so it cannot be attended to"* — for the attention output, masking a chunk
  out and never loading it produce the same result.
* The toy's purpose is to make the **decode-time control flow** concrete: where the
  retriever fires, what it consumes (decode hidden state + compressed CSA keys),
  what it produces (a keep/drop mask), and how that mask sparsifies the next
  window of decode steps.

### What it is / is NOT

* **IS:** a minimal, torch-only illustration of the decode-time control flow that
  drives memory recall with the real FlashMemory DS-V4 retriever.
* **IS NOT:** a runnable DeepSeek-V4. The "decoder" is a couple of layers of
  randomly-initialized toy attention/MLP whose only jobs are (a) to emit a
  `[B, 4096]` hidden state for the retriever and (b) to own a memory attention we
  can sparsify. The generated tokens are meaningless.

> **The production version cannot be released.** It depends on the internal sglang
> + DeepSeek-V4 CSA framework (native FP8 indexer, real compressed KV-cache,
> attention-sink, threshold fallback, per-request routing, and the actual KV swap
> engine). This file shows the *algorithmic role* of the retriever only.

### Run

```bash
python toy_flashmemory_inference.py --ckpt weights/flashmemory_ds_v4.safetensors
```

Runs on CPU by default; pass `--device cuda` for GPU.

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-chunks` | `256` | number of CSA memory chunks (the long history) |
| `--steps` | `192` | decode steps to generate |
| `--retrieval-interval` | `64` | run the retriever every N steps (FlashMemory default) |
| `--select-mode` | `topk` | `topk` or `threshold` |
| `--top-k` | `64` | chunks to recall per cycle (`select-mode=topk`) |
| `--threshold` | `0.5` | sigmoid keep threshold (`select-mode=threshold`) |
| `--ensemble` | `max` | cross-layer ensemble mode (`max` / `mean`) |
| `--batch` | `1` | parallel decode sequences |

Example output (CPU, default args — `top-K=64` out of `256` chunks):

```
FlashMemory DS-V4 — toy sparse-decode loop
[load] weights/flashmemory_ds_v4.safetensors
[load] layers=['l10', 'l12', 'l20']  n_heads=128  head_dim=128
[init] decoder: 2 layers, 8 heads  |  CSA memory: 256 chunks [132] uint8

[decode] 192 steps, retriever every 64 steps (topk [top-K=64], ensemble=max)
------------------------------------------------------------
[cycle  0] pos     8..71    |  keep 25.0% (64/256)  |  score mean=0.4910 max=0.5445
[cycle  1] pos    72..135   |  keep 25.0% (64/256)  |  score mean=0.4910 max=0.5445
...
------------------------------------------------------------
[done] 192 tokens, 3 cycles, avg keep/cycle: 25.0%  →  ~75% CSA KV dropped
[note] Dropped chunks are masked to -inf in attention (= KV not recalled to GPU).
```

> As in `demo.py`, the scores come from **random mock K** and cluster near 0.5;
> they are only meaningful on real CSA keys. The toy's value is the *control flow*
> — watch each retrieval cycle report how many chunks were scored, recalled, and
> masked out.

---


## License

MIT — see [`LICENSE`](./LICENSE).
