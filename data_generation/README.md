# Retriever Training Data Dump — Guide

This document explains how to use the modified sglang server + DeepSeek-V4 to **dump the data needed for training a standalone retriever (Lightning Indexer)** during inference.

Each produced record contains: the prompt-phase hidden states, the compressed K-cache, and per-decode-token golden-chunk labels.

---

## 0. How It Works (Quick Overview)

DeepSeek-V4's CSA (Compressed Sparse Attention) scores historical chunks and does top-k selection at every decode step, at every CSA layer. We attach a hook inside sglang:

- **Prefill phase**: records the prompt's compressed K-cache (`compk_layer_*`) and hidden states.
- **Decode phase**: for each token, does top-p filtering across the 21 CSA layers and counts how many layers select each chunk; a chunk selected by `≥ min_layers` layers is recorded as that token's **golden chunk** (label).

The client tells the server to enter dump mode via a file protocol at `/tmp/dsv4_tracker_cmd.json`; the server accumulates per-request isolated state and flushes to `.pkl` once a request finishes.

---

## 1. File Manifest

| File | Purpose |
|---|---|
| `start_dump_server.sh` | Launches a **dump-dedicated** sglang server (distinct from `start_server.sh`'s inference-verification mode) |
| `run_dump_training_data.py` | Client: sends concurrent inference requests, drives the dump, collects outputs |
| `experiment_utils.py` | Client dependency: input-format detection, message building, cmd-file read/write |
| `example_data/creative_writing_multiturn_filtered.subset10.jsonl` | 10 example inputs (see §3) |

---

## 2. Quick Start

### Step 1 — Launch the dump server

```bash
# MODEL points to the local DeepSeek-V4 FP8 weights directory
MODEL=path/to/ds_fp8 TP=8 PORT=30000 bash start_dump_server.sh
```

Key points (already baked into the script — do not break them manually):
- **`--disable-cuda-graph`**: required. Otherwise decode gets captured by CUDA graph and the dump hook's Python side-effects never run → you only get empty directories.
- **Do NOT set `SGLANG_RETRIEVER_INLINE` / `SGLANG_PATHP_CUDAGRAPH`**: these cause the dump `score_hook` to be skipped. The script already `unset`s them.
- **`no_proxy` includes local addresses**: prevents the warmup self-request from being hijacked by an HTTP proxy, which would get the server Killed. The script already sets this.

Wait for the log to show `Application startup complete` and warmup to pass, then confirm:

```bash
curl -s http://localhost:30000/health    # returns OK when ready
```

### Step 2 — Run the client to produce data

```bash
# MODEL_PATH must match the model loaded by the server
MODEL_PATH=path/to/ds_fp8 python run_dump_training_data.py \
  --input  example_data/creative_writing_multiturn_filtered.subset10.jsonl \
  --output-dir /ABSOLUTE/path/to/output/creative_writing \
  --start-idx 0 --end-idx 10 \
  --thinking --topp 0.6 --min-layers 3 \
  --concurrency 32 --batch-size 100
```

> ⚠️ **`--output-dir` must be an absolute path.** This path is written into the cmd file by the client and then used by the **server process**, which writes files relative to its own cwd. With a relative path, the client's and server's cwd may differ, so outputs land elsewhere (symptom: "directory got created but there are no pkl files").

### Step 3 — Self-check (strongly recommended: run 1 sample first)

Before scaling up, run a single sample with `--end-idx 1` to confirm the pipeline:

```bash
rm -f /tmp/debug_rids.txt
# ... run one sample ...
cat /tmp/debug_rids.txt         # has content and rids=[...] is not None → hook path OK
ls /ABSOLUTE/path/to/output/creative_writing/*.pkl   # doc_00000.pkl should appear
```

---

## 3. Input Format

The input is **JSONL**, one JSON object per line (i.e., experiment_utils' "formated / MRCR" format). Fields:

| Field | Type | Description |
|---|---|---|
| `prompt` | str | A **JSON string** that parses into a messages list (`[{"role","content"}, ...]`). **Does NOT include the final assistant answer** (to prevent answer leakage into the prompt). |
| `answer` | str | Gold answer (optional; not strictly required by the dump itself, used for alignment/evaluation). |
| `random_string_to_prepend` | str | The presence of this field triggers the `is_mrcr=True` branch, making the client take the `json.loads(prompt)` path. May be an empty string `""`. |
| `golden_chunks` | list | Optional. `[{"prefix","suffix"}, ...]`, used for downstream label filtering; not needed during the dump phase. |

Example (taken from the 1st record of `example_data/creative_writing_multiturn_filtered.subset10.jsonl`, a multi-turn conversation):

```json
{
  "prompt": "[{\"role\": \"system\", \"content\": \"You are a helpful assistant...\"}, {\"role\": \"user\", \"content\": \"I want to create a commercial appraisal report...\"}, {\"role\": \"assistant\", \"content\": \"Sure, I'd be happy to help...\"}, ... ]",
  "answer": "To create user interface wireframes, you can follow these steps: ...",
  "golden_chunks": [{"prefix": "As we have updated the project charter ...", "suffix": "..."}],
  "random_string_to_prepend": ""
}
```

- The 4 top-level fields: `prompt` / `answer` / `golden_chunks` / `random_string_to_prepend`
- The `prompt` above parses into **136 messages** (system + multiple user/assistant turns); the client sends it as the inference input to the server.

> To build your own input: serialize any messages list into a string with `json.dumps` and put it in `prompt`, add an empty `random_string_to_prepend`, and you're done.

---

## 4. Output Format

The following are generated under `--output-dir`:

```
<output-dir>/
├── doc_00000.pkl          # one pkl per input (a tensor dict)
├── doc_00000.json         # corresponding prompt/response/original index
├── doc_00001.pkl
├── doc_00001.json
├── ...
└── server_metadata.jsonl  # one line of metadata per output
```

(A temporary `<output-dir>_compk_full/` directory is also created, used by the second pass to fetch the full compressed K; it can be ignored after being merged back into the pkl.)

### 4.1 `doc_XXXXX.pkl` (a `pickle`-stored dict[str, torch.Tensor])

Using `--target-csa-layers 6 8 10 12 14 16 18 20` (default), with `P` compressed blocks in the prompt and `T` decoded tokens:

| key | shape | dtype | Meaning |
|---|---|---|---|
| `hidden_layer_{L}` | `[T, 4096]` | bfloat16 | Hidden state per decode token at the L-th CSA layer |
| `compk_layer_{L}` | `[P, 132]` | uint8 | Prompt's compressed K at layer L (first 128 = FP8 key, last 4 = bytes of the fp32 scale) |
| `positions_layer_{L}` | `[T]` | int64 | Absolute position of each decode token |
| `label_indices` | `[N]` | int32 | Golden chunk ids across all decode tokens (CSR-flattened) |
| `label_scores` | `[N]` | int32 | How many layers selected the corresponding chunk (layer hits) |
| `label_pointers` | `[T+1]` | int32 | CSR row pointers: token t's labels are `label_indices[pointers[t]:pointers[t+1]]` |
| `logits_layer_{L}_token_{0,1,2}` | `[P]` | bfloat16 | Full chunk logits of the first 3 decode tokens per layer (for debugging/verification) |
| `q_layer_{L}_token_{0,1,2}` | `[1, 8192]` | bfloat16 | Query of the first 3 tokens (with RoPE+Hadamard, for verification) |
| `weights_layer_{L}_token_{0,1,2}` | `[1, n_heads]` | bfloat16 | Fused weights of the first 3 tokens (for verification) |

**How to read the CSR labels**: the golden chunks of the `t`-th decode token =
`label_indices[label_pointers[t] : label_pointers[t+1]]`, with the corresponding scores in the same range of `label_scores`.

### 4.2 `doc_XXXXX.json`

| Field | Description |
|---|---|
| `prompt` | The messages list actually sent to the server |
| `response` | The text generated by the model |
| `original_index` | The line number of this record in the input jsonl (0-based, includes the start-idx offset) |
| `rid` | The server-side request id (used to align with metadata) |

### 4.3 `server_metadata.jsonl`

Each line corresponds to one `doc_XXXXX.pkl`, with fields: `doc_index` / `rid` / `filename` / `n_prompt_blocks`(=P) / `n_decode_tokens`(=T) / `n_total_labels`(=N) / `hidden_shapes` / `compk_shapes` / `logits_shapes` / `target_csa_layers`.

---

## 5. Common Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Input jsonl (see §3) |
| `--output-dir` | required | Output directory, **use an absolute path** |
| `--start-idx` / `--end-idx` | 0 / end | Processes `items[start:end]`, supports resumable segmented runs |
| `--n-samples` | None | Take only the first N records (mutually exclusive with start/end) |
| `--thinking` | off | Enable thinking mode (matches the server's chat template) |
| `--topp` | 0.6 | Per-layer top-p filtering threshold |
| `--min-layers` | 3 | A chunk must be selected by ≥ this many layers to count as golden |
| `--target-csa-layers` | `6 8 10 12 14 16 18 20` | Which CSA layers (0-indexed) to record hidden/compk for |
| `--concurrency` | 4 | Number of concurrent requests |
| `--batch-size` | 10 | Records per batch (each batch runs pass-1 generation + pass-2 compk + merge) |

The output directory supports **resumable appending**: the client reads the maximum `doc_index` in an existing `server_metadata.jsonl` and continues from the next number.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Directory created but no `.pkl`/`.json` | `--output-dir` used a relative path; or `--disable-cuda-graph` was omitted; or `SGLANG_RETRIEVER_INLINE`/`SGLANG_PATHP_CUDAGRAPH` was set by mistake | Launch with `start_dump_server.sh` + use an absolute `--output-dir` |
| Server is `Killed` immediately on startup, log shows Squid/503 `ERR_CONNECT_FAIL` | The warmup self-request was hijacked by an HTTP proxy | `start_dump_server.sh` already sets `no_proxy`; confirm it wasn't overridden |
| `HFValidationError: Repo id must be...` | The `MODEL` path doesn't exist locally and was treated as an HF repo | Use `MODEL=` to point to the real local weights directory |
| `rids=None` in `/tmp/debug_rids.txt` | The `rids` field isn't in effect (incomplete sglang files) | Confirm the loaded sglang includes the dump changes (`deepseek_v4.py` / `indexer.py` / `forward_batch_info.py` / `schedule_batch.py`) |
