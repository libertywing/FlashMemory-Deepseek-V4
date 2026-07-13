# Retriever 训练数据生产 (Data Dump) 指南

本文档介绍如何用改造过的 sglang server + DeepSeek-V4，从推理过程中 **dump 出训练独立 retriever（Lightning Indexer）所需的数据**。

产出的每条数据包含：prompt 阶段的 hidden states、compressed K-cache、以及逐 decode-token 的 golden-chunk 标签。

---

## 0. 原理速览

DeepSeek-V4 的 CSA（Compressed Sparse Attention）在每个 decode step、每个 CSA layer 会对历史 chunk 打分并 top-k 选择。我们在 sglang 里挂了一个 hook：

- **Prefill 阶段**：记录 prompt 的 compressed K-cache（`compk_layer_*`）和 hidden states。
- **Decode 阶段**：每个 token 在 21 个 CSA layer 上做 top-p 过滤，统计每个 chunk 被多少层选中；被 `≥ min_layers` 层选中的 chunk 记为该 token 的 **golden chunk**（label）。

Client 通过 `/tmp/dsv4_tracker_cmd.json` 文件协议告诉 server 进入 dump 模式；server 按 per-request 隔离状态累积，请求结束后 flush 成 `.pkl`。

---

## 1. 文件清单

| 文件 | 作用 |
|---|---|
| `start_dump_server.sh` | 启动 **dump 专用** sglang server（区别于 `start_server.sh` 的推理验证模式） |
| `run_dump_training_data.py` | Client：并发发推理请求、驱动 dump、收集产物 |
| `experiment_utils.py` | Client 依赖：输入格式检测、消息构建、cmd 文件读写 |
| `example_data/creative_writing_multiturn_filtered.subset10.jsonl` | 10 条示例输入（见 §3） |

---

## 2. 快速开始

### Step 1 — 启动 dump server

```bash
# MODEL 指向本机的 DeepSeek-V4 FP8 权重目录
MODEL=path/to/ds_fp8 TP=8 PORT=30000 bash start_dump_server.sh
```

关键点（脚本已内置，勿手动破坏）：
- **`--disable-cuda-graph`**：必须。否则 decode 被 CUDA graph 捕获，dump hook 的 Python side-effect 不执行 → 只产出空目录。
- **不设 `SGLANG_RETRIEVER_INLINE` / `SGLANG_PATHP_CUDAGRAPH`**：这两个会让 dump 的 `score_hook` 被跳过。脚本已 `unset`。
- **`no_proxy` 含本地地址**：避免 warmup 自请求被 HTTP 代理劫持导致 server 被 Killed。脚本已设。

等待日志出现 `Application startup complete` + warmup 通过，然后确认：

```bash
curl -s http://localhost:30000/health    # 返回 OK 即 ready
```

### Step 2 — 跑 client 生产数据

```bash
# MODEL_PATH 必须和 server 加载的模型一致
MODEL_PATH=path/to/ds_fp8 python run_dump_training_data.py \
  --input  example_data/creative_writing_multiturn_filtered.subset10.jsonl \
  --output-dir /ABSOLUTE/path/to/output/creative_writing \
  --start-idx 0 --end-idx 10 \
  --thinking --topp 0.6 --min-layers 3 \
  --concurrency 32 --batch-size 100
```

> ⚠️ **`--output-dir` 必须用绝对路径**。该路径由 client 写进 cmd 文件、再由 **server 进程**按自己的 cwd 落盘；用相对路径时 client 和 server 的 cwd 可能不一致，导致产物写到别处（表现为"目录建了但没有 pkl"）。

### Step 3 — 自检（强烈建议先跑 1 条）

正式上量前，先 `--end-idx 1` 跑一条，确认通路：

```bash
rm -f /tmp/debug_rids.txt
# ... 跑一条 ...
cat /tmp/debug_rids.txt         # 有内容且 rids=[...] 非 None → hook 通路 OK
ls /ABSOLUTE/path/to/output/creative_writing/*.pkl   # 应出现 doc_00000.pkl
```

---

## 3. 输入格式

输入是 **JSONL**，每行一个 JSON 对象（即 experiment_utils 的 "formated / MRCR" 格式）。字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `prompt` | str | **JSON 字符串**，parse 后是一个 messages list（`[{"role","content"}, ...]`）。**末尾不含最终 assistant 答案**（防止答案泄漏进 prompt）。 |
| `answer` | str | gold answer（可选，dump 本身不强制用，用于对齐/评测） |
| `random_string_to_prepend` | str | 存在此字段即触发 `is_mrcr=True` 分支，让 client 走 `json.loads(prompt)` 路径。可为空串 `""`。 |
| `golden_chunks` | list | 可选。`[{"prefix","suffix"}, ...]`，供下游标签过滤用；dump 阶段不需要。 |

示例（取自 `example_data/creative_writing_multiturn_filtered.subset10.jsonl` 第 1 条，多轮对话）：

```json
{
  "prompt": "[{\"role\": \"system\", \"content\": \"You are a helpful assistant...\"}, {\"role\": \"user\", \"content\": \"I want to create a commercial appraisal report...\"}, {\"role\": \"assistant\", \"content\": \"Sure, I'd be happy to help...\"}, ... ]",
  "answer": "To create user interface wireframes, you can follow these steps: ...",
  "golden_chunks": [{"prefix": "As we have updated the project charter ...", "suffix": "..."}],
  "random_string_to_prepend": ""
}
```

- 顶层 4 个字段：`prompt` / `answer` / `golden_chunks` / `random_string_to_prepend`
- 上面这条 `prompt` parse 后是 **136 条 messages**（system + user/assistant 多轮）；client 会把它作为推理输入发给 server。

> 想自己造输入：把任意 messages list 用 `json.dumps` 序列化成字符串塞进 `prompt`，加上空的 `random_string_to_prepend`，即可。

---

## 4. 输出格式

`--output-dir` 下会生成：

```
<output-dir>/
├── doc_00000.pkl          # 每条输入一个 pkl（tensor 字典）
├── doc_00000.json         # 对应的 prompt/response/原始索引
├── doc_00001.pkl
├── doc_00001.json
├── ...
└── server_metadata.jsonl  # 每行一条产物的元信息
```

（另外还会临时生成一个 `<output-dir>_compk_full/` 目录，用于第二轮取完整 compressed K，合并回 pkl 后可忽略。）

### 4.1 `doc_XXXXX.pkl`（`pickle` 存的 dict[str, torch.Tensor]）

以 `--target-csa-layers 6 8 10 12 14 16 18 20`（默认）、prompt 有 `P` 个 compressed block、decode 出 `T` 个 token 为例：

| key | shape | dtype | 含义 |
|---|---|---|---|
| `hidden_layer_{L}` | `[T, 4096]` | bfloat16 | 第 L 个 CSA layer、每个 decode token 的 hidden state |
| `compk_layer_{L}` | `[P, 132]` | uint8 | 第 L 层 prompt 的 compressed K（前 128 = FP8 key，后 4 = fp32 scale 的字节） |
| `positions_layer_{L}` | `[T]` | int64 | 每个 decode token 的绝对位置 |
| `label_indices` | `[N]` | int32 | 所有 decode token 的 golden chunk id（CSR 展平） |
| `label_scores` | `[N]` | int32 | 对应 chunk 被多少层选中（layer hits） |
| `label_pointers` | `[T+1]` | int32 | CSR 行指针：token t 的 label 是 `label_indices[pointers[t]:pointers[t+1]]` |
| `logits_layer_{L}_token_{0,1,2}` | `[P]` | bfloat16 | 前 3 个 decode token 在各层的完整 chunk logits（调试/验证用） |
| `q_layer_{L}_token_{0,1,2}` | `[1, 8192]` | bfloat16 | 前 3 token 的 query（含 RoPE+Hadamard，验证用） |
| `weights_layer_{L}_token_{0,1,2}` | `[1, n_heads]` | bfloat16 | 前 3 token 的 fused weights（验证用） |

**labels 的 CSR 读法**：第 `t` 个 decode token 的 golden chunks =
`label_indices[label_pointers[t] : label_pointers[t+1]]`，对应分数在 `label_scores` 同区间。

### 4.2 `doc_XXXXX.json`

| 字段 | 说明 |
|---|---|
| `prompt` | 实际发给 server 的 messages list |
| `response` | 模型生成的文本 |
| `original_index` | 该条在输入 jsonl 中的行号（0-based，含 start-idx 偏移） |
| `rid` | server 端 request id（与 metadata 对齐用） |

### 4.3 `server_metadata.jsonl`

每行对应一个 `doc_XXXXX.pkl`，字段：`doc_index` / `rid` / `filename` / `n_prompt_blocks`(=P) / `n_decode_tokens`(=T) / `n_total_labels`(=N) / `hidden_shapes` / `compk_shapes` / `logits_shapes` / `target_csa_layers`。

---

## 5. 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | 必填 | 输入 jsonl（见 §3） |
| `--output-dir` | 必填 | 输出目录，**用绝对路径** |
| `--start-idx` / `--end-idx` | 0 / 末尾 | 处理 `items[start:end]`，支持分段续跑 |
| `--n-samples` | None | 只取前 N 条（与 start/end 二选一） |
| `--thinking` | off | 开启 thinking 模式（与 server 的 chat template 对应） |
| `--topp` | 0.6 | 每层 top-p 过滤阈值 |
| `--min-layers` | 3 | chunk 需被 ≥ 该层数选中才记为 golden |
| `--target-csa-layers` | `6 8 10 12 14 16 18 20` | 记录哪些 CSA layer（0-indexed）的 hidden/compk |
| `--concurrency` | 4 | 并发请求数 |
| `--batch-size` | 10 | 每批处理条数（一批走完一轮生成+二轮 compk+合并） |

输出目录支持**追加续跑**：client 会读已有 `server_metadata.jsonl` 的最大 `doc_index`，从下一个编号继续。

---

## 6. 排错

| 现象 | 原因 | 解决 |
|---|---|---|
| 目录建了但没有 `.pkl`/`.json` | `--output-dir` 用了相对路径；或漏了 `--disable-cuda-graph`；或误设了 `SGLANG_RETRIEVER_INLINE`/`SGLANG_PATHP_CUDAGRAPH` | 用 `start_dump_server.sh` 启动 + `--output-dir` 绝对路径 |
| server 启动即被 `Killed`，日志有 Squid/503 `ERR_CONNECT_FAIL` | warmup 自请求被 HTTP 代理劫持 | `start_dump_server.sh` 已设 `no_proxy`；确认没被覆盖 |
| `HFValidationError: Repo id must be...` | `MODEL` 路径在本机不存在，被当成 HF repo | 用 `MODEL=` 指向本机真实权重目录 |
| `/tmp/debug_rids.txt` 里 `rids=None` | `rids` 字段未生效（sglang 文件不完整） | 确认加载的 sglang 含 dump 改动（`deepseek_v4.py` / `indexer.py` / `forward_batch_info.py` / `schedule_batch.py`） |
