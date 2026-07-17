# CHANGES — release cleanup (2026-07-17)

本次对 release 代码库的整理改动汇总，供 review 后更新 git。所有改动已在本机用
**本仓库自带的 sglang** 冒烟验证跑通（推理 + 数据生成 + 训练全链路）。

## 1. 移除 masking（单一推理路径）

删除单机 score-masking「伪推理」路径，避免与 offload（PD 分离）路径并存造成困惑：

- **删** `start_server.sh`（单机 masking 启动脚本）。
- **删** `deepseek_v4.py:_make_score_hook` 里的 `SGLANG_RETRIEVER_INLINE=1` inline 推理入口块
  （masking 的唯一真实入口）。保留 `pass1/pass2/dump_training` 逻辑（data_generation 依赖）。
- **保留** `inline_retriever_hook.py` 的 `InlineRetrieverHook` 类 —— 它被 offload 路径的
  `resident_mask_capturer`（Level-1 side-band scorer）复用，不是独立的 masking 路径。
- README / TECHNICAL_REPORT（zh+en）删去 Mode A / score-masking 段落。

唯一保留的推理路径 = **PD 分离 offload**（Path-P c4 KV offload + Path-I index-K offload +
两级 page-recall + cuda-graph）。

## 2. 更新 sglang（本机验证版）

- 用本机跑通的 sglang（editable `/workspace/sglang`）整份替换仓库 `sglang/python/`。
- 相比旧版主要差异：删除了 `disaggregation/decode.py` 里的 benchmark barrier / admission
  相关代码（`SGLANG_PD_DECODE_BARRIER` / `ADMIT_CAP` / `ADMIT_STRIDE`）—— 实测无 barrier
  下自然到达即可稳态测吞吐，且 barrier 在 1M 会触发启动死锁。

## 3. 脚本上移主目录

- `high_concurrency/{launch_decode,launch_prefill,launch_router}.sh` → 移到仓库主目录。
- 删除空的 `high_concurrency/`。
- `launch_decode.sh`：修 checkpoints 相对路径（`../checkpoints` → `checkpoints`）；
  更新头部注释（去 masking 对比）+ 尾部实测甜点配置。

## 4. 更新 technical_report + requirements

- report（zh+en）：删 masking 描述 + benchmark barrier 描述；6.1 结果表更新为最新实测
  （256K 76/2759, 512K 60/2008, 1M 40/1537 tok/s）；上限成因改述（cuda-graph 预分配 batch≈80 /
  1M host-DRAM 墙）。
- `requirements.txt`：补 data_generation（`openai`）+ training（torch/safetensors/numpy 已覆盖）依赖。

## 5. 新增 training/

- 从内部 `release/training/` 拷入 `training/`（train.py / dataloader.py / inference.py /
  eval.py / utils.py / verify.py / train_ddp.sh / mine_hard_negs.py）。
- `train.py` 的 `DATA_CONFIGS` 加一个 `smoke` 示例配置（占位路径，指向 data_generation 输出目录）。

## 验证（本机冒烟，全部用本仓库 sglang）

| 环节 | 结果 |
|------|------|
| **推理**（256K PD offload） | pred 正确（correct=1/1），两级 recall 正常跑，无崩溃 |
| **数据生成**（example 10 条） | 生成 `doc_*.pkl`，字段/shape/dtype 全对（hidden [T,4096] bf16 / compk [N,132] u8 / label CSR int32） |
| **训练**（生成数据） | data_gen→training 衔接通，loss 下降（avg_loss=0.055），ckpt 落盘 |
| **全量 py_compile** | 所有改动文件通过 |

## 目录结构（最终）

```
FlashMemory-Deepseek-V4/
├── sglang/                     # 本机验证版 sglang 源码
├── launch_decode.sh            # PD-D (decode, all offload/recall)
├── launch_prefill.sh           # PD-P (prefill)
├── launch_router.sh            # router
├── data_generation/            # Stage-1 数据生产
├── training/                   # retriever 训练
├── TECHNICAL_REPORT_{en,zh}.md
├── requirements.txt
└── (无 start_server.sh / 无 high_concurrency/)
```
