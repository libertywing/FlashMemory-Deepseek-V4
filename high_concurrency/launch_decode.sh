#!/bin/bash
# =============================================================================
# launch_decode.sh — PD 分离 · D (decode) server, 高并发推理配置.
#
# 这是 ours 全部并发/显存优化的落地端 (Path-P c4 offload + Path-I index-K
# offload + 两级 page-recall + cuda-graph)。P (prefill) 端只管把长 prompt
# prefill 完、把 KV 传过来; 所有 offload/召回都在 D 端。
#
# 与「单机准确率验证」(../start_server.sh, 纯 score-masking) 的区别:
#   * 那个: 全量 KV 常驻 GPU, retriever 只 mask 选择, 不省显存。
#   * 这个: c4 classical KV + 非-target index-K 真 offload 到 CPU mirror,
#           GPU 只留召回页 → 每请求显存骤降 → 并发上限翻数倍。
#
# 用法 (在 D 机上跑, GPU0-7):
#   TGT_CONC=60 TGT_CTX=524288 CTX_LEN=1100000 bash launch_decode.sh
# 常用配置见文件尾部注释。
# =============================================================================
set -euo pipefail

# ── DeepSeek-V4 模型模式 (CSA + MoE, FP4 experts) ────────────────────────────
export SGLANG_DSV4_MODE=2604 SGLANG_DSV4_2604_SUBMODE=2604B SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_OPT_USE_ONLINE_COMPRESS=1

# ── 核心开关: Path-P (c4 KV offload) + cuda-graph 两级召回 ───────────────────
export SGLANG_DECODE_SWAP_P=1            # 启用 Path-P: c4 classical KV → CPU mirror + GPU reserve
export SGLANG_PATHP_CUDAGRAPH=1          # 两级召回在 cuda-graph 内跑 (vs eager hook)

# ── 两级 Memory-Indexer 召回 (Level-1 选页 side-band, Level-2 in-graph mask) ──
export SGLANG_PATHP_SCORE_RESIDENT="${SCORE_RESIDENT:-1}"  # decode 只对 resident_set 打分 (非全历史)
export SGLANG_PATHP_PAGE_RECALL="${PAGE_RECALL:-1}"        # 页粒度召回 (vs chunk 粒度; 64× 小的 page->block 表)
export SGLANG_PATHP_PAGE_KMAX="${PAGE_KMAX:-96}"           # 每请求最多召回 K_max 个最密页 (96*64=6144 chunk)
export SGLANG_PATHP_RS_NATIVE_TOPK="${RS_NATIVE_TOPK:-1}"  # 用 fused native topk_transform_512
export SGLANG_PATHP_ASYNC_RECALL="${ASYNC_RECALL:-1}"      # 召回移到后台线程 (与下一 64 步 decode 重叠)
export SGLANG_PATHP_PREBUILD_CAPTURE="${PREBUILD_CAPTURE:-1}"  # 预建 capture buffer (score-resident 能进 graph)
export SGLANG_PATHP_FUSED_REMAP="${FUSED_REMAP:-0}"        # top-512 列直出 reserve cell (跳 backend remap)
export SGLANG_RETRIEVER_LAST_KEEP="${LAST_KEEP:-2048}"     # recency tail: 最近 N c4 token 永远保留
export SGLANG_PATHP_DECODE_RESIDENT_CHUNKS=512             # decode-resident 区: 每请求最多 512 decode chunk
export SGLANG_PATHP_DECODE_MAX_CONC=128                    # decode-resident 区支持的最大并发

# ── Path-P reserve / 池尺寸 ──────────────────────────────────────────────────
export SGLANG_PD_RESERVE_TOKENS="${PD_RESERVE:-524288}"    # D 端独立 reserve buffer (召回页落这里, 与传输落点解耦)
export SGLANG_RETRIEVER_C4_DEVICE_TOKENS=8192             # 缩 c4 GPU 物理池 (只留传输 landing, 真池在 CPU mirror)
export SGLANG_PD_CREDIT_OFFLOADED_C4=1                    # admission 预算里把已 offload 的 c4 记为 0 (放行更高并发)
export SGLANG_PD_SWA_WINDOW_PREALLOC=1                    # SWA 池按窗口预分配 (解耦 SWA 与 context 长度)
export SGLANG_DSV4_SWA_MAX_TOKENS="${SWA_MAX:-131072}"    # SWA 池上限 (只跟并发相关, 与 context 无关)
export SGLANG_DSV4_TARGET_CONCURRENCY="${TGT_CONC:-60}"   # admission 目标并发 (× TGT_CTX = full_token 预算)
export SGLANG_DSV4_TARGET_CONTEXT="${TGT_CTX:-524288}"    # admission 目标 context

# ── Path-I: 选择性 index-K offload (可选, 1M 高并发才需要) ───────────────────
# offload 非-target 的 18 层 index-K 打分池到 CPU (target 层 10/12/20 保留全量给 Level-1)。
# 只在需要冲 1M 高并发时开; gate-off (不设) = index-K 全量常驻 GPU, 字节不变。
if [ -n "${PATHP_INDEX_K_OFFLOAD:-}" ]; then
  export SGLANG_PATHP_INDEX_K_OFFLOAD="${PATHP_INDEX_K_OFFLOAD}"
  export SGLANG_RETRIEVER_INDEX_K_DEVICE_TOKENS="${INDEX_K_DEVICE_TOKENS:-1572864}"
fi

# ── retriever (Memory-Indexer) ckpt ──────────────────────────────────────────
# 注意: PD 高并发路径由 resident_mask_capturer 驱动 scorer, INLINE=0 (不走 eager inline hook)。
export SGLANG_RETRIEVER_INLINE=0
export SGLANG_RETRIEVER_INLINE_CKPT="${CKPT:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints/top3_R930_joint.pt}"
export SGLANG_RETRIEVER_INLINE_LAYERS="${LAYERS:-10,12,20}"
export SGLANG_RETRIEVER_SIGMOID_THRESH="${THRESH:-0.5}"
export SGLANG_RETRIEVER_INTERVAL="${INTERVAL:-64}"
export SGLANG_RETRIEVER_FIRST_KEEP=0
export SGLANG_RETRIEVER_ENSEMBLE_MODE=or

# ── NIXL / UCX (跨机 KV 传输) ────────────────────────────────────────────────
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_bond_1:1,bond1}"
export UCX_IB_GID_INDEX="${UCX_IB_GID_INDEX:-3}"
export UCX_TLS="${UCX_TLS:-rc_x,tcp,cuda_copy,cuda_ipc}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=6000
export NO_PROXY=",0.0.0.0,127.0.0.1,localhost" no_proxy=",0.0.0.0,127.0.0.1,localhost"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

MODEL="${MODEL:-/dockerdata/models/ds_fp8}"
echo "[launch_decode] TP=8 port=31200 TGT_CONC=${SGLANG_DSV4_TARGET_CONCURRENCY} TGT_CTX=${SGLANG_DSV4_TARGET_CONTEXT} index_k_offload=${SGLANG_PATHP_INDEX_K_OFFLOAD:-off}"

exec sglang serve --trust-remote-code --model-path "$MODEL" \
  --tp 8 --base-gpu-id 0 --moe-runner-backend marlin \
  --disaggregation-mode decode --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 8998 \
  --num-reserved-decode-tokens 1024 --decode-log-interval 40 \
  --max-running-requests 128 --context-length "${CTX_LEN:-1100000}" --skip-server-warmup \
  --host 0.0.0.0 --port 31200

# =============================================================================
# 常用配置 (8×H20 TP8, 跨机 PD):
#   256K:  TGT_CONC=60  TGT_CTX=262144
#   512K:  TGT_CONC=60  TGT_CTX=524288
#   1M:    TGT_CONC=40  TGT_CTX=1100000  PATHP_INDEX_K_OFFLOAD=1 INDEX_K_DEVICE_TOKENS=1572864
# =============================================================================
