#!/bin/bash
# =============================================================================
# start_server.sh — 启动一个 sglang server, 加载我们训练的 Memory-Indexer retriever
#                    (score-masking 模式), 用于「验证准确率」。
#
# 这是最简单的推理路径:
#   * 每 SGLANG_RETRIEVER_INTERVAL 个 decode step, 用训练好的 Lightning-Indexer
#     ckpt 给 CSA chunks 打分, 把「非选中」chunk 的 logits mask 成 -inf,
#     于是 sglang 原生 top-K 只能从 retriever 选中的集合里挑。
#   * 全量 KV 仍然常驻 GPU —— 只改变「选择」, 不做 KV offload / swap / PD 分离。
#     (那些是另一套并发/显存 infra, 不在本 release 范围内。)
#
# 生效路径 = deepseek_v4.py:_make_score_hook -> InlineRetrieverHook
#   (SGLANG_RETRIEVER_INLINE=1)。
# 注意: 不要设 SGLANG_PATHP_CUDAGRAPH / SGLANG_RETRIEVER_INDEX_K_DEVICE_TOKENS 等
#       变量 —— 那会切到 offload/swap 路径, 就不是纯 masking 的「准确率验证」配置了。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ── 可覆盖参数 (环境变量) ────────────────────────────────────────────────────
MODEL="${MODEL:-/dockerdata/models/ds_fp8}"                       # DeepSeek-V4 FP8 权重路径 (按需修改)
CKPT="${CKPT:-$HERE/checkpoints/top3_R930_joint.pt}"             # retriever ckpt (默认 3-layer joint)
LAYERS="${LAYERS:-10,12,20}"                                     # joint ckpt 的 CSA layer 子键 l{id}
PORT="${PORT:-30000}"
TP="${TP:-4}"                                                    # tensor parallel (4 或 8 都可; 128 heads 需能整除)
THRESH="${THRESH:-0.5}"                                          # sigmoid 阈值
INTERVAL="${INTERVAL:-64}"                                       # retriever 打分间隔 (decode steps)
FIRST_KEEP="${FIRST_KEEP:-0}"                                    # attention-sink: 前 N 个 c4 chunk 永远保留 (实测 0 最好)
LAST_KEEP="${LAST_KEEP:-2048}"                                   # recency: 最近 N 个 c4 token 永远保留
ENSEMBLE_MODE="${ENSEMBLE_MODE:-or}"                             # 多层 ensemble: or (取并集) / mean

# ── sanity check ─────────────────────────────────────────────────────────────
if [[ ! -e "$CKPT" ]]; then
  echo "[start_server] ERROR: ckpt 不存在: $CKPT" >&2
  echo "  用 CKPT=/path/to/xxx.pt 指定, 或确认 checkpoints/ 已就位。" >&2
  exit 1
fi
if [[ ! -e "$MODEL" ]]; then
  echo "[start_server] WARN: 模型路径不存在: $MODEL —— 请用 MODEL=/your/ds_fp8 覆盖。" >&2
fi

# ── DeepSeek-V4 模型运行模式 (CSA + MoE) ─────────────────────────────────────
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export SGLANG_DSV4_FP4_EXPERTS=1

# ── Retriever: inline score-masking hook (唯一真正接进 score_hook 的路径) ─────
export SGLANG_RETRIEVER_INLINE=1
export SGLANG_RETRIEVER_INLINE_CKPT="$CKPT"
export SGLANG_RETRIEVER_INLINE_LAYERS="$LAYERS"
export SGLANG_RETRIEVER_SIGMOID_THRESH="$THRESH"
export SGLANG_RETRIEVER_INTERVAL="$INTERVAL"
export SGLANG_RETRIEVER_FIRST_KEEP="$FIRST_KEEP"
export SGLANG_RETRIEVER_LAST_KEEP="$LAST_KEEP"
export SGLANG_RETRIEVER_ENSEMBLE_MODE="$ENSEMBLE_MODE"

echo "[start_server] MODEL=$MODEL"
echo "[start_server] CKPT=$CKPT  LAYERS=$LAYERS  TP=$TP  PORT=$PORT"
echo "[start_server] thresh=$THRESH interval=$INTERVAL first_keep=$FIRST_KEEP last_keep=$LAST_KEEP ensemble=$ENSEMBLE_MODE"
echo "[start_server] 日志里应出现: [InlineRetriever] init ...  和随后的 [InlineRetriever] resident#N ..."

# --disable-cuda-graph 是必须的: eager score_hook 会做 host-sync (.item()),
# 在 cuda-graph capture 内不允许 (且 graph 路径会切到另一套 mask 实现)。
exec sglang serve --trust-remote-code --model-path "$MODEL" \
  --tp "$TP" --moe-runner-backend marlin --disable-cuda-graph \
  --host 0.0.0.0 --port "$PORT"
