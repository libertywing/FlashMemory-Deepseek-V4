#!/bin/bash
# =============================================================================
# start_dump_server.sh — 启动 sglang server, 用于「生产 retriever 训练数据」(Stage-1 dump)
#
# 本脚本 = 数据生产模式: 走 deepseek_v4.py 的文件协议 tracker (pass1_dump_training),
#          把 hidden / compressed-K / golden-label dump 成 pkl, 供离线训练 retriever 用
#          (供 training/ 消费)。它加载原生 DeepSeek-V4, 不做 offload / 推理选择。
#
# ★ dump hook 生效的两个硬门槛 (见 indexer.py:_forward_c4_indexer 里
#   `if _rmc is None and not get_is_capture_mode(): ... c4_indexer.score_hook(...)`):
#     1) _rmc is None            → 绝不能设 SGLANG_PATHP_CUDAGRAPH (否则走 resident-mask
#                                   加速路径, 你的 dump score_hook 被跳过 → 空目录)
#     2) not cuda-graph capture  → 必须 --disable-cuda-graph (否则 decode 被 graph 捕获,
#                                   hook 的 Python side-effect 全部不执行 → 空目录)
#
# ★ 数据生产走 dump 路径, 不走推理选择路径。为防环境里残留的加速/inline env 关掉
#   dump hook, 下面显式 unset 它们 (SGLANG_PATHP_CUDAGRAPH / SGLANG_RETRIEVER_INLINE)。
#
# 起服务后自检:
#   rm -f /tmp/debug_rids.txt ; (发一条推理) ; cat /tmp/debug_rids.txt
#     → 有内容且 rids=[...] 非 None ⇒ dump 通路 OK
#   跑完第一个 batch 后 ls <output-dir>/*.pkl 应能看到 doc_00000.pkl
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ── 可覆盖参数 (环境变量) ────────────────────────────────────────────────────
MODEL="${MODEL:-path/to/ds_fp8}"                                      # DeepSeek-V4 FP8 权重 (必填, 用 MODEL=/your/ds_fp8 覆盖)
PORT="${PORT:-30000}"
TP="${TP:-8}"                                                          # tensor parallel (4 或 8; 128 heads 需能整除)

# ── sanity check ─────────────────────────────────────────────────────────────
if [[ ! -e "$MODEL" ]]; then
  echo "[start_dump_server] WARN: 模型路径不存在: $MODEL —— 请用 MODEL=/your/ds_fp8 覆盖。" >&2
fi

# ── 防污染: 确保不残留 Stage-2 / 加速路径的 env (它们会关掉 dump hook) ────────
unset SGLANG_RETRIEVER_INLINE            || true   # 若=1 会抢占 score_hook, dump 不触发
unset SGLANG_RETRIEVER_INLINE_CKPT       || true
unset SGLANG_RETRIEVER_INLINE_LAYERS     || true
unset SGLANG_PATHP_CUDAGRAPH             || true   # 若=1 会让 _rmc 非 None, dump 被跳过
unset SGLANG_PATHP_SCORE_RESIDENT        || true
unset SGLANG_PATHP_PAGE_RECALL           || true

# ── 让本地自请求绕过 HTTP 代理 ───────────────────────────────────────────────
# sglang 启动后 warmup 会自请求 http://0.0.0.0:30000/model_info；若环境里有
# http_proxy/https_proxy 而 no_proxy 不含本地地址，这个请求会被代理劫持,
# 返回 503 (ERR_CONNECT_FAIL) → warmup 失败 → server 被 Killed。
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,${no_proxy:-}"
export NO_PROXY="$no_proxy"

# ── DeepSeek-V4 模型运行模式 (CSA + MoE, FP4 experts), 加载权重必需 ──
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export SGLANG_DSV4_FP4_EXPERTS=1

echo "[start_dump_server] MODEL=$MODEL  TP=$TP  PORT=$PORT"
echo "[start_dump_server] 模式 = Stage-1 dump (pass1_dump_training via /tmp/dsv4_tracker_cmd.json)"
echo "[start_dump_server] INLINE=unset  PATHP_CUDAGRAPH=unset  --disable-cuda-graph=ON"
echo "[start_dump_server] 起服务后跑 run_dump_training_data.py 生产数据；自检见脚本头部注释。"

# --disable-cuda-graph 必须: eager 的 dump score_hook 会做 host-sync (.item() / 文件IO),
# 在 cuda-graph capture 内不允许, 且 graph 路径会切到 resident-mask 实现绕开 dump。
exec sglang serve --trust-remote-code --model-path "$MODEL" \
  --tp "$TP" --moe-runner-backend marlin --disable-cuda-graph \
  --host 0.0.0.0 --port "$PORT"
