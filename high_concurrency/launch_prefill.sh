#!/bin/bash
# =============================================================================
# launch_prefill.sh — PD 分离 · P (prefill) server.
#
# P 端只负责: 把长 prompt prefill 完, 通过 NIXL 把 KV 传给 D。P 不做 offload/召回
# (那些都在 D 端)。P 是标准 sglang prefill server + retriever OFF (INLINE=0)。
#
# 关键: --swa-full-tokens-ratio 决定 P 能 prefill 的最大 prompt 长度。
#   默认 0.1 → cap ~655K (拒 >655K 的 prompt); 跑 900K-1M prompt 需 SWA_RATIO=0.4。
#
# 用法 (在 P 机上跑, GPU0-7):
#   CTX_LEN=1100000 SWA_RATIO=0.4 HOST=<P_IP> bash launch_prefill.sh
# =============================================================================
set -euo pipefail

export SGLANG_DSV4_MODE=2604 SGLANG_DSV4_2604_SUBMODE=2604B SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_RETRIEVER_INLINE=0          # P 端不跑 retriever (选择在 D 端做)
export SGLANG_OPT_USE_ONLINE_COMPRESS=1
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_bond_1:1,bond1}"
export UCX_IB_GID_INDEX="${UCX_IB_GID_INDEX:-3}"
export UCX_TLS="${UCX_TLS:-rc_x,tcp,cuda_copy,cuda_ipc}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=6000
export NO_PROXY=",0.0.0.0,127.0.0.1,localhost" no_proxy=",0.0.0.0,127.0.0.1,localhost"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

MODEL="${MODEL:-/dockerdata/models/ds_fp8}"
HOST="${HOST:-0.0.0.0}"
MEM_FRAC="${MEM_FRAC:-0.75}"
CTX_LEN="${CTX_LEN:-1100000}"
MAX_RUN="${MAX_RUN:-48}"
SWA_RATIO="${SWA_RATIO:-0.1}"   # 0.1→cap~655K (512K 用); 0.4→cap~2.6M (900K-1M 用)

echo "[launch_prefill] TP=8 mem_frac=$MEM_FRAC ctx=$CTX_LEN swa_ratio=$SWA_RATIO host=$HOST bootstrap=8998 port=31100"

exec sglang serve --trust-remote-code --model-path "$MODEL" \
  --tp 8 --base-gpu-id 0 --moe-runner-backend marlin --disable-cuda-graph \
  --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 8998 \
  --swa-full-tokens-ratio "${SWA_RATIO}" \
  --max-running-requests "${MAX_RUN}" --mem-fraction-static "${MEM_FRAC}" \
  --context-length "${CTX_LEN}" --skip-server-warmup \
  --host "$HOST" --port 31100
