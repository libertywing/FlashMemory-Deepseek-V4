#!/bin/bash
# =============================================================================
# launch_router.sh — PD 分离 · router (mini load-balancer).
#
# 把 client 请求路由到 P (prefill) + D (decode)。用 sglang_router 的 mini-lb。
# client 只连 router (默认 :31503), 由它协调 P prefill → NIXL 传 KV → D decode。
#
# 用法:
#   PREFILL_IP=<P_IP> DECODE_IP=<D_IP> bash launch_router.sh
# =============================================================================
set -euo pipefail

PREFILL_IP="${PREFILL_IP:-127.0.0.1}"
DECODE_IP="${DECODE_IP:-127.0.0.1}"
PORT="${PORT:-31503}"

export NO_PROXY=",0.0.0.0,127.0.0.1,localhost,${PREFILL_IP},${DECODE_IP}" \
       no_proxy=",0.0.0.0,127.0.0.1,localhost,${PREFILL_IP},${DECODE_IP}"

echo "[launch_router] prefill=http://${PREFILL_IP}:31100 (bootstrap 8998)  decode=http://${DECODE_IP}:31200  port=${PORT}"

exec python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill "http://${PREFILL_IP}:31100" 8998 \
  --decode "http://${DECODE_IP}:31200" \
  --request-timeout-secs 14400 \
  --host 0.0.0.0 --port "${PORT}"
