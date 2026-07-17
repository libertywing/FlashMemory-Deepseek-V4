#!/bin/bash
# train_ddp.sh — Launch joint training with DDP (DistributedDataParallel)
#
# Usage examples:
#   bash train_ddp.sh single                    # single-GPU baseline (no DDP)
#   bash train_ddp.sh ddp 2 0,1                 # 2-card DDP on GPUs 0,1
#   bash train_ddp.sh ddp 4 0,1,2,3             # 4-card DDP on GPUs 0,1,2,3
#
# Notes:
# - --batch-size is PER-RANK; global = bs × world_size
# - DDP auto-detects via torchrun env vars (RANK / LOCAL_RANK / WORLD_SIZE)
# - val/ckpt/print only on rank 0
# - find_unused_parameters=True (pairwise loss may skip a layer)
#
# Speed expectations:
#   1× H20: ~57 step/min joint, ~75 step/min single
#   2× H20 DDP: ~95 step/min joint (1.65× speedup, IO/sync bound)
#   4× H20 DDP: ~150 step/min joint (2.6× speedup)
# (estimates; actual may vary)

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-single}"     # single | ddp
NPROC="${2:-1}"         # number of GPUs (DDP)
GPUS="${3:-0}"          # comma-separated GPU ids (CUDA_VISIBLE_DEVICES)

EXP_NAME="${EXP_NAME:-ddp_test}"
EXP_DIR="experiments/expR_${EXP_NAME}"
mkdir -p "$EXP_DIR"

# Common training args (edit as needed)
COMMON_ARGS=(
    --joint-layers 10,12,20
    --data-config combined_v2
    --weight-dir ./weights
    --output-dir "$EXP_DIR/ckpts"
    --epochs 5 --batch-size 512 --lr 1e-4 --wd 0.01
    --sample-interval 1 --label-interval 64 --max-pos 512 --n-heads 64
    --seed 42 --num-workers 0 --log-interval 50
    --neg-ratio 3 --weighted-loss --val-fullset
    --val-every-steps 1000 --patience-vals 3 --bf16
)

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

if [ "$MODE" = "single" ]; then
    echo "[launcher] Single-GPU mode on GPU $GPUS"
    CUDA_VISIBLE_DEVICES="$GPUS" \
        python3 -u train.py --device cuda:0 "${COMMON_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/train.log"
elif [ "$MODE" = "ddp" ]; then
    echo "[launcher] DDP mode: nproc=$NPROC GPUs=$GPUS"
    CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node "$NPROC" --master_port 29500 \
        train.py --device cuda "${COMMON_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/train.log"
else
    echo "Usage: $0 {single|ddp} [nproc] [gpu_ids]"
    exit 1
fi
