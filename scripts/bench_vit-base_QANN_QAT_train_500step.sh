#!/usr/bin/env bash
set -euo pipefail

# Bench: train QANN (QAT) for 1 epoch, 500 steps; report peak GPU memory and avg time/step.
#
# Usage:
#   ./scripts/bench_vit-base_QANN_QAT_train_500step.sh
#   CUDA_VISIBLE_DEVICES=0 DATA_PATH=/path/to/imagenet ./scripts/bench_vit-base_QANN_QAT_train_500step.sh

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

FINETUNE_PATH="${FINETUNE_PATH:-}"
DATA_PATH="${DATA_PATH:-/home/andyblocker/data}"
OUT_DIR="${OUT_DIR:-/home/andyblocker/ESR-SNN/output}"
LOG_DIR="${LOG_DIR:-/home/andyblocker/ESR-SNN/output}"

FINETUNE_ARGS=()
if [ -n "${FINETUNE_PATH:-}" ]; then
  if [ ! -f "${FINETUNE_PATH:-}" ]; then
    echo "FINETUNE_PATH not found: ${FINETUNE_PATH:-}" >&2
    exit 1
  fi
  FINETUNE_ARGS+=(--finetune "$FINETUNE_PATH")
fi

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUM_ITER="${ACCUM_ITER:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OMP_NUM_THREADS="$OMP_NUM_THREADS" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port="$MASTER_PORT" \
    main_finetune_distill_snn.py \
    --epochs 1 --max_train_steps "$MAX_TRAIN_STEPS" \
    --batch_size "$BATCH_SIZE" --accum_iter "$ACCUM_ITER" --num_workers "$NUM_WORKERS" \
    --benchmark --skip_eval --no_save --no_log \
    --ddp_find_unused_parameters 0 \
    --model vit_base_patch16 \
    "${FINETUNE_ARGS[@]}" \
    --blr 3e-4 --layer_decay 1.0 --warmup_epochs 0 --min_lr 1e-6 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
    --project_name "T-SNN-Bench" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --log_dir "$LOG_DIR" \
    --mode "QANN_QAT" --global_pool --act_layer relu \
    --level 10 --weight_quantization_bit 32 \
    --print_freq 500 \
    $EXTRA_ARGS
