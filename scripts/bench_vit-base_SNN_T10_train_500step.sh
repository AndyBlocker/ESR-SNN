#!/usr/bin/env bash
set -euo pipefail

# Bench: train 10-timestep SNN (no freezing) for 1 epoch, 500 steps; report peak GPU memory and avg time/step.
#
# Usage:
#   ./scripts/bench_vit-base_SNN_T10_train_500step.sh
#   CUDA_VISIBLE_DEVICES=0 DATA_PATH=/path/to/imagenet ./scripts/bench_vit-base_SNN_T10_train_500step.sh

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

FINETUNE_PATH="${FINETUNE_PATH:-/home/andyblocker/ESR-SNN/checkpoint-19.pth}"
DATA_PATH="${DATA_PATH:-/home/andyblocker/data}"
OUT_DIR="${OUT_DIR:-/home/andyblocker/ESR-SNN/output}"
LOG_DIR="${LOG_DIR:-/home/andyblocker/ESR-SNN/output}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM_ITER="${ACCUM_ITER:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TIME_STEP="${TIME_STEP:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OMP_NUM_THREADS="$OMP_NUM_THREADS" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port="$MASTER_PORT" \
    main_finetune_distill_snn.py \
    --epochs 1 --max_train_steps "$MAX_TRAIN_STEPS" \
    --batch_size "$BATCH_SIZE" --accum_iter "$ACCUM_ITER" --num_workers "$NUM_WORKERS" \
    --benchmark --skip_eval --no_save --no_log \
    --ddp_find_unused_parameters 0 \
    --snn_verbose 0 \
    --model vit_base_patch16 \
    --finetune "$FINETUNE_PATH" \
    --blr 3e-4 --layer_decay 1.0 --warmup_epochs 0 --min_lr 1e-6 \
    --time_step "$TIME_STEP" --encoding_type analog \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
    --project_name "T-SNN-Bench" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --log_dir "$LOG_DIR" \
    --neuron_impl torch \
    --mode "SNN" --neuron_type "ST-BIF" --global_pool --act_layer relu \
    --level 10 --weight_quantization_bit 32 \
    --print_freq 500 \
    $EXTRA_ARGS
