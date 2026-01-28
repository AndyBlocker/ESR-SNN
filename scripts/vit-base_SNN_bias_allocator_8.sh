#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/vit-base_SNN_bias_allocator_8.sh
#   TORCH_PROFILER=1 PROFILE_DIR=./torch_profiler ./scripts/vit-base_SNN_bias_allocator_8.sh
#   SNN_MODEL_PATH=./output/.../checkpoint-best.pth ./scripts/vit-base_SNN_bias_allocator_8.sh
#   BIAS_RESUME=./output/.../checkpoint-9.pth ./scripts/vit-base_SNN_bias_allocator_8.sh

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

FINETUNE_PATH="${FINETUNE_PATH:-/home/andyblocker/ESR-SNN/checkpoint-19.pth}"
DATA_PATH="${DATA_PATH:-/home/andyblocker/data}"
OUT_DIR="${OUT_DIR:-/home/andyblocker/ESR-SNN/output}"
LOG_DIR="${LOG_DIR:-/home/andyblocker/ESR-SNN/output}"

SNN_MODEL_PATH="${SNN_MODEL_PATH:-}"
BIAS_RESUME="${BIAS_RESUME:-}"

RESUME_ARGS=()
if [ -n "$SNN_MODEL_PATH" ]; then
  RESUME_ARGS+=(--snn_model_path "$SNN_MODEL_PATH")
fi
if [ -n "$BIAS_RESUME" ]; then
  RESUME_ARGS+=(--resume "$BIAS_RESUME")
fi

TORCH_PROFILER="${TORCH_PROFILER:-0}"
PROFILE_DIR="${PROFILE_DIR:-./torch_profiler}"
PROFILE_PRINT="${PROFILE_PRINT:-0}"
PROFILE_PRINT_LIMIT="${PROFILE_PRINT_LIMIT:-30}"
PROFILE_PRINT_SORT="${PROFILE_PRINT_SORT:-}"
PROFILE_ARGS=()
if [ "$TORCH_PROFILER" -eq 1 ]; then
  PROFILE_ARGS=(
    --profile
    --profile_dir "$PROFILE_DIR"
    --profile_wait 1
    --profile_warmup 1
    --profile_active 5
    --profile_repeat 1
    --profile_record_shapes
    --profile_profile_memory
  )
  if [ "$PROFILE_PRINT" -eq 1 ]; then
    PROFILE_ARGS+=(--profile_print --profile_print_limit "$PROFILE_PRINT_LIMIT")
    if [ -n "$PROFILE_PRINT_SORT" ]; then
      PROFILE_ARGS+=(--profile_print_sort "$PROFILE_PRINT_SORT")
    fi
  fi
fi

NSYS="${NSYS:-0}"
NSYS_ARGS=()
if [ "$NSYS" -eq 1 ]; then
  NSYS_ARGS=(
    nsys profile -o nsys_train_bias_allocator_1gpu --force-overwrite=true --stats=true
    --trace=cuda,nvtx,osrt,cudnn,cublas --delay=10 --duration=30
    --cuda-memory-usage=true
  )
fi

OMP_NUM_THREADS="$OMP_NUM_THREADS" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "${NSYS_ARGS[@]}" \
  python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port="$MASTER_PORT" \
    main_train_bias_allocator_snn.py \
    --accum_iter 4 --batch_size 1 \
    --model vit_base_patch16 \
    --finetune "$FINETUNE_PATH" \
    --epochs 1 --num_workers 8 \
    --blr 3e-4 --warmup_epochs 0 --min_lr 1e-6 \
    --time_step 10 --encoding_type analog \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
    --project_name "T-SNN-BiasAllocator" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --log_dir "$LOG_DIR" \
    --ddp_find_unused_parameters 0 \
    --neuron_impl torch \
    --snn_verbose 0 \
    --mode "SNN" --level 10 --global_pool --act_layer relu --weight_quantization_bit 32 --print_freq 10 \
    "${RESUME_ARGS[@]}" \
    "${PROFILE_ARGS[@]}"
