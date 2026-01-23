#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/vit-base_SNN_eval_8.sh
#   NSYS=1 ./scripts/vit-base_SNN_eval_8.sh
#   TORCH_PROFILER=1 PROFILE_DIR=./torch_profiler ./scripts/vit-base_SNN_eval_8.sh

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

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
    nsys profile -o nsys_eval_snn_1gpu --force-overwrite=true --stats=true
    --trace=cuda,nvtx,osrt,cudnn,cublas --delay=5 --duration=30
    --cuda-memory-usage=true
  )
fi

OMP_NUM_THREADS="$OMP_NUM_THREADS" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "${NSYS_ARGS[@]}" \
  python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port="$MASTER_PORT" \
    main_eval_snn.py \
    --batch_size 1 \
    --model vit_base_patch16 \
    --finetune ~/gpfs-share/ckpts/vit-base/checkpoint-19.pth \
    --resume ~/gpfs-share/ckpts/vit-base/checkpoint-19.pth \
    --num_workers 8 \
    --time_step 10 --encoding_type analog \
    --weight_quantization_bit 32 \
    --level 10 \
    --data_path ~/gpfs-share/data/ \
    --output_dir ~/gpfs-share/code/SpikeZIP_transformer_Hybrid_CVPR/output \
    --log_dir ~/gpfs-share/code/SpikeZIP_transformer_Hybrid_CVPR/output \
    --mode "SNN" --global_pool --act_layer relu \
    --dist_eval \
    "${PROFILE_ARGS[@]}"
