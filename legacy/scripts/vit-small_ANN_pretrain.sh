# ===== NCCL / CUDA 安全配置（排查 & 稳定优先）=====
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_IB_DISABLE=1          # 禁 IB（你之前 log 里 IB 明确报错）
export NCCL_P2P_DISABLE=0         # 不要关 P2P（你之前关了更容易炸）
export NCCL_SHM_DISABLE=1         # 避免 /dev/shm 问题
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1     # 同步暴露真正的 CUDA 错误

# ===== 用 torchrun（不要再用 torch.distributed.launch）=====
torchrun \
  --nproc_per_node=6 \
  --master_port=29500 \
  main_pretrain.py \
    --accum_iter 1 \
    --batch_size 512 \
    --model mae_vit_large_patch16 \
    --epochs 400 \
    --mask_ratio 0.75 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --weight_decay 0.05 \
    --data_path /home/youkang/ImageNet/imagenet_extracted \
    --output_dir /home/youkang/gpfs-share/SpikeZIP_transformer_Hybrid_CVPR/vit_large_pretrain \
    --log_dir /home/youkang/gpfs-share/SpikeZIP_transformer_Hybrid_CVPR/vit_large_pretrain \
    --act_layer relu \
    --NormType dyht \
    --remove_softmax
