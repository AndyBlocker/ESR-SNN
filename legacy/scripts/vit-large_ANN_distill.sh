export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export NCCL_IB_DISABLE=1          # 禁 IB（你之前 log 里 IB 明确报错）
export NCCL_P2P_DISABLE=0         # 不要关 P2P（你之前关了更容易炸）
export NCCL_SHM_DISABLE=1         # 避免 /dev/shm 问题
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1     # 同步暴露真正的 CUDA 错误


torchrun  --nproc_per_node=5 --master_port='29500' main_finetune_distill.py \
    --accum_iter 1 \
    --batch_size 224 \
    --model vit_base_patch16 \
    --model_teacher vit_base_patch16 \
    --finetune /home/youkang/gpfs-share/SpikeZIP_transformer_Hybrid_CVPR/timmvit_base_patch16_224.augreg2_in21k_ft_in1k.bin \
    --epochs 50 --num_workers 8 \
    --blr 2.5e-6 --layer_decay 1.0 --warmup_epochs 0 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --clip_grad 1.0 \
    --dist_eval --data_path /home/youkang/gpfs-share/ImageNet/imagenet_extracted --project_name "T-SNN-DyHT-Test1" --output_dir /home/youkang/gpfs-share/SpikeZIP_transformer_Hybrid_CVPR/output --log_dir /home/youkang/gpfs-share/SpikeZIP_transformer_Hybrid_CVPR/output \
    --mode ANN --global_pool --act_layer relu --act_layer_teacher gelu --temp 2.0 --print_freq 200
 