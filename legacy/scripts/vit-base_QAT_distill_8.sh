export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_DISABLE=1          # 禁 IB（你之前 log 里 IB 明确报错）
export NCCL_P2P_DISABLE=0         # 不要关 P2P（你之前关了更容易炸）
export NCCL_SHM_DISABLE=1         # 避免 /dev/shm 问题
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1     # 同步暴露真正的 CUDA 错误

python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port='29500' main_finetune_distill.py \
    --accum_iter 1 \
    --batch_size 128 \
    --model vit_base_patch16 \
    --model_teacher vit_base_patch16 \
    --finetune /home/youkang/gpfs-share/SpikeZIP_transformer_Hybrid_CVPR/output/T-SNN-DyHT-Test1_vit_base_patch16_imagenet_relu_ANN_act32_weightbit32_NormTypelayernorm/checkpoint-best.pth \
    --resume /home/youkang/gpfs-share/SpikeZIP_transformer_Hybrid_CVPR/output/T-SNN-ori-18Level-bias_vit_base_patch16_imagenet_relu_QANN_QAT_act18_weightbit32_NormTypelayernorm/checkpoint-83.42.pth \
    --epochs 50 \
    --blr 5e-6 --layer_decay 1.0 --warmup_epochs 0  \
    --weight_decay 0.0001 --drop_path 0.1 --mixup 0.2 --cutmix 0.2 --reprob 0.25 --clip_grad 1.0 \
    --dist_eval --data_path /home/youkang/gpfs-share/ImageNet/imagenet_extracted --project_name "T-SNN-ori-18Level-bias" --output_dir /data/kang_you/SpikeZIP_transformer_resnet1/output/ --log_dir /data/kang_you/SpikeZIP_transformer_resnet1/output/ \
    --mode "QANN_QAT" --level 10 --act_layer relu --NormType layernorm --weight_quantization_bit 32 --act_layer_teacher relu --temp 2.0 --print_freq 10