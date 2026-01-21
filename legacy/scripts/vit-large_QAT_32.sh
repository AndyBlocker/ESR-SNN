cd /root/SNN/code/SpikeZIP_transformer
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, python -m torch.distributed.launch --nproc_per_node=8 --master_port='29500' main_finetune.py \
    --accum_iter 4 \
    --batch_size 20 \
    --model vit_large_patch16 \
    --finetune /root/autodl-tmp/SpikeZIP_transformer/output_dir/T-SNN_vit_large_patch16_imagenet_relu_ANN_32/checkpoint-39.pth \
    --epochs 50 \
    --blr 1.6e-3 --layer_decay 0.75 \
    --weight_decay 5e-4 --drop_path 0.05 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /root/autodl-tmp/imagenet --output_dir /root/autodl-tmp/SpikeZIP_transformer/output_dir/ --log_dir /root/autodl-tmp/SpikeZIP_transformer/output_dir/ \
    --mode "QANN_QAT" --level 32 --act_layer relu --wandb --print_freq 200
