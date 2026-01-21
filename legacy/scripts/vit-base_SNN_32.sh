cd /home/zekai_xu/SNN/code/SpikeZIP_transformer
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port='29500' main_finetune.py \
    --accum_iter 1 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_base_patch16_relu_QANN_QAT_32/vit-base-imagenet-relu-q32-82.78.pth \
    --resume /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_base_patch16_relu_QANN_QAT_32/vit-base-imagenet-relu-q32-82.78.pth \
    --epochs 100 \
    --blr 3.536e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data1/ImageNet/ --output_dir /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/ --log_dir /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/ \
    --mode "SNN" --act_layer relu --eval --time_step 150 --encoding_type rate --level 32