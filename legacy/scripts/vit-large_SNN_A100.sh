cd /root/SNN/code/SpikeZIP_transformer
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 --master_port='29500' main_finetune.py  \
    --accum_iter 4 \
    --batch_size 16 \
    --model vit_large_patch16 \
    --finetune /root/SNN/code/SpikeZIP_transformer/output_dir/vit-large-imagenet-relu-q32-83.74.pth \
    --resume /root/SNN/code/SpikeZIP_transformer/output_dir/vit-large-imagenet-relu-q32-83.74.pth \
    --epochs 50 \
    --blr 1.67e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /root/data/imagenet --output_dir /root/SNN/code/SpikeZIP_transformer/output_dir/ --log_dir /root/SNN/code/SpikeZIP_transformer/output_dir/ \
    --mode "SNN" --eval --act_layer relu --wandb --print_freq 200 --level 32 --time_step 200
