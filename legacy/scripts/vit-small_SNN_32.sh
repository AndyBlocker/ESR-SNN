cd /root/SNN/code/SpikeZIP_transformer
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port='29501' main_finetune.py \
    --accum_iter 4 \
    --batch_size 64 \
    --model vit_small_patch16 \
    --finetune /root/autodl-tmp/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_imagenet_relu_QANN_QAT_32/vit-small-imagenet-relu-q32-81.59.pth \
    --resume /root/autodl-tmp/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_imagenet_relu_QANN_QAT_32/vit-small-imagenet-relu-q32-81.59.pth \
    --epochs 100 \
    --blr 3.536e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /root/autodl-tmp/imagenet/ --output_dir /root/autodl-tmp/SpikeZIP_transformer/output_dir/ --log_dir /root/autodl-tmp/SpikeZIP_transformer/output_dir/ \
    --mode "SNN" --act_layer relu --eval --time_step 200 --encoding_type rate --level 32 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5