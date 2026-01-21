cd /home/zekai_xu/SNN/code/SpikeZIP_transformer
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, python -m torch.distributed.launch --nproc_per_node=8 --master_port='29500' main_finetune_distill.py \
    --accum_iter 4 \
    --batch_size 48 \
    --model vit_base_patch16 \
    --model_teacher vit_base_patch16 \
    --finetune /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_base_patch16_imagenet_relu_ANN_32/vit-base-imagenet-relu-83.75.pth \
    --pretrain_teacher /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_base_patch16_imagenet_relu_ANN_32/vit-base-imagenet-relu-83.75.pth \
    --epochs 100 \
    --blr 2.667e-4 --layer_decay 0.65 \
    --weight_decay 1e-4 --drop_path 0.05 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data1/ImageNet/ --output_dir /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/ --log_dir /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/ \
    --mode "QANN_QAT" --level 32 --act_layer relu --act_layer_teacher relu --temp 2.0 --wandb --print_freq 200