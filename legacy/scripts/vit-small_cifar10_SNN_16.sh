cd /home/zekai_xu/SNN/code/SpikeZIP_transformer
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29500' main_finetune.py \
    --accum_iter 4 \
    --batch_size 64 \
    --model vit_small_patch16 --nb_classes 10 \
    --finetune /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_cifar10_relu_QANN_QAT_16/vit-small-cifar10-relu-q16-98.72.pth \
    --resume /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_cifar10_relu_QANN_QAT_16/vit-small-cifar10-relu-q16-98.72.pth \
    --epochs 100 \
    --blr 3.536e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data1/ --dataset cifar10 --output_dir /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/ --log_dir /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/ \
    --mode "SNN" --act_layer relu --eval --time_step 2000 --encoding_type rate --level 16 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5