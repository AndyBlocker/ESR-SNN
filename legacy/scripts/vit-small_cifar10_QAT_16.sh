cd /home/zekai_xu/SNN/code/SpikeZIP_transformer
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, python -m torch.distributed.launch --nproc_per_node=8 --master_port='29500' main_finetune_distill.py \
    --accum_iter 4 \
    --batch_size 128 \
    --model vit_small_patch16 --nb_classes 10 \
    --model_teacher vit_small_patch16 \
    --finetune /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_cifar10_relu_ANN_32/vit-small-cifar10-relu-99.24.pth \
    --pretrain_teacher /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_cifar10_relu_ANN_32/vit-small-cifar10-relu-99.24.pth \
    --epochs 300 \
    --blr 1.5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data1/ --dataset cifar10 --output_dir /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/ --log_dir /home/zekai_xu/SNN/code/SpikeZIP_transformer/output_dir/ \
    --mode "QANN_QAT" --level 16 --act_layer relu --act_layer_teacher relu --temp 2.0 --wandb --print_freq 200 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5