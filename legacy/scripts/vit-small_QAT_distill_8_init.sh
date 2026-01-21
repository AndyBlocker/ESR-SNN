cd /home/zkxu/SNN/code/SpikeZIP_transformer
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29500' main_finetune_distill.py \
    --accum_iter 4 \
    --batch_size 96 \
    --model vit_small_patch16 \
    --model_teacher vit_small_patch16 \
    --finetune /home/zkxu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_relu_ANN_32/vit-small-patch16-relu-82.34.pth \
    --pretrain_teacher /home/zkxu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_relu_ANN_32/vit-small-patch16-relu-82.34.pth \
    --epochs 100 \
    --blr 2.25e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data/zkxu/DG/dataset/imagenet/ --output_dir /data/zkxu/SNN/code/SpikeZIP_transformer/output_dir/ --log_dir /data/zkxu/SNN/code/SpikeZIP_transformer/output_dir/ \
    --mode "QANN_QAT" --level 8 --act_layer relu --act_layer_teacher relu --wandb --temp 2.0 --print_freq 200 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5