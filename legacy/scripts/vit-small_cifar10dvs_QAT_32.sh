cd /home/zkxu/SNN/code/SpikeZIP_transformer
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29500' main_finetune_distill_dvs.py \
    --accum_iter 4 \
    --batch_size 92 \
    --model  vit_small_patch16_dvs --nb_classes 10 \
    --model_teacher  vit_small_patch16_dvs \
    --finetune /home/zkxu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_dvs_cifar10dvs_relu_ANN_32/vit-small-cifar10dvs-relu-90.40.pth \
    --pretrain_teacher /home/zkxu/SNN/code/SpikeZIP_transformer/output_dir/T-SNN_vit_small_patch16_dvs_cifar10dvs_relu_ANN_32/vit-small-cifar10dvs-relu-90.40.pth \
    --epochs 300 \
    --blr 2.25e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data/zkxu/cifar-10-dvs --dataset cifar10dvs --output_dir /data/zkxu/SNN/code/SpikeZIP_transformer/output_dir/ --log_dir /data/zkxu/SNN/code/SpikeZIP_transformer/output_dir/ \
    --mode "QANN_QAT" --level 32 --act_layer relu --act_layer_teacher relu --temp 2.0 --wandb --print_freq 200 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5