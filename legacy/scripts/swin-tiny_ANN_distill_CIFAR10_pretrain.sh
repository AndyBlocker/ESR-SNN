NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --use-env --nproc_per_node=7 --master_port='29502' main_finetune_distill.py \
    --accum_iter 1 \
    --batch_size 144 \
    --model swin_tiny_cifar --convEmbedding --dataset imagenet --nb_classes 1000 \
    --model_teacher swin_tiny_cifar \
    --epochs 100 \
    --blr 1e-3 --layer_decay 1.0 --warmup_epochs 10 --input_size 224 \
    --weight_decay 0.5 --drop_path 0.1 --mixup 0.5 --cutmix 1.0 --reprob 0.25 --clip_grad 5.0 \
    --dist_eval --data_path /data/ --output_dir /data/kang_you1/SpikeZIP_transformer_resnet1/output/ --log_dir /data/kang_you1/SpikeZIP_transformer_resnet1/output/ \
    --mode "ANN" --act_layer relu --NormType dyt --remove_softmax --act_layer_teacher gelu --temp 2.0 --print_freq 10
