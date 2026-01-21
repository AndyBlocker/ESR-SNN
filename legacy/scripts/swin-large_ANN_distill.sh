NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --use-env --nproc_per_node=7 --master_port='29500' main_finetune_distill.py \
    --accum_iter 4 \
    --batch_size 48 \
    --model swin_large --convEmbedding \
    --model_teacher swin_large \
    --epochs 300 \
    --blr 2.5e-4 --layer_decay 1.0 --warmup_epochs 20 \
    --weight_decay 0.2 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --clip_grad 1.0 --min_lr 1e-5 \
    --dist_eval --data_path /data/ImageNet --output_dir /data/kang_you1/SpikeZIP_transformer_resnet1/output/ --log_dir /data/kang_you1/SpikeZIP_transformer_resnet1/output/ \
    --mode "ANN" --act_layer relu --NormType dyht --remove_softmax --act_layer_teacher gelu --temp 2.0 --print_freq 200
