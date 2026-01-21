NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --use-env --nproc_per_node=7 --master_port='29500' main_finetune.py \
    --accum_iter 4 \
    --batch_size 146 \
    --model vit_base_patch16 \
    --resume /home/kang_you1/vit_small_patch16_224_no_softmax_no_ln_1744544323.5130663_in21k_74.93.pth \
    --epochs 300 \
    --blr 1e-4 --layer_decay 1.0 --warmup_epochs 0 --clip_grad 1.0 \
    --weight_decay 0.03 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data/ImageNet --output_dir /home/kang_you/SpikeZIP_transformer_resnet1/output/ --log_dir /home/kang_you/SpikeZIP_transformer_resnet1/output/ \
    --mode "ANN" --act_layer relu --remove_softmax --NormType mybatchnorm --print_freq 200 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5
