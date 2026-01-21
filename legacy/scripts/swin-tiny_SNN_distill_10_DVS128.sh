NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --use-env --nproc_per_node=7 --master_port='29503' main_finetune_distill_snn.py \
    --accum_iter 1 \
    --batch_size 2 \
    --model swin_tiny_dvs --convEmbedding --dataset dvs128 --nb_classes 11 \
    --model_teacher swin_tiny_dvs \
    --epochs 100 --hybrid_training --warmup_epochs 0 --print_freq 10 \
    --blr 5e-3 --layer_decay 1.0  \
    --weight_decay 0.5 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data1/DVS128 --output_dir /data/kang_you1/SpikeZIP_transformer_resnet1/output/ --log_dir /data/kang_you1/SpikeZIP_transformer_resnet1/output/ \
    --mode "SNN" --act_layer relu --remove_softmax --NormType dyt --time_step 16 --act_layer_teacher relu --temp 2.0 --encoding_type rate --level 32 --weight_quantization_bit 32