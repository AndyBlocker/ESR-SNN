# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port='29500' main_finetune_distill_snn.py \
#     --accum_iter 4 \
#     --batch_size 1 \
#     --model vit_base_patch16 \
#     --finetune ~/gpfs-share/ckpts/vit-base/checkpoint-19.pth \
#     --resume ~/gpfs-share/ckpts/vit-base/checkpoint-19.pth \
#     --epochs 5 --num_workers 8 \
#     --blr 3e-4 --layer_decay 1.0 --warmup_epochs 0 --time_step 10 --encoding_type analog \
#     --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
#     --dist_eval --project_name "T-SNN-DyHT-ori-Training" --data_path ~/gpfs-share/data/ --output_dir ~/gpfs-share/code/SpikeZIP_transformer_Hybrid_CVPR/output --log_dir ~/gpfs-share/code/SpikeZIP_transformer_Hybrid_CVPR/output \
#     --mode "SNN" --level 10 --global_pool --act_layer relu --weight_quantization_bit 32 --print_freq 10



OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
nsys profile -o nsys_vit_snn_1gpu --force-overwrite=true --stats=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --delay=60 --duration=30 \
  --cuda-memory-usage=true \
  python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=29500 \
    main_finetune_distill_snn.py \
    --accum_iter 4 --batch_size 1 \
    --model vit_base_patch16 \
    --finetune ~/gpfs-share/ckpts/vit-base/checkpoint-19.pth \
    --resume  ~/gpfs-share/ckpts/vit-base/checkpoint-19.pth \
    --epochs 5 --num_workers 8 \
    --blr 3e-4 --layer_decay 1.0 --warmup_epochs 0 --time_step 10 --encoding_type analog \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
    --dist_eval --project_name "T-SNN-DyHT-ori-Training" \
    --data_path ~/gpfs-share/data/ \
    --output_dir ~/gpfs-share/code/SpikeZIP_transformer_Hybrid_CVPR/output \
    --log_dir    ~/gpfs-share/code/SpikeZIP_transformer_Hybrid_CVPR/output \
    --mode "SNN" --level 10 --global_pool --act_layer relu --weight_quantization_bit 32 --print_freq 10