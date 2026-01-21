cd /home/zekai_xu/SNN/code/mae-snn/
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' main_finetune.py --eval --resume mae_finetuned_vit_base.pth \
                                              --model vit_base_patch16 --batch_size 64 --data_path /data1/ImageNet/ --act_layer gelu