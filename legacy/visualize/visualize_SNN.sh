python swin_visualize.py \
    --model_path /data/kang_you1/SpikeZIP_transformer_resnet2/output/T-SNN_swin_tiny_imagenet_relu_SNN_act10_weightbit32/checkpoint-best.pth \
    --level 10 \
    --weight_quantization_bit 32 \
    --NormType dyt \
    --remove_softmax \
    --act_layer relu \
    --use-cuda \
    --mode SNN \
    --time_step 4 \
    --encoding_type rate \
    --image-path /home/kang_you/SpikeZIP_transformer_Hybrid/visualize/n02086646_280.JPEG \

