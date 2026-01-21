python swin_visualize.py \
    --model_path /data/kang_you1/SpikeZIP_transformer_resnet1/output/T-SNN_swin_tiny_imagenet_relu_QANN_QAT_act10_weightbit32_NormTypedyt/checkpoint-best.pth \
    --level 10 \
    --weight_quantization_bit 32 \
    --NormType dyt \
    --remove_softmax \
    --act_layer relu \
    --use-cuda \
    --mode QANN \
    --image-path /home/kang_you/SpikeZIP_transformer_Hybrid/visualize/n02086646_280.JPEG \

