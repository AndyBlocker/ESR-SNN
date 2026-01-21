import os
os.chdir("../")
import cv2
import timm
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from functools import partial
import torch.nn as nn
from spike_quan_layer import DyT
from spike_quan_wrapper_ICML import remove_softmax, add_bn_in_mlp, swap_BN_MLP_MHSA, myquan_replace, SNNWrapper_MS

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./images/both.png',
        help='Input image path')
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--act_layer', type=str, default="relu",
                        help='Using ReLU or GELU as activation')
    parser.add_argument('--NormType', default='layernorm', type=str,
                        help='the normalization type')
    parser.add_argument('--remove_softmax', action='store_true',
                        help='need softmax or not')
    parser.add_argument('--mode', type=str, default="ANN",
                        help='QANN,ANN,SNN mode')
    parser.add_argument('--model', type=str, default="swin_tiny",
                        help='model type')
    parser.add_argument('--level', default=32, type=int,
                        help='the quantization levels')
    parser.add_argument('--weight_quantization_bit', default=32, type=int, help="the weight quantization bit")
    parser.add_argument('--encoding_type', default="analog", type=str,
                        help='encoding type for snn')
    parser.add_argument('--time_step', default=2000, type=int,
                        help='time-step for snn')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def reshape_transform(tensor, height=7, width=7):
    '''
    不同参数的Swin网络的height和width是不同的，具体需要查看所对应的配置文件yaml
    height = width = config.DATA.IMG_SIZE / config.MODEL.NUM_HEADS[-1]
    比如该例子中IMG_SIZE: 224  NUM_HEADS: [4, 8, 16, 32]
    height = width = 224 / 32 = 7
    '''
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
    """

    args = get_args()

    if args.act_layer == "relu":
        activation = nn.ReLU
    elif args.act_layer == "gelu":
        activation = nn.GELU
    else:
        raise NotImplementedError

    normLayer = partial(nn.LayerNorm, eps=1e-6)
    if args.NormType == "layernorm":
        normLayer = partial(nn.LayerNorm, eps=1e-6)
    elif args.NormType == "dyt":
        normLayer = partial(DyT)    
    

    model = timm.create_model('swin_tiny_patch4_window7_224', norm_layer=normLayer,act_layer=activation,pretrained=False)
    if args.remove_softmax:
        remove_softmax(model)
    if args.NormType == "mybatchnorm" or args.NormType == "dyt":
        add_bn_in_mlp(model, normLayer)
    if args.mode == "ANN":
        checkpoint = torch.load(args.model_path, map_location='cpu')
        checkpoint_model = checkpoint if ".bin" in args.model_path else checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=True)
        print("msg:", msg) 
        print(model)
    elif args.mode == "QANN":
        myquan_replace(model, args.level, args.weight_quantization_bit, is_softmax = not args.remove_softmax)
        checkpoint = torch.load(args.model_path, map_location='cpu')
        checkpoint_model = checkpoint if ".bin" in args.model_path else checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=True)
        print("msg:", msg) 
        print(model)        
    elif args.mode == "SNN":
        myquan_replace(model, args.level, args.weight_quantization_bit, is_softmax = not args.remove_softmax)
        model = SNNWrapper_MS(ann_model=model, cfg=args, time_step=args.time_step, \
                           Encoding_type=args.encoding_type, level=args.level, neuron_type="ST-BIF", \
                           model_name="test", is_softmax = not args.remove_softmax, suppress_over_fire = False, \
                           record_inout=False,learnable=True,record_dir="")
        checkpoint = torch.load(args.model_path, map_location='cpu')
        checkpoint_model = checkpoint if ".bin" in args.model_path else checkpoint['model']
        msg = model.model.load_state_dict(checkpoint_model, strict=True)
        print("msg:", msg) 
        print(model)        

    model.eval()

    if args.use_cuda:
        model = model.cuda()
	
    # 作者这个地方应该写错了，误导了我很久，输出model结构能发现正确的target_layers应该为最后一个stage后的LayerNorm层
    # target_layers = [model.layers[-1].blocks[-1].norm2]
    if args.mode == "SNN":
        target_layers = [model.model.norm]
    else:
        target_layers = [model.norm]
	
    # transformer会比CNN额外多输入参数reshape_transform
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)
	
    # 保证图片输入后为RGB格式，cv2.imread读取后为BGR
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
	
    class_map = {151: "Chihuahua", 281: "tobby cat"}
    class_id = 151
    class_name = class_map[class_id]
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(class_id)],
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.title(class_name)
    plt.savefig(f"/home/kang_you/SpikeZIP_transformer_Hybrid/visualize/swin_cam_{args.mode}.png")
