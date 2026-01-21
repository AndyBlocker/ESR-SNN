import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from spike_quan_wrapper import myquan_replace, SNNWrapper

import models_vit
import wandb

from engine_finetune import train_one_epoch, evaluate

from copy import deepcopy

import warnings
from PIL import Image
import PIL
import torchvision.transforms as transforms
warnings.filterwarnings("ignore", category=UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def build_transform():
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    input_size = 224
    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def read_image(path):
    img = Image.open(path)
    T = build_transform()
    x = T(img).unsqueeze(0)
    # print(x.shape)
    return x

model = models_vit.__dict__["vit_base_patch16"](
    num_classes=1000,
    drop_path_rate=0.1,
    global_pool=True,
    act_layer=nn.ReLU,
)

myquan_replace(model,level=32)


checkpoint = torch.load("/home/kang_you/mae-main-my/output_dir/qvit_base_patch16_ReLU/model_best_epoch97_83.064.pth", map_location='cpu')
print("Load pre-trained checkpoint from: %s" % "/home/kang_you/mae-main-my/output_dir/qvit_base_patch16_ReLU/model_best_epoch97_83.064.pth")
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
interpolate_pos_embed(model, checkpoint_model)

# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=True)
print(msg)
f = open("./qann_model_server.txt","w+")
f.write(str(model))
f.close()

snn_model = SNNWrapper(ann_model=deepcopy(model),cfg=None,level=32,neuron_type="ST-BIF",model_name="vit_base_patch16",Encoding_type="rate")

# x = torch.randn(1,3,224,224).cuda()

x = read_image("/home/kang_you/vit_snn/ILSVRC2012_val_00000014.JPEG").cuda()

model.cuda()
model.eval()

snn_model.cuda()
snn_model.eval()

ann_output = model(x)
print("ann_output.shape",ann_output.shape)

snn_output,_ = snn_model(x)

print(ann_output[0,0:20])

print(snn_output[0,0:20])

print(torch.argmax(ann_output,dim=-1))
print(torch.argmax(snn_output,dim=-1))
print(torch.sum(~(torch.abs(ann_output - snn_output)<1e-3)))

print(torch.sum(torch.abs(ann_output - snn_output)))


