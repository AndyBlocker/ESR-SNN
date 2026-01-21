# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torchvision

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import timm

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from spike_quan_wrapper_ICML import myquan_replace, SNNWrapper, remove_softmax, MyBatchNorm1d, add_bn_in_mlp, swap_BN_MLP_MHSA, adjust_LN2BN_Ratio, add_convEmbed
from spike_quan_layer import MyLayerNorm, set_init_false, LN2BNorm, MyQuan, DyT, MLP_BN, WindowAttention_no_softmax, DyHT, DyHT_ReLU
import timm.optim.optim_factory as optim_factory
from torchvision import transforms
from util.SNNaugment import SNNAugmentWide
from util.swin_optimizer import build_optimizer as build_swin_optimizer
from copy import deepcopy

import models_vit
import wandb

from engine_finetune import train_one_epoch, evaluate, train_one_epoch_distill
from PowerNorm import MaskPowerNorm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--print_freq', default=1000, type=int,
                        help='print_frequency')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--project_name', default='T-SNN', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Model parameters
    parser.add_argument('--model', default='vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_teacher', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--encoding_type', default="analog", type=str,
                        help='encoding type for snn')
    parser.add_argument('--time_step', default=2000, type=int,
                        help='time-step for snn')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--act_layer', type=str, default="relu",
                        help='Using ReLU or GELU as activation')
    parser.add_argument('--act_layer_teacher', type=str, default="gelu",
                        help='Using ReLU or GELU as activation for teacher model')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--temp', type=float, default=2.0, metavar='T',
                        help='temperature for distillation')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--no_aug', action='store_true', default=False,
                        help='do not apply augmentation')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--pretrain_teacher', default='',
                        help='pretrained teacher model')
    parser.add_argument('--global_pool', default=False, action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--define_params', action='store_true')
    parser.add_argument('--mean', nargs='+', type=float)
    parser.add_argument('--std', nargs='+', type=float)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--fintune_from_QANN', default='',
                        help='fintune from QANN checkpoint')
    parser.add_argument('--convEmbedding', action='store_true',
                        help='ConvEmbedding from QKFormer')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Using wandb or not')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # training mode
    parser.add_argument('--mode', default="ANN", type=str,
                        help='the running mode of the script["ANN", "QANN_PTQ", "QANN_QAT", "SNN"]')

    # LSQ quantization
    parser.add_argument('--level', default=32, type=int,
                        help='the quantization levels')
    parser.add_argument('--weight_quantization_bit', default=32, type=int, help="the weight quantization bit")
    parser.add_argument('--neuron_type', default="ST-BIF", type=str,
                        help='neuron type["ST-BIF", "IF"]')
    parser.add_argument('--remove_softmax', action='store_true',
                        help='need softmax or not')
    parser.add_argument('--NormType', default='layernorm', type=str,
                        help='the normalization type')
    parser.add_argument('--prune', action='store_true',
                        help='prune or not')
    parser.add_argument('--swap_bn', action='store_true',default=False, 
                        help='swap bn for fusion')
    parser.add_argument('--prune_ratio', type=float, default="3e8",
                        help='prune ratio')
    parser.add_argument('--suppress_over_fire', action='store_true', default=False,
                        help='suppress_over_fire')
    return parser


class SparsityMonitor:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.total_ones = 0
        self.total_elements = 0
        
        # 注册钩子到所有MultiStepLIFNode模块
        for module in self.model.modules():
            if isinstance(module, MyQuan):
                hook = module.register_forward_hook(self._hook)
                self.hooks.append(hook)
    
    def _hook(self, module, input, output):
        # 统计1的数量和总元素数
        if module.record:
            self.total_ones += (output/module.s.data).abs().sum().item()
            self.total_elements += output.numel() * module.pos_max
    
    def get_sparsity(self):
        """获取当前统计的稀疏度"""
        if self.total_elements == 0:
            return 0.0  # 防止除以零
        return self.total_ones / self.total_elements
    
    def reset(self):
        """重置统计计数器"""
        self.total_ones = 0
        self.total_elements = 0
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_dir = os.path.join(args.log_dir,
                                    "{}_{}_{}_{}_{}_act{}_weightbit{}_NormType{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit,args.NormType))
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        if args.wandb:
            wandb.init(config=args, project=args.project_name,
                       name="{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit),
                       dir=args.output_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.act_layer == "relu":
        activation = nn.ReLU
    elif args.act_layer == "gelu":
        activation = nn.GELU
    else:
        raise NotImplementedError

    if args.act_layer_teacher == "relu":
        activation_teacher = nn.ReLU
    elif args.act_layer_teacher == "gelu":
        activation_teacher = nn.GELU
    else:
        raise NotImplementedError

    normLayer = partial(nn.LayerNorm, eps=1e-6)
    if args.NormType == "layernorm":
        normLayer = partial(nn.LayerNorm, eps=1e-6)
    elif args.NormType == "powernorm":
        normLayer = partial(MaskPowerNorm, eps=1e-6)
    elif args.NormType == "mylayernorm":
        normLayer = partial(MyLayerNorm, eps=1e-6)
    elif args.NormType == "mybatchnorm":
        normLayer = partial(MyBatchNorm1d, eps=1e-6)
    elif args.NormType == "ln2bn":
        normLayer = partial(LN2BNorm, eps=1e-6)
    elif args.NormType == "dyt":
        normLayer = partial(DyT)    
    elif args.NormType == "dyht":
        normLayer = partial(DyHT)    
    elif args.NormType == "dyht_relu":
        normLayer = partial(DyHT_ReLU)    
    
    print("args.drop_path",args.drop_path)
    if "vit_small" in args.model:
            model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            drop_rate = args.drop_rate,
            global_pool=args.global_pool,
            act_layer=activation,
            norm_layer=normLayer,
        )
    elif "vit_base" in args.model:
            model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            drop_rate = args.drop_rate,
            global_pool=args.global_pool,
            act_layer=activation,
            norm_layer=normLayer,
        )
    elif "swin_tiny_cifar" in args.model:
        model = timm.create_model("swin_cifar_patch4_window7_224",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path, img_size=args.input_size, num_classes=args.nb_classes)
    elif "swin_tiny_dvs" in args.model:
        model = timm.create_model("swin_dvs_patch4_window4_128",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path, in_chans=3, img_size=args.input_size, num_classes=args.nb_classes)
    elif "swin_tiny" in args.model:
        if args.NormType != "layernorm" or args.remove_softmax:
            model = timm.create_model("swin_tiny_patch4_window7_224",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path)
        else:
            model = timm.create_model("swin_tiny_patch4_window7_224",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path, checkpoint_path="/data/kang_you1/swin_tiny_patch4_window7_224.pth")
    elif "swin_small" in args.model:
        if args.NormType != "layernorm" or args.remove_softmax:
            model = timm.create_model("swin_small_patch4_window7_224_Hybrid",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path)
        else:
            model = timm.create_model("swin_small_patch4_window7_224_Hybrid",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path, checkpoint_path="/data/kang_you1/swin_base_patch4_window7_224_22kto1k.pth")
    elif "swin_base" in args.model:
        if args.NormType != "layernorm" or args.remove_softmax:
            model = timm.create_model("swin_base_patch4_window7_224_Hybrid",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path)
        else:
            model = timm.create_model("swin_base_patch4_window7_224_Hybrid",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path, checkpoint_path="/data/kang_you1/swin_base_patch4_window7_224_22kto1k.pth")
    elif "swin_large" in args.model:
        if args.NormType != "layernorm" or args.remove_softmax:
            model = timm.create_model("swin_large_patch4_window7_224_Hybrid",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path)
        else:
            model = timm.create_model("swin_large_patch4_window7_224_Hybrid",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path)
    else:
        print("global_pool",args.global_pool)
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            drop_rate = args.drop_rate,
            global_pool=args.global_pool,
            act_layer=activation,
            norm_layer=normLayer,
        )
        # model = timm.create_model("vit_large_patch16_224", pretrained=False, drop_path_rate=args.drop_path, act_layer=activation, norm_layer=normLayer)

    if args.remove_softmax:
        remove_softmax(model)
    if args.NormType == "mybatchnorm" or args.NormType == "dyt" or args.NormType == "dyht":
        add_bn_in_mlp(model, normLayer)
        if args.swap_bn:
            swap_BN_MLP_MHSA(model)
    if args.convEmbedding:
        add_convEmbed(model)
    # elif args.NormType == "layernorm" or args.NormType == "ln2bn":
    #     add_bn_in_mlp(model, normLayer)

    # if "swin_tiny" in args.model_teacher:
    #     model_teacher = timm.create_model("swin_tiny_patch4_window7_224",pretrained=False, checkpoint_path="/data/kang_you1/swin_tiny_patch4_window7_224.pth",act_layer=activation_teacher)
    # elif "swin_base" in args.model_teacher:
    #     model_teacher = timm.create_model("swin_base_patch4_window7_224",pretrained=False, checkpoint_path="/data/kang_you1/swin_base_patch4_window7_224_22kto1k.pth",act_layer=activation_teacher)
    # else:
    #     model_teacher = models_vit.__dict__[args.model_teacher](
    #         num_classes=args.nb_classes,
    #         drop_path_rate=args.drop_path,
    #         global_pool=False if "vit_small" in args.model_teacher else args.global_pool,
    #         act_layer=activation_teacher,
    #         norm_layer=nn.LayerNorm,
    #     )

    model_teacher = timm.create_model("vit_base_patch16_224",pretrained=False, checkpoint_path="/home/youkang/gpfs-share/SpikeZIP_transformer_Hybrid_CVPR/timmvit_base_patch16_224.augreg2_in21k_ft_in1k.bin")
    # model_teacher = deepcopy(model)
    model_teacher.eval()
    # model_teacher = None

    assert args.pretrain_teacher is not None
    # print("Load pre-trained checkpoint from: %s" % args.finetune)
    # checkpoint = torch.load(args.finetune, map_location='cpu')
    # checkpoint_model = checkpoint if ".bin" in args.finetune else checkpoint['model']
    # model.load_state_dict(checkpoint_model, strict=True)
    if len(args.pretrain_teacher) > 0:
        print("Load pre-trained teacher checkpoint from: %s" % args.pretrain_teacher)    
        checkpoint_teacher = torch.load(args.pretrain_teacher, map_location='cpu')
        checkpoint_model_teacher = checkpoint_teacher if ".bin" in args.pretrain_teacher else checkpoint_teacher['model']
        state_dict_teacher = model_teacher.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model_teacher and checkpoint_model_teacher[k].shape != state_dict_teacher[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model_teacher[k]

        # interpolate position embedding
        interpolate_pos_embed(model_teacher, checkpoint_model_teacher)

        # load pre-trained model
        msg_teacher = model_teacher.load_state_dict(checkpoint_model_teacher, strict=False)
        print(msg_teacher)

    if args.finetune and not args.eval and not (args.mode == "SNN") and not (args.mode == "QANN-QAT" and args.eval):
        if (args.dataset == "cifar10dvs" or args.dataset == "dvs128") and args.mode == "QANN_QAT":
            if not args.convEmbedding:
                model.patch_embed.proj = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), model.patch_embed.proj)        
            else:
                model.patch_embed.proj_conv = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), model.patch_embed.proj_conv)        

        checkpoint = torch.load(args.finetune, map_location='cpu',weights_only=False)

        print("Load finetune Full precision checkpoint from: %s !!!!!!!" % args.finetune)
        if "model" in checkpoint.keys():
            checkpoint_model = checkpoint if ".bin" in args.finetune else checkpoint['model']
        else:
            checkpoint_model = checkpoint if ".bin" in args.finetune else checkpoint
        state_dict = model.state_dict()

        for suffix in ['alpha', 'gamma', 'beta','weight', 'bias']:
            old_key = f'norm.{suffix}'
            new_key = f'fc_norm.{suffix}'
            
            # 逻辑：如果 checkpoint 里有 'norm.xxx'，但是没有 'fc_norm.xxx'
            # 并且当前模型 (state_dict) 里确实需要 'fc_norm.xxx'
            if old_key in checkpoint_model and new_key in state_dict:
                # 只有当 checkpoint 中还没这个新 key 时才赋值，避免覆盖
                if new_key not in checkpoint_model:
                    print(f"Detected key mismatch: mapping '{old_key}' to '{new_key}'")
                    checkpoint_model[new_key] = checkpoint_model[old_key]
                    # print(f"checkpoint_model[{old_key}]",checkpoint_model[old_key])
                    # print(f"state_dict[{new_key}]",state_dict[new_key])

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        model_teacher = deepcopy(model)

    if args.rank == 0:
        print("======================== ANN model ========================")
        f = open(f"{args.log_dir}/ann_model_arch.txt", "w+")
        f.write(str(model))
        f.close()
    if args.mode.count("QANN") > 0:
        myquan_replace(model, args.level, args.weight_quantization_bit, is_softmax = not args.remove_softmax)
        if args.rank == 0:
            print("======================== QANN model =======================")
            f = open(f"{args.log_dir}/qann_model_arch.txt", "w+")
            f.write(str(model))
            f.close()
        if args.prune:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print("Load pre-trained checkpoint for QANN from: %s !!!!!!!" % args.finetune)
            checkpoint_model = checkpoint if ".bin" in args.finetune else checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)            
            set_init_false(model)
    elif args.mode == "SNN":
        myquan_replace(model, args.level, args.weight_quantization_bit, is_softmax = not args.remove_softmax)
        checkpoint = torch.load(args.finetune, map_location='cpu') if not args.eval else torch.load(args.resume,
                                                                                                    map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
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
        if args.rank == 0:
            print("======================== QANN model =======================")
            f = open(f"{args.log_dir}/qann_model_arch.txt", "w+")
            f.write(str(model))
            f.close()

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        # trunc_normal_(model.head.weight, std=2e-5)
        model = SNNWrapper(ann_model=model, cfg=None, time_step=args.time_step, Encoding_type=args.encoding_type,
                           level=args.level, neuron_type=args.neuron_type, model_name=args.model)
        if args.rank == 0:
            print("======================== SNN model =======================")
            f = open(f"{args.log_dir}/snn_model_arch.txt", "w+")
            f.write(str(model))
            f.close()

    if args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load checkpoint from: %s !!!!!!!" % args.finetune)
        checkpoint_model = checkpoint if ".bin" in args.finetune else checkpoint['model']
        state_dict = model.state_dict()
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)            
        set_init_false(model)

    if len(args.fintune_from_QANN) > 0:
        checkpoint = torch.load(args.fintune_from_QANN, map_location='cpu')
        print("fintune_from_QANN Load checkpoint from: %s !!!!!!!" % args.fintune_from_QANN)
        checkpoint_model = checkpoint if ".bin" in args.fintune_from_QANN else checkpoint['model']
        state_dict = model.state_dict()
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        set_init_false(model)


    if (args.dataset == "cifar10dvs" or args.dataset == "dvs128") and args.mode == "ANN":
        if not args.convEmbedding:
            model.patch_embed.proj = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), model.patch_embed.proj)        
        else:
            model.patch_embed.proj_conv = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), model.patch_embed.proj_conv) 

    for name,module in list(model.named_modules()):
        if isinstance(module, MLP_BN) or isinstance(module, WindowAttention_no_softmax):
            module.name = name

    model.to(device)
    model_teacher.to(device)

    model_without_ddp = model if args.mode != "SNN" else model.model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module if args.mode != "SNN" else model.module.model

    # build optimizer with layer-wise lr decay (lrd)
    if "vit" in args.model:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        # optimizer = optim_factory.Lamb(param_groups, trust_clip=True, lr=args.lr)
        # optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    else:
        # optimizer = optim_factory.Lamb(model_without_ddp.parameters(), trust_clip=True, lr=args.lr)
        param_groups = []
        for n, p in model_without_ddp.named_parameters():
            if p.requires_grad:
                param_groups.append({'params': [p], 'lr': args.lr, 'weight_decay': args.weight_decay, 'layer_name': n})
        
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        # optimizer = build_swin_optimizer(args, model)

    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # if args.mode != "SNN":
    #     misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if len(args.resume) > 0:
        checkpoint = torch.load(args.resume,map_location="cpu",weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # loss_scaler.load_state_dict(checkpoint["scaler"])
        # args = checkpoint["args"]
        # args.start_epoch = 61
        set_init_false(model)
        # print(model)
        # exit(0)
    
    if args.eval:
        monitor = SparsityMonitor(model)
        test_stats = evaluate(data_loader_val, model, device, None, "ANN", args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        sparsity = 1 - monitor.get_sparsity()
        print("the sparsity of the model is: ", sparsity)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    

    # define data augment for DVS:
    if args.dataset == "cifar10dvs" or args.dataset == "dvs128":
        train_snn_aug = transforms.Compose([
                        transforms.Resize(size=(args.input_size, args.input_size)),
                        transforms.RandomHorizontalFlip(p=0.5)
                        ])
        train_trivalaug = SNNAugmentWide()    
        test_snn_aug = transforms.Compose([
                        transforms.Resize(size=(args.input_size, args.input_size)),
                        ])
    else:
        train_snn_aug = None
        train_trivalaug = None
        test_snn_aug = None

    test_stats = evaluate(data_loader_val, model_teacher, device, test_snn_aug ,"ANN", args)
    print(f"Accuracy of the teacher network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    # # # #
    # print("model_teacher",model_teacher)
    # print("model",model)
    test_stats = evaluate(data_loader_val, model, device, test_snn_aug ,"ANN", args)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    # # exit(0)

    args.choose_prune = 1.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.NormType == "ln2bn":
            adjust_LN2BN_Ratio(args.warmup_epochs, epoch, model)

        train_stats = train_one_epoch_distill(
            model, model_teacher, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer, aug=train_snn_aug, trival_aug=train_trivalaug,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch == args.epochs - 1):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device, test_snn_aug, "ANN",args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        if max_accuracy == test_stats["acc1"]:
            print("find the best accuracy, save model")
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="best")
            
        args.choose_prune = 1.0 if max_accuracy == test_stats["acc1"] else 0.0

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            if args.wandb:
                epoch_1000x = int(((len(data_loader_train) - 1) / len(data_loader_train) + epoch) * 1000)
                wandb.log({'test_acc1_curve': test_stats['acc1']}, step=epoch_1000x)
                wandb.log({'test_acc5_curve': test_stats['acc5']}, step=epoch_1000x)
                wandb.log({'test_loss_curve': test_stats['loss']}, step=epoch_1000x)
                if args.mode == "SNN":
                    for t in range(model.max_T):
                        wandb.log({'acc1@{}_curve'.format(t + 1): test_stats['acc@{}'.format(t + 1)]}, step=epoch_1000x)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

import torch.multiprocessing as mp

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    mp.set_start_method('forkserver', force=True)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.output_dir = os.path.join(args.log_dir,
                                    "{}_{}_{}_{}_{}_act{}_weightbit{}_NormType{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit,args.NormType))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(args.output_dir)
    main(args)
