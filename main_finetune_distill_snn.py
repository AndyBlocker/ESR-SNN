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
from copy import deepcopy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import timm
from torchvision import transforms
from util.SNNaugment import SNNAugmentWide
# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from snn.layer import DyHT, DyHT_ReLU, DyT, MyBatchNorm1d, MyLayerNorm, set_init_false
from snn.wrapper import add_bn_in_mlp, add_convEmbed, myquan_replace, remove_softmax, SNNWrapper, SNNWrapper_MS
import timm.optim.optim_factory as optim_factory

import models_vit
import wandb

from engine_finetune import evaluate, train_one_epoch_distill, train_one_epoch_distill_mse, Align_QANN_SNN, train_one_epoch_distill_snn
from PowerNorm import MaskPowerNorm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# torch.set_default_dtype(torch.double)
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=15)

def build_state_dict_with_time_allocator(input_state_dict, model_state_dict, keyword="time_allocator", fill_value=1.0):
    """
    Build new_state_dict such that:
      - keys not containing `keyword` prefer values from input_state_dict (if present),
        otherwise use model_state_dict.
      - for keys containing `keyword`, construct a tensor of same shape as model_state_dict[key]:
        first part along dim=0 copied from input_state_dict (if found and shape-compatible),
        remaining rows filled with fill_value.
    Returns new_state_dict (cpu tensors).
    """
    new_state = {}
    # collect input keys that mention the keyword for easier matching
    input_time_keys = [k for k in input_state_dict.keys() if keyword in k]
    # collect model keys that mention the keyword
    model_time_keys = [k for k in model_state_dict.keys() if keyword in k]

    print(f"Found {len(input_state_dict)} keys in input_state_dict, {len(model_state_dict)} keys in model_state_dict")
    print(f"Found {len(input_time_keys)} input time_allocator keys, {len(model_time_keys)} model time_allocator keys")

    # helper: find best-matching input key for a given model key that contains keyword
    def find_matching_input_key(model_key):
        # exact match
        if model_key in input_state_dict:
            return model_key
        # suffix match (match final name)
        model_suffix = model_key.split('.')[-1]
        for k in input_time_keys:
            if k.split('.')[-1] == model_suffix:
                return k
        # if there's exactly one input_time_key, use it
        if len(input_time_keys) == 1:
            return input_time_keys[0]
        # otherwise fallback: try any input key that contains keyword
        if len(input_time_keys) > 0:
            return input_time_keys[0]
        return None

    # process all keys present in the model (we follow model_state_dict keys order)
    for k, model_val in model_state_dict.items():
        if keyword in k:
            # handle time_allocator-like key
            in_key = find_matching_input_key(k)
            target_shape = tuple(model_val.shape)
            target_dtype = model_val.dtype
            target_device = model_val.device if isinstance(model_val, torch.Tensor) else torch.device('cpu')

            if in_key is None:
                # no input provided, just fill entire tensor with fill_value
                print(f"[WARN] No input_time key found for model key '{k}'; filling all with {fill_value}")
                new_t = torch.full(target_shape, fill_value, dtype=target_dtype, device='cpu')
            else:
                in_val = input_state_dict[in_key]
                # ensure tensor
                if not isinstance(in_val, torch.Tensor):
                    in_val = torch.tensor(in_val)
                in_shape = tuple(in_val.shape)

                # Check compatibility beyond first dim
                if in_shape[1:] != target_shape[1:]:
                    print(f"[WARN] shape mismatch for key '{k}' vs input key '{in_key}': model shape {target_shape}, input shape {in_shape}")
                    # try to broadcast/copy minimum suffix shape
                    # conservative fallback: use model_val entirely
                    new_t = torch.full(target_shape, fill_value, dtype=target_dtype, device='cpu')
                else:
                    # allocate full tensor and copy prefix
                    new_t = torch.full(target_shape, fill_value, dtype=target_dtype, device='cpu')
                    n_copy = min(in_shape[0], target_shape[0])
                    # cast input values to target dtype if needed and copy first n_copy along dim=0
                    new_t[:n_copy, ...] = in_val[:n_copy].to(dtype=target_dtype, device='cpu')
                    if in_shape[0] < target_shape[0]:
                        print(f"[INFO] key '{k}': copied {n_copy} rows from input '{in_key}', filled remaining {target_shape[0]-n_copy} rows with {fill_value}")
                    else:
                        print(f"[INFO] key '{k}': input has >= rows, copied first {n_copy} rows")
            # store cpu tensor in new_state (state_dict entries are usually cpu tensors)
            new_state[k] = new_t
        else:
            # non-time_allocator key: prefer input_state_dict if available
            if k in input_state_dict:
                v = input_state_dict[k]
                # ensure tensor
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                # if shapes match use input, else fallback to model's (with a warning)
                if tuple(v.shape) == tuple(model_val.shape):
                    new_state[k] = v.clone().detach().cpu()
                else:
                    print(f"[WARN] non-time key '{k}' shape mismatch: input {tuple(v.shape)} vs model {tuple(model_val.shape)}; using model's value")
                    new_state[k] = model_val.clone().detach().cpu()
            else:
                # not in input, use model's value
                new_state[k] = model_val.clone().detach().cpu()

    # optionally report keys that were in input but not used
    unused_input_keys = set(input_state_dict.keys()) - set(new_state.keys())
    if unused_input_keys:
        print(f"[WARN] {len(unused_input_keys)} keys found in input_state_dict but not used (not present in model):")
        for kk in list(unused_input_keys)[:10]:
            print("  -", kk)

    return new_state


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
    parser.add_argument('--global_pool', action='store_true')
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
    parser.add_argument('--convEmbedding', action='store_true',
                        help='ConvEmbedding from QKFormer')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--wandb', action='store_true',
                        help='Using wandb or not')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
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
    parser.add_argument('--prune_ratio', type=float, default="3e8",
                        help='prune ratio')
    parser.add_argument('--hybrid_training', action='store_true', default=False,
                        help='training after conversion')
    parser.add_argument('--record_inout', action='store_true', default=False,
                        help='record the snn input and output or not')
    parser.add_argument('--energy', action='store_true', default=False,
                        help='calculate energy or not')
    parser.add_argument('--snn_model_path', default="", type=str,
                        help='snn_model_path')
    parser.add_argument('--suppress_over_fire', action='store_true', default=False,
                        help='suppress_over_fire')
    return parser


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
                                    "{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit))
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
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
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
    elif "swin_tiny_cifar" in args.model:
        model = timm.create_model("swin_cifar_patch4_window7_224",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path, img_size=args.input_size, num_classes=args.nb_classes)
    elif "swin_tiny_dvs" in args.model:
        model = timm.create_model("swin_dvs_patch4_window4_128",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path, in_chans=3, img_size=args.input_size, num_classes=args.nb_classes)
    elif "swin_tiny" in args.model:
        model = timm.create_model("swin_tiny_patch4_window7_224",norm_layer=normLayer,act_layer=activation,pretrained=False,drop_path_rate=args.drop_path)
    elif "swin_small" in args.model:
        if args.NormType != "layernorm" or args.remove_softmax:
            model = timm.create_model("swin_small_patch4_window7_224_Hybrid",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path)
        else:
            model = timm.create_model("swin_small_patch4_window7_224_Hybrid",norm_layer=normLayer,act_layer=activation,pretrained=False, drop_path_rate=args.drop_path, checkpoint_path="/data/kang_you1/swin_base_patch4_window7_224_22kto1k.pth")
    elif "swin_base" in args.model:
        model = timm.create_model("swin_base_patch4_window7_224",norm_layer=normLayer,act_layer=activation,pretrained=False,checkpoint_path="/data/kang_you1/swin_base_patch4_window7_224_22kto1k.pth",drop_path_rate=args.drop_path)
    else:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            drop_rate = args.drop_rate,
            global_pool=args.global_pool,
            act_layer=activation,
            norm_layer=normLayer,
        )
    if args.remove_softmax:
        remove_softmax(model)
    if args.NormType == "mybatchnorm" or args.NormType == "dyt":
        add_bn_in_mlp(model, normLayer)
    if args.convEmbedding:
        add_convEmbed(model)

    # if "swin_tiny" in args.model:
    #     model_teacher = timm.create_model("swin_tiny_patch4_window7_224",pretrained=False, checkpoint_path="/data/kang_you1/swin_tiny_patch4_window7_224.pth",act_layer=activation_teacher)
    # elif "swin_base" in args.model:
    #     model_teacher = timm.create_model("swin_base_patch4_window7_224",pretrained=False, checkpoint_path="/data/kang_you1/swin_base_patch4_window7_224_22kto1k.pth",act_layer=activation_teacher)
    # else:
    #     model_teacher = models_vit.__dict__[args.model_teacher](
    #     num_classes=args.nb_classes,
    #     drop_path_rate=args.drop_path,
    #     global_pool=False if "vit_small" in args.model_teacher else args.global_pool,
    #     act_layer=activation_teacher,
    #     norm_layer=nn.LayerNorm,
    #     )

    model_teacher = None
    # model_teacher.eval()

    # assert args.pretrain_teacher is not None
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
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s !!!!!!!" % args.finetune)
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

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        # if not args.mode == "QANN-QAT":
        #     trunc_normal_(model.head.weight, std=2e-5)

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
            print("Load pre-trained checkpoint from: %s !!!!!!!" % args.finetune)
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
        if (args.dataset == "cifar10dvs" or args.dataset == "dvs128") and args.mode == "SNN":
            if not args.convEmbedding:
                model.patch_embed.proj = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), model.patch_embed.proj)        
            else:
                model.patch_embed.proj_conv = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), model.patch_embed.proj_conv) 
        
        myquan_replace(model, args.level, args.weight_quantization_bit, is_softmax = not args.remove_softmax)

        if len(args.finetune) > 0:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False) if not args.eval else torch.load(args.resume,
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
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
        if args.rank == 0:
            print("======================== QANN model =======================")
            f = open(f"qann_model_arch.txt", "w+")
            f.write(str(model))
            f.close()
        # model_teacher = deepcopy(model)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        # trunc_normal_(model.head.weight, std=2e-5)
        # print(model)
        model = SNNWrapper_MS(ann_model=model, cfg=args, time_step=args.time_step, \
                           Encoding_type=args.encoding_type, level=args.level, neuron_type=args.neuron_type, \
                           model_name=args.model, is_softmax = not args.remove_softmax, suppress_over_fire = args.suppress_over_fire, \
                           record_inout=args.record_inout,learnable=args.hybrid_training,record_dir=args.log_dir+f"/output_bin_snn_{args.model}_w{args.weight_quantization_bit}_a{int(torch.log2(torch.tensor(args.level)))}_T{args.time_step}/")
        # print(model)
        
        if len(args.snn_model_path) > 0:
            checkpoint = torch.load(args.snn_model_path, map_location='cpu', weights_only=False)
            print("Load SNN checkpoint from: %s" % args.snn_model_path)
            checkpoint_model = checkpoint['model']
            new_state_dict = {}
            for k,v in checkpoint_model.items():
                new_state_dict["model."+k] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(msg)

        if args.rank == 0:
            print("======================== SNN model =======================")
            f = open(f"snn_model_arch.txt", "w+")
            f.write(str(model))
            f.close()

    if (args.dataset == "cifar10dvs" or args.dataset == "dvs128") and args.mode == "ANN":
        if not args.convEmbedding:
            model.patch_embed.proj = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), model.patch_embed.proj)        
        else:
            model.patch_embed.proj_conv = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False), model.patch_embed.proj_conv) 

    model.to(device)
    # model_teacher.to(device)

    model_without_ddp = model if args.mode != "SNN" else model.model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
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
        # model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module if args.mode != "SNN" else model.module.model

    # build optimizer with layer-wise lr decay (lrd)
    if "vit" in args.model:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        # optimizer = optim_factory.Lamb(param_groups, trust_clip=True, lr=args.lr)
        # optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9)
    else:
        param_groups = []
        for n, p in model_without_ddp.named_parameters():
            if p.requires_grad:
                param_groups.append({'params': [p], 'lr': args.lr, 'weight_decay': args.weight_decay, 'layer_name': n})
        # optimizer = optim_factory.Lamb(model_without_ddp.parameters(), trust_clip=True, lr=args.lr)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        # optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    # optimizer_align = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.mode != "SNN":
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # if len(args.resume) > 0:
    #     checkpoint = torch.load(args.resume)
    #     model_without_ddp.load_state_dict(checkpoint["model"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     loss_scaler.load_state_dict(checkpoint["scaler"])
    #     time_step = args.time_step + 0
    #     args = checkpoint["args"]
    #     args.time_step = time_step + 0
    #     args.start_epoch = checkpoint["epoch"]
    #     set_init_false(model)
    
    if args.eval:
        checkpoint = torch.load(args.finetune)
        input_state_dict = checkpoint["model"]
        model_state_dict = model_without_ddp.state_dict()
        new_state_dict = build_state_dict_with_time_allocator(input_state_dict, model_state_dict)
        
        model_without_ddp.load_state_dict(new_state_dict)
        test_stats = evaluate(data_loader_val, model, device, None,"SNN", args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

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

    # if not args.energy:
    #     # print("args.time_step",args.time_step)
    #     print("model",model)
    #     # test_stats = evaluate(data_loader_val, model_teacher, device, test_snn_aug, "ANN",  args)
    #     # print(f"Accuracy of the teacher network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    #     test_stats = evaluate(data_loader_val, model, device, test_snn_aug, args.mode,  args)
    #     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    #     for t in range(args.time_step):
    #         print(f"Accuracy of the network on the {len(dataset_val)} test images at time step {t + 1}: {test_stats['acc@{}'.format(t + 1)]:.1f}%")
    #         # wandb.log({'acc1@{}_curve'.format(t + 1): test_stats['acc@{}'.format(t + 1)]}, step=epoch_1000x)
        
    #     exit(0)
        
    if args.energy:
        from energy_consumption_calculation import get_model_complexity_info
        if len(args.snn_model_path) > 0:
            checkpoint = torch.load(args.snn_model_path, map_location='cpu')
            print("Load SNN checkpoint from: %s" % args.snn_model_path)
            checkpoint_model = checkpoint['model']
            new_state_dict = {}
            for k,v in checkpoint_model.items():
                new_state_dict["module.model."+k] = v
            msg = model.load_state_dict(new_state_dict, strict=True)
            print(msg)
            # msg = model.load_state_dict(checkpoint_model, strict=True)

        # test_stats = evaluate(data_loader_val, model, device,None,args.mode, args)
        # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        ts1 = time.time()
        Nops, Nparams = get_model_complexity_info(model, (3, 224, 224), data_loader_val,ost = open(f"{args.log_dir}/energy_info.txt","w+"), as_strings=True, print_per_layer_stat=True, verbose=True, syops_units='Mac', param_units=' ', output_precision=3)
        print("Nops: ", Nops)
        print("Nparams: ", Nparams)
        t_cost = (time.time() - ts1) / 60
        print(f"Time cost: {t_cost} min")
        exit(0)
    # if args.mode == "SNN" and misc.is_main_process():
    #     for k, v in test_stats.items():
    #         print(k, v)
    #     with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
    #         json.dump(test_stats, f)
    #     exit(0)
    # print(f"begin to align QANN and SNN!!!!!")
    # Align_QANN_SNN(model, model_teacher, criterion, data_loader_train,
    #         optimizer, device, 0, loss_scaler,
    #         args.clip_grad, mixup_fn,
    #         log_writer=log_writer,
    #         args=args)

    # test_stats = evaluate(data_loader_val, model_teacher, device, test_snn_aug ,"ANN", args)
    # print(f"Accuracy of the teacher network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
    # test_stats = evaluate(data_loader_val, model, device, test_snn_aug, args.mode, args)
    # print(f"Accuracy of the spiking neural network after alignment on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
    # exit(0)

    args.choose_prune = 1.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_distill_snn(
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

        test_stats = evaluate(data_loader_val, model, device, test_snn_aug, args.mode ,args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        args.choose_prune = 1.0 if max_accuracy == test_stats["acc1"] else 0.0
        if max_accuracy == test_stats["acc1"]:
            print("find the best accuracy, save model")
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="best")

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
        args.output_dir = os.path.join(args.output_dir,
                                       "{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(args.output_dir)
    main(args)
