# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import inspect
import json
import os
import re
import time
from functools import partial
from pathlib import Path

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models_vit
import util.misc as misc
from engine_finetune import train_one_epoch_distill_snn
from PowerNorm import MaskPowerNorm
from snn.layer import DyHT, DyHT_ReLU, DyT, MyBatchNorm1d, MyLayerNorm, set_init_false
from snn.nvtx import profiler_range
from snn.wrapper import SNNWrapper_MS, add_bn_in_mlp, add_convEmbed, myquan_replace, remove_softmax
from util.SNNaugment import SNNAugmentWide
from util.datasets import TransformFirstDataset, build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed

import wandb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(precision=15)

def _fmt_bytes(num_bytes: int) -> str:
    mib = num_bytes / (1024 ** 2)
    gib = num_bytes / (1024 ** 3)
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    return f"{mib:.1f} MiB"


def _get_profile_base_dir(args):
    if args is not None and getattr(args, "profile_dir", ""):
        return args.profile_dir
    if args is not None and getattr(args, "output_dir", ""):
        return args.output_dir
    return "."


def _build_profiler(args, rank):
    if args is None or not getattr(args, "profile", False):
        return None
    if not getattr(args, "profile_all_ranks", False) and rank != 0:
        return None

    base_dir = _get_profile_base_dir(args)
    trace_dir = os.path.join(base_dir, "profiler", "train_bias_allocator", f"rank{rank}")
    os.makedirs(trace_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=getattr(args, "profile_wait", 1),
        warmup=getattr(args, "profile_warmup", 1),
        active=getattr(args, "profile_active", 3),
        repeat=getattr(args, "profile_repeat", 1),
    )

    sig = inspect.signature(torch.profiler.profile)
    profile_kwargs = {
        "record_shapes": getattr(args, "profile_record_shapes", False),
        "with_stack": getattr(args, "profile_with_stack", False),
        "profile_memory": getattr(args, "profile_profile_memory", False),
    }
    if "with_flops" in sig.parameters:
        profile_kwargs["with_flops"] = getattr(args, "profile_with_flops", False)

    return torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            trace_dir, worker_name=f"rank{rank}"
        ),
        **profile_kwargs,
    )


def _print_profile_summary(profiler, args, rank):
    if profiler is None or args is None or not getattr(args, "profile_print", False):
        return
    if not getattr(args, "profile_all_ranks", False) and rank != 0:
        return
    row_limit = getattr(args, "profile_print_limit", 30)
    sort_by = getattr(args, "profile_print_sort", "")
    if not sort_by:
        sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    try:
        print(f"[Profiler][train_bias_allocator] Top ops by {sort_by}:")
        print(profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit))
    except Exception as exc:
        print(f"[Profiler][train_bias_allocator] summary failed: {exc}")
    if getattr(args, "profile_profile_memory", False):
        mem_sort = "self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage"
        try:
            print(f"[Profiler][train_bias_allocator] Top ops by {mem_sort}:")
            print(profiler.key_averages().table(sort_by=mem_sort, row_limit=row_limit))
        except Exception as exc:
            print(f"[Profiler][train_bias_allocator] memory summary failed: {exc}")


def get_args_parser():
    parser = argparse.ArgumentParser('SNN BiasAllocator training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--print_freq', default=1000, type=int,
                        help='print_frequency')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--project_name', default='T-SNN', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Model parameters
    parser.add_argument('--model', default='vit_small_patch16', type=str, metavar='MODEL',
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
    parser.add_argument('--act_layer', type=str, default="relu",
                        help='Using ReLU or GELU as activation')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

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
                        help='layer-wise lr decay (unused for BiasAllocator-only training)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--no_aug', action='store_true', default=False,
                        help='do not apply augmentation')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Checkpoints / IO
    parser.add_argument('--finetune', default='',
                        help='checkpoint to load before SNN conversion')
    parser.add_argument('--resume', default='',
                        help='resume BiasAllocator training checkpoint')
    parser.add_argument('--snn_model_path', default="", type=str,
                        help='optional SNN checkpoint to load (BiasAllocator warm-start)')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--wandb', action='store_true',
                        help='Using wandb or not')

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

    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor)')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--ddp_find_unused_parameters', default=0, type=int, choices=[0, 1],
                        help='DDP find_unused_parameters (0/1). For BiasAllocator training this should usually be 0.')

    parser.add_argument('--bias_allocator_include_regex', default='biasAllocator', type=str,
                        help='regex for trainable BiasAllocator params (matched against parameter names)')
    parser.add_argument('--bias_allocator_exclude_regex', default='', type=str,
                        help='optional regex to exclude BiasAllocator params from training')
    parser.add_argument('--bias_allocator_auto_disable_unused', default=1, type=int, choices=[0, 1],
                        help='auto-disable trainable BiasAllocator params that do not receive grads in a short probe step (recommended when ddp_find_unused_parameters=0)')
    parser.add_argument('--bias_allocator_probe_steps', default=1, type=int,
                        help='number of probe batches used to detect unused BiasAllocator params (default: 1)')

    # SNN / quantization parameters
    parser.add_argument('--mode', default="SNN", type=str,
                        help='running mode (BiasAllocator expects SNN)')
    parser.add_argument('--level', default=32, type=int,
                        help='the quantization levels')
    parser.add_argument('--weight_quantization_bit', default=32, type=int,
                        help="the weight quantization bit")
    parser.add_argument('--neuron_type', default="ST-BIF", type=str,
                        help='neuron type["ST-BIF", "IF"]')
    parser.add_argument('--neuron_impl', default="torch", type=str, choices=["auto", "torch"],
                        help='MS neuron implementation: "auto" tries CuPy CUDA then falls back to torch; "torch" forces torch op.')
    parser.add_argument('--remove_softmax', action='store_true',
                        help='need softmax or not')
    parser.add_argument('--NormType', default='layernorm', type=str,
                        help='the normalization type')
    parser.add_argument('--convEmbedding', action='store_true',
                        help='ConvEmbedding from QKFormer')
    parser.add_argument('--hybrid_training', action='store_true', default=False,
                        help='training after conversion')
    parser.add_argument('--record_inout', action='store_true', default=False,
                        help='record the snn input and output or not')
    parser.add_argument('--suppress_over_fire', action='store_true', default=False,
                        help='suppress_over_fire')
    parser.add_argument('--max_train_steps', default=100, type=int,
                        help='limit number of train iterations per epoch (useful for profiling)')
    parser.add_argument('--save_checkpoint', action='store_true', default=False,
                        help='save checkpoints (off by default for profiling)')
    parser.add_argument('--snn_verbose', default=0, type=int, choices=[0, 1],
                        help='SNN forward verbose mode (may early-exit). For BiasAllocator training default is 0 (full-T).')
    parser.add_argument('--grad_checkpointing', action='store_true', default=False,
                        help='Enable activation checkpointing for transformer blocks (reduces memory, increases compute).')

    parser.add_argument('--profile', action='store_true', default=False,
                        help='enable torch profiler')
    parser.add_argument('--profile_dir', default='', type=str,
                        help='output dir for profiler traces')
    parser.add_argument('--profile_wait', default=1, type=int,
                        help='profiler schedule: wait steps')
    parser.add_argument('--profile_warmup', default=1, type=int,
                        help='profiler schedule: warmup steps')
    parser.add_argument('--profile_active', default=3, type=int,
                        help='profiler schedule: active steps')
    parser.add_argument('--profile_repeat', default=1, type=int,
                        help='profiler schedule: repeat cycles')
    parser.add_argument('--profile_record_shapes', action='store_true', default=False,
                        help='record tensor shapes in profiler')
    parser.add_argument('--profile_profile_memory', action='store_true', default=False,
                        help='record memory in profiler')
    parser.add_argument('--profile_with_stack', action='store_true', default=False,
                        help='record python stacks in profiler')
    parser.add_argument('--profile_with_flops', action='store_true', default=False,
                        help='record flops in profiler (if supported)')
    parser.add_argument('--profile_all_ranks', action='store_true', default=False,
                        help='profile all ranks (default: rank 0 only)')
    parser.add_argument('--profile_print', action='store_true', default=False,
                        help='print profiler summary after run')
    parser.add_argument('--profile_print_limit', default=30, type=int,
                        help='row limit for profiler summary tables')
    parser.add_argument('--profile_print_sort', default='', type=str,
                        help='sort key for profiler summary')

    return parser


def _build_base_model(args):
    if args.act_layer == "relu":
        activation = nn.ReLU
    elif args.act_layer == "gelu":
        activation = nn.GELU
    else:
        raise NotImplementedError

    if args.NormType == "layernorm":
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
    elif args.NormType == "powernorm":
        norm_layer = partial(MaskPowerNorm, eps=1e-6)
    elif args.NormType == "mylayernorm":
        norm_layer = partial(MyLayerNorm, eps=1e-6)
    elif args.NormType == "mybatchnorm":
        norm_layer = partial(MyBatchNorm1d, eps=1e-6)
    elif args.NormType == "dyt":
        norm_layer = partial(DyT)
    elif args.NormType == "dyht":
        norm_layer = partial(DyHT)
    elif args.NormType == "dyht_relu":
        norm_layer = partial(DyHT_ReLU)
    else:
        raise NotImplementedError

    if "vit_small" in args.model:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            drop_rate=args.drop_rate,
            global_pool=args.global_pool,
            act_layer=activation,
            norm_layer=norm_layer,
        )
    elif "swin_tiny_cifar" in args.model:
        model = timm.create_model(
            "swin_cifar_patch4_window7_224",
            norm_layer=norm_layer,
            act_layer=activation,
            pretrained=False,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            num_classes=args.nb_classes,
        )
    elif "swin_tiny_dvs" in args.model:
        model = timm.create_model(
            "swin_dvs_patch4_window4_128",
            norm_layer=norm_layer,
            act_layer=activation,
            pretrained=False,
            drop_path_rate=args.drop_path,
            in_chans=3,
            img_size=args.input_size,
            num_classes=args.nb_classes,
        )
    elif "swin_tiny" in args.model:
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            norm_layer=norm_layer,
            act_layer=activation,
            pretrained=False,
            drop_path_rate=args.drop_path,
        )
    elif "swin_small" in args.model:
        model = timm.create_model(
            "swin_small_patch4_window7_224_Hybrid",
            norm_layer=norm_layer,
            act_layer=activation,
            pretrained=False,
            drop_path_rate=args.drop_path,
        )
    elif "swin_base" in args.model:
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            norm_layer=norm_layer,
            act_layer=activation,
            pretrained=False,
            drop_path_rate=args.drop_path,
        )
    else:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            drop_rate=args.drop_rate,
            global_pool=args.global_pool,
            act_layer=activation,
            norm_layer=norm_layer,
        )

    if args.remove_softmax:
        remove_softmax(model)
    if args.NormType in {"mybatchnorm", "dyt"}:
        add_bn_in_mlp(model, norm_layer)
    if args.convEmbedding:
        add_convEmbed(model)
    if args.grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    elif args.grad_checkpointing:
        setattr(model, "grad_checkpointing", True)
    return model


def _load_checkpoint_into_model(model, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in state and k in state_dict and state[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from checkpoint (shape mismatch)")
            del state[k]
    interpolate_pos_embed(model, state)
    msg = model.load_state_dict(state, strict=False)
    print(msg)


def _freeze_for_bias_allocator(model, include_regex: str = "biasAllocator", exclude_regex: str = ""):
    include_pat = re.compile(include_regex) if include_regex else None
    exclude_pat = re.compile(exclude_regex) if exclude_regex else None
    trainable = []
    for name, param in model.named_parameters():
        keep = bool(include_pat.search(name)) if include_pat is not None else False
        if keep and exclude_pat is not None and exclude_pat.search(name):
            keep = False

        param.requires_grad = keep
        if param.requires_grad:
            trainable.append(name)
    return trainable


def _auto_disable_unused_trainables(model, data_loader_train, criterion, device, args, mixup_fn=None):
    if not bool(getattr(args, "bias_allocator_auto_disable_unused", 1)):
        return []

    probe_steps = int(getattr(args, "bias_allocator_probe_steps", 1) or 1)
    trainables = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    if not trainables:
        return []

    model.train(True)
    model.zero_grad(set_to_none=True)

    # Keep only params that consistently receive grads across probe steps (safe for DDP find_unused_parameters=0).
    used_mask = torch.ones(len(trainables), dtype=torch.int32, device=device)
    data_iter = iter(data_loader_train)
    for _ in range(probe_steps):
        try:
            samples, targets = next(data_iter)
        except StopIteration:
            break

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            out = model(samples, verbose=bool(getattr(args, "snn_verbose", 0)))
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits.float(), targets)

        loss.backward()
        for i, (_, param) in enumerate(trainables):
            if param.grad is None:
                used_mask[i] = 0
        model.zero_grad(set_to_none=True)

    if bool(getattr(args, "distributed", False)):
        try:
            torch.distributed.all_reduce(used_mask, op=torch.distributed.ReduceOp.MIN)
        except Exception:
            pass

    disabled = []
    for i, (name, param) in enumerate(trainables):
        if int(used_mask[i].item()) == 0:
            param.requires_grad = False
            disabled.append(name)

    return disabled


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.mode != "SNN":
        raise ValueError("BiasAllocator training expects --mode SNN")
    # Ensure training loop keeps only BiasAllocator trainable (redundant safety).
    args.freeze_except_bias_allocator = True

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset_train = build_dataset(is_train=True, args=args)

    # For DVS-style frame datasets, run augmentations inside DataLoader workers
    # (avoid per-batch Python loops in the training loop).
    if args.dataset in {"cifar10dvs", "dvs128"}:
        train_snn_aug = transforms.Compose([
            transforms.Resize(size=(args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        train_trivalaug = SNNAugmentWide()
        dataset_train = TransformFirstDataset(
            dataset_train, transforms.Compose([train_snn_aug, train_trivalaug])
        )

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_dir = os.path.join(
            args.log_dir,
            "{}_{}_{}_{}_{}_act{}_weightbit{}".format(
                args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level, args.weight_quantization_bit
            ),
        )
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        if args.wandb:
            wandb.init(
                config=args,
                project=args.project_name,
                name="{}_{}_{}_{}_{}_act{}_weightbit{}".format(
                    args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level, args.weight_quantization_bit
                ),
                dir=args.output_dir,
            )
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    model = _build_base_model(args)

    if (args.dataset in {"cifar10dvs", "dvs128"}) and args.mode == "SNN":
        if not args.convEmbedding:
            model.patch_embed.proj = torch.nn.Sequential(
                torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
                model.patch_embed.proj,
            )
        else:
            model.patch_embed.proj_conv = torch.nn.Sequential(
                torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
                model.patch_embed.proj_conv,
            )

    myquan_replace(model, args.level, args.weight_quantization_bit, is_softmax=not args.remove_softmax)

    if args.finetune:
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        _load_checkpoint_into_model(model, args.finetune)

    if global_rank == 0:
        try:
            f = open(f"{args.log_dir}/qann_model_arch.txt", "w+")
            f.write(str(model))
            f.close()
        except Exception:
            pass

    model = SNNWrapper_MS(
        ann_model=model,
        cfg=args,
        time_step=args.time_step,
        Encoding_type=args.encoding_type,
        level=args.level,
        neuron_type=args.neuron_type,
        neuron_impl=args.neuron_impl,
        model_name=args.model,
        is_softmax=not args.remove_softmax,
        suppress_over_fire=args.suppress_over_fire,
        record_inout=args.record_inout,
        learnable=args.hybrid_training,
        record_dir=args.log_dir + f"/output_bin_snn_{args.model}_w{args.weight_quantization_bit}_a{int(torch.log2(torch.tensor(args.level)))}_T{args.time_step}/",
    )

    if args.snn_model_path:
        checkpoint = torch.load(args.snn_model_path, map_location='cpu', weights_only=False)
        state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        if any(k.startswith("model.") for k in state.keys()):
            load_state = state
        else:
            load_state = {"model." + k: v for k, v in state.items()}
        print("Load SNN checkpoint from: %s" % args.snn_model_path)
        msg = model.load_state_dict(load_state, strict=False)
        print(msg)

    model.to(device)

    # Freeze BEFORE wrapping with DDP so DDP only buckets trainable params.
    trainable_names = _freeze_for_bias_allocator(
        model,
        include_regex=getattr(args, "bias_allocator_include_regex", "biasAllocator"),
        exclude_regex=getattr(args, "bias_allocator_exclude_regex", ""),
    )
    if global_rank == 0:
        print(f"[BiasAllocator] Trainable params: {len(trainable_names)}")
        for name in trainable_names[:50]:
            print("  -", name)
        if len(trainable_names) > 50:
            print(f"  ... ({len(trainable_names) - 50} more)")

    disabled = _auto_disable_unused_trainables(model, data_loader_train, criterion, device, args, mixup_fn=mixup_fn)
    if global_rank == 0 and len(disabled) > 0:
        print(f"[BiasAllocator] Auto-disabled unused trainables: {len(disabled)}")
        for name in disabled[:50]:
            print("  -", name)
        if len(disabled) > 50:
            print(f"  ... ({len(disabled) - 50} more)")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )

    model_without_ddp = model.module.model if args.distributed else model.model

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable BiasAllocator parameters found. Did the MS wrapper conversion run?")

    n_parameters = sum(p.numel() for p in trainable_params)
    print('number of trainable params (M): %.4f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start BiasAllocator training for {args.epochs} epochs (max_train_steps={args.max_train_steps})")
    start_time = time.time()

    # DVS augmentations are applied in DataLoader workers via TransformFirstDataset above.

    rank = misc.get_rank()
    profiler = _build_profiler(args, rank)
    if profiler is not None:
        profiler.start()

    mem_baseline = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        mem_baseline = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        if misc.is_main_process():
            print(
                f"[Mem][baseline] allocated={_fmt_bytes(mem_baseline[0])} reserved={_fmt_bytes(mem_baseline[1])}"
            )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        t0 = time.perf_counter()
        with profiler_range("phase/train_one_epoch_distill_snn", enabled=args.profile):
            train_stats = train_one_epoch_distill_snn(
                model,
                None,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad,
                mixup_fn,
                log_writer=log_writer,
                aug=None,
                trival_aug=None,
                args=args,
                profiler=profiler,
            )
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        t1 = time.perf_counter()
        steps = int(train_stats.get("steps", 0) or 0)
        if misc.is_main_process() and steps > 0:
            avg_ms = (t1 - t0) * 1000.0 / steps
            print(f"[Time][train] steps={steps} total={(t1 - t0):.3f}s avg={avg_ms:.3f}ms/step")

        if args.save_checkpoint and args.output_dir and (epoch % 10 == 0 or epoch == args.epochs - 1):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if profiler is not None:
        profiler.stop()
        _print_profile_summary(profiler, args, rank)

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        cur_alloc = torch.cuda.memory_allocated()
        cur_reserved = torch.cuda.memory_reserved()
        if misc.is_main_process():
            if mem_baseline is not None:
                base_alloc, base_reserved = mem_baseline
                print(
                    f"[Mem][peak] allocated={_fmt_bytes(peak_alloc)} reserved={_fmt_bytes(peak_reserved)} "
                    f"(baseline allocated={_fmt_bytes(base_alloc)} reserved={_fmt_bytes(base_reserved)})"
                )
            else:
                print(f"[Mem][peak] allocated={_fmt_bytes(peak_alloc)} reserved={_fmt_bytes(peak_reserved)}")
            print(f"[Mem][end] allocated={_fmt_bytes(cur_alloc)} reserved={_fmt_bytes(cur_reserved)}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SNN BiasAllocator training', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.output_dir = os.path.join(
            args.output_dir,
            "{}_{}_{}_{}_{}_act{}_weightbit{}".format(
                args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level, args.weight_quantization_bit
            ),
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(args.output_dir)

    main(args)
