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
import inspect
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import timm
from timm.utils import accuracy

import models_vit
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from snn.layer import DyHT, DyHT_ReLU, DyT, MyBatchNorm1d, MyLayerNorm, cal_overfire_loss
from snn.wrapper import add_bn_in_mlp, add_convEmbed, myquan_replace, remove_softmax, SNNWrapper
from PowerNorm import MaskPowerNorm


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
    input_time_keys = [k for k in input_state_dict.keys() if keyword in k]
    model_time_keys = [k for k in model_state_dict.keys() if keyword in k]

    print(f"Found {len(input_state_dict)} keys in input_state_dict, {len(model_state_dict)} keys in model_state_dict")
    print(f"Found {len(input_time_keys)} input time_allocator keys, {len(model_time_keys)} model time_allocator keys")

    def find_matching_input_key(model_key):
        if model_key in input_state_dict:
            return model_key
        model_suffix = model_key.split('.')[-1]
        for k in input_time_keys:
            if k.split('.')[-1] == model_suffix:
                return k
        if len(input_time_keys) == 1:
            return input_time_keys[0]
        if len(input_time_keys) > 0:
            return input_time_keys[0]
        return None

    for k, model_val in model_state_dict.items():
        if keyword in k:
            in_key = find_matching_input_key(k)
            target_shape = tuple(model_val.shape)
            target_dtype = model_val.dtype

            if in_key is None:
                print(f"[WARN] No input_time key found for model key '{k}'; filling all with {fill_value}")
                new_t = torch.full(target_shape, fill_value, dtype=target_dtype, device='cpu')
            else:
                in_val = input_state_dict[in_key]
                if not isinstance(in_val, torch.Tensor):
                    in_val = torch.tensor(in_val)
                in_shape = tuple(in_val.shape)

                if in_shape[1:] != target_shape[1:]:
                    print(f"[WARN] shape mismatch for key '{k}' vs input key '{in_key}': model shape {target_shape}, input shape {in_shape}")
                    new_t = torch.full(target_shape, fill_value, dtype=target_dtype, device='cpu')
                else:
                    new_t = torch.full(target_shape, fill_value, dtype=target_dtype, device='cpu')
                    n_copy = min(in_shape[0], target_shape[0])
                    new_t[:n_copy, ...] = in_val[:n_copy].to(dtype=target_dtype, device='cpu')
                    if in_shape[0] < target_shape[0]:
                        print(f"[INFO] key '{k}': copied {n_copy} rows from input '{in_key}', filled remaining {target_shape[0]-n_copy} rows with {fill_value}")
                    else:
                        print(f"[INFO] key '{k}': input has >= rows, copied first {n_copy} rows")
            new_state[k] = new_t
        else:
            if k in input_state_dict:
                v = input_state_dict[k]
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                if tuple(v.shape) == tuple(model_val.shape):
                    new_state[k] = v.clone().detach().cpu()
                else:
                    print(f"[WARN] non-time key '{k}' shape mismatch: input {tuple(v.shape)} vs model {tuple(model_val.shape)}; using model's value")
                    new_state[k] = model_val.clone().detach().cpu()
            else:
                new_state[k] = model_val.clone().detach().cpu()

    unused_input_keys = set(input_state_dict.keys()) - set(new_state.keys())
    if unused_input_keys:
        print(f"[WARN] {len(unused_input_keys)} keys found in input_state_dict but not used (not present in model):")
        for kk in list(unused_input_keys)[:10]:
            print("  -", kk)

    return new_state


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
    trace_dir = os.path.join(base_dir, "profiler", "eval", f"rank{rank}")
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
        print(f"[Profiler][eval] Top ops by {sort_by}:")
        print(profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit))
    except Exception as exc:
        print(f"[Profiler][eval] summary failed: {exc}")
    if getattr(args, "profile_profile_memory", False):
        mem_sort = "self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage"
        try:
            print(f"[Profiler][eval] Top ops by {mem_sort}:")
            print(profiler.key_averages().table(sort_by=mem_sort, row_limit=row_limit))
        except Exception as exc:
            print(f"[Profiler][eval] memory summary failed: {exc}")


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
        if args.NormType != "layernorm" or args.remove_softmax:
            model = timm.create_model(
                "swin_small_patch4_window7_224_Hybrid",
                norm_layer=norm_layer,
                act_layer=activation,
                pretrained=False,
                drop_path_rate=args.drop_path,
            )
        else:
            model = timm.create_model(
                "swin_small_patch4_window7_224_Hybrid",
                norm_layer=norm_layer,
                act_layer=activation,
                pretrained=False,
                drop_path_rate=args.drop_path,
                checkpoint_path="/data/kang_you1/swin_base_patch4_window7_224_22kto1k.pth",
            )
    elif "swin_base" in args.model:
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            norm_layer=norm_layer,
            act_layer=activation,
            pretrained=False,
            checkpoint_path="/data/kang_you1/swin_base_patch4_window7_224_22kto1k.pth",
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
    if args.NormType in ("mybatchnorm", "dyt"):
        add_bn_in_mlp(model, norm_layer)
    if args.convEmbedding:
        add_convEmbed(model)

    return model


def _load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)


def _unwrap_snn_model(model):
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def evaluate(data_loader, model, device, snn_aug, args, profiler=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    total_num = 0
    correct_per_timestep = None
    correct_per_timestep_sum = None
    max_T = 0

    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.dataset == "cifar10dvs" or args.dataset == "dvs128":
            N = images.shape[0]
            if snn_aug is not None:
                images = torch.stack([(snn_aug(images[i])) for i in range(N)])
            if args.mode == "SNN" and args.dataset == "dvs128":
                pass
            else:
                images = images.sum(dim=1)

        if args.mode != "SNN":
            output = model(images)
        else:
            snn_model = _unwrap_snn_model(model)
            output, count, _, accu_per_timestep = snn_model(images, verbose=True)
            overfire_loss = cal_overfire_loss(model) * 0.1
            print(overfire_loss)

            max_T = max(max_T, count)
            if accu_per_timestep.shape[0] < max_T:
                padding_per_timestep = accu_per_timestep[-1].unsqueeze(0)
                padding_length = max_T - accu_per_timestep.shape[0]
                accu_per_timestep = torch.cat(
                    [accu_per_timestep, padding_per_timestep.repeat(padding_length, 1, 1)], dim=0)

            _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
            correct_per_timestep = torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)
            if correct_per_timestep_sum is None:
                correct_per_timestep_sum = correct_per_timestep.detach().clone()
            else:
                if correct_per_timestep_sum.shape[0] < correct_per_timestep.shape[0]:
                    pad = correct_per_timestep_sum.new_zeros(
                        (correct_per_timestep.shape[0] - correct_per_timestep_sum.shape[0],)
                    )
                    correct_per_timestep_sum = torch.cat([correct_per_timestep_sum, pad], dim=0)
                correct_per_timestep_sum[:correct_per_timestep.shape[0]] += correct_per_timestep.detach()

        loss = criterion(output, target)

        total_num += images.shape[0]

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.detach())
        metric_logger.meters['acc1'].update(acc1.detach(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.detach(), n=batch_size)
        if args.mode == "SNN":
            metric_logger.meters['AvgTime'].update(float(count), n=batch_size)
            _unwrap_snn_model(model).reset()

        if profiler is not None:
            profiler.step()

    print("Evaluation End")
    metric_logger.synchronize_between_processes()
    if args.mode == "SNN" and correct_per_timestep_sum is not None and args.distributed:
        correct_per_timestep_sum = correct_per_timestep_sum.to(device)
        torch.distributed.all_reduce(correct_per_timestep_sum)
        total_num_tensor = torch.tensor(
            total_num, device=device, dtype=correct_per_timestep_sum.dtype
        )
        torch.distributed.all_reduce(total_num_tensor)
        total_num = int(total_num_tensor.item())
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.mode == "SNN" and correct_per_timestep_sum is not None:
        acc_per_timestep = (correct_per_timestep_sum / float(total_num)) * 100.0
        acc_per_timestep = acc_per_timestep.detach().cpu().tolist()
        for t, val in enumerate(acc_per_timestep):
            stats['acc@{}'.format(t + 1)] = val
    return stats


def get_args_parser():
    parser = argparse.ArgumentParser('SNN eval', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for evaluation')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--model', default='vit_small_patch16', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--act_layer', type=str, default="relu",
                        help='Using ReLU or GELU as activation')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--define_params', action='store_true')
    parser.add_argument('--mean', nargs='+', type=float)
    parser.add_argument('--std', nargs='+', type=float)

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to log')

    parser.add_argument('--mode', default="SNN", type=str,
                        help='the running mode of the script["ANN", "QANN_PTQ", "QANN_QAT", "SNN"]')
    parser.add_argument('--level', default=32, type=int, help='the quantization levels')
    parser.add_argument('--weight_quantization_bit', default=32, type=int)
    parser.add_argument('--neuron_type', default="ST-BIF", type=str,
                        help='neuron type["ST-BIF", "IF"]')
    parser.add_argument('--neuron_impl', default="auto", type=str, choices=["auto", "torch"],
                        help='neuron implementation: auto uses CUDA/custom path when available; torch uses pure torch ops')
    parser.add_argument('--remove_softmax', action='store_true',
                        help='need softmax or not')
    parser.add_argument('--NormType', default='layernorm', type=str,
                        help='the normalization type')
    parser.add_argument('--convEmbedding', action='store_true',
                        help='ConvEmbedding from QKFormer')
    parser.add_argument('--time_step', default=2000, type=int,
                        help='time-step for snn')
    parser.add_argument('--encoding_type', default="analog", type=str,
                        help='encoding type for snn')
    parser.add_argument('--suppress_over_fire', action='store_true', default=False,
                        help='suppress_over_fire')
    parser.add_argument('--record_inout', action='store_true', default=False,
                        help='record the snn input and output or not')
    parser.add_argument('--hybrid_training', action='store_true', default=False,
                        help='training after conversion')

    parser.add_argument('--finetune', default='', help='checkpoint to load before eval')
    parser.add_argument('--resume', default='', help='alternate checkpoint to load before SNN conversion')
    parser.add_argument('--snn_model_path', default="", type=str,
                        help='snn_model_path')

    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended for speed)')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

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


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_val = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
        drop_last=False
    )

    model = _build_base_model(args)

    if args.mode == "SNN":
        if (args.dataset == "cifar10dvs" or args.dataset == "dvs128") and args.mode == "SNN":
            if not args.convEmbedding:
                model.patch_embed.proj = torch.nn.Sequential(
                    torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    model.patch_embed.proj
                )
            else:
                model.patch_embed.proj_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    model.patch_embed.proj_conv
                )

        myquan_replace(model, args.level, args.weight_quantization_bit, is_softmax=not args.remove_softmax)

        load_path = args.resume if args.resume else args.finetune
        if len(load_path) > 0:
            print("Load pre-trained checkpoint from: %s" % load_path)
            _load_checkpoint(model, load_path)

        model = SNNWrapper(
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

        if len(args.snn_model_path) > 0:
            checkpoint = torch.load(args.snn_model_path, map_location='cpu', weights_only=False)
            print("Load SNN checkpoint from: %s" % args.snn_model_path)
            checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            new_state_dict = {}
            for k, v in checkpoint_model.items():
                new_state_dict["model." + k] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(msg)

    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    if args.mode != "SNN":
        model_without_ddp = model.module if args.distributed else model
    else:
        model_without_ddp = model.module.model if args.distributed else model.model

    if len(args.finetune) > 0:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        input_state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model_state_dict = model_without_ddp.state_dict()
        new_state_dict = build_state_dict_with_time_allocator(input_state_dict, model_state_dict)
        model_without_ddp.load_state_dict(new_state_dict)

    rank = misc.get_rank()
    profiler = _build_profiler(args, rank)
    if profiler is not None:
        profiler.start()

    test_stats = evaluate(data_loader_val, model, device, None, args, profiler=profiler)

    if profiler is not None:
        profiler.stop()
        _print_profile_summary(profiler, args, rank)

    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SNN eval', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
