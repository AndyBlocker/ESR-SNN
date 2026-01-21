# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import wandb
from timm.data import Mixup
from timm.utils import accuracy
from torch.utils.tensorboard import SummaryWriter
import util.misc as misc
import util.lr_sched as lr_sched
from copy import deepcopy
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from snn.layer import cal_overfire_loss, MyQuan, ST_BIFNeuron_MS
from snn.wrapper import cal_l1_loss, open_dropout
from torch.nn.utils import prune
import re
from torch.utils.checkpoint import checkpoint


def get_logits_loss(fc_t, fc_s, one_hot_label, temp, num_classes=1000):
    s_input_for_softmax = fc_s / temp
    t_input_for_softmax = fc_t / temp

    softmax = torch.nn.Softmax(dim=1)
    logsoftmax = torch.nn.LogSoftmax()

    t_soft_label = softmax(t_input_for_softmax)

    softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)

    fc_s_auto = fc_s.detach()
    fc_t_auto = fc_t.detach()
    log_softmax_s = logsoftmax(fc_s_auto)
    log_softmax_t = logsoftmax(fc_t_auto)
    # one_hot_label = F.one_hot(label, num_classes=num_classes).float()
    softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
    softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

    focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
    ratio_lower = torch.zeros(1).cuda()
    focal_weight = torch.max(focal_weight, ratio_lower)
    focal_weight = 1 - torch.exp(- focal_weight)
    softmax_loss = focal_weight * softmax_loss

    soft_loss = (temp ** 2) * torch.mean(softmax_loss)

    return soft_loss

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.profile_time and data_iter_step >= 2:
            break
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.mode != "SNN":
                outputs = model(samples)
            else:
                outputs, counts = model(samples, verbose=False)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # 对于模型的每个参数，计算其梯度的L2范数
        # for param in model.parameters():
        #     if param.grad is not None:
        #         grad_norm = torch.norm(param.grad, p=2)
        #         print(grad_norm)
        # loss_scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        # loss_scaler.step(optimizer)
        # loss_scaler.update()
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        if args.mode == "SNN":
            model.module.reset()

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.mode == "SNN":
                log_writer.add_scalar('counts', counts, epoch_1000x)
            if args.wandb:
                wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                wandb.log({'lr_curve': max_lr}, step=epoch_1000x)
                if args.mode == "SNN":
                    wandb.log({'counts': counts}, step=epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def replace_decimal_strings(input_string):
    pattern = r'\.(\d+)'
    
    replaced_string = re.sub(pattern, r'[\1]', input_string)

    return replaced_string

def unstruct_prune(model,ratio):
    
    # reset weight_mask
    for name, m in model.named_modules():
        if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
            if hasattr(m,"weight_mask"):
                print(m)
                m.weight.data = m.weight_orig
                m.weight_mask[m.weight_mask==0] = 1

    parameters_to_prune = []
    for name, m in model.named_modules():
        # if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
        #     parameters_to_prune.append((m ,'weight'))
        if name.count("proj")>0 or name.count("fc2")>0:
            if isinstance(m,torch.nn.Sequential) and isinstance(m[0],torch.nn.Linear):
                # print(name,m)
                parameters_to_prune.append((m[0],'weight'))
            elif isinstance(m,torch.nn.Linear):
                # print(name,m)
                parameters_to_prune.append((m ,'weight'))
    # print(tuple(parameters_to_prune),ratio)

    # global_unstructured
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
        amount=ratio,
    )
    zero_number = 0
    total_bumber = 0
    for name, m in model.named_modules():
        if name.count("proj")>0 or name.count("fc2")>0:
            if isinstance(m,torch.nn.Sequential) and isinstance(m[0],torch.nn.Linear):
                zero_number = zero_number + torch.sum(m[0].weight==0)
                total_bumber = total_bumber + m[0].weight.numel()
            elif isinstance(m,torch.nn.Linear):
                zero_number = zero_number + torch.sum(m.weight==0)
                total_bumber = total_bumber + m.weight.numel()

    print("prune finish!!!!! global sparsity:",(zero_number/total_bumber)*100)
    
def train_one_epoch_distill_prune(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    model_teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    # first prune for a certain ratio
    unstruct_prune(model,args.ratio[epoch])
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs_teacher = model_teacher(samples)
            loss = criterion(outputs, targets)
            loss_distill = get_logits_loss(outputs_teacher, outputs, targets, args.temp)
            loss_all = loss + loss_distill

        loss_value = loss.item()
        loss_distill_value = loss_distill.item()
        loss_all_value = loss_all.item()

        if not math.isfinite(loss_all_value):
            print("Loss is {}, stopping training".format(loss_all_value))
            sys.exit(1)

        loss_all /= accum_iter
        loss_scaler(loss_all, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_all=loss_all_value, loss=loss_value, loss_distill=loss_distill_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_all_value_reduce = misc.all_reduce_mean(loss_all_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_distill_value_reduce = misc.all_reduce_mean(loss_distill_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_all', loss_all_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_distill', loss_distill_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.wandb:
                wandb.log({'loss_all_curve': loss_all_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_distill_curve': loss_distill_value_reduce}, step=epoch_1000x)
                wandb.log({'lr_curve': max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class FeatureHooker:
    """
    用于自动 Hook 指定类型的层并提取特征
    """
    def __init__(self, model, target_layer_type=Block):
        self.model = model
        self.features = {}  # 用于存储当前 batch 的特征 {layer_idx: feature}
        self.hooks = []
        self.layer_ids = [] # 记录被 Hook 的层的名字或索引
        
        # 1. 自动注册 Hook
        self._register_hooks(target_layer_type)

    def _register_hooks(self, target_layer_type):
        """遍历模型，找到所有 Block 并注册 hook"""
        layer_idx = 0
        for name, module in self.model.named_modules():
            # 使用 isinstance 判断是否是你指定的 Block
            if isinstance(module, target_layer_type):
                # 使用闭包保存 layer_idx
                hook = module.register_forward_hook(self._get_hook_fn(layer_idx))
                self.hooks.append(hook)
                self.layer_ids.append(name) # 方便 debug 打印层名
                layer_idx += 1
        
        print(f"Success: Hooked {len(self.hooks)} blocks of type {target_layer_type.__name__}")

    def _get_hook_fn(self, layer_idx):
        """生成 hook 函数"""
        def hook_fn(module, input, output):
            # 注意：如果 Block 返回的是 tuple (x, weights)，取 output[0]
            # 这里默认 Block 返回的是 Tensor，如果不是请根据情况修改
            if isinstance(output, tuple):
                self.features[layer_idx] = output[0]
            else:
                self.features[layer_idx] = output
        return hook_fn

    def get_features(self, selected_indices=None):
        """
        获取特征
        selected_indices: list, 例如 [3, 7, 11]，如果为 None 则返回所有
        """
        if selected_indices is None:
            # 返回所有层，按索引排序
            return [self.features[i] for i in range(len(self.features))]
        else:
            # 返回指定层
            return [self.features[i] for i in selected_indices if i in self.features]

    def clear(self):
        """清空特征（通常在每个 batch 开始前不需要，因为会被覆盖，但为了保险可以调）"""
        self.features = {}

    def remove_hooks(self):
        """移除所有 hooks，释放内存"""
        for hook in self.hooks:
            hook.remove()

def cosine_loss(s_feat, t_feat):
    # s_feat, t_feat shape: [B, N, C]
    # 在最后一个维度 (C) 上做归一化，然后算相似度
    
    # target=1 表示我们希望它们相似
    # PyTorch 的 CosineEmbeddingLoss 需要输入 target
    target = torch.ones(s_feat.shape[0], device=s_feat.device)
    
    # 如果是 (B, N, C)，建议先展平为 (B*N, C) 再算，或者手动写 normalize
    s_norm = torch.nn.functional.normalize(s_feat, dim=-1)
    t_norm = torch.nn.functional.normalize(t_feat, dim=-1)
    
    # Loss = 1 - cos_sim (因为我们希望相似度越高 loss 越小)
    loss = 1.0 - (s_norm * t_norm).sum(dim=-1).mean()
    return loss

def train_one_epoch_distill(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, aug=None, trival_aug=None,
                    args=None):
    model.train(True)
    # model_teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    # # --- Step 1: 初始化 Hooker ---
    # # 只需要在 epoch 开始前初始化一次
    # # 如果 model 是 DDP 包装过的，可能需要传入 model.module
    # real_model_s = model.module if hasattr(model, 'module') else model
    # real_model_t = model_teacher.module if hasattr(model_teacher, 'module') else model_teacher
    
    # # 分别 Hook 学生和老师
    # hooker_s = FeatureHooker(real_model_s, Block)
    # hooker_t = FeatureHooker(real_model_t, Block)

    # --- 设定你想蒸馏的层索引 ---
    # 假设 ViT Base 有 12 个 Block，我们想对最后几层或者均匀分布的层做蒸馏
    # 例如：选择索引为 [3, 7, 11] 的层
    # selected_layers = [3, 7, 11]

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # if data_iter_step > 500:
        #     break
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args.dataset == "cifar10dvs" or args.dataset == "dvs128":
            N = samples.shape[0]
            if aug != None:
                # image = image.flatten(1, 2).contiguous() # 合并T,C
                samples = torch.stack([(aug(samples[i])) for i in range(N)])
                # image = image.reshape(N,T,C,H,W)

            if trival_aug != None:
                # image = image.flatten(0,1).contiguous()
                samples = torch.stack([(trival_aug(samples[i])) for i in range(N)])
                # image = image.reshape(N,T,C,H,W).contiguous()
            # print("samples",samples.shape)
            if args.mode == "SNN" and args.dataset == "dvs128":
                pass
            else:
                samples = samples.sum(dim=1)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            targets_one_hot = targets
        else:
            targets_one_hot = torch.nn.functional.one_hot(targets, args.nb_classes)
        # print(model.module.blocks[0].norm1[0].weight)

        # 记得每次 forward 前清空（可选，因为 key 会被覆盖）
        # hooker_s.clear()
        # hooker_t.clear()

        with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16, enabled=True):
            if args.mode != "SNN":
                outputs = model(samples)
                with torch.no_grad():
                    outputs_teacher = model_teacher(samples)
                loss = criterion(outputs.float(), targets)
            else:
                outputs, counts, output_ts = model(samples, verbose=True)
                with torch.no_grad():
                    outputs_teacher = model_teacher(samples)
                # loss = criterion(output_ts[7].float(), targets) + (torch.abs(output_ts[8]) - torch.abs(output_ts[7])).sum()/output_ts[8:].numel()
                loss = criterion(outputs.float(), targets)

            # feats_s_list = hooker_s.get_features(selected_layers)
            # feats_t_list = hooker_t.get_features(selected_layers)
            # loss_distill = 0.0
            # for fs, ft in zip(feats_s_list, feats_t_list):
            #     # 简单的 MSE
            #     loss_distill += cosine_loss(fs, ft)            
            # loss_distill = loss_distill / len(selected_layers)
            
            loss_distill = get_logits_loss(outputs_teacher, outputs, targets_one_hot, args.temp, num_classes=args.nb_classes)
            # loss_distill = torch.tensor(0.0).to(loss.device)
            if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
                if epoch < 10:
                    overfire_loss = cal_l1_loss(model)
                elif epoch < 20:
                    overfire_loss = cal_l1_loss(model)
                elif epoch < 40:
                    overfire_loss = cal_l1_loss(model)
                else:
                    overfire_loss = cal_l1_loss(model)
            loss_all = loss + loss_distill + (overfire_loss if hasattr(args, "suppress_over_fire") and args.suppress_over_fire else 0.0)
        loss_value = loss.item()
        loss_distill_value = loss_distill.item()
        if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
            overfire_loss_value  = overfire_loss.item()
        loss_all_value = loss_all.item()

        if not math.isfinite(loss_all_value):
            print("Loss is {}, stopping training".format(loss_all_value))
            sys.exit(1)

        loss_all /= accum_iter
        loss_scaler(loss_all, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0,data_iter_step=data_iter_step)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
            metric_logger.update(loss_all=loss_all_value, loss=loss_value, loss_distill=loss_distill_value, overfire_loss = overfire_loss_value)
        else:
            metric_logger.update(loss_all=loss_all_value, loss=loss_value, loss_distill=loss_distill_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_all_value_reduce = misc.all_reduce_mean(loss_all_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_distill_value_reduce = misc.all_reduce_mean(loss_distill_value)
        if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
            overfire_loss_value_reduce = misc.all_reduce_mean(overfire_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_all', loss_all_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_distill', loss_distill_value_reduce, epoch_1000x)
            if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
                log_writer.add_scalar('loss_overfire', overfire_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.wandb:
                wandb.log({'loss_all_curve': loss_all_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_distill_curve': loss_distill_value_reduce}, step=epoch_1000x)
                if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
                    wandb.log({'loss_overfire_curve': overfire_loss_value_reduce}, step=epoch_1000x)
                wandb.log({'lr_curve': max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _compute_L1(logits, reduce_batch=True, eps=1e-8):
    """
    logits: [T, B, N]
    returns L1: [T, B] (per timestep per batch)
    """
    L1 = logits.abs().sum(dim=2)  # [T, B]
    return L1

def time_decay_loss(logits,
                    mode='hinge',          # 'hinge' | 'neg_diff' | 'target'
                    reduction='mean',      # 'mean' | 'sum' | 'none'
                    weight_schedule=None,  # None -> auto linear; else tensor of shape [T-1]
                    hinge_margin=0.0,      # used for hinge mode: allow small increase
                    target_schedule=None,  # used for 'target' mode: shape [T] or scalar ratio r
                    eps=1e-8):
    """
    Encourage L1 norm to decrease over time, with stronger pressure on earlier time steps.

    Args:
        logits: [T, B, N]
        mode:
            'hinge'   : loss = sum_t alpha_t * relu(L1_{t+1}-L1_t + margin)
                        (penalize non-decrease; encourages earlier decrease by alpha_t)
            'neg_diff': loss = sum_t alpha_t * (L1_{t+1} - L1_t)  (encourage larger positive diff)
                        (this can be negative; we minimize it -> maximize differences)
            'target'  : you provide target_schedule of length T (or scalar ratio r)
                        loss = sum_t alpha_t * relu(L1_t - target_t)
        weight_schedule:
            tensor-like of length T-1 giving weight for each adjacent pair.
            If None, default linear decreasing weights puts larger weight on earlier diffs:
                alpha_t = (T - t - 1) / sum_{k=0}^{T-2} (T - k - 1)
        hinge_margin:
            small positive slack; if L1_{t+1} <= L1_t - margin considered OK.
        target_schedule:
            If None, and mode=='target', you can pass r (0<r<1) as scalar and target_t = L1_0 * r^t
    Returns:
        scalar loss (or per-sample if reduction='none')
    """
    T, B, N = logits.shape
    assert T >= 2, "Need at least 2 timesteps"

    L1 = _compute_L1(logits)  # [T, B]

    # # construct alpha weights (length T-1)
    # if weight_schedule is None:
    #     # default: linear decreasing, larger weight on earlier diffs
    #     alphas = torch.tensor([float(T - t - 1) for t in range(T - 1)],
    #                           device=logits.device, dtype=logits.dtype)
    #     alphas = alphas / (alphas.sum() + eps)  # normalized
    # else:
    #     alphas = torch.as_tensor(weight_schedule, device=logits.device, dtype=logits.dtype)
    #     assert alphas.shape[0] == T - 1

    # # pairwise diffs: D_t = L1_t - L1_{t+1} -> want D_t large and positive
    # D = L1[:-1, :] - L1[1:, :]  # [T-1, B]  (positive if decreased)

    # if mode == 'hinge':
    #     # penalize when not decreased by margin: loss = relu(-D + margin) = relu(L1_{t+1} - L1_t + margin)
    #     # shape [T-1, B]
    #     per_pair_loss = F.relu(-D + hinge_margin)  # zeros when D >= margin
    #     # apply alphas per time (broadcast to batch)
    #     per_pair_weighted = alphas.unsqueeze(1) * per_pair_loss  # [T-1, B]
    #     per_sample = per_pair_weighted.sum(dim=0)  # [B]
    # elif mode == 'neg_diff':
    #     # minimize negative diff => encourage large positive D
    #     per_pair = (L1[1:, :] - L1[:-1, :])  # = -D
    #     per_pair_weighted = alphas.unsqueeze(1) * per_pair  # can be negative
    #     per_sample = per_pair_weighted.sum(dim=0)  # [B]
    #     # we want to minimize this => larger positive D reduces loss (good)
    # elif mode == 'target':
    #     # build target_schedule: shape [T]
    #     if target_schedule is None or (isinstance(target_schedule, (int,float))):
    #         # scalar ratio r given: target_t = L1_0 * r^t
    #         r = target_schedule if target_schedule is not None else 0.5
    #         L1_0 = L1[0:1, :]  # [1,B]
    #         # target shape [T,B]
    #         exps = torch.tensor([r ** t for t in range(T)], device=logits.device, dtype=logits.dtype).unsqueeze(1)
    #         target = (L1_0 * exps)  # [T,B]
    #     else:
    #         target = torch.as_tensor(target_schedule, device=logits.device, dtype=logits.dtype)
    #         if target.dim() == 1:
    #             target = target.unsqueeze(1).expand(T, B)
    #         assert target.shape == (T, B)
    #     # loss = sum_t alpha_{t} * relu(L1_t - target_t)  (we penalize being above target)
    #     # we can set alpha for t : map t->alpha_t (length T)
    #     alpha_full = torch.cat([alphas, alphas[-1:]])  # naive expand to T, or user-supplied
    #     alpha_full = torch.ones(T, device=logits.device, dtype=logits.dtype) / T  # simpler default
    #     per_t_loss = F.relu(L1 - target) * alpha_full.unsqueeze(1)
    #     per_sample = per_t_loss.sum(dim=0)
    # else:
    #     raise ValueError("mode must be in {'hinge','neg_diff','target'}")

    # reduction
    if reduction == 'mean':
        return L1.mean() * 0.1
    elif reduction == 'sum':
        return L1.mean() * 0.1
    else:
        return L1   # [B]



def train_one_epoch_distill_snn(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, aug=None, trival_aug=None,
                    args=None):
    model.train(True)
    if model_teacher is not None:
        model_teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    keep_keyword = "biasAllocator"
    for name, param in model.named_parameters():
        if keep_keyword not in name:
            param.requires_grad = False
        else:
            # parts = name.split(".")
            # try:
            #     block_id = int(parts[3])  # e.g. "12"
            # except (IndexError, ValueError, TypeError):
            #     block_id = 0  # 默认值
            
            # if block_id <= 7:
            if param.requires_grad:
                print(name) 
            # param.requires_grad = True
            # else:
            #     param.requires_grad = False

    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    
        # if data_iter_step > 2000:
        #     break
    
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args.dataset == "cifar10dvs" or args.dataset == "dvs128":
            N = samples.shape[0]
            if aug != None:
                # image = image.flatten(1, 2).contiguous() # 合并T,C
                samples = torch.stack([(aug(samples[i])) for i in range(N)])
                # image = image.reshape(N,T,C,H,W)

            if trival_aug != None:
                # image = image.flatten(0,1).contiguous()
                samples = torch.stack([(trival_aug(samples[i])) for i in range(N)])
                # image = image.reshape(N,T,C,H,W).contiguous()
            # print("samples",samples.shape)
            if args.mode == "SNN" and args.dataset == "dvs128":
                pass
            else:
                samples = samples.sum(dim=1)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            targets_one_hot = targets
        else:
            targets_one_hot = torch.nn.functional.one_hot(targets, args.nb_classes)
        # print(model.module.blocks[0].norm1[0].weight)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            if args.mode != "SNN":
                outputs = model(samples)
                # outputs_teacher = model_teacher(samples)
                loss = criterion(outputs.float(), targets)
            else:
                outputs, counts, output_ts, _ = model(samples, verbose=True)
                # print("training images.abs().mean()",samples.abs().mean(),outputs.abs().mean())
                # outputs_teacher = model_teacher(samples)
                # loss = criterion(output_ts[7].float(), targets) + (torch.abs(output_ts[8]) - torch.abs(output_ts[7])).sum()/output_ts[8:].numel()
                loss = criterion((outputs).float(), targets)
                loss_value = loss.item()
                # loss = loss*0.0
            # loss_distill = get_logits_loss(outputs_teacher, outputs, targets_one_hot, args.temp, num_classes=args.nb_classes)
            loss_distill = torch.tensor(0.0).to(loss.device)
            # loss_distill = time_decay_loss(output_ts, mode='hinge', reduction='mean',hinge_margin=0.0)
            if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
                overfire_loss = cal_overfire_loss(model)
            loss_all = loss + loss_distill + (overfire_loss if hasattr(args, "suppress_over_fire") and args.suppress_over_fire else 0.0)
            # loss_value = loss.item()
            loss_distill_value = loss_distill.item()
            if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
                overfire_loss_value  = overfire_loss.item()
            loss_all_value = loss_all.item()

        if not math.isfinite(loss_all_value):
            print("Loss is {}, stopping training".format(loss_all_value))
            sys.exit(1)

        loss_all /= accum_iter
        loss_scaler(loss_all, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0,data_iter_step=data_iter_step)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
            metric_logger.update(loss_all=loss_all_value, loss=loss_value, loss_distill=loss_distill_value, overfire_loss = overfire_loss_value)
        else:
            metric_logger.update(loss_all=loss_all_value, loss=loss_value, loss_distill=loss_distill_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_all_value_reduce = misc.all_reduce_mean(loss_all_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_distill_value_reduce = misc.all_reduce_mean(loss_distill_value)
        if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
            overfire_loss_value_reduce = misc.all_reduce_mean(overfire_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_all', loss_all_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_distill', loss_distill_value_reduce, epoch_1000x)
            if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
                log_writer.add_scalar('loss_overfire', overfire_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.wandb:
                wandb.log({'loss_all_curve': loss_all_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_distill_curve': loss_distill_value_reduce}, step=epoch_1000x)
                if hasattr(args, "suppress_over_fire") and args.suppress_over_fire:
                    wandb.log({'loss_overfire_curve': overfire_loss_value_reduce}, step=epoch_1000x)
                wandb.log({'lr_curve': max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_distill_mse(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    model_teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # print(model.module.blocks[0].norm1[0].weight)
        with torch.cuda.amp.autocast():
            if args.mode != "SNN":
                outputs = model(samples)
            else:
                outputs, counts = model(samples, verbose=False)
            outputs_teacher = model_teacher(samples)
            loss = criterion(outputs, targets).detach()
            # loss_distill = get_logits_loss(outputs_teacher, outputs, targets, args.temp)
            loss_distill = torch.nn.functional.mse_loss(outputs, outputs_teacher)
            loss_all = loss + loss_distill
        loss_value = loss.item()
        loss_distill_value = loss_distill.item()
        loss_all_value = loss_all.item()

        if not math.isfinite(loss_all_value):
            print("Loss is {}, stopping training".format(loss_all_value))
            sys.exit(1)

        loss_all /= accum_iter
        loss_scaler(loss_all, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_all=loss_all_value, loss=loss_value, loss_distill=loss_distill_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_all_value_reduce = misc.all_reduce_mean(loss_all_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_distill_value_reduce = misc.all_reduce_mean(loss_distill_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_all', loss_all_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_distill', loss_distill_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.wandb:
                wandb.log({'loss_all_curve': loss_all_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_distill_curve': loss_distill_value_reduce}, step=epoch_1000x)
                wandb.log({'lr_curve': max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def Align_QANN_SNN(model: torch.nn.Module, QANN_model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer:Optional[SummaryWriter] = None,
                    args=None):
    torch.set_printoptions(precision=6)
    model_layer_out = None    
    ideal_snn_out = None   
    name1 = ""
    name2 = ""
    def qann_layer_hook(module, inp, out):
        nonlocal ideal_snn_out, model, name2, log_writer
        s_scale = module.s.type(out.dtype)
        qann_model_layer_out = out+0.0
        ideal_snn_out = []
        # if log_writer is not None:
        #     log_writer.add_histogram(tag=f"{name1}/quan_sum", values=(qann_model_layer_out/s_scale).detach().cpu(), global_step=0)
        for i in range(model.module.T):
            out1 = qann_model_layer_out*0.0
            out1[qann_model_layer_out>=s_scale-1e-3] = s_scale
            out1[qann_model_layer_out<=-s_scale+1e-3] = -s_scale
            qann_model_layer_out = qann_model_layer_out - out1
            # if log_writer is not None:
            #     log_writer.add_histogram(tag=f"{name1}/quan", values=(out1/s_scale).detach().cpu(), global_step=i)
            ideal_snn_out.append(out1)

        ideal_snn_out = torch.stack(ideal_snn_out,dim=0).detach()

    def snn_layer_hook(module, inp, out):
        nonlocal model_layer_out, name1, log_writer
        T = model.module.T
        model_layer_out = out.reshape(torch.Size([T,out.shape[0]//T]) + out.shape[1:])
        # for i in range(T):
        #     if log_writer is not None:
        #         log_writer.add_histogram(tag=f"{name1}/ST_BIF", values=(model_layer_out[i]/module.q_threshold).detach().cpu(), global_step=i)
        # if log_writer is not None:
        #     log_writer.add_histogram(tag=f"{name1}/ST_BIF_sum", values=((model_layer_out/module.q_threshold).sum(dim=0)).detach().cpu(), global_step=0)
        # model_layer_out = out.sum(dim=0)
    
    model.train(True)
    QANN_model.eval()
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    index1 = -1
    index2 = -1
    
    # for param in model.parameters():
    #     param.requires_grad = False
    
    while(1):
                
        for i, (name1, module1) in enumerate(list(model.named_modules())):
            if isinstance(module1, ST_BIFNeuron_MS) and i > index1:
                index1 = i
                break

        for i, (name2, module2) in enumerate(list(QANN_model.named_modules())):
            if isinstance(module2, MyQuan) and i > index2:
                index2 = i
                break
        
        if name1.count("attn_IF") > 0:
            continue

        print(name1,name2)
        if i == len(list(QANN_model.named_modules())) - 1:
            break
        
        h1 = module1.register_forward_hook(snn_layer_hook)
        h2 = module2.register_forward_hook(qann_layer_hook)

        # if epoch < 20:
        #     print(f"skip layer {name1}!!!")
        #     epoch = epoch + 1
        #     continue
        
        # module1.time_allocator.requires_grad = True
        # for param in module1.parameters():
        #     param.requires_grad = True
        
        print(f"aligning {name1} layer.....")

        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        epoch = epoch + 1
        for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):            
            
            
            # if name1.count("after_attn_IF") > 0 or name1.count("proj_IF") > 0 or name1.count("norm2") > 0:
            #     if data_iter_step > 1000:
            #         break
            # else:
            if data_iter_step > 200:
                break
            # if data_iter_step % accum_iter == 0:
            #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            # print(model.module.blocks[0].norm1[0].weight)
            with torch.cuda.amp.autocast():
                # print("=========================SNN==========================")
                outputs_snn, counts = model(samples, verbose=False)
                # print("=========================QANN==========================")
                outputs_qann = QANN_model(samples)
                # print(model_layer_out.shape, ideal_snn_out.shape)
                loss = torch.nn.functional.l1_loss(model_layer_out, ideal_snn_out,reduction="mean") + \
                        torch.nn.functional.l1_loss(outputs_snn, outputs_qann,reduction="mean")*0.0

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # if loss_value < 1e-4:
            #     pass
            # else:
            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)

            # print("module1.time_allocator.grad",module1.time_allocator.grad)
            
            optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', max_lr, epoch_1000x)
                if args.wandb:
                    wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                    wandb.log({'lr_curve': max_lr}, step=epoch_1000x)

            if loss_value_reduce < 1e-4:
                print(f"loss is less than 1e-4 during training, {name1} layer alignment finish!!!!")
                break

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()                
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if stats['loss'] < 1e-3:
            print(f"loss is less than 1e-3 after training, {name1} layer alignment finish!!!!")
            #     break
        h1.remove()
        h2.remove()    

        # for param in module1.parameters():
        #     param.requires_grad = False
        # module1.time_allocator.requires_grad = False

    # for param in model.parameters():
    #     param.requires_grad = True


@torch.no_grad()
def evaluate(data_loader, model, device, snn_aug, mode, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # open_dropout(model)
    # if args.mode == "SNN":
    #     model.max_T = 0
    total_num = 0
    correct_per_timestep = None
    
    max_T = 0
    count1 = 0

    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if args.dataset == "cifar10dvs" or args.dataset == "dvs128":
            N = images.shape[0]
            if snn_aug != None:
                # image = image.flatten(1, 2).contiguous() # 合并T,C
                images = torch.stack([(snn_aug(images[i])) for i in range(N)])
            if args.mode == "SNN" and args.dataset == "dvs128":
                pass
            else:
                images = images.sum(dim=1)
        # compute output
        # with torch.cuda.amp.autocast():
        if mode != "SNN":
            output = model(images)
        else:
            # accu_per_timestep: cur_T * B * n_classes
            output, count, _ ,accu_per_timestep = model.module(images, verbose=True)
            # print("evaluate images.abs().mean()",images.abs().mean(),output.abs().mean())
            # print(accu_per_timestep.shape, count)
            overfire_loss = cal_overfire_loss(model) * 0.1
            print(overfire_loss)
            
            max_T = max(max_T, count)
            # print(max_T)
            if accu_per_timestep.shape[0] < max_T:
                padding_per_timestep = accu_per_timestep[-1].unsqueeze(0)
                padding_length = max_T - accu_per_timestep.shape[0]
                accu_per_timestep = torch.cat(
                    [accu_per_timestep, padding_per_timestep.repeat(padding_length, 1, 1)], dim=0)

            if correct_per_timestep is not None and correct_per_timestep.shape[0] < max_T:
                for t in range(correct_per_timestep.shape[0], max_T):
                    metric_logger.meters['acc@{}'.format(t + 1)] = deepcopy(metric_logger.meters['acc@{}'.format(correct_per_timestep.shape[0])])

            _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
            correct_per_timestep = torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)

            # if correct_per_timestep is None:
            #     _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
            #     correct_per_timestep = torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)
            # else:
            #     _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
            #     # print(correct_per_timestep.shape, predicted_per_time_step.shape, target.unsqueeze(0).shape)
            #     correct_per_timestep = torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)
        # output = model(images)
        loss = criterion(output, target)

        total_num += images.shape[0]

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(output.argmax(-1).reshape(-1))
        # print(target)
        # print("acc1, acc5",acc1, acc5)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if mode == "SNN":
            metric_logger.meters['AvgTime'].update(float(count), n=batch_size)
        if mode == "SNN":
            for t in range(max_T):
                metric_logger.meters['acc@{}'.format(t + 1)].update(
                    correct_per_timestep[t].cpu().item() * 100. / batch_size, n=batch_size)
            model.module.reset()

        # break

        # count1 += 1
        # if count1 >= 3:
        #     break
    # gather the stats from all processes
    # accuracy_per_timestep = correct_per_timestep.float().cpu().data / float(total_num)
    print("Evaluation End")
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
