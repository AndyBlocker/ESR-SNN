import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
import numpy as np
import scipy
import glo
from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA
from neuron_cupy.st_bif_ss_cuda import ST_BIFNodeATGF_SS_CUDA
from neuron_cupy.cuda_operator_LIF import LIFNodeATGF_MS_CUDA
from timm.models.layers.helpers import to_2tuple
from typing import Optional
from timm.models.layers import trunc_normal_
from torch import Tensor
from einops import rearrange
from timm.models.layers.trace_utils import _assert

# torch.set_default_dtype(torch.double)
# torch.set_default_tensor_type(torch.DoubleTensor)

class ORIIFNeuron(nn.Module):
    def __init__(self,q_threshold,level,sym=False):
        super(ORIIFNeuron,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.level = torch.tensor(level)
        self.sym = sym
        self.pos_max = torch.tensor(level - 1)
        self.neg_min = torch.tensor(0)
            
        self.eps = 0

    # def __repr__(self):
    #         return f"IFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.spike_position = None
        # self.neg_spike_position = None

    def forward(self,input):
        x = input/self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape,dtype=torch.float32).to(x.device)
            self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.edvice) + 0.5

        self.is_work = True
        
        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0)
        # neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        # self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        # self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        # print((x == 0).all(), (self.cur_output==0).all())
        if (x == 0).all() and (self.cur_output==0).all():
            self.is_work = False
        
        # print("self.cur_output",self.cur_output)
        
        return self.cur_output*self.q_threshold


class IFNeuron(nn.Module):
    def __init__(self,q_threshold,level,sym=False):
        super(IFNeuron,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.pos_max = torch.tensor(level//2 - 1)
            self.neg_min = torch.tensor(-level//2)
        else:
            self.pos_max = torch.tensor(level//2 - 1)
            self.neg_min = torch.tensor(0)
            
        self.eps = 0

    def __repr__(self):
            return f"ST-BIFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.spike_position = None
        self.neg_spike_position = None

    def forward(self,input):
        x = input/self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x*self.q_threshold
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape,dtype=torch.float32).to(x.device)
            self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + 0.5

        self.is_work = True
        
        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0) & (self.acc_q < self.pos_max)
        neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        # print((x == 0).all(), (self.cur_output==0).all())
        if (x == 0).all() and (self.cur_output==0).all():
            self.is_work = False
        
        # print("self.cur_output",self.cur_output)
        
        return self.cur_output*self.q_threshold

def theta_backward(x):
    sigmoid = torch.sigmoid(4*x)
    return 4*sigmoid*(1-sigmoid)
    # tanh = F.tanh(2*x)
    # return 1/(1+(2*x)*(2*x))
    # return 1 - F.tanh(2*x)*F.tanh(2*x)

def theta(x):
    # return (x > 0).int()
    return 1.0*(torch.gt(x,0))
 
def theta_eq(x):
    # return (x >= 0).int()
    return 1.0*(torch.ge(x,0))


class ST_BIFNodeATGF_SS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_t: torch.Tensor, V_t_1: torch.Tensor, T_t_1: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, t: torch.Tensor):

        spike = x_t * 0.0
        H_t = V_t_1 + x_t
        
        # Group_size = 4
        # if H_t.shape[-1] == 3072:
        #     B,H,C = H_t.shape
        #     H_t_group = H_t.reshape(B,H,C//Group_size,Group_size).abs()
        #     H_t_group = H_t_group.mean(-1)
        #     mid_value = torch.quantile(H_t_group,0.5,interpolation='nearest')
        #     mask = (H_t_group >= mid_value).unsqueeze(-1)
        #     spike_condition = ((H_t >= v_th) & (T_t_1-T_max < 0)).reshape(B,H,C//Group_size,Group_size) & mask
        #     neg_spike_condition = ((H_t < 0) & (T_t_1-T_min > 0)).reshape(B,H,C//Group_size,Group_size) & mask
        #     spike_condition = spike_condition.reshape(B,H,C)
        #     neg_spike_condition = neg_spike_condition.reshape(B,H,C)
        # else:
        spike_condition = (H_t >= v_th) & (T_t_1-T_max < 0)
        neg_spike_condition = (H_t < 0) & (T_t_1-T_min > 0) 
        
        # spike[torch.logical_and((torch.ge(H_t-v_th,0)), (torch.lt(T_t_1-T_max,0)))] = 1
        # spike[torch.logical_and((torch.lt(H_t,0)), (torch.gt(T_t_1-T_min,0)))] = -1


        spike = torch.where(spike_condition, torch.ones_like(H_t),
                                      torch.where(neg_spike_condition, -torch.ones_like(H_t),
                                                  torch.zeros_like(H_t)))

        V_t = H_t - v_th * spike
        T_t = T_t_1 + spike

        ctx.save_for_backward(T_t_1,H_t,v_th,T_max,T_min,t)
        
        return spike, V_t, T_t

    @staticmethod
    def backward(ctx, grad_spike_t: torch.Tensor, grad_v_t: torch.Tensor, grad_T_t: torch.Tensor):

        T_t_1,H_t,v_th,T_max,T_min,t = ctx.saved_tensors
        
        grad_T_t_to_H_t = (theta_backward(H_t - v_th)*theta(T_max - T_t_1)+theta_backward(-H_t)*theta(T_t_1 - T_min))
        grad_Y_t_to_T_t_1 = -(theta_eq(H_t-v_th)*theta_backward(T_max - T_t_1)+theta(-H_t)*theta_backward(T_t_1 - T_min))
        
        tmp = grad_spike_t - v_th*grad_v_t + grad_T_t
        grad_X_t = tmp*grad_T_t_to_H_t + grad_v_t
        grad_T_t_1 = tmp*grad_Y_t_to_T_t_1 + grad_T_t
        grad_V_t_1 = grad_X_t + 0.0
        # print("t:",t,"grad_V_t_1",grad_X_t.mean().item(),"grad_T_t_1",grad_T_t_1.mean().item(),"grad_v_t",grad_v_t.mean().item(),"grad_T_t",grad_T_t.mean().item())
        return grad_X_t, grad_V_t_1, grad_T_t_1, None, None, None, None

class ST_BIFNeuron_SS(nn.Module):
    def __init__(self,q_threshold,level,sym=False):
        super(ST_BIFNeuron_SS,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = nn.Parameter(torch.tensor(q_threshold),requires_grad=False)
        self.level = torch.tensor(level)
        self.T = self.level//2 - 1
        self.sym = sym
        if sym:
            self.register_buffer("pos_max",torch.tensor(level//2 - 1))
            self.register_buffer("neg_min",torch.tensor(-level//2 + 1))
            # self.pos_max = torch.tensor(level//2 - 1)
            # self.neg_min = torch.tensor(-level//2)
        else:
            self.register_buffer("pos_max",torch.tensor(level - 1))
            self.register_buffer("neg_min",torch.tensor(0))
            # self.pos_max = torch.tensor(level - 1)
            # self.neg_min = torch.tensor(0)
        self.init = True
        # self.steps = max(self.pos_max,torch.abs(self.neg_min))
        self.eps = 0
        self.t = 0
        self.init_state = 0
        self.init_batch = 20

    # def __repr__(self):
    #         return f"ST_BIFNeuron_SS(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.acc_q = 0.0
        self.t = 0
        self.init_state = 0

    def forward(self,input):
        # print("input.mean()",input.abs().mean().item(),"self.q_threshold",self.q_threshold.data.item())
        
        # s_grad_scale = 1.0 / (((input).detach().abs().mean() * input.numel()) ** 0.5)
        # if self.init:
        #     if self.init_state == 0:
        #         self.q_threshold.data = (((input).detach().abs().mean() * 2) / (self.pos_max.detach().abs() ** 0.5))
        #         self.init_state += 1
        #     elif self.init_state < self.init_batch:
        #         self.q_threshold.data = 0.1*self.q_threshold.data + 0.9*(((input).detach().abs().mean() * 2) / (self.pos_max.detach().abs() ** 0.5))
        #         self.init_state += 1
        #     else:
        #         self.init = False

        if not torch.is_tensor(self.q) and not torch.is_tensor(self.acc_q):
            self.q = input * 0.0 + 0.5*self.q_threshold
            # self.q = input * 0.0
            self.acc_q = input * 0.0

        # if self.steps > 0:
        #     self.q = self.q + 0.5*self.q_threshold/self.steps
        #     self.steps = self.steps - 1
        
        # s_scale = grad_scale(self.q_threshold, s_grad_scale)
        self.t = self.t + 1
        spikes, self.q, self.acc_q = ST_BIFNodeATGF_SS_CUDA.apply(input, self.q, self.acc_q, self.q_threshold, self.pos_max, self.neg_min, torch.tensor(self.t))



        return spikes*self.q_threshold



class ST_BIFNodeATGF_MS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, prefire: torch.Tensor):

        Time = x_seq.shape[0]
        device = x_seq.device
        total_neurons = x_seq[0].numel()  # B * N * C
        max_spikes = max(1, int(total_neurons * 0.4))  # 至少允许 1 个
        
        v_seq = []
        T_seq = []
        H_seq = []
        spike_seq = []
        v = x_seq[0]*0 + 0.5*v_th
        T = x_seq[0]*0
        spike = x_seq[0]*0
        T_seq.append(T)
        spike_seq.append(spike)
        H_seq.append(v)
        
        for t in range(Time):
            spike = spike * 0.0
            v = v + x_seq[t]
            H_seq.append(v)
            spike[torch.logical_and((torch.ge(v-v_th,0)), (torch.lt(T-T_max,0)))] = 1
            spike[torch.logical_and((torch.lt(v,0)), (torch.gt(T-T_min,0)))] = -1

            # ---------------- Gate机制：全局最多10%神经元发放 ----------------
            spike_mask = spike != 0
            num_spike = spike_mask.sum().item()
            if num_spike > max_spikes:
                # 扁平化
                v_flat = v.reshape(-1)            # (total_neurons,)
                mask_flat = spike_mask.reshape(-1)  # bool

                # 把非脉冲位置的 v 置为 -inf，保证 topk 只从脉冲位置中选
                # 使用 masked_fill 更安全（保持 dtype/device）
                v_masked = v_flat.clone()
                v_masked = v_masked.masked_fill(~mask_flat, float('-inf'))

                # 选出 top-k 的扁平索引
                topk = torch.topk(v_masked, k=max_spikes, largest=True)
                keep_idx_flat = topk.indices  # 长度 = max_spikes

                # 构造保留 mask（扁平）
                keep_mask_flat = torch.zeros_like(mask_flat, dtype=torch.bool, device=device)
                keep_mask_flat[keep_idx_flat] = True

                # reshape 回原始形状并应用到 spike（保留符号）
                keep_mask = keep_mask_flat.view_as(spike_mask)
                spike = spike * keep_mask
            # ---------------------------------------------------------------

            if t < T_max:
                v = v - v_th * spike
            else:
                v = v - v_th * spike
            T = T + spike
            T_seq.append(T)
            spike_seq.append(spike)

        H_seq = torch.stack(H_seq,dim=0)
        T_seq = torch.stack(T_seq,dim=0)
        spike_seq = torch.stack(spike_seq,dim=0)
        
        ctx.save_for_backward(spike_seq,T_seq,H_seq,v_th,T_max,T_min)
        
        return spike_seq[1:,], v, T_seq[1:,]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_T_seq: torch.Tensor):

        spike_seq, T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        Time = spike_seq.shape[0] - 1
        grad_x_seq = []

        grad_Y_seq = grad_spike_seq
        
        grad_V = 0.0 # t
        grad_T = 0.0 # t
        for t in range(Time, 0, -1):
            grad_T_t_to_H_t = (theta_backward(H_seq[t] - v_th)*theta(T_max - T_seq[t-1])+theta_backward(-H_seq[t])*theta(T_seq[t-1] - T_min))
            grad_Y_t_to_T_t_1 = -(theta_eq(H_seq[t]-v_th)*theta_backward(T_max - T_seq[t-1])+theta(-H_seq[t])*theta_backward(T_seq[t-1] - T_min))
            
            grad_X = (grad_Y_seq[t-1] - v_th*grad_V + grad_T)*grad_T_t_to_H_t + grad_V
            grad_T = (grad_Y_seq[t-1] - v_th*grad_V + grad_T)*grad_Y_t_to_T_t_1 + grad_T
            grad_V = grad_X + 0.0
            grad_x_seq.append(grad_X)
        
        grad_x_seq = torch.flip(torch.stack(grad_x_seq,dim=0),dims=[0])
        return grad_x_seq, None, None, None


class ST_BIFNeuron_MS(nn.Module):
    def __init__(self,q_threshold,level,sym=False, first_neuron=False, need_spike_tracer=False, T=8, C=768):
        super(ST_BIFNeuron_MS,self).__init__()
        # self.q = 0.0
        self.need_spike_tracer = need_spike_tracer
        if self.need_spike_tracer:
            self.acc_q = 0.0
        self.T = T
        self.step = level//2 - 1
        self.first_neuron = first_neuron
        self.suppress_over_fire = True
        self.overfireLoss = 0.0
        self.name = ""

        self.dim = 197
        print("self.T",self.T, "C",C)
        # self.time_allocator = nn.Parameter(torch.ones(self.T - 1, 1, 1, 1),requires_grad=True)
        self.bias_channel = nn.Parameter(torch.zeros(C), requires_grad=False).to(torch.float32)

        self.param_number = 2
        init_list = []
        for i in range(self.param_number - 1):
            if i < self.step - 1:
                init_list.append(1/(self.step))
            else:
                init_list.append(0.0)

        if self.T > 1:
            self.biasAllocator = nn.Parameter(torch.tensor(init_list),requires_grad=True)
        else:
            self.biasAllocator = 1.0


        self.q_threshold = nn.Parameter(torch.tensor(q_threshold),requires_grad=False)
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.register_buffer("pos_max",torch.tensor(level//2 - 1))
            self.register_buffer("neg_min",torch.tensor(-level//2 - 1))
            # self.pos_max = torch.tensor(level//2 - 1)
            # self.neg_min = torch.tensor(-level//2)
        else:
            self.register_buffer("pos_max",torch.tensor(level - 1))
            self.register_buffer("neg_min",torch.tensor(0))
            # self.pos_max = torch.tensor(level - 1)
            # self.neg_min = torch.tensor(0)
        self.register_buffer("prefire",torch.tensor(0.0))
        self.init = True
        self.eps = 0

    def __repr__(self):
        return f"ST_BIFNeuron_MS(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        # self.q = 0.0
        if self.need_spike_tracer:
            self.acc_q = 0.0

    def forward(self,input):
        N = input.shape[0]
        ori_shape = input.shape
        # print(self.q_threshold.data.item())

        input = input.reshape(torch.Size([int((self.T)),N//int((self.T))]) + input.shape[1:])
        # # print("ST_BIFNeuron_MS input.sum(dim=0).abs().mean()",input.sum(dim=0).abs().mean(),input.dtype)
        biasAllocator = torch.cat([1 - torch.sum(self.biasAllocator,dim=0,keepdim=True), self.biasAllocator], dim=0)

        if len(input.shape) == 4:
            bias_term = biasAllocator.view(-1, 1, 1, 1) * self.bias_channel.view(1, 1, 1, -1)
            input = torch.cat([
                input[:self.param_number] + bias_term,
                input[self.param_number:]
            ], dim=0)
        elif len(input.shape) == 5:
            if input.shape[-1] != input.shape[-2]:
                T1,B1,Head1,N1,C1 = input.shape
                bias_term = biasAllocator.view(-1, 1, 1, 1) * self.bias_channel.view(1, 1, 1, -1)
                input = input.transpose(2,3).reshape(T1,B1,N1,C1*Head1)
                input = torch.cat([
                    input[:self.param_number] + bias_term,
                    input[self.param_number:]
                ], dim=0)
                input = input.reshape(T1,B1,N1,Head1,C1).transpose(2,3)

        spike_seq, v_seq, T_seq = ST_BIFNodeATGF_MS_CUDA.apply(input.flatten(2), self.q_threshold, self.pos_max, self.neg_min, self.prefire)
        # self.q = v
        # print(self.q[self.q>0].mean())
        if self.need_spike_tracer:
            self.acc_q = T_seq.reshape(ori_shape)
        # print("ST_BIFNeuron_MS output.sum(dim=0).abs().mean()",(spike_seq*self.q_threshold).sum(dim=0).abs().mean(),spike_seq.dtype)
        spike_seq = spike_seq.reshape(ori_shape)
        if len(input.shape) == 4 and self.suppress_over_fire:
            spike_seq_1 = spike_seq.reshape(torch.Size([int((self.T)),N//int((self.T))]) + spike_seq.shape[1:])
            # print(spike_seq_1.shape)
            # self.overfireLoss = (spike_seq_1.abs().sum(dim=0) - spike_seq_1.sum(dim=0).abs()).sum() * 1e-5
            # if len(spike_seq.shape) == 3:
            #     # print(spike_seq.shape)
            #     channel_dist = (spike_seq).abs().mean(dim=1)
            #     mask = channel_dist < 1.0 * channel_dist.mean(dim=1,keepdim=True)
            #     # print(channel_dist.max(), channel_dist.min(), channel_dist.mean())
            #     self.overfireLoss = (channel_dist * mask).sum() * 2e-5
            self.overfireLoss = spike_seq.abs().sum() / spike_seq.numel()
            # self.overfireLoss = spike_seq[6:].abs().sum() * 1e-6

        return spike_seq*self.q_threshold


class spiking_BatchNorm2d_MS(nn.Module):
    def __init__(self,bn:nn.BatchNorm1d,level,input_allcate=False):
        super(spiking_BatchNorm2d_MS, self).__init__()
        self.level = level
        self.bn = bn
        self.bn.running_mean.requires_grad = False
        self.bn.running_var.requires_grad = False

    def forward(self, input):
        P = input.shape[0]
        # x = input.reshape(torch.Size([int((self.level)),P//int((self.level))]) + input.shape[1:]).detach()
        # self.bn(x.sum(dim=0))

        # print("before BatchNorm",input.reshape(4,32,384,197).sum(dim=0).abs().mean())
        input_shape = len(input.shape)
        if input_shape == 4:
            B,H,N,C = input.shape
            input = input.reshape(B*H,N,C)
        if input_shape == 2:
            input = input.unsqueeze(1)
        input = input.transpose(-1,-2)
        # output = self.bn(input)
        output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()/self.level,running_var=self.bn.running_var.detach(),weight=self.bn.weight,bias=self.bn.bias/self.level,eps=self.bn.eps)
        output = output.transpose(-1,-2)
        # print(input.mean(),output.mean())
        if input_shape == 2:
            output = output.squeeze(1)
        if input_shape == 4:
            output = output.reshape(B,H,N,C)
        
        # print("after BatchNorm",output.reshape(4,32,384,197).detach().sum(dim=0).abs().mean())
        return output
        
        
class spiking_BatchNorm2d(nn.Module):
    def __init__(self,bn:nn.BatchNorm2d,level,input_allcate):
        super(spiking_BatchNorm2d, self).__init__()
        self.level = level
        self.fire_time = 0
        self.bn = bn
        self.bn.momentum = None
        self.input_allcate = input_allcate
        if self.input_allcate:
            self.input_allocator = torch.nn.Parameter(torch.ones(torch.Size([self.level-1, self.bn.running_mean.shape[0]]))/self.level,requires_grad=True)
        self.eps = bn.eps
        self.accu_input = 0.0
    
    def reset(self):
        # print("spiking_BatchNorm2d reset")
        self.fire_time = 0
        self.accu_input = 0.0
    
    def forward(self, input):
        self.bn.eval()
        self.accu_input = self.accu_input + input
        # print("self.bn.running_mean",self.bn.running_mean.mean())
        # print("self.fire_time",self.fire_time,"self.level",self.level,"self.bn.running_mean",self.bn.running_mean.mean().item(),"self.bn.running_var",self.bn.running_var.mean().item(),"self.bn.weight",self.bn.weight.mean().item(),"self.bn.bias",self.bn.bias.mean().item())
        if self.fire_time < self.level - 1:
            # self.bn.running_mean.data = self.bn.running_mean*self.input_allocator[self.fire_time].detach()
            # self.bn.bias.data = self.bn.bias*self.input_allocator[self.fire_time]
            input_shape = len(input.shape)
            if input_shape == 4:
                B,H,N,C = input.shape
                input = input.reshape(B*H,N,C)
            if input_shape == 2:
                input = input.unsqueeze(1)
            if self.input_allcate:
                input = input.transpose(-1,-2)
                output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()*self.input_allocator[self.fire_time].detach(),running_var=self.bn.running_var,weight=self.bn.weight,bias=self.bn.bias*self.input_allocator[self.fire_time],eps=self.eps)
                output = output.transpose(-1,-2)
            else:
                input = input.transpose(-1,-2)
                output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()/self.level,running_var=self.bn.running_var,weight=self.bn.weight,bias=self.bn.bias/self.level,eps=self.eps)
                output = output.transpose(-1,-2)
            if input_shape == 2:
                output = output.squeeze(1)
            if input_shape == 4:
                output = output.reshape(B,H,N,C)
            self.fire_time = self.fire_time + 1
            return output
        elif self.fire_time == self.level - 1:
            input_shape = len(input.shape)
            if input_shape == 4:
                B,H,N,C = input.shape
                input = input.reshape(B*H,N,C)
            if input_shape == 2:
                input = input.unsqueeze(1)
            if self.input_allcate:
                input = input.transpose(-1,-2)
                allocate = 1.0 - torch.sum(self.input_allocator,dim=0)
                output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()*allocate.detach(),running_var=self.bn.running_var,weight=self.bn.weight,bias=self.bn.bias*allocate,eps=self.eps)
                output = output.transpose(-1,-2)
            else:
                input = input.transpose(-1,-2)
                output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()/self.level,running_var=self.bn.running_var,weight=self.bn.weight,bias=self.bn.bias/self.level,eps=self.eps)
                output = output.transpose(-1,-2)
            self.fire_time = self.fire_time + 1
            if input_shape == 2:
                output = output.squeeze(1)
            if self.training:
                self.bn.train()
                accu_input = self.accu_input.transpose(-1,-2)
                if input_shape == 4:
                    B,H,N,C = self.accu_input.shape
                    accu_input = self.accu_input.reshape(B*H,N,C).transpose(-1,-2)
                if input_shape == 2:
                    accu_input = self.accu_input.unsqueeze(0).transpose(-1,-2)
                self.bn(accu_input)
            if input_shape == 4:
                output = output.reshape(B,H,N,C)
            return output
        else:
            input_shape = len(input.shape)
            if input_shape == 4:
                B,H,N,C = input.shape
                input = input.reshape(B*H,N,C)
            if input_shape == 2:
                input = input.unsqueeze(1)
            input = input.transpose(-1,-2)
            temp1 = self.bn.running_mean.clone()
            temp2 = self.bn.bias.clone()
            self.bn.running_mean.data = self.bn.running_mean*0.0
            self.bn.bias.data = self.bn.bias*0.0
            output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()/self.level,running_var=self.bn.running_var,weight=self.bn.weight,bias=self.bn.bias/self.level,eps=self.eps)
            self.bn.running_mean.data = temp1
            self.bn.bias.data = temp2
            output = output.transpose(-1,-2)
            if input_shape == 2:
                output = output.squeeze(1)
            if input_shape == 4:
                output = output.reshape(B,H,N,C)
            return output


class Release_attn(nn.Module):
    def __init__(self,step):
        super(Release_attn, self).__init__()
        self.X = 0.0
        self.Y_pre = None
        self.step = step
        self.t = 0
    
    def reset(self):
        # print("Spiking_LayerNorm reset")
        self.X = 0.0
        self.Y_pre = None
        self.t = 0
        
    def forward(self,input):
        self.t = self.t + 1
        self.X = input + (self.X.detach() if torch.is_tensor(self.X) else self.X)
        if self.t <= self.step:
            Y = self.X * (self.step/self.t)
        else:
            Y = self.X
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

# class Spiking_LayerNorm(nn.Module):
#     def __init__(self,dim,T,step, eps=1e-5, elementwise_affine=True):
#         super(Spiking_LayerNorm, self).__init__()
#         # self.layernorm = nn.LayerNorm(dim)
#         self.X = 0.0
#         self.Y_pre = None
#         self.eps = eps
#         self.T = T
#         self.step = step
#         self.t = 0 
#         self.last_mean = 0.0
#         self.last_var = 0.0
#         self.elementwise_affine = elementwise_affine
#         if self.elementwise_affine:
#             self.weight = nn.Parameter(torch.ones(dim))
#             self.bias = nn.Parameter(torch.zeros(dim))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#     def reset(self):
#         # print("Spiking_LayerNorm reset")
#         self.X = 0.0
#         self.Y_pre = None
#         self.t = 0
        
#     def forward(self,input):
#         output = []
#         input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
#         self.X = 0.0 
#         for t in range(self.T):
#             self.X = input[t] + self.X
#             if torch.is_tensor(self.last_mean):
#                 this_mean = self.X.mean(dim=-1, keepdim=True) * (t/(self.T-1)) + self.last_mean * ((self.T-1-t)/(self.T-1))
#                 this_var = self.X.var(dim=-1, unbiased=False, keepdim=True) * (t/(self.T-1)) + self.last_var * ((self.T-1-t)/(self.T-1))
                
#                 Y = (self.X - this_mean) / torch.sqrt(this_var + self.eps)

#                 if self.elementwise_affine:
#                     Y = Y * self.weight + self.bias                
#             else:
#                 this_mean = self.X.mean(dim=-1, keepdim=True)
#                 this_var = self.X.var(dim=-1, unbiased=False, keepdim=True)                

#                 Y = (self.X - this_mean) / torch.sqrt(this_var + self.eps)

#                 if self.elementwise_affine:
#                     Y = Y * self.weight + self.bias                

#             if self.Y_pre is not None:
#                 Y_pre = self.Y_pre.detach().clone()
#             else:
#                 Y_pre = 0.0
#             self.Y_pre = Y
#             output.append(Y - Y_pre)

#         self.last_mean = self.X.mean(dim=[0,-1], keepdim=True)
#         self.last_var = self.X.var(dim=[0,-1], unbiased=False, keepdim=True)

#         return torch.cat(output, dim=0)

class Spiking_LayerNorm(nn.Module):
    def __init__(self,dim,T):
        super(Spiking_LayerNorm, self).__init__()
        self.layernorm = None
        self.X = 0.0
        self.Y_pre = None
        self.weight = None
        self.bias = None
        self.T = T
        self.t = 0

    def reset(self):
        # print("Spiking_LayerNorm reset")
        self.X = 0.0
        self.Y_pre = None
        self.t = 0
        
    def forward(self,input):
        output = []
        input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        # print("input.sum(dim=0).abs().mean()",input.sum(dim=0).abs().mean(), "after layernorm:", self.layernorm(input.sum(dim=0)).abs().mean())
        for t in range(self.T):
            self.X = input[t] + self.X
            if t < 4:
                Y = self.layernorm(self.X)*((t+1)/4)
            else:
                Y = self.layernorm(self.X)
            if self.Y_pre is not None:
                Y_pre = self.Y_pre.detach().clone()
            else:
                Y_pre = 0.0
            self.Y_pre = Y
            output.append(Y - Y_pre)
        return torch.cat(output, dim=0)
            # print("t=",t,"Y - Y_pre",(Y - Y_pre).mean())
        # ori_shape = input.shape
        # input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        # input = torch.cumsum(input, dim=0)
        # output = self.layernorm(input)
        # output = torch.diff(output,dim=0,prepend=(output[0]*0.0).unsqueeze(0))
        # return output.reshape(ori_shape)

class spiking_dyt(nn.Module):
    def __init__(self,dyt,step,T):
        super(spiking_dyt, self).__init__()
        self.X = 0.0
        # self.gamma = dyt.gamma
        # self.alpha = dyt.alpha
        self.beta = nn.Parameter(dyt.beta.data, requires_grad=True)
        self.dyt = dyt
        self.dyt.beta.data = self.dyt.beta * 0.0
        self.dyt.beta.requires_grad = False
        # self.dyt.gamma.requires_grad = False
        # self.dyt.alpha.requires_grad = False
        self.step = step
        self.t = 0
        self.T = T
        # self.divide = torch.tensor([min(1.0,1.0*(i+1)/self.step) for i in range(self.T)])
        # print(self.divide)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.t = 0
    
    def forward(self,input):
        # ori_shape = input.shape
        # input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        # self.X = torch.cumsum(input, dim=0) - input
        # Y = self.gamma * torch.sinh(self.alpha * input) / (torch.cosh(self.alpha*self.X) * torch.cosh(self.alpha*(input + self.X))) + self.beta/self.T
        # return Y.reshape(ori_shape)
        ori_shape = input.shape
        input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        input = torch.cumsum(input, dim=0)
        output = self.dyt(input)
        output = torch.diff(output,dim=0,prepend=(output[0]*0.0).unsqueeze(0))
        output[:self.step] = output[:self.step] + self.beta/self.step
        return output.reshape(ori_shape)
        
        
class SpikeMaxPooling_SS(nn.Module):
    def __init__(self,maxpool:nn.MaxPool2d,step,T):
        # take from 
        super(SpikeMaxPooling_SS, self).__init__()
        self.X = None
        self.Y_pre = None
        self.step = step
        self.maxpool = maxpool
        self.stride = maxpool.stride
        self.t = 0
        self.T = T
        # print(self.divide)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = None
        self.Y_pre = None       
        self.t = 0
    
    def forward(self,input):
        if self.X is None:
            output = self.maxpool(input)
            self.X = input
        else:
            output = self.maxpool(self.X + input) - self.maxpool(self.X)
            self.X = self.X + input
        return output



class spiking_softmax(nn.Module):
    def __init__(self,step,T):
        super(spiking_softmax, self).__init__()
        self.X = 0.0
        self.Y_pre = None
        self.step = step
        self.t = 0
        self.T = T
        # print(self.divide)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.Y_pre = None       
        self.t = 0
    
    def forward(self,input):
        ori_shape = input.shape
        input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        input = torch.cumsum(input, dim=0)
        output = F.softmax(input,dim=-1)
        output = torch.diff(output,dim=0,prepend=(output[0]*0.0).unsqueeze(0))
        return output.reshape(ori_shape)

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def threshold_optimization(data, quantization_level=255, n_trial=300, eps=1e-10):
    '''
    This function collect the activation data and find the optimized clipping
    threshold using KL_div as metric. Since this method is originated from
    post-training quantization which adopted in Tensor-RT, we keep the number of
    bits here.
    Args:
        data(numpy array): activation data
        n_bit(int):
        n_trial(int): the searching steps.
        eps(float): add eps at the average bin step for numberical stability.

    '''

    n_lvl = quantization_level  # quantization levels
    n_half_lvls = (quantization_level)//2
    n_bin_edge = n_lvl * n_trial + 1

    data_max = np.max(np.abs(data))
    hist, bin_edge = np.histogram(data.flatten(),
                                  bins=np.linspace(-data_max,
                                                   data_max,
                                                   num=n_bin_edge))

    mid_idx = int((len(hist)) / 2)
    start_idx = 100
    # log the threshold and corresponding KL-divergence
    kl_result = np.empty([len(range(start_idx, n_trial + 1)), 2])

    for i in range(start_idx, n_trial + 1):
        ref_dist = np.copy(hist[mid_idx - i * n_half_lvls:mid_idx +
                                i * n_half_lvls])
        # merge the outlier
        ref_dist[0] += hist[:mid_idx - i * n_half_lvls].sum()
        ref_dist[-1] += hist[mid_idx + i * n_half_lvls:].sum()
        # perform quantization: bins merge and expansion
        reshape_dim = int(len(ref_dist) / n_lvl)
        ref_dist_reshape = ref_dist.reshape(n_lvl, i)
        # merge bins for quantization
        ref_dist_merged = ref_dist_reshape.sum(axis=1)
        nonzero_mask = (ref_dist_reshape != 0
                        )  # obtain the mask of non-zero bins
        # in each merged large bin, get the average bin_count
        average_bin_count = ref_dist_merged / (nonzero_mask.sum(1) + eps)
        # expand the merged bins back
        expand_bin_count = np.expand_dims(average_bin_count,
                                          axis=1).repeat(i, axis=1)
        candidate_dist = (nonzero_mask * expand_bin_count).flatten()
        kl_div = scipy.stats.entropy(candidate_dist / candidate_dist.sum(),
                                     ref_dist / ref_dist.sum())
        #log threshold and KL-divergence
        current_th = np.abs(
            bin_edge[mid_idx - i * n_half_lvls])  # obtain current threshold
        kl_result[i -
                  start_idx, 0], kl_result[i -
                                           start_idx, 1] = current_th, kl_div

    # based on the logged kl-div result, find the threshold correspond to the smallest kl-div
    th_sel = kl_result[kl_result[:, 1] == kl_result[:, 1].min()][0, 0]
    print(f"Threshold calibration of current layer finished!, calculate threshold {th_sel}")

    return th_sel

def set_init_false(model):
    def set_init_false_inner(model):
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, QAttention_without_softmax) or isinstance(child, QWindowAttention):
                child.init = True
            if isinstance(child, MyQuan) or isinstance(child, MyQuanRound):
                child.init_state = child.batch_init
                # child.s.data = child.s.data * 1.5
            else:
                set_init_false_inner(child)
    set_init_false_inner(model)

def cal_overfire_loss(model):
    l2_loss = 0.0
    def l2_regularization_inner(model):
        nonlocal l2_loss
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, ST_BIFNeuron_MS):
                l2_loss = l2_loss + child.overfireLoss
            else:
                l2_regularization_inner(child)
    l2_regularization_inner(model)
    return l2_loss

def clip(x, eps):
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()

class MyQuan(nn.Module):
    def __init__(self,level,sym = False, cal_loss=False, channel_num=768,**kwargs):
        super(MyQuan,self).__init__()
        # self.level_init = level
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        self.channel_num = channel_num
        if level >= 512:
            print("level",level)
            self.pos_max = 'full'
        else:
            print("level",level)
            self.pos_max = torch.tensor(level)
            if sym:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(-level//2 + 1))
            else:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0)).to(torch.float32)
        self.batch_init = 20
        self.s_max = nn.Parameter(torch.tensor(1000.0), requires_grad=False).to(torch.float32)
        self.bias_channel = nn.Parameter(torch.zeros(self.channel_num), requires_grad=True).to(torch.float32)
        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"
        self.record = True
        self.cal_loss = cal_loss
        self.print_dist = False

    def __repr__(self):
        return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data},s_max={self.s_max})"

    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self,name,tfwriter,global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    def forward(self, x):        
        # print(self.bias_channel.mean())
        input_detype = x.dtype
        if input_detype == torch.float16:
            x = x.to(torch.bfloat16)
        # print(x.shape)
        
        if self.training and input_detype == torch.float16:
            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda', enabled=True):
                if self.pos_max == 'full':
                    return x
                if str(self.neg_min.device) == 'cpu':
                    self.neg_min = self.neg_min.to(x.device)
                if str(self.pos_max.device) == 'cpu':
                    self.pos_max = self.pos_max.to(x.device)
                min_val = self.neg_min
                max_val = self.pos_max

                # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
                s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)

                if self.init_state == 0 and self.training:
                    # print("x.detach().abs().mean()",x.detach().abs().mean())
                    self.s.data = torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max ** 0.5)).cuda() if self.sym \
                                    else torch.tensor(x.detach().abs().mean() * 4 / (self.pos_max ** 0.5)).cuda()
                    self.s.data = max(torch.tensor(1e-3,device=x.device), self.s.data)
                    self.init_state += 1
                    return x

                s_scale = grad_scale(self.s, s_grad_scale)
                bias_scale = grad_scale(self.bias_channel, s_grad_scale)
                if len(x.shape) == 3:
                    x = x + bias_scale.reshape(1,1,-1)
                elif len(x.shape) == 4:
                    if x.shape[-1] != x.shape[-2]:
                        B,Head,N,C = x.shape
                        x = x.transpose(1,2).reshape(B,N,C*Head) + bias_scale.reshape(1,1,1,-1)
                        x = x.reshape(B,N,Head,C).transpose(1,2)
                output = torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)*s_scale
        else:
            if self.pos_max == 'full':
                return x
            if str(self.neg_min.device) == 'cpu':
                self.neg_min = self.neg_min.to(x.device)
            if str(self.pos_max.device) == 'cpu':
                self.pos_max = self.pos_max.to(x.device)
            min_val = self.neg_min
            max_val = self.pos_max

            # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
            s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)

            if self.init_state == 0 and self.training:
                self.s.data = (torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max ** 0.5)).cuda() if self.sym \
                                else torch.tensor(x.detach().abs().mean() * 4 / (self.pos_max ** 0.5)).cuda())
                self.s.data = max(torch.tensor(1e-3,device=x.device), self.s.data)
                self.init_state += 1
                return x

            s_scale = grad_scale(self.s, s_grad_scale)
            bias_scale = grad_scale(self.bias_channel, s_grad_scale)
            if len(x.shape) == 3:
                x = x + bias_scale.reshape(1,1,-1)
            elif len(x.shape) == 4:
                if x.shape[-1] != x.shape[-2]:
                    B,Head,N,C = x.shape
                    x = x.transpose(1,2).reshape(B,N,C*Head) + bias_scale.reshape(1,1,1,-1)
                    x = x.reshape(B,N,Head,C).transpose(1,2)

            output = torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)*s_scale

        if self.debug and self.tfwriter is not None:
            self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
            # self.tfwriter.add_histogram(tag="after_clip/".format(s_scale.item())+self.name+'_data', values=(floor_pass(x/s_scale)).detach().cpu(), global_step=self.global_step)
            self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
            # print("(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))",(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)))
            self.debug = False
            self.tfwriter = None
            self.name = ""
            self.global_step = 0.0

        output = output.to(input_detype)
        # print("MyQuan output.abs().mean()",output.abs().mean(),output.dtype)
        if self.cal_loss:
            self.l1_loss = (torch.abs(output)/s_scale.detach()).sum() * 1e-8

        # x_abs = torch.abs(output)/self.s
        # self.l2_loss = l2_loss1 + (x_abs - (1/147)*x_abs*x_abs*x_abs).sum()
        # self.absvalue = (torch.abs(output)/self.s).sum()
        # output = floor_pass(x/s_scale)*s_scale
        # print(output.abs().mean(), self.s.data.item())
        # print(self.print_dist)        
        return output


class MyQuanRound(nn.Module):
    def __init__(self,level,sym = False, **kwargs):
        super(MyQuanRound,self).__init__()
        # self.level_init = level
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 512:
            print("level",level)
            self.pos_max = 'full'
        else:
            print("level",level)
            self.pos_max = torch.tensor(level)
            if sym:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(-level//2 + 1))
            else:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0)).to(torch.float32)
        self.batch_init = 20
        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"
        self.record = True

    def __repr__(self):
        return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data})"

    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self,name,tfwriter,global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    def forward(self, x):
        input_detype = x.dtype
        if self.pos_max == 'full':
            return x
        if str(self.neg_min.device) == 'cpu':
            self.neg_min = self.neg_min.to(x.device)
        if str(self.pos_max.device) == 'cpu':
            self.pos_max = self.pos_max.to(x.device)
        min_val = self.neg_min
        max_val = self.pos_max

        # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
        s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)

        if self.init_state == 0 and self.training:
            self.s.data = (torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max ** 0.5)).cuda() if self.sym \
                            else torch.tensor(x.detach().abs().mean() * 4 / (self.pos_max ** 0.5)).cuda())
            self.init_state += 1
            print("myquanRound init")
            return x

        self.s.data = min(torch.tensor(1/self.pos_max,device=x.device), self.s.data)
        s_scale = grad_scale(self.s, s_grad_scale)
        output = torch.clamp(torch.round(x/s_scale), min=min_val, max=max_val)*s_scale

        if self.debug and self.tfwriter is not None:
            self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
            self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
            self.debug = False
            self.tfwriter = None
            self.name = ""
            self.global_step = 0.0

        output = output.to(input_detype)
        # print("MyQuan output.abs().mean()",output.abs().mean(),output.dtype)

        # x_abs = torch.abs(output)/self.s
        # self.l2_loss = l2_loss1 + (x_abs - (1/147)*x_abs*x_abs*x_abs).sum()
        # self.absvalue = (torch.abs(output)/self.s).sum()
        # output = floor_pass(x/s_scale)*s_scale
        # print(output.abs().mean(), self.s.data.item())
        return output


class QAttention_without_softmax(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            level = 2,
            is_softmax = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.level = level
        self.is_softmax = is_softmax
        self.qkv_bias = qkv_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.quan_q = MyQuan(self.level,sym=True)
        self.quan_k = MyQuan(self.level,sym=True)
        self.quan_v = MyQuan(self.level,sym=True)
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.attn_quan = MyQuan(self.level,sym=False)
        self.attn_quan.s_max.data = 1.0/self.attn_quan.pos_max
        # self.attn_quan.s.data = torch.tensor(0.125)
        # self.attn_quan.init_state = self.attn_quan.batch_init
        # self.attn_quan.s.requires_grad = False
        self.proj_drop = nn.Dropout(proj_drop)
        if self.is_softmax:
            self.attn_softmax_quan = MyQuan(self.level,sym=True)
        self.after_attn_quan = MyQuan(self.level,sym=True)
        # self.quan_proj = MyQuan(self.level,sym=True)
        self.init = False
                
    def forward(self, x):
        B, N, C = x.shape
        # print("x.abs().mean()",x.abs().mean())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.quan_q(q)
        k = self.quan_k(k)
        v = self.quan_v(v)
        # print("q.abs().mean()",q.abs().mean())
        # print("k.abs().mean()",k.abs().mean())
        # print("v.abs().mean()",v.abs().mean())

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = self.attn_drop(attn)
        # print("attn.abs().mean() before",attn.abs().mean())
        if self.init == False and self.training:
            attn = torch.clamp(attn/N, max=0.99, min=-0.01)
            attn = self.attn_quan(attn)
            self.init = True
            print("QAttention_without_softmax init")
        else:
            attn = self.attn_quan(attn/N)
        # print("attn.abs().mean()",attn.abs().mean())

        x = attn @ v
        x = self.after_attn_quan(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        # print("x.abs().mean()",x.abs().mean())

        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.quan_proj(x)
        # print("x.abs().mean()",x.abs().mean())
        return x


class QAttention_without_softmax_qkvfusion(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            level = 2,
            is_softmax = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.level = level
        self.is_softmax = is_softmax
        self.qkv_bias = qkv_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.quan_qkv = MyQuan(self.level,sym=True)
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.attn_quan = MyQuanRound(self.level,sym=False)
        self.attn_quan.s.data = torch.tensor(0.125)
        self.attn_quan.init_state = self.attn_quan.batch_init
        self.attn_quan.s.requires_grad = False
        self.proj_drop = nn.Dropout(proj_drop)
        if self.is_softmax:
            self.attn_softmax_quan = MyQuan(self.level,sym=True)
        self.after_attn_quan = MyQuan(self.level,sym=True)
        self.quan_proj = MyQuan(self.level,sym=True)
        self.init = False
                
    def forward(self, x):
        B, N, C = x.shape
        # print("x.abs().mean()",x.abs().mean())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv = self.quan_qkv(qkv)
        q, k, v = qkv.unbind(0)
        # print("q.abs().mean()",q.abs().mean())
        # print("k.abs().mean()",k.abs().mean())
        # print("v.abs().mean()",v.abs().mean())

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = self.attn_drop(attn)
        # print("attn.abs().mean() before",attn.abs().mean())
        if self.init == False and self.training:
            attn = torch.clamp(attn/N, max=0.99, min=-0.01)
            self.init = True
        else:
            attn = self.attn_quan(attn/N)
        # print("attn.abs().mean()",attn.abs().mean())

        x = attn @ v
        x = self.after_attn_quan(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        # print("x.abs().mean()",x.abs().mean())

        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.quan_proj(x)
        # print("x.abs().mean()",x.abs().mean())
        return x


class QAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            level = 2,
            is_softmax = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.level = level
        self.is_softmax = is_softmax
        self.qkv_bias = qkv_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.quan_q = MyQuan(self.level,sym=True)
        self.quan_k = MyQuan(self.level,sym=True)
        self.quan_v = MyQuan(self.level,sym=True)
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.attn_quan = MyQuan(self.level,sym=False)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.is_softmax:
            self.attn_softmax_quan = MyQuan(self.level,sym=True)
        self.after_attn_quan = MyQuan(self.level,sym=True)
        self.quan_proj = MyQuan(self.level,sym=True)
        
    def forward(self, x):
        B, N, C = x.shape
        # print("x.abs().mean()",x.abs().mean())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # q, k = self.q_norm(q), self.k_norm(k)
        q = self.quan_q(q)
        k = self.quan_k(k)
        v = self.quan_v(v)
        # print("q.abs().mean()",q.abs().mean())
        # print("k.abs().mean()",k.abs().mean())
        # print("v.abs().mean()",v.abs().mean())
        # if self.training:
        #     print("q,k,v",q.abs().mean().item(),k.abs().mean().item(),v.abs().mean().item())
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # if self.training:
        #     print("attn",attn.abs().mean().item())
        if self.is_softmax:
            # attn = self.attn_quan(attn)
            attn = attn.softmax(dim=-1)
            attn = self.attn_softmax_quan(attn)
        else:
            attn = self.attn_quan(attn)/N
        
        attn = self.attn_drop(attn)
        x = attn @ v
        # if self.training:
        #     print("after_attn",x.abs().mean().item())
        x = self.after_attn_quan(x)
        # if self.training:
        #     print("after_attn_quan",x.abs().mean().item())

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # if self.training:
        #     print("proj",x.abs().mean().item())
        x = self.proj_drop(x)
        x = self.quan_proj(x)

        return x

class AttentionMulti(nn.Module):
    def __init__(self):
        super(AttentionMulti,self).__init__()

    def forward(self, x1_t,x2_t,x1_sum_t,x2_sum_t):
        return (x1_t + x1_sum_t) @ x2_t.transpose(-2, -1) + x1_t @ x2_sum_t.transpose(-2, -1)
    
class AttentionMulti1(nn.Module):
    def __init__(self):
        super(AttentionMulti1,self).__init__()

    def forward(self, x1_t,x2_t,x1_sum_t,x2_sum_t):
        return  (x1_t + x1_sum_t) @ x2_t + x1_t @ x2_sum_t

def multi(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return (x1_t + x1_sum_t) @ x2_t.transpose(-2, -1) + x1_t @ x2_sum_t.transpose(-2, -1)

def multi1(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return  (x1_t + x1_sum_t) @ x2_t + x1_t @ x2_sum_t


class SAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            neuron_layer = ST_BIFNeuron_MS,
            level = 2,
            is_softmax = True,
            T = 32,
            
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim ** -0.5)
        self.neuron_layer = neuron_layer
        self.level = level
        self.is_softmax = is_softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True)
        # self.spikeBN_q = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(self.head_dim),level=self.level//2-1,input_allcate=False)
        # self.spikeBN_k = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(self.head_dim),level=self.level//2-1,input_allcate=False)
        # self.spikeBN_v = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(self.head_dim),level=self.level//2-1,input_allcate=False)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.attn_ReLU = nn.ReLU()
        self.attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False, need_spike_tracer=not is_softmax)
        # self.attn_IF.prefire = 0.4
        # self.spikeBN_attn = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(197),level=self.level,input_allcate=False)
        # if self.is_softmax:
        self.attn_softmax_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=is_softmax)
        self.attn_softmax_IF.prefire.data = torch.tensor(0.135)
        # self.spikeBN_after_attn = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(self.head_dim),level=self.level//2-1,input_allcate=False)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.spikeBN_proj = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(dim),level=self.level//2-1,input_allcate=False)
        # self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        if self.is_softmax:
            self.Ssoftmax = spiking_softmax(self.level//2 - 1, T)
        self.T = T
        # self.Release_attn1 = Release_attn(self.level//2 - 1)
        # self.Release_attn2 = Release_attn(self.level//2 - 1)

        # saving mid feature
        self.t = 0
        self.first = False        
        self.accu_input = []
        self.accu_qkv = []
        self.accu_q = []
        self.accu_k = []
        self.accu_v = []
        self.accu_q_scale = []
        self.accu_q_scale_acc = []
        self.accu_k_acc = []
        self.accu_v_acc = []
        self.accu_qk = []
        self.accu_qk_softmax = []
        self.accu_qk_acc = []
        self.accu_attn = []
        self.accu_proj_input = []
        self.accu_proj = []
        self.accu = []
        self.accu1 = []
        self.accu_q_in = None
        self.accu_k_in = None
        self.accu_v_in = None
        self.accu_attn_in = None
        self.name = ""

    def reset(self):
        # print("SAttention reset")
        self.q_IF.reset()
        self.k_IF.reset()
        self.v_IF.reset()
        # self.spikeBN_q.reset()
        # self.spikeBN_k.reset()
        # self.spikeBN_v.reset()
        # self.spikeBN_attn.reset()
        # self.spikeBN_after_attn.reset()
        # self.spikeBN_proj.reset()
        self.attn_IF.reset()
        self.attn_softmax_IF.reset()
        self.after_attn_IF.reset()
        # self.proj_IF.reset()
        if self.is_softmax:
            self.Ssoftmax.reset()
        # self.qkv.reset()
        # self.proj.reset()
        self.t = 0
        self.accu_q_in = None
        self.accu_k_in = None
        self.accu_v_in = None
        self.accu_attn_in = None

    def forward(self, x):
        self.t = self.t + 1
        B, N, C = x.shape
        
        if self.first:
            self.accu_input.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn_4dim(torch.stack(self.accu_input), glo.get_value("output_bin_snn_dir"),self.name+"_qkv.in")
                del self.accu_input

        # print("x.abs().mean()",x.reshape(self.T,32,197,384).sum(dim=0).abs().mean())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3 B self.num_heads N self.head_dim

        q, k, v = qkv.unbind(0)
        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)
        # print("q.abs().mean()",q.reshape(self.T,32,6,197,64).sum(dim=0).abs().mean())
        # print("k.abs().mean()",k.reshape(self.T,32,6,197,64).sum(dim=0).abs().mean())
        # print("v.abs().mean()",v.reshape(self.T,32,6,197,64).sum(dim=0).abs().mean())
        
        if self.first:
            self.accu_q.append(q[0].unsqueeze(0)+0)
            self.accu_k.append(k[0].unsqueeze(0)+0)
            self.accu_v.append(v[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn_5dim(torch.stack(self.accu_q), glo.get_value("output_bin_snn_dir"),self.name+"_qkv_q.out")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_k), glo.get_value("output_bin_snn_dir"),self.name+"_qkv_k.out")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_v), glo.get_value("output_bin_snn_dir"),self.name+"_qkv_v.out")
                del self.accu_q
        
        
        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold
        if self.first:
            self.accu_q_scale.append(q[0].unsqueeze(0)+0)
            self.accu_q_scale_acc.append(q_acc[0].unsqueeze(0)+0)
            self.accu_k_acc.append((self.k_IF.acc_q*self.k_IF.q_threshold)[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn_5dim(torch.stack(self.accu_q_scale), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_q.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_q_scale_acc), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_q_accu.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_k), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_k.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_k_acc), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_k_accu.in")
                del self.accu_q_scale
                del self.accu_q_scale_acc
                del self.accu_k
                del self.accu_k_acc

        # if self.accu_q_in is None:
        #     self.accu_q_in = q*0.0
        #     self.accu_k_in = k*0.0
        # print(f"=============================Timestep={self.t}===================================")
        # print("q_acc.mean()",q_acc.mean())
        # print("q.mean()",q.mean())
        # print("self.k_IF.acc_q*self.k_IF.q_threshold.mean()",(self.k_IF.acc_q*self.k_IF.q_threshold).mean())
        # print("k.mean()",k.mean())
        # print(q.shape, k.shape, q_acc.shape, self.k_IF.acc_q.shape)
        attn = multi(q,k,q_acc - q.detach(),self.k_IF.acc_q*self.k_IF.q_threshold - k.detach())
        # attn = self.spikeBN_attn(attn)

        # attn = multi(q,k,self.accu_q_in,self.accu_k_in)
        # attn = self.Release_attn1(attn)
        # print("attn.shape",attn.shape,"attn.abs().mean()",torch.abs(attn).mean())
        # self.accu_q_in = self.accu_q_in + q
        # self.accu_k_in = self.accu_k_in + k
        # attn = q@k.transpose(-2, -1)
        # print("attn",attn.abs().mean())
        if not self.is_softmax:
            attn = self.attn_IF(attn)

        if self.first:
            self.accu_qk.append(attn[0].unsqueeze(0)+0)
            if self.t == self.T:    
                save_input_for_bin_snn_5dim(torch.stack(self.accu_qk), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti.out")        
        
        if self.is_softmax:
            attn = self.Ssoftmax(attn)
            attn = self.attn_softmax_IF(attn)
            if self.first:
                self.accu_qk_softmax.append(attn[0].unsqueeze(0)+0)
                if self.t == self.T:    
                    save_input_for_bin_snn_5dim(torch.stack(self.accu_qk_softmax), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_softmax.out")

        if not self.is_softmax:
            attn = attn/N
        #     print("/N",attn.abs().mean())
            # attn = self.attn_softmax_IF(attn)
            # acc_attn = self.attn_IF.acc_q*self.attn_IF.q_threshold/N

        attn = self.attn_drop(attn)

        if self.first:
            if not self.is_softmax:
                self.accu_qk_acc.append((self.attn_IF.acc_q*self.attn_IF.q_threshold)[0].unsqueeze(0)+0)
            else:
                self.accu_qk_acc.append((self.attn_softmax_IF.acc_q*self.attn_softmax_IF.q_threshold)[0].unsqueeze(0)+0)
            self.accu_v_acc.append((self.v_IF.acc_q*self.v_IF.q_threshold)[0].unsqueeze(0)+0)
            if self.t == self.T:
                if not self.is_softmax:
                    save_input_for_bin_snn_5dim(torch.stack(self.accu_qk), glo.get_value("output_bin_snn_dir"),self.name+"_attn_qk.in")
                else:
                    save_input_for_bin_snn_5dim(torch.stack(self.accu_qk_softmax), glo.get_value("output_bin_snn_dir"),self.name+"_attn_qk.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_qk_acc), glo.get_value("output_bin_snn_dir"),self.name+"_attn_qk_acc.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_v), glo.get_value("output_bin_snn_dir"),self.name+"_attn_v.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_v_acc), glo.get_value("output_bin_snn_dir"),self.name+"_attn_v_acc.in")
                del self.accu_qk
                del self.accu_qk_acc
                del self.accu_v
                del self.accu_v_acc

        # if self.accu_attn_in is None:
        #     self.accu_attn_in = attn*0.0
        #     self.accu_v_in = v*0.0
        # if not self.is_softmax:
        #     x = multi1(attn,v,self.accu_attn_in,self.accu_v_in)
        # else:
        #     x = multi1(attn,v,self.accu_attn_in,self.accu_v_in)
        if not self.is_softmax:
            x = multi1(attn,v,(self.attn_IF.acc_q*self.attn_IF.q_threshold - attn.detach()),(self.v_IF.acc_q*self.v_IF.q_threshold - v.detach()))
        else:
            x = multi1(attn,v,(self.attn_softmax_IF.acc_q*self.attn_softmax_IF.q_threshold - attn.detach()),(self.v_IF.acc_q*self.v_IF.q_threshold - v.detach()))
        # x = attn @ v
        # x = self.Release_attn2(x)
        # self.accu_attn_in = self.accu_attn_in + attn
        # self.accu_v_in = self.accu_v_in + v

        # print("after multi1",x.abs().mean())
        # x = self.spikeBN_after_attn(x)
        x = self.after_attn_IF(x)
        # print("after after_attn_IF",x.abs().mean())
        if self.first:
            self.accu_attn.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:    
                save_input_for_bin_snn_5dim(torch.stack(self.accu_attn), glo.get_value("output_bin_snn_dir"),self.name+"_attn.out")
                del self.accu_attn

        x = x.transpose(1, 2).reshape(B, N, C)

        if self.first:
            self.accu_proj_input.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:    
                save_input_for_bin_snn_4dim(torch.stack(self.accu_proj_input), glo.get_value("output_bin_snn_dir"),self.name+"_proj.in")
                del self.accu_proj_input

        x = self.proj(x)
        # print("after proj",x.abs().mean())
        x = self.proj_drop(x)
        # x = self.spikeBN_proj(x)
        # print("after spikeBN_proj",x.abs().mean())
        # x = self.proj_IF(x)
        # print("after proj_IF",x.abs().mean())

        if self.first:
            self.accu_proj.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:    
                save_input_for_bin_snn_4dim(torch.stack(self.accu_proj), glo.get_value("output_bin_snn_dir"),self.name+"_proj.out")
                del self.accu_proj
                self.first = False
                local_rank = torch.distributed.get_rank()
                if local_rank == 0:
                    torch.save(self.qkv.quan_w_fn(self.qkv.weight),f'{glo.get_value("output_bin_snn_dir")}/{self.name}_qkv_weight.pth')
                    torch.save(self.qkv.bias,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_qkv_bias.pth')
                    torch.save(self.proj.quan_w_fn(self.proj.weight),f'{glo.get_value("output_bin_snn_dir")}/{self.name}_proj_weight.pth')
                    torch.save(self.proj.bias,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_proj_bias.pth')

        return x


class SAttention_without_softmax(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            neuron_layer = ST_BIFNeuron_MS,
            level = 2,
            is_softmax = True,
            T = 32,
            
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim ** -0.5)
        self.neuron_layer = neuron_layer
        self.level = level
        self.is_softmax = is_softmax

        self.T = T
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False, need_spike_tracer=True, T = self.T, C=768)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False, need_spike_tracer=True, T = self.T, C=768)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True, T = self.T, C=768)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False, need_spike_tracer=not is_softmax, T = self.T, C=768)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, T = self.T, C=768)
        self.feature_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, T = self.T)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, T = self.T, C=197)
        self.multi = AttentionMulti()
        self.multi1 = AttentionMulti1()

        # saving mid feature
        self.t = 0
        self.first = False        
        self.accu_input = []
        self.accu_qkv = []
        self.accu_q = []
        self.accu_k = []
        self.accu_v = []
        self.accu_q_scale = []
        self.accu_q_scale_acc = []
        self.accu_k_acc = []
        self.accu_v_acc = []
        self.accu_qk = []
        self.accu_qk_softmax = []
        self.accu_qk_acc = []
        self.accu_attn = []
        self.accu_proj_input = []
        self.accu_proj = []
        self.accu = []
        self.accu1 = []
        self.accu_q_in = None
        self.accu_k_in = None
        self.accu_v_in = None
        self.accu_attn_in = None
        self.name = ""

    def reset(self):
        self.q_IF.reset()
        self.k_IF.reset()
        self.v_IF.reset()
        self.attn_IF.reset()
        self.after_attn_IF.reset()
        self.t = 0
        self.accu_q_in = None
        self.accu_k_in = None
        self.accu_v_in = None
        self.accu_attn_in = None

    def forward(self, x):
        self.t = self.t + 1
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3 B self.num_heads N self.head_dim
        # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())

        q, k, v = qkv.unbind(0)
        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)
        # print("q.abs().mean()",q.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
        # print("k.abs().mean()",k.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
        # print("v.abs().mean()",v.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
                        
        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold

        attn = self.multi(q,k,q_acc - q.detach(),self.k_IF.acc_q*self.k_IF.q_threshold - k.detach())
        attn = self.attn_drop(attn)
        # print("attn.abs().mean() before",attn.reshape(self.T,8,12,197,197).sum(dim=0).abs().mean())
        attn = self.attn_IF(attn/N)
        # print("attn.abs().mean()",attn.reshape(self.T,8,12,197,197).sum(dim=0).abs().mean())


        x = self.multi1(attn,v,(self.attn_IF.acc_q*self.attn_IF.q_threshold - attn.detach()),(self.v_IF.acc_q*self.v_IF.q_threshold - v.detach()))

        x = self.after_attn_IF(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())

        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.proj_IF(x)
        # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())

        return x


class SAttention_without_softmax_SS(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            neuron_layer = ST_BIFNeuron_SS,
            level = 2,
            is_softmax = True,
            T = 32,
            
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim ** -0.5)
        self.neuron_layer = neuron_layer
        self.level = level
        self.is_softmax = is_softmax

        self.T = T
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)

        # saving mid feature
        self.t = 0
        self.name = ""

    def reset(self):
        self.q_IF.reset()
        self.k_IF.reset()
        self.v_IF.reset()
        self.attn_IF.reset()
        self.after_attn_IF.reset()
        self.proj_IF.reset()
        self.qkv.reset()
        self.proj.reset()
        self.t = 0

    def forward(self, x):
        self.t = self.t + 1
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3 B self.num_heads N self.head_dim
        # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())

        q, k, v = qkv.unbind(0)
        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)
        # print("q.abs().mean()",q.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
        # print("k.abs().mean()",k.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
        # print("v.abs().mean()",v.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
                        
        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold

        attn = multi(q,k,q_acc - q.detach(),self.k_IF.acc_q*self.k_IF.q_threshold - k.detach())
        attn = self.attn_drop(attn)
        # print("attn.abs().mean() before",attn.reshape(self.T,8,12,197,197).sum(dim=0).abs().mean())
        attn = self.attn_IF(attn/N)
        # print("attn.abs().mean()",attn.reshape(self.T,8,12,197,197).sum(dim=0).abs().mean())


        x = multi1(attn,v,(self.attn_IF.acc_q*self.attn_IF.q_threshold - attn.detach()),(self.v_IF.acc_q*self.v_IF.q_threshold - v.detach()))

        x = self.after_attn_IF(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())

        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.proj_IF(x)
        # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())

        return x


class DyT(nn.Module):
    def __init__(self, C, init_alpha=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(torch.ones(1) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(C)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))

    def forward(self,x):
        x = torch.tanh(self.alpha*x)
        return self.gamma * x + self.beta

class DyHT_ReLU(nn.Module):
    def __init__(self, C, init_alpha=0.25):
        super(DyHT_ReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(torch.ones(C) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(1)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))

    def forward(self,x):
        # B = x.shape[0]
        # print("QANN INPUT DyHT.abs().mean()",x.abs().mean())
        x = (torch.nn.functional.hardtanh(self.alpha*x + self.beta, min_val=0.0, max_val=1.0))
        # print("QANN DyHT.abs().mean()",x.abs().mean())
        # print("self.gamma min",self.gamma.min(),"self.gamma max",self.gamma.max())
        
        return x * self.gamma

class DyHT(nn.Module):
    def __init__(self, C, init_alpha=0.25):
        super(DyHT, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(torch.ones(C)))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(1) * init_alpha))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))

    def forward(self,x):
        # B = x.shape[0]
        # print("QANN INPUT DyHT.abs().mean()",x.abs().mean())
        x = torch.nn.functional.hardtanh(self.alpha*x + self.beta, min_val=-1.0, max_val=1.0)
        # print("QANN DyHT.abs().mean()",x.abs().mean())
        # print("self.gamma min",self.gamma.min(),"self.gamma max",self.gamma.max())
        
        return x * self.gamma

class SDyHT_SS(nn.Module):
    def __init__(self, C, init_alpha=0.25):
        super(SDyHT_SS, self).__init__()
        self.step = 1
        self.T = 1
        self.t = 0
        self.alpha = nn.Parameter(torch.tensor(torch.ones(C) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(1)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))
        self.accu1 = 0.0
        self.accu2 = 0.0
    
    def reset(self):
        self.t = 0
    
    def forward(self,x):
        self.t = self.t + 1
        # self.accu1 = self.accu1 + x
        # if self.t == 12:
        #     print("SNN INPUT DyHT.abs().mean()",self.accu1.abs().mean())
        x = self.alpha * self.gamma * x
        if self.t <= self.step:
            x = x + self.beta * self.gamma/self.step
        # self.accu2 = self.accu2 + x
        # if self.t == 12:
        #     print("SNN OUTPUT DyHT.abs().mean()",self.accu2.abs().mean())
        #     self.t = 0
        return x

class SDyHT(nn.Module):
    def __init__(self, C, init_alpha=0.25, step=3, T=32):
        super(SDyHT, self).__init__()
        self.step = step
        self.T = T
        self.alpha = nn.Parameter(torch.tensor(torch.ones(C) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(1)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))

        self.param_number = 2
        init_list = []
        for i in range(self.param_number - 1):
            if i < self.step - 1:
                init_list.append(1/(self.step))
            else:
                init_list.append(0.0)
        if self.T > 1:
            self.biasAllocator = nn.Parameter(torch.tensor(init_list),requires_grad=True)
        else:
            self.biasAllocator = 1.0

    def __repr__(self):
        return f"SDyHT(self.gamma={self.gamma.data[0]})"

    def forward(self,x):
        biasAllocator = torch.cat([1 - torch.sum(self.biasAllocator,dim=0,keepdim=True), self.biasAllocator], dim=0)
        if x.dim() == 3:
            B,N,C = x.shape
            # print("SNN INPUT DyHT.abs().mean()",x.reshape(32,8,-1).sum(dim=0).abs().mean())
            x = x.reshape(self.T,B//self.T,N,C)
            x = self.alpha * self.gamma * x
            bias_term = biasAllocator.view(-1, 1, 1, 1) * self.gamma * self.beta.view(1, 1, 1, -1)
            x = torch.cat([
                x[:self.param_number] + bias_term,
                x[self.param_number:]
            ], dim=0)
            x = x.reshape(B,N,C)
            # print("SNN DyHT.abs().mean()",x.reshape(32,8,-1).sum(dim=0).abs().mean())
        else:
            B,C = x.shape
            # print("SNN INPUT DyHT.abs().mean()",x.reshape(32,8,-1).sum(dim=0).abs().mean())
            x = x.reshape(self.T,B//self.T,C)
            x = self.alpha * self.gamma * x
            bias_term = biasAllocator.view(-1, 1, 1) * self.gamma * self.beta.view(1, 1, -1)
            x = torch.cat([
                x[:self.param_number] + bias_term,
                x[self.param_number:]
            ], dim=0)
            x = x.reshape(B,C)
        
        # print("self.beta",self.beta.data)        
        
        return x


class WindowAttention_no_softmax(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.attnBatchNorm = MyBatchNorm1d(dim=coords_h.shape[0]*coords_w.shape[0])
        self.attnBatchNorm = DyT(coords_h.shape[0]*coords_w.shape[0])
        self.tokenNum = coords_h.shape[0]*coords_w.shape[0]
        self.ReLU = nn.ReLU()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.name = "attention"

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # if torch.isnan(qkv).any():
        #     print(f"{self.name}.qkv: NAN!!!!")

        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # q = torch.clamp(q,min=-6.0,max=6.0)
        # k = torch.clamp(k,min=-6.0,max=6.0)
        # v = torch.clamp(v,min=-6.0,max=6.0)

        # print("q",self.name, q.abs().max())
        # print("k",self.name, k.abs().max())
        # print("v",self.name, v.abs().max())
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # if torch.isnan(attn).any():
        #     print(f"{self.name}.attn: NAN!!!!")
        # print("attn1.abs()",self.name, attn.abs().max())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        # if torch.isnan(attn).any():
        #     print(f"{self.name}.attn_after_softmax: NAN!!!!")

        attn = self.attn_drop(attn)
        # print("attn2.abs()",self.name, attn.abs().max())

        attn = torch.clamp(attn/N, max=0.99, min=-0.01)

        # print("attn3",self.name, attn.abs().max())
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # if torch.isnan(x).any():
        #     print(f"{self.name}.proj: NAN!!!!")
        
        return x

class QWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., level=10):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.level = level
        self.init = False

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.quan_q = MyQuan(self.level,sym=True)
        self.quan_k = MyQuan(self.level,sym=True)
        self.quan_v = MyQuan(self.level,sym=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.after_attn_quan = MyQuan(self.level,sym=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.quan_proj = MyQuan(self.level,sym=True)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_softmax_quan = MyQuan(self.level,sym=False)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        q = self.quan_q(q)
        k = self.quan_k(k)
        v = self.quan_v(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print("attn",attn.abs().mean())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(attn.shape,relative_position_bias.unsqueeze(0).shape)
        attn = attn + relative_position_bias.unsqueeze(0)
        # print("relative_position_bias",attn.abs().mean())

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        # print("softmax std",attn.abs().std())
        if self.init == False and self.training:
            attn = torch.clamp(attn/N, max=0.99, min=-0.01)
            attn = self.attn_softmax_quan(attn)
            self.init = True
            print("QAttention_without_softmax init")
        else:
            attn = self.attn_softmax_quan(attn/(N))

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.after_attn_quan(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.quan_proj(x)
        return x


class SWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., level=10, T = 32, step=4, neuron_layer = ST_BIFNeuron_MS):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.level = level
        self.T = T
        self.neuron_layer = neuron_layer
        self.step = step

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True,T=self.T)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True,T=self.T)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True,T=self.T)
        self.attn_drop = nn.Dropout(attn_drop)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True,T=self.T)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True,T=self.T)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.Ssoftmax = spiking_softmax(self.step, T)
        self.attn_softmax_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True,T=self.T)
        self.attn_multi = AttentionMulti()
        self.attn_multi1 = AttentionMulti1()
        # self.attn_softmax_IF.prefire.data = torch.tensor(0.025)

    def reset(self):
        # print("SAttention reset")
        self.q_IF.reset()
        self.k_IF.reset()
        self.v_IF.reset()
        self.attn_softmax_IF.reset()
        self.after_attn_IF.reset()
        self.proj_IF.reset()
        self.Ssoftmax.reset()
        self.t = 0


    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)
        # print("======================q,k,v======================")

        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold
        # attn = (q @ k.transpose(-2, -1))
        attn = self.attn_multi(q,k,q_acc - q.detach(),self.k_IF.acc_q*self.k_IF.q_threshold - k.detach())
        # attn1 = attn.reshape(torch.Size([self.T, B_//self.T]) + attn.shape[1:])
        # print("SNN multi",attn1.sum(dim=0).abs().mean())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn.reshape(torch.Size([self.T, B_//self.T]) + attn.shape[1:])
        for t in range(self.T):
            if t < self.step:
                attn[t] = attn[t] + relative_position_bias.unsqueeze(0)/self.step
                # print(attn[t].shape,relative_position_bias.unsqueeze(0).shape)
        attn = attn.reshape(torch.Size([attn.shape[0]*attn.shape[1]]) + attn.shape[2:])
        # attn = attn + relative_position_bias.unsqueeze(0)

        # attn1 = attn.reshape(torch.Size([self.T, B_//self.T]) + attn.shape[1:])
        # print("SNN relative_position_bias",attn1.sum(dim=0).abs().mean())
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.Ssoftmax(attn)
        # else:
        #     attn = self.Ssoftmax(attn)
        
        # attn1 = attn.reshape(torch.Size([self.T, B_//self.T]) + attn.shape[1:])
        # print("Ssoftmax std",attn1.sum(dim=0).abs().std())
        attn = self.attn_softmax_IF(attn/N)

        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.attn_multi1(attn,v,(self.attn_softmax_IF.acc_q*self.attn_softmax_IF.q_threshold - attn.detach()),(self.v_IF.acc_q*self.v_IF.q_threshold - v.detach())).transpose(1, 2).reshape(B_, N, C)
        x = self.after_attn_IF(x)
        # print("======================after_attn_IF======================")
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.proj_IF(x)
        return x

class SWindowAttention_SS(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., level=10, T = 32, step=4, neuron_layer = ST_BIFNeuron_SS):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.level = level
        self.T = T
        self.neuron_layer = neuron_layer
        self.step = step

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.Ssoftmax = spiking_softmax(self.step, T)
        self.attn_softmax_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.attn_multi = AttentionMulti()
        self.attn_multi1 = AttentionMulti1()
        self.t = 0
        # self.attn_softmax_IF.prefire.data = torch.tensor(0.025)

    def reset(self):
        self.qkv.reset()
        self.q_IF.reset()
        self.proj.reset()
        self.k_IF.reset()
        self.v_IF.reset()
        self.attn_softmax_IF.reset()
        self.after_attn_IF.reset()
        self.proj_IF.reset()
        self.t = 0


    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        self.t = self.t + 1
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)
        # print("======================q,k,v======================")

        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold
        # attn = (q @ k.transpose(-2, -1))
        attn = self.attn_multi(q,k,q_acc - q.detach(),self.k_IF.acc_q*self.k_IF.q_threshold - k.detach())
        # attn1 = attn.reshape(torch.Size([self.T, B_//self.T]) + attn.shape[1:])
        # print("SNN multi",attn1.sum(dim=0).abs().mean())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        if self.t <= self.step:
            attn = attn + relative_position_bias.unsqueeze(0)/self.step
        # attn1 = attn.reshape(torch.Size([self.T, B_//self.T]) + attn.shape[1:])
        # print("SNN relative_position_bias",attn1.sum(dim=0).abs().mean())
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.Ssoftmax(attn)
        # else:
        #     attn = self.Ssoftmax(attn)
        
        # attn1 = attn.reshape(torch.Size([self.T, B_//self.T]) + attn.shape[1:])
        # print("Ssoftmax std",attn1.sum(dim=0).abs().std())
        attn = self.attn_softmax_IF(attn/N)

        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.attn_multi1(attn,v,(self.attn_softmax_IF.acc_q*self.attn_softmax_IF.q_threshold - attn.detach()),(self.v_IF.acc_q*self.v_IF.q_threshold - v.detach())).transpose(1, 2).reshape(B_, N, C)
        x = self.after_attn_IF(x)
        # print("======================after_attn_IF======================")
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.proj_IF(x)
        return x


class SpikeMaxPooling(nn.Module):
    def __init__(self,maxpool:nn.MaxPool2d,step,T):
        # take from 
        super(SpikeMaxPooling, self).__init__()
        self.X = 0.0
        self.Y_pre = None
        self.step = step
        self.maxpool = maxpool
        self.stride = maxpool.stride
        self.t = 0
        self.T = T
        # print(self.divide)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.Y_pre = None       
        self.t = 0
    
    def forward(self,input):
        ori_shape = input.shape
        input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        input = torch.cumsum(input, dim=0).reshape(ori_shape)
        output = self.maxpool(input)
        ori_shape = output.shape
        output = output.reshape(torch.Size([self.T, output.shape[0]//self.T]) + output.shape[1:])
        output = torch.diff(output,dim=0,prepend=(output[0]*0.0).unsqueeze(0))
        return output.reshape(ori_shape)


# class SpikeMaxPooling(nn.Module):
#     def __init__(self,maxpool):
#         super(SpikeMaxPooling,self).__init__()
#         self.maxpool = maxpool
        
#         self.accumulation = None
    
#     def reset(self):
#         self.accumulation = None

#     def forward(self,x):
#         old_accu = self.accumulation
#         if self.accumulation is None:
#             self.accumulation = x
#         else:
#             self.accumulation = self.accumulation + x
        
#         if old_accu is None:
#             output = self.maxpool(self.accumulation)
#         else:
#             output = self.maxpool(self.accumulation) - self.maxpool(old_accu)

#         # print("output.shape",output.shape)
#         # print(output[0][0][0:4][0:4])
        
#         return output


class Addition(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(self,x):
        return x[0]+x[1]


class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return self._conv_forward(x, quantized_weight, self.bias)


class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, quan_w_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return torch.nn.functional.linear(x, quantized_weight, self.bias)


class LLConv2d(nn.Module):
    def __init__(self,conv:nn.Conv2d,**kwargs):
        super(LLConv2d,self).__init__()
        self.conv = conv
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level//2 - 1
        self.realize_time = self.steps
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        # self.quan_w_fn = self.conv.quan_w_fn
        
    def reset(self):
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self,input):
        # print("LLConv2d.steps",self.steps)
        x = input
        N,C,H,W = x.shape
        F_h,F_w = self.conv.kernel_size
        S_h,S_w = self.conv.stride
        P_h,P_w = self.conv.padding
        C = self.conv.out_channels
        H = math.floor((H - F_h + 2*P_h)/S_h)+1
        W = math.floor((W - F_w + 2*P_w)/S_w)+1

        if self.zero_output is None:
            # self.zero_output = 0.0
            self.zero_output = torch.zeros(size=(N,C,H,W),device=x.device,dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        # output = self.conv(x)
        if self.realize_time > 0:
            output = torch.nn.functional.conv2d(input, self.conv.weight, (self.conv.bias/self.steps if self.conv.bias is not None else 0.0), stride=self.conv.stride, \
                padding=self.conv.padding, dilation=self.conv.dilation,groups=self.conv.groups)
            self.realize_time = self.realize_time - 1
        else:
            output = torch.nn.functional.conv2d(input, self.conv.weight, None, stride=self.conv.stride, \
                padding=self.conv.padding, dilation=self.conv.dilation,groups=self.conv.groups)            
        # if self.neuron_type == 'IF':
        #     pass
        # else:
        #     if self.conv.bias is None:
        #         pass
        #     else:
        #         # if not self.first:
        #         #     output = output - self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        #         output = output - (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if self.conv.bias is not None else 0.0)
        #         if self.realize_time > 0:
        #             output = output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
        #             self.realize_time = self.realize_time - 1
        #             # print("conv2d self.realize_time",self.realize_time)
                    

        self.is_work = True
        self.first = False

        return output


# class LLLinear_MS(nn.Module):
#     def __init__(self,linear:nn.Linear,**kwargs):
#         super(LLLinear_MS,self).__init__()
#         self.linear = linear
#         self.level = kwargs["level"]
#         self.T = kwargs["time_step"]
#         self.steps = kwargs["step"]
#         self.overfireLoss = torch.tensor(0.0)
#         self.group_size = 4
    
#     def forward(self,input):
#         B = input.shape[0]//self.T        
#         # print("self.steps",self.steps,"B",B,"self.T",self.T)
#         group_size = self.group_size
        
#         if len(input.shape) == 3:        
#             B1, N, L = input.shape  # 假设 input 是 [B, N, L] 形状
#             pad = (group_size - (N % group_size)) % group_size  # 需要补齐的通道数

#             if pad > 0:
#                 # 在 channel 维度上 padding，pad 后形状为 [B, N + pad, L]
#                 input_pad = F.pad(input+0.0, (0, 0, 0, pad))  # (last two are for (dim=-2,-1))

#             # reshape 成 [B, group_size, num_groups, T]
#             input_group = input_pad.reshape(B1, group_size, (N + pad) // group_size, L)
#             self.overfireLoss = torch.sqrt((input_group ** 2).sum(dim=1) + 1e-8).mean()
                
#         output = torch.cat([nn.functional.linear(input[:B*self.steps], self.linear.weight, self.linear.bias),\
#                             nn.functional.linear(input[B*self.steps:], self.linear.weight)])
#         return output


class LLLinear_MS(nn.Module):
    def __init__(self,linear:nn.Linear,**kwargs):
        super(LLLinear_MS,self).__init__()
        self.linear = linear
        self.level = kwargs["level"]
        self.T = kwargs["time_step"]
        self.steps = kwargs["step"]
        self.linear.bias.data = self.linear.bias * self.steps
        self.group_size = 4
        self.param_number = 2
        init_list = []
        for i in range(self.param_number - 1):
            if i < self.steps - 1:
                init_list.append(1/(self.steps))
            else:
                init_list.append(0.0)
        if self.T > 1:
            self.biasAllocator = nn.Parameter(torch.tensor(init_list),requires_grad=True)
        else:
            self.biasAllocator = 1.0
        self.overfireLoss = torch.tensor(0.0)
    
    def forward(self,input):
        B = input.shape[0]//self.T
        biasAllocator = torch.cat([1 - torch.sum(self.biasAllocator,dim=0,keepdim=True), self.biasAllocator], dim=0)
        outputlist = []
        
        # group_size = self.group_size
        
        # if len(input.shape) == 3:        
        #     B1, N, L = input.shape  # 假设 input 是 [B, N, L] 形状
        #     pad = (group_size - (N % group_size)) % group_size  # 需要补齐的通道数

        #     if pad > 0:
        #         # 在 channel 维度上 padding，pad 后形状为 [B, N + pad, L]
        #         input_pad = F.pad(input+0.0, (0, 0, 0, pad))  # (last two are for (dim=-2,-1))

        #     # reshape 成 [B, group_size, num_groups, T]
        #     input_group = input_pad.reshape(B1, group_size, (N + pad) // group_size, L)
            
        #     abs_x = torch.abs(input_group)
        #     s = torch.tanh(10.0 * abs_x)  # approximate nonzero indicator
        #     p_zero = torch.prod(1 - s, dim=1)  # high when all zero
        #     self.overfireLoss = (1 - p_zero).mean()  # encourage all-zero groups
            
            # self.overfireLoss = torch.sqrt((input_group ** 2).sum(dim=1) + 1e-8).mean()
        # print(self.linear.bias.mean().item(),biasAllocator)
        for t in range(self.T):
            if t < self.param_number:
                outputlist.append(nn.functional.linear(input[B*t:B*(t+1)], self.linear.weight, self.linear.bias * biasAllocator[t]))
            else:
                outputlist.append(nn.functional.linear(input[B*t:B*(t+1)], self.linear.weight))
        
        # print(biasAllocator, biasAllocator.sum())
        # print("self.linear.bias",self.linear.bias.data)
        output = torch.cat(outputlist, dim=0)
        return output

class LLConv2d_MS(nn.Module):
    def __init__(self,conv:nn.Conv2d,**kwargs):
        super(LLConv2d_MS,self).__init__()
        self.conv = conv
        self.level = kwargs["level"]
        self.T = kwargs["time_step"]
        self.steps = kwargs["step"]
    
    def forward(self,input):
        B = input.shape[0]//self.T
        # print("LLConv2d_MS.input",input.reshape(torch.Size([self.T,B])+input.shape[1:]).sum(dim=0).abs().mean())
        output = torch.cat([nn.functional.conv2d(input[:B*self.steps], self.conv.weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,groups=self.conv.groups),\
                            nn.functional.conv2d(input[B*self.steps:], self.conv.weight, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,groups=self.conv.groups)],dim=0)
        # print("LLConv2d_MS.output",output.reshape(torch.Size([self.T,B])+output.shape[1:]).sum(dim=0).abs().mean())
        return output


class LLLinear(nn.Module):
    def __init__(self,linear,**kwargs):
        super(LLLinear,self).__init__()
        self.linear = linear
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level//2 - 1
        self.realize_time = self.steps
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        # self.quan_w_fn = self.linear.quan_w_fn
        
    def reset(self):
        # print("LLLinear reset")
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self,input):
        # print("LLLinear", input.mean())
        # print("LLLinear.steps",self.steps)
        x = input
        if x.ndim == 2:
            B,N = x.shape
        elif x.ndim == 3:
            B,C,N = x.shape
        N = self.linear.out_features
        if x.dim() == 3:
            B, N, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, N, D)
        elif x.dim() == 2:
            B, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, D)
        if self.zero_output is None:
            self.zero_output = torch.zeros(size=shape_new,device=x.device,dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            return self.zero_output

        if self.realize_time > 0:
            output = torch.nn.functional.linear(x,self.linear.weight,self.linear.bias/self.steps)
            self.realize_time = self.realize_time - 1
        else:
            output = torch.nn.functional.linear(x,self.linear.weight,None)

        # if self.neuron_type == 'IF':
        #     pass
        # else:
        #     if self.linear.bias is None:
        #         pass
        #     else:
        #         if self.realize_time > 0:
        #             output = output - (self.linear.bias.data.unsqueeze(0) if self.linear.bias is not None else 0.0) * (1 - 1/(self.steps)) 
        #             self.realize_time = self.realize_time - 1
        #         else:
        #             output = output - (self.linear.bias.data.unsqueeze(0) if self.linear.bias is not None else 0.0)

        self.is_work = True
        self.first = False

        return output


class save_module_inout(nn.Module):
    def __init__(self, m, T):
        super().__init__()
        self.m = m
        self.T = T
        self.name = ""
        self.t = 0
        self.accu = []
        self.accu2 = []
        self.accu1 = []
        self.first = False
    
    def forward(self,x):
        if isinstance(self.m[0],Addition):
            dimNum = len(x[0].shape) + 1
        else:
            dimNum = len(x.shape) + 1
        self.t = self.t + 1
        if self.first:
            if isinstance(self.m[0],Addition):
                self.accu.append(x[0][0].unsqueeze(0)+0)
            else:
                self.accu.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                if dimNum == 3:
                    save_fc_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                if dimNum == 4:
                    save_input_for_bin_snn_4dim(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                if dimNum == 5:
                    save_input_for_bin_snn_5dim(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
            if isinstance(self.m[0],Addition):
                self.accu2.append(x[1][0].unsqueeze(0)+0)
                if self.t == self.T:
                    if dimNum == 3:
                        save_fc_input_for_bin_snn(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                    if dimNum == 4:
                        save_input_for_bin_snn_4dim(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                    if dimNum == 5:
                        save_input_for_bin_snn_5dim(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                    del self.accu2
                
        x = self.m(x)
        if self.first:
            self.accu1.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                if dimNum == 3:
                    save_fc_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                if dimNum == 4:
                    save_input_for_bin_snn_4dim(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                if dimNum == 5:
                    save_input_for_bin_snn_5dim(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                self.first = False

                # saving weight and bias
                local_rank = torch.distributed.get_rank()
                if local_rank == 0 and not isinstance(self.m[0],Addition):
                    if hasattr(self.m[0], "quan_w_fn"):
                        torch.save(self.m[0].quan_w_fn(self.m[0].weight),f'{glo.get_value("output_bin_snn_dir")}/{self.name}_weight.pth')
                    else:
                        torch.save(self.m[0].weight,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_weight.pth')
                        
                    if self.m[0].bias is not None:
                        torch.save(self.m[0].bias,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_bias.pth')
                        
                del self.accu
                del self.accu1
        return x

class Attention_no_softmax(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_Relu = nn.ReLU(inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):   
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn)
        attn = torch.clamp(attn/N, max=0.99, min=-0.01)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# class Attention_no_softmax(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.head_dim = head_dim
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.attn_Relu = nn.ReLU(inplace=True)
#         self.proj = nn.Linear(dim, dim)
#         # self.attnBatchnorm = MyBatchNorm1d(dim=dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         # self.scale_size_v = nn.Parameter(torch.tensor([(2*i+1)*1.0 for i in range(197)], requires_grad=False).unsqueeze(0).unsqueeze(0).unsqueeze(-1)) # 1,1,N,1
#         self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=5,
#                         groups=head_dim, padding=5 // 2)
#         # self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.num_heads, 14 * 14 + 1, head_dim)))

#     def forward(self, x):   
#         B, N, C = x.shape
#         input_detype = x.dtype
#         if input_detype == torch.float16:
#             x = x.to(torch.bfloat16)
#         if self.training and input_detype == torch.float16:
#             with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda', enabled=True):
#                 qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
#                 q, k, v = qkv[0], qkv[1], qkv[2]    # make torchscript happy (cannot use tensor as tuple)
#                 q = F.relu6(q)
#                 k = F.relu6(k)
#                 # v = torch.clamp(v,min=-6.0,max=6.0)

#                 attn = (q @ k.transpose(-2, -1)).contiguous() * self.scale / ((float(q.shape[-1]))*36)
#                 attn = self.attn_drop(attn)

#                 x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
#                 classtoken = v[:,:,0,:].unsqueeze(2).contiguous()
#                 feature_map = v[:,:,1:,:].reshape(B*self.num_heads,int(math.sqrt(N-1)),int(math.sqrt(N-1)),self.head_dim).permute(0,3,1,2).contiguous() # B*H,C,N,N
#                 feature_map = torch.cat([classtoken, self.dwc(feature_map).permute(0,2,3,1).reshape(B,self.num_heads,N-1,self.head_dim)],dim=2).transpose(1, 2).reshape(B, N, C).contiguous()
#                 x = x.to(input_detype)
#                 feature_map = feature_map.to(input_detype)
#         else:
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
#             q, k, v = qkv[0], qkv[1], qkv[2]    # make torchscript happy (cannot use tensor as tuple)
#             q = F.relu6(q)
#             k = F.relu6(k)
#             # v = torch.clamp(v,min=-6.0,max=6.0)

#             attn = (q @ k.transpose(-2, -1)).contiguous() * self.scale / ((float(q.shape[-1]))*36)
#             attn = self.attn_drop(attn)

#             x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
#             classtoken = v[:,:,0,:].unsqueeze(2).contiguous()
#             feature_map = v[:,:,1:,:].reshape(B*self.num_heads,int(math.sqrt(N-1)),int(math.sqrt(N-1)),self.head_dim).permute(0,3,1,2).contiguous() # B*H,C,N,N
#             feature_map = torch.cat([classtoken, self.dwc(feature_map).permute(0,2,3,1).reshape(B,self.num_heads,N-1,self.head_dim)],dim=2).transpose(1, 2).reshape(B, N, C).contiguous()

#         x = self.proj(x + feature_map)
#         # x = self.proj(x+self.v_proj(F.relu(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2))).transpose(1, 2))
#         x = self.proj_drop(x)
#         return x


# class Attention_no_softmax(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.head_dim = head_dim
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.attn_Relu = nn.ReLU(inplace=True)
#         self.proj = nn.Linear(dim, dim)
#         self.attnBatchnorm = MyBatchNorm1d(dim=197)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.scale_size_v = nn.Parameter(torch.tensor([(i+1)*1.0 for i in range(197)], requires_grad=False).unsqueeze(0).unsqueeze(0).unsqueeze(-1)) # 1,1,N,1
#         # self.v_proj = nn.Linear(197, 197)

#         # self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=5,
#         #                 groups=head_dim, padding=5 // 2)
#         # self.positional_encoding = nn.Parameter(torch.zeros(size=(1, 14 * 14, dim)))

#     def forward(self, x):   
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#         q = F.relu6(q)
#         k = F.relu6(k)
#         # v = F.relu6(v)
#         # print("q,k,v",q.abs().mean(),k.abs().mean(),v.abs().mean())

#         attn = q @ k.transpose(-2, -1) * self.scale / (q.shape[-1]*36)
#         attn = self.attn_drop(attn)
#         attn = (torch.tril(attn) + torch.tril(attn.transpose(-2, -1)))/2

#         x = (attn @ torch.cumsum(v, dim=-2)/self.scale_size_v).transpose(1, 2).reshape(B, N, C)
#         # classtoken = v[:,:,0,:].unsqueeze(2)
#         # feature_map = v[:,:,1:,:].reshape(B*self.num_heads,int(math.sqrt(N-1)),int(math.sqrt(N-1)),self.head_dim).permute(0,3,1,2) # B*H,C,N,N
#         # feature_map = torch.cat([classtoken, self.dwc(feature_map).permute(0,2,3,1).reshape(B,self.num_heads,N-1,self.head_dim)],dim=2).transpose(1, 2).reshape(B, N, C)
#         # x = self.proj(x+feature_map)
#         # x = self.proj(x+self.v_proj(F.relu(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2))).transpose(1, 2))
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class MyBatchNorm1d_SS(nn.BatchNorm1d):
    def __init__(self, dim, **kwargs):
        super(MyBatchNorm1d_SS, self).__init__(dim, **kwargs)
        self.spike = False
        self.T = 0
        self.step = 0
        self.momentum = 0.1
        self.eps = 1e-5
        self.t = 0
        
    def reset(self):
        self.t = 0
    
    def forward(self,x):
        # self.training = False
        input_shape = len(x.shape)
        if input_shape == 4:
            B,H,N,C = x.shape
            x = x.reshape(B*H,N,C)
        if input_shape == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1,2)
        # if self.spike:
        #     print("before mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
        # else:
        #     print("before mybatchnorm1d:",x.abs().mean())
        self.t = self.t + 1
        if self.t <= self.step:
            x = F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
        else:
            x = F.batch_norm(x,torch.zeros_like(self.running_mean),self.running_var,self.weight,torch.zeros_like(self.bias),self.training,self.momentum,self.eps)
        # if self.spike:
        #     print("after mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
        # else:
        #     print("after mybatchnorm1d:",x.abs().mean())
        x = x.transpose(1,2)
        if input_shape == 2:
            x = x.squeeze(1)
        if input_shape == 4:
            x = x.reshape(B,H,N,C)
        # print("self.running_mean",self.running_mean.abs().mean())
        return x




class MyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, dim, **kwargs):
        super(MyBatchNorm1d, self).__init__(dim, **kwargs)
        self.spike = False
        self.T = 0
        self.step = 0
        self.momentum = 0.1
        self.eps = 1e-5
    
    def forward(self,x):
        # self.training = False
        input_shape = len(x.shape)
        if input_shape == 4:
            B,H,N,C = x.shape
            x = x.reshape(B*H,N,C)
        if input_shape == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1,2)
        # if self.spike:
        #     print("before mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
        # else:
        #     print("before mybatchnorm1d:",x.abs().mean())
        if not self.spike:
            x = F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
        else:
            Fd = x.shape[0]
            if self.step >= self.T:
                x = F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,False,self.momentum,self.eps)
            else:
                x = torch.cat([F.batch_norm(x[:int(Fd*(self.step/self.T))],self.running_mean,self.running_var,self.weight,self.bias,False,self.momentum,self.eps), \
                            F.batch_norm(x[int(Fd*(self.step/self.T)):],torch.zeros_like(self.running_mean),self.running_var,self.weight,torch.zeros_like(self.bias),False,self.momentum,self.eps)])
        # if self.spike:
        #     print("after mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
        # else:
        #     print("after mybatchnorm1d:",x.abs().mean())
        x = x.transpose(1,2)
        if input_shape == 2:
            x = x.squeeze(1)
        if input_shape == 4:
            x = x.reshape(B,H,N,C)
        # print("self.running_mean",self.running_mean.abs().mean())
        return x


class LN2BNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LN2BNorm,self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        self.Eta = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("running_mean",torch.zeros(self.dim))
        self.register_buffer("running_var",torch.ones(self.dim))
        self.Lambda = 1.0
        self.momentum = 0.1
        self.eps = eps
        
    def forward(self,x):
        out_LN = F.layer_norm(x, (self.dim,), self.weight, self.bias)
        out_Identity = x + 0.0
        input_shape = len(x.shape)
        if input_shape == 4:
            B,H,N,C = x.shape
            x = x.reshape(B*H,N,C)
        if input_shape == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1,2)
        out_BN = F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
        out_BN = out_BN.transpose(1,2)
        if input_shape == 2:
            out_BN = out_BN.squeeze(1)
        if input_shape == 4:
            out_BN = out_BN.reshape(B,H,N,C)
        return out_LN*self.Lambda + out_BN*(1 - self.Lambda)
        
class MLP_BN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=MyBatchNorm1d, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])
        self.name = "MLP"

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class MyBachNorm(nn.Module):
    def __init__(self,bn,T):
        super(MyBachNorm,self).__init__()
        # bn.bias.data = bn.bias/T
        # bn.running_mean = bn.running_mean/T
        self.bn = bn
        self.T = T
        self.t = 0
    
    def forward(self,x):
        self.bn.eval()
        if self.t == 0:
            self.bn.train()
        self.t = self.t + 1
        if self.t == self.T:
            self.t = 0
        x = self.bn(x)
        return x
    

class PatchEmbedConv(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.proj_conv = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = norm_layer(embed_dim // 2)
        self.proj_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_ReLU = nn.ReLU(inplace=True)

        self.proj1_conv = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = norm_layer(embed_dim)
        self.proj1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_ReLU = nn.ReLU(inplace=True)

        self.proj2_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = norm_layer(embed_dim)

        self.proj_res_conv = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = norm_layer(embed_dim)
        self.proj_res_ReLU = nn.ReLU(inplace=True)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj_conv(x).reshape(B, self.embed_dim//2, H * W).transpose(1,2).contiguous()
        x = self.proj_bn(x).transpose(1,2).reshape(B, self.embed_dim//2, H, W).contiguous()
        x = self.proj_maxpool(x)
        x = self.proj_ReLU(x)
        
        x_feat = x + 0.0
        x = self.proj1_conv(x).reshape(B, self.embed_dim, (H//2) * (W//2)).transpose(1,2).contiguous()
        x = self.proj1_bn(x).transpose(1,2).reshape(B, self.embed_dim, H//2, W//2).contiguous()
        x = self.proj1_maxpool(x)
        x = self.proj1_ReLU(x)

        x = self.proj2_conv(x).reshape(B, self.embed_dim, (H//4) * (W//4)).transpose(1,2).contiguous()
        x = self.proj2_bn(x).transpose(1,2).reshape(B, self.embed_dim, H//4, W//4).contiguous()

        x_feat = self.proj_res_conv(x_feat).reshape(B, self.embed_dim, (H//4) * (W//4)).transpose(1,2).contiguous()
        x_feat = self.proj_res_bn(x_feat).transpose(1,2).reshape(B, self.embed_dim, H//4, W//4).contiguous()

        x = self.proj_res_ReLU(x + x_feat)    
    
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)
        return x

class PatchMergingConv(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm): 
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.proj1_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = norm_layer(dim)
        self.proj1_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_ReLU = nn.ReLU(inplace=True)

        self.proj2_conv = nn.Conv2d(dim, 2*dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = norm_layer(2*dim)

        self.proj_res_conv = nn.Conv2d(dim, 2*dim, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = norm_layer(2*dim)
        self.proj_res_ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # B C H W
        x_feat = x + 0.0

        x = self.proj1_conv(x).reshape(B, C, H * W).transpose(1,2).contiguous()
        x = self.proj1_bn(x).transpose(1,2).reshape(B, C, H, W).contiguous()
        x = self.proj1_maxpool(x)
        x = self.proj1_ReLU(x)

        x = self.proj2_conv(x).reshape(B, 2*C, (H//2) * (W//2)).transpose(1,2).contiguous()
        x = self.proj2_bn(x).transpose(1,2).reshape(B, 2*C, H//2, W//2).contiguous()

        x_feat = self.proj_res_conv(x_feat).reshape(B, 2*C, (H//2) * (W//2)).transpose(1,2).contiguous()
        x_feat = self.proj_res_bn(x_feat).transpose(1,2).reshape(B, 2*C, H//2, W//2).contiguous()

        x = self.proj_res_ReLU(x_feat + x)
        return x.reshape(B, 2*C, (H//2) * (W//2)).transpose(1,2).contiguous()

class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)
        self.running_mean = None
        self.running_var = None
        self.momentum = 0.9
        self.eps = 1e-6
    
    def forward(self,x):        
        if self.training:
            if self.running_mean is None:
                self.running_mean = nn.Parameter((1-self.momentum) * x.mean([-1], keepdim=True),requires_grad=False)
                self.running_var = nn.Parameter((1-self.momentum) * x.var([-1], keepdim=True),requires_grad=False)
            else:
                self.running_mean.data = (1-self.momentum) * x.mean([-1], keepdim=True) + self.momentum * self.running_mean # mean: [1, max_len, 1]
                self.running_var.data = (1-self.momentum) * x.var([-1], keepdim=True) + self.momentum * self.running_var # std: [1, max_len, 1]
            return self.weight * (x - self.running_mean) / (self.running_var + self.eps) + self.bias
        else:
            # if self.running_mean is None:
            self.running_mean = nn.Parameter(x.mean([-1], keepdim=True),requires_grad=False)
            self.running_var = nn.Parameter(x.var([-1], keepdim=True),requires_grad=False)
            running_mean = self.running_mean
            running_var = self.running_var
            return self.weight * (x) / (running_var + self.eps).sqrt() + self.bias    
        # 注意这里也在最后一个维度发生了广播

def save_input_for_bin_snn_5dim(input,dir,name):
    T,B,L1,L2,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_L1={L1}_L2={L2}_N={N}.pth')

    
def save_input_for_bin_snn_4dim(input,dir,name):
    T,B,L,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_L={L}_N={N}.pth')
    
def save_fc_input_for_bin_snn(input,dir,name):
    T,B,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_N={N}.pth')

