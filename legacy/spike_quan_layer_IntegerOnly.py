import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
import numpy as np
import scipy
import glo
from PowerNorm import MaskPowerNorm
from timm.models.vision_transformer import Block


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def get_M_N(s):
    T = s + 0.0
    N = torch.tensor(0)
    while(torch.abs(T)<128):
        N += 1
        T *= 2
    return N,torch.round(T)

def get_M1_N1_M2_N2(s1,s2):
    T1 = s1 + 0.0
    N = torch.tensor(0)
    T2 = s2 + 0.0
    while(torch.abs(T1)<128 or torch.abs(T2)<128):
        N += 1
        T1 *= 2
        T2 *= 2
    return N,torch.round(T1),torch.round(T2)


def get_multi_M_N(slist):
    NList = []
    MList = []
    for sIndex in range(slist.shape[0]):
        s = slist[sIndex]
        T = s + 0.0
        N = torch.tensor(0)
        while(torch.abs(T)<128):
            N += 1
            T *= 2        
        NList.append(N)
        MList.append(torch.round(T))
    return torch.tensor(NList), torch.tensor(MList)
    
    
def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def reshape_to_activation(inputs):
    return inputs.reshape(1, -1, 1, 1)

def reshape_to_weight(inputs):
    return inputs.reshape(-1, 1, 1, 1)

def reshape_to_bias(inputs):
    return inputs.reshape(-1)


def save_input_for_bin(input,dir,name):
    B,C,H,W = input.shape
    local_rank = torch.distributed.get_rank()
    
    if local_rank == 0:
        input_list = input.tolist()
        input_binfile = open(f'{dir}/input_{name}_B={B}_C={C}_H={H}_W={W}.bin','wb')
        for i in range(B):
            for j in range(C):
                for n in range(H):
                    for m in range(W):
                        input_binfile.write(int(round(float(input_list[i][j][n][m]))).to_bytes(length=1,byteorder='big',signed =True))
        input_binfile.close()
    
def save_fc_input_for_bin(input,dir,name):
    B,N = input.shape
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        input_list = input.tolist()
        input_binfile = open(f'{dir}/input_{name}_B={B}_N={N}.bin','wb')
        for i in range(B):
            for j in range(N):
                input_binfile.write(int(round(float(input_list[i][j]))).to_bytes(length=1,byteorder='big',signed =True))
        input_binfile.close()


class LsqQuan(t.nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__()
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(torch.tensor(1.0),requires_grad=True)

    def __repr__(self):
        return f"LsqQuan(thd_pos={self.thd_pos}, thd_neg={self.thd_neg}, s={self.s.data}, per_channel={self.per_channel})"

    
    def init_from(self, x, *args, **kwargs):
        # threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.thd_pos), n_trial=300, eps=1e-10)
        # self.s.data = torch.tensor(threshold / (self.thd_pos),dtype=torch.float32).cuda()
        
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x, clip=True):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(self.s, s_grad_scale)
        # print(s_scale,s_scale.grad)
        # print("self.thd_neg",self.thd_neg, "self.thd_pos", self.thd_pos)
        x = x / s_scale
        if clip:
            x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x

class QuanAvgPool(t.nn.Module):
    def __init__(self,m, quan_out_fn):
        super(QuanAvgPool,self).__init__()
        assert isinstance(m, t.nn.AvgPool2d) or isinstance(m, t.nn.AdaptiveAvgPool2d), "average pooling!!!"
        self.m = m
        self.quan_out_fn = quan_out_fn
        self.is_init = False
    def forward(self,x):
        if self.is_init == False:
            x = self.m(x)
            self.quan_out_fn.init_from(x)
            self.is_init = True
            return x

        x = self.m(x)
        x = self.quan_out_fn(x)
        # print("train AvgPool output:",(x/self.quan_out_fn.s).abs()[0,0,0,:])
        return x
        
class QuanInferAvgPool(t.nn.Module):
    def __init__(self,m:QuanAvgPool, last_quan_out_fn, name):
        super(QuanInferAvgPool,self).__init__()
        self.m = m
        self.last_quan_out_fn = last_quan_out_fn
        self.kernel_size = self.m.m.kernel_size
        self.s = last_quan_out_fn.s/(m.quan_out_fn.s*self.kernel_size*self.kernel_size)
        self.thd_pos = m.quan_out_fn.thd_pos
        self.thd_neg = m.quan_out_fn.thd_neg
        self.name = name
        self.first = True
        N,M = get_M_N(self.s)

        if torch.is_tensor(M):
            self.M = t.nn.Parameter(torch.tensor(M.item(),dtype=int),requires_grad=False)
        else:
            self.M = t.nn.Parameter(torch.tensor(M,dtype=int),requires_grad=False)
        if torch.is_tensor(N):
            self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        else:
            self.N = t.nn.Parameter(torch.tensor(N,dtype=int),requires_grad=False)

        # *self.m.kernel_size*self.m.kernel_size
    def forward(self,x):
        if self.first:
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"in")        

        # print("Infer: pooling",self.m.m(x).mean())
        # print("Infer: quantized pooling",((float(self.M)/float(2**self.N))*self.m.m(x))[0:4,0,0,:])
        x = torch.round((float(self.M)/float(2**self.N))*(self.m.m(x)*self.kernel_size*self.kernel_size))
        x = torch.clip(x,max=self.thd_pos,min=self.thd_neg)
        # print("Infer: quantized output",x[0:4,0,0,:])
        # print("Infer: quantized output",x.mean())

        if self.first:
            self.first = False
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"out")
        # print("Infer AvgPool output:", x.abs()[0,0,0,:])
        
        return x

class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None, quan_out_fn=None,is_first=False):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_out_fn = quan_out_fn
        self.quan_a_fn = quan_a_fn
        
        self.weight = t.nn.Parameter(m.weight.detach())
        # print(self.weight.mean())
        self.quan_w_fn.init_from(m.weight)
        self.quan_out_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())
        else:
            self.bias = None
        self.is_init = False
        self.is_first = is_first
        if self.is_first:
            self.quan_a_fn.init_from(m.weight)
        # self.l1_loss = 0
        self.l2_loss = 0
        self.absoluteValue = 0

    def forward(self, x):

        if self.is_init == False:
            if self.is_first:
                self.quan_a_fn.init_from(x)
            out = self._conv_forward(x, self.weight,bias = self.bias)
            self.quan_out_fn.init_from(out)
            self.is_init = True
            return out

        quantized_weight = self.quan_w_fn(self.weight)
                                
        if self.is_first:
            x = self.quan_a_fn(x)

        out = self._conv_forward(x, quantized_weight,bias = None)        
        quantized_bias = self.quan_out_fn(self.bias)
        quantized_out = torch.clip(self.quan_out_fn(out,clip=False) + quantized_bias.reshape(1,-1,1,1),min=self.quan_out_fn.s*self.quan_out_fn.thd_neg,max=self.quan_out_fn.s*self.quan_out_fn.thd_pos)
        self.l2_loss = (torch.nn.functional.relu(quantized_out)/self.quan_out_fn.s - self.quan_out_fn.thd_pos).sum()
        self.absoluteValue = torch.abs(quantized_out.detach()/self.quan_out_fn.s.detach()).sum().item()
        
        return quantized_out


class QuanInferConv2d(t.nn.Conv2d):
    def __init__(self, m: QuanConv2d, last_quan_out_fn, is_first=False, name="act"):
        assert type(m) == QuanConv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=True if m.bias is not None else False,
                    padding_mode=m.padding_mode)

        self.is_first = is_first
        self.m = m  
        self.name = name 
        self.last_quan_out_fn = last_quan_out_fn
        self.thd_neg = m.quan_out_fn.thd_neg
        self.thd_pos = m.quan_out_fn.thd_pos

        self.weight.data = m.weight.cuda()
        # print(self.weight.mean())

        self.weight = t.nn.Parameter(m.quan_w_fn(self.weight.detach())/m.quan_w_fn.s,requires_grad=False)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.quan_out_fn(m.bias.detach())/m.quan_out_fn.s,requires_grad=False)
        else:
            self.bias = None
            #print(self.bias)

        if is_first:
            s = m.quan_a_fn.s*m.quan_w_fn.s/m.quan_out_fn.s
        else:
            s = last_quan_out_fn.s*m.quan_w_fn.s/m.quan_out_fn.s
        
        N,M = get_M_N(s)

        if torch.is_tensor(M):
            self.M = t.nn.Parameter(torch.tensor(M.item(),dtype=int),requires_grad=False)
        else:
            self.M = t.nn.Parameter(torch.tensor(M,dtype=int),requires_grad=False)
        if torch.is_tensor(N):
            self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        else:
            self.N = t.nn.Parameter(torch.tensor(N,dtype=int),requires_grad=False)
        self.s = s
        self.first = True
    
    def forward(self,x):
        if self.is_first:
            x = self.m.quan_a_fn(x)/self.m.quan_a_fn.s
        input = x + 0.0
        if self.first:
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"in")        
        wx = self._conv_forward(x, self.weight,bias=None)
        wx = torch.round((float(self.M)/float(2**self.N))*wx)
        if self.bias is not None:
            x = wx + self.bias.reshape(1,-1,1,1)
        x = torch.clip(x,max=self.thd_pos,min=self.thd_neg)
        if self.first:
            self.first = False
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"out")

        return x



class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_out_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_out_fn = quan_out_fn
        self.m = m

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        self.quan_out_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

        self.is_init = False
        self.l2_loss = 0.0
        self.absoluteValue = 0.0
        
    def forward(self, x):
        if self.is_init == False:
            out = t.nn.functional.linear(x, self.weight, bias=None) + self.bias.reshape(1,-1)
            self.quan_out_fn.init_from(out)
            self.is_init = True
            return out
        
        quantized_weight = self.quan_w_fn(self.weight)
        out = t.nn.functional.linear(x, quantized_weight, bias=None)
        if self.bias is not None:
            quantized_bias = self.quan_out_fn(self.bias)
        else:
            quantized_bias = None
        quantized_out = self.quan_out_fn(out,clip=False) + quantized_bias.reshape(1,-1)
        quantized_out = torch.clip(quantized_out,min=self.quan_out_fn.s*self.quan_out_fn.thd_neg,max=self.quan_out_fn.s*self.quan_out_fn.thd_pos)

        self.l2_loss = 0
        self.absoluteValue = torch.abs(quantized_out.detach()/self.quan_out_fn.s.detach()).sum().item()
        return quantized_out

class QuanInferLinear(t.nn.Linear):
    def __init__(self, m: QuanLinear, last_quan_out_fn, is_first=False, name="act"):
        assert type(m) == QuanLinear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.m = m
        self.is_first = is_first
        self.first = True
        self.name = name
        self.last_quan_out_fn = last_quan_out_fn
        self.thd_neg = m.quan_out_fn.thd_neg
        self.thd_pos = m.quan_out_fn.thd_pos
        self.weight = t.nn.Parameter(m.quan_w_fn(m.weight.detach())/m.quan_w_fn.s,requires_grad=False)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.quan_out_fn(m.bias.detach())/m.quan_out_fn.s,requires_grad=False)
        s = last_quan_out_fn.s*m.quan_w_fn.s/m.quan_out_fn.s
        N,M = get_M_N(s)

        if torch.is_tensor(M):
            self.M = t.nn.Parameter(torch.tensor(M.item(),dtype=int),requires_grad=False)
        else:
            self.M = t.nn.Parameter(torch.tensor(M,dtype=int),requires_grad=False)
        if torch.is_tensor(N):
            self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        else:
            self.N = t.nn.Parameter(torch.tensor(N,dtype=int),requires_grad=False)

        self.s_b = m.quan_out_fn.s
    
    def forward(self,x):
        if self.first:
            save_fc_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name)

        out = torch.round((float(self.M)/float(2**self.N))*t.nn.functional.linear(x, self.weight)) + self.bias
        out = torch.clip(out,max=self.thd_pos,min=self.thd_neg)
        if self.first:
            self.first = False
            save_fc_input_for_bin(out[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name)

        return out

class QAttention(nn.Module):

    def __init__(
            self,
            dim,
            quan_qkv_weight:LsqQuan,
            quan_proj_weight:LsqQuan,
            quan_q:LsqQuan,
            quan_k:LsqQuan,
            quan_v:LsqQuan,
            quan_proj:LsqQuan,
            attn_quan:LsqQuan,
            after_attn_quan:LsqQuan,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            bit = 4,
            is_softmax = False,
            
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bit = bit
        self.is_softmax = is_softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.quan_qkv_weight = quan_qkv_weight
        self.quan_qkv_weight.init_from(self.qkv.weight)

        self.quan_q = quan_q
        self.quan_k = quan_k
        self.quan_v = quan_v
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim, bias=True)
        self.quan_proj_weight = quan_proj_weight
        self.quan_proj_weight.init_from(self.proj.weight)

        self.quan_proj = quan_proj
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_quan = attn_quan
        self.after_attn_quan = after_attn_quan
        self.is_init = False
        self.relu = nn.ReLU()
        self.tokenLength = 0
        
    def forward(self, x):
        if self.is_init == False:
            B, N, C = x.shape
            self.tokenLength = N
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            # q, k = self.q_norm(q), self.k_norm(k)
            self.quan_q.init_from(q)
            self.quan_k.init_from(k)
            self.quan_v.init_from(v)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.is_softmax:
                attn = attn.softmax(dim=-1)
                self.attn_quan.init_from(attn)
            else:
                # print("no softmax!!!!")
                attn = attn/N
                self.attn_quan.init_from(attn)
                attn = self.relu(attn)
                
            attn = self.attn_drop(attn)
            x = attn @ v
            self.after_attn_quan.init_from(x)

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            self.quan_proj.init_from(x)
            self.is_init = True
            
            return x
        else:
            B, N, C = x.shape
            self.tokenLength = N

            quantized_qkv_weight = self.quan_qkv_weight(self.qkv.weight)
            qkv = t.nn.functional.linear(x, quantized_qkv_weight, bias=None).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            q = q * self.scale
            self.q1 = q
            quantized_q = self.quan_q(q)
            quantized_k = self.quan_k(k)
            quantized_v = self.quan_v(v)
            self.q = quantized_q/self.quan_q.s
            self.k = quantized_k/self.quan_k.s
            self.v = quantized_v/self.quan_v.s
            attn = quantized_q @ quantized_k.transpose(-2, -1)
            if self.is_softmax:
                attn = attn.softmax(dim=-1)
                attn = self.attn_quan(attn)
            else:
                # print("no softmax!!!!")
                attn = self.attn_quan(attn/N)
                attn = self.relu(attn)
            
            self.attn = attn/self.attn_quan.s
            x = attn @ quantized_v
            x = self.after_attn_quan(x)

            x = x.transpose(1, 2).reshape(B, N, C)

            quantized_proj_weight = self.quan_proj_weight(self.proj.weight)
            x = t.nn.functional.linear(x, quantized_proj_weight, bias=None)

            if self.proj.bias is not None:
                quantized_bias = self.quan_proj(self.proj.bias)
            else:
                quantized_bias = None
            quantized_out = self.quan_proj(x,clip=False) + quantized_bias.reshape(1,-1)
            quantized_out = torch.clip(quantized_out,min=self.quan_proj.s*self.quan_proj.thd_neg,max=self.quan_proj.s*self.quan_proj.thd_pos)
            quantized_out = self.quan_proj(quantized_out)
            
            return quantized_out

class QInferAttention(nn.Module):

    def __init__(
            self,
            m:QAttention,
            last_act_quan:LsqQuan,
            name="attention"
    ):
        super().__init__()
        self.last_act_quan = last_act_quan

        # Initialize the M,N for QKV
        self.neuron_q_N,self.neuron_q_M = get_M_N(last_act_quan.s*m.quan_qkv_weight.s*m.scale/m.quan_q.s)
        self.neuron_k_N,self.neuron_k_M = get_M_N(last_act_quan.s*m.quan_qkv_weight.s/m.quan_k.s)
        self.neuron_v_N,self.neuron_v_M = get_M_N(last_act_quan.s*m.quan_qkv_weight.s/m.quan_v.s)
        
        self.neuron_q_N = t.nn.Parameter(torch.tensor(self.neuron_q_N.item(),dtype=int),requires_grad=False)
        self.neuron_q_M = t.nn.Parameter(torch.tensor(self.neuron_q_M.item(),dtype=int),requires_grad=False)
                
        self.neuron_k_N = t.nn.Parameter(torch.tensor(self.neuron_k_N.item(),dtype=int),requires_grad=False)
        self.neuron_k_M = t.nn.Parameter(torch.tensor(self.neuron_k_M.item(),dtype=int),requires_grad=False)
        
        self.neuron_v_N = t.nn.Parameter(torch.tensor(self.neuron_v_N.item(),dtype=int),requires_grad=False)
        self.neuron_v_M = t.nn.Parameter(torch.tensor(self.neuron_v_M.item(),dtype=int),requires_grad=False)
        
        # Initialize the qkv weight
        self.qkv_weight = t.nn.Parameter(m.quan_qkv_weight(m.qkv.weight.detach())/m.quan_qkv_weight.s,requires_grad=False)
        
        # Initialize the M,N for q,k multiplication
        self.neuron_attn_N, self.neuron_attn_M = get_M_N(m.quan_q.s*m.quan_k.s/(m.attn_quan.s*m.tokenLength))
                
        # Initialize the M,N for attention 
        self.neuron_after_attn_N, self.neuron_after_attn_M = get_M_N(m.attn_quan.s*m.quan_v.s/(m.after_attn_quan.s))
        
        # Initialize the M,N for projection layer
        self.neuron_proj_N, self.neuron_proj_M = get_M_N(m.after_attn_quan.s*m.quan_proj_weight.s/(m.quan_proj.s))

        # Initialize the proj weight and bias
        self.proj_weight = t.nn.Parameter(m.quan_proj_weight(m.proj.weight.detach())/m.quan_proj_weight.s,requires_grad=False)
        self.proj_bias = t.nn.Parameter(m.quan_proj(m.proj.bias.detach())/m.quan_proj.s,requires_grad=False)

        # Define other tools
        self.relu = nn.ReLU()
        self.name = name
        self.num_heads = m.num_heads
        self.head_dim = m.head_dim
        self.thd_pos = m.quan_proj.thd_pos
        self.thd_neg = m.quan_proj.thd_neg
        self.first = True
        
    def forward(self, x):
        # if self.first:
        #     save_fc_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"_input")

        B, N, C = x.shape

        # calculate the qkv (int16)
        qkv = t.nn.functional.linear(x, self.qkv_weight).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  
        
        # quantize the q,k,v     
        self.q1 = q
        q = torch.round((np.float64(self.neuron_q_M)/np.float64(2**self.neuron_q_N))*q)
        k = torch.round((np.float64(self.neuron_k_M)/np.float64(2**self.neuron_k_N))*k)
        v = torch.round((np.float64(self.neuron_v_M)/np.float64(2**self.neuron_v_N))*v)
        self.q = q
        self.k = k
        self.v = v
        # if self.first:
        #     save_fc_input_for_bin(q[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"_q")
        #     save_fc_input_for_bin(k[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"_k")
        #     save_fc_input_for_bin(v[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"_v")
        
        # calculate multiplication between q,k
        attn = q @ k.transpose(-2, -1)
        attn = torch.round((np.float64(self.neuron_attn_M)/np.float64(2**self.neuron_attn_N))*attn)
        attn = self.relu(attn)
        self.attn = attn
        # if self.first:
        #     save_fc_input_for_bin(attn[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"_qkmulti")
        
        # calculate attention
        x = attn @ v
        x = torch.round((np.float64(self.neuron_after_attn_M)/np.float64(2**self.neuron_after_attn_N))*x)
        x = x.transpose(1, 2).reshape(B, N, C)        
        # if self.first:
        #     save_fc_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"_attn")
        
        # calculate the projection
        x = torch.round((np.float64(self.neuron_proj_M)/np.float64(2**self.neuron_proj_N))*t.nn.functional.linear(x, self.proj_weight)) + self.proj_bias
        x = torch.clip(x,max=self.thd_pos,min=self.thd_neg)        
        # if self.first:
        #     save_fc_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"_proj")
        #     self.first = False

        return x



class PowerNormQuan(nn.Module):
    def __init__(
            self,
            m:MaskPowerNorm,
            quan_fn:LsqQuan,
    ):
        super().__init__()
        self.m = m
        self.num_features = m.num_features
        self.quan_fn = quan_fn
        self.thd_pos = self.quan_fn.thd_pos
        self.thd_neg = self.quan_fn.thd_neg
        self.is_init = False
    
    def forward(self,x):
        shaped_input = (len(x.shape) == 2)
        if shaped_input:
            x = x.unsqueeze(0)

        x = x.transpose(0,1)

        x = self.m.gp(x)
        
        x = x.permute(1, 2, 0).contiguous()
        input_shape = x.size()
        x = x.reshape(x.size(0), self.num_features, -1)
        x = x.unsqueeze(-1)        
        
        N, C, H, W = x.size()
        var = self.m.running_phi
        if not self.is_init:
            x = x / (var + self.m.eps).sqrt()
            x = self.m.weight.reshape(1,C,1,1) * x
            output = x+self.m.bias.reshape(1,C,1,1)
            self.quan_fn.init_from(output)
            self.is_init = True
        else:
            x = self.quan_fn(self.m.weight.reshape(1,C,1,1) * x/(var + self.m.eps).sqrt(),clip=False)
            bias = self.quan_fn(self.m.bias)            
            output = torch.clip(x+bias.reshape(1,C,1,1),self.thd_neg*self.quan_fn.s,self.thd_pos*self.quan_fn.s)        
        
        output = output.reshape(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        # Reshape it.
        if shaped_input:
            output = output.squeeze(0)

        # add by SpikeZIP-TF: output T x B x C -> B x T x C
        output = output.transpose(0,1)

        return output
            

class PowerNormInfer(nn.Module):
    def __init__(
            self,
            m:PowerNormQuan,
            last_act_quan:LsqQuan,
            name="PowerNorm"
    ):
        super().__init__()
        self.running_phi = m.m.running_phi
        self.eps = m.m.eps
        self.weight = m.m.weight
        self.bias = m.m.bias
        self.last_act_quan = last_act_quan

        self.num_features = m.num_features
        var = self.running_phi
    
        self.N1,self.M1 = get_multi_M_N(last_act_quan.s*self.weight/((var.reshape(-1) + self.eps).sqrt()*m.quan_fn.s))

        self.N1 = t.nn.Parameter(self.N1.int(),requires_grad=False)
        self.M1 = t.nn.Parameter(self.M1.int(),requires_grad=False)
        
        self.bias = t.nn.Parameter(m.quan_fn(self.bias)/m.quan_fn.s,requires_grad=False)
                
        self.thd_pos = m.quan_fn.thd_pos    
        self.thd_neg = m.quan_fn.thd_neg    
    def forward(self,x):
        shaped_input = (len(x.shape) == 2)
        if shaped_input:
            x = x.unsqueeze(0)        

        # reshape for x1
        x = x.transpose(0,1)
        x = x.permute(1, 2, 0).contiguous()
        input_shape = x.size()
        x = x.reshape(x.size(0), self.num_features, -1)
        x = x.unsqueeze(-1)
        N, C, H, W = x.size()


        output = torch.round(x*self.M1.reshape(1,C,1,1)/2**self.N1.reshape(1,C,1,1)) + self.bias.reshape(1,C,1,1)        
        output = torch.clip(output,self.thd_neg,self.thd_pos)
        
        output = output.reshape(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        # Reshape it.
        if shaped_input:
            output = output.squeeze(0)

        # add by SpikeZIP-TF: output T x B x C -> B x T x C
        output = output.transpose(0,1)

        return output        
    

class AdditionQuan(nn.Module):
    def __init__(
        self,
        quan_a_fn:LsqQuan,
    ):
        super().__init__()
        self.quan_a_fn = quan_a_fn
        self.is_init = False
        self.thd_pos = quan_a_fn.thd_pos
        self.thd_neg = quan_a_fn.thd_neg
    
    def forward(self,x1,x2):
        if self.is_init == False:
            x = x1 + x2
            self.quan_a_fn.init_from(x)
            self.is_init = True
            return x
        else:
            return self.quan_a_fn(x1+x2)

class AdditionInfer(nn.Module):
    def __init__(
        self,
        m:AdditionQuan,
        quan_input1_fn:LsqQuan,
        quan_input2_fn:LsqQuan,
    ):
        super().__init__()

        s1 = quan_input1_fn.s/m.quan_a_fn.s
        s2 = quan_input2_fn.s/m.quan_a_fn.s
        N,M1,M2 = get_M1_N1_M2_N2(s1,s2)
        self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        self.M1 = t.nn.Parameter(torch.tensor(M1.item(),dtype=int),requires_grad=False)
        self.M2 = t.nn.Parameter(torch.tensor(M2.item(),dtype=int),requires_grad=False)
        self.thd_neg = m.thd_neg
        self.thd_pos = m.thd_pos

    def forward(self,x1,x2):
        x = torch.round((x1*self.M1+x2*self.M2)/2**self.N)
        x = torch.clip(x,self.thd_neg,self.thd_pos)
        return x

def myquan_replace_integerOnly(model,act_bit,weight_bit=8, is_softmax = True):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model,act_bit,weight_bit):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, Block):
                # print(children)
                qattn = QAttention(dim=child.attn.num_heads*child.attn.head_dim,num_heads=child.attn.num_heads,is_softmax=is_softmax,bit=act_bit, \
                                    quan_qkv_weight=LsqQuan(weight_bit, all_positive=False, symmetric=False, per_channel=False), \
                                    quan_proj_weight=LsqQuan(weight_bit, all_positive=False, symmetric=False, per_channel=False), \
                                    quan_q=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False), \
                                    quan_k=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False), \
                                    quan_v=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False), \
                                    quan_proj=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False), \
                                    attn_quan=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False), \
                                    after_attn_quan=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False))

                qattn.qkv = child.attn.qkv
                # qattn.q_norm = child.q_norm
                # qattn.k_norm = child.k_norm
                qattn.attn_drop = child.attn.attn_drop
                qattn.proj = child.attn.proj
                qattn.proj_drop = child.attn.proj_drop
                
                model._modules[name].attn = qattn
                model._modules[name].addition1 = AdditionQuan(quan_a_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False))
                model._modules[name].addition2 = AdditionQuan(quan_a_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False))
                # model._modules[name].act1 = MyQuan(level, sym=True)
                # model._modules[name].act2 = MyQuan(level, sym=True)
                model._modules[name].norm1 = PowerNormQuan(m=child.norm1,quan_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False))
                model._modules[name].norm2 = PowerNormQuan(m=child.norm2,quan_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False))
                model._modules[name].mlp.fc1 = QuanLinear(m=child.mlp.fc1, quan_out_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False), quan_w_fn=LsqQuan(weight_bit, all_positive=False, symmetric=False, per_channel=False))
                model._modules[name].mlp.fc2 = QuanLinear(m=child.mlp.fc2, quan_out_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False), quan_w_fn=LsqQuan(weight_bit, all_positive=False, symmetric=False, per_channel=False))
                print("index",cur_index,"myquan replace finish!!!!")
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child,quan_w_fn=LsqQuan(weight_bit, all_positive=False, symmetric=False, per_channel=False), \
                                                 quan_out_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False), \
                                                 quan_a_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False),is_first=(cur_index==0))
                is_need = True
                cur_index = cur_index + 1
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child, quan_out_fn=LsqQuan(weight_bit, all_positive=False, symmetric=False, per_channel=False), quan_w_fn=LsqQuan(weight_bit, all_positive=False, symmetric=False, per_channel=False))
                is_need = True
                cur_index = cur_index + 1
            elif isinstance(child, nn.LayerNorm) or isinstance(child, MaskPowerNorm):
                model._modules[name] = PowerNormQuan(m=child,quan_fn=LsqQuan(act_bit, all_positive=False, symmetric=False, per_channel=False))
                is_need = True
            if not is_need:
                _myquan_replace(child,act_bit,weight_bit)

    get_index(model)
    _myquan_replace(model,act_bit,weight_bit)


