import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch._dynamo import allow_in_graph
except Exception:
    def allow_in_graph(fn):
        return fn

import glo
from snn.nvtx import nvtx_range
from .quant import MyQuan
from .softmax import spiking_softmax, spiking_softmax_ss
from .st_bifneuron_ms import ST_BIFNeuron_MS
from .st_bifneuron_ss import ST_BIFNeuron_SS
from .record import save_input_for_bin_snn_4dim
from .dyt import DyHT_Softmax


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
        self.attn_quan.s_max.data = torch.tensor(1.0/self.attn_quan.pos_max)
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
        with nvtx_range("snn.layer.attention.QAttention_without_softmax.forward"):
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
        # self.quan_proj = MyQuan(self.level,sym=True)
        
    def forward(self, x):
        with nvtx_range("snn.layer.attention.QAttention.forward"):
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
            # x = self.quan_proj(x)

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

@allow_in_graph
def multi(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return (x1_t + x1_sum_t) @ x2_t.transpose(-2, -1) + x1_t @ x2_sum_t.transpose(-2, -1)

@allow_in_graph
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
        self.is_single_step = issubclass(neuron_layer, ST_BIFNeuron_SS) if isinstance(neuron_layer, type) else isinstance(neuron_layer, ST_BIFNeuron_SS)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True, T=T, C=dim)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True, T=T, C=dim)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=True, T=T, C=dim)
        # self.spikeBN_q = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(self.head_dim),level=self.level//2-1,input_allcate=False)
        # self.spikeBN_k = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(self.head_dim),level=self.level//2-1,input_allcate=False)
        # self.spikeBN_v = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(self.head_dim),level=self.level//2-1,input_allcate=False)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.attn_ReLU = nn.ReLU()
        self.attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False, need_spike_tracer=not is_softmax, T=T, C=dim)
        # self.attn_IF.prefire.data = torch.tensor(0.2)
        # self.spikeBN_attn = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(197),level=self.level,input_allcate=False)
        # if self.is_softmax:
        self.attn_softmax_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, need_spike_tracer=is_softmax, T=T, C=dim)
        self.attn_softmax_IF.prefire.data = torch.tensor(0.2)
        # self.spikeBN_after_attn = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(self.head_dim),level=self.level//2-1,input_allcate=False)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True, T=T, C=dim)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.spikeBN_proj = spiking_BatchNorm2d(bn=torch.nn.BatchNorm1d(dim),level=self.level//2-1,input_allcate=False)
        # self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        if self.is_softmax:
            if self.is_single_step:
                self.Ssoftmax = spiking_softmax_ss(self.level//2 - 1, T)
            else:
                self.Ssoftmax = spiking_softmax(self.level//2 - 1, T)
        self.T = T
        # self.Release_attn1 = Release_attn(self.level//2 - 1)
        # self.Release_attn2 = Release_attn(self.level//2 - 1)

        # saving mid feature
        self.t = 0
        self.first = False        
        self.name = ""

    def reset(self):
        # print("SAttention reset")
        self.q_IF.reset()
        self.k_IF.reset()
        self.v_IF.reset()
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
        with nvtx_range("snn.layer.attention.SAttention.forward"):
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
            q_if = self.q_IF
            k_if = self.k_IF
            v_if = self.v_IF
            q = q_if(q)
            k = k_if(k)
            v = v_if(v)

            scale = self.scale
            q = q * scale
            q_acc = q_if.acc_q * q_if.q_threshold * scale
            k_acc = k_if.acc_q * k_if.q_threshold
            v_acc = v_if.acc_q * v_if.q_threshold
            q_det = q.detach()
            k_det = k.detach()
            v_det = v.detach()
            attn = multi(q, k, q_acc - q_det, k_acc - k_det)

            if self.is_softmax:
                attn = self.Ssoftmax(attn)
                attn_if = self.attn_softmax_IF
                # attn_print = attn.reshape(4,B//4,self.num_heads,N,N)
                # for t in range(4):
                #     print(f"ST_BIFNeuron_MS attn_print[{t}].abs().mean()",attn_print[t].abs().mean(),attn_print.dtype)
            else:
                attn_if = self.attn_IF

            attn = attn_if(attn)
            if not self.is_softmax:
                attn = attn * (1.0 / float(N))

            attn = self.attn_drop(attn)

            attn_acc = attn_if.acc_q * attn_if.q_threshold
            attn_det = attn.detach()
            x = multi1(attn, v, attn_acc - attn_det, v_acc - v_det)
            # x_print = x.reshape(4,B//4,self.num_heads,N,self.head_dim)
            # for t in range(4):
            #     print(f"ST_BIFNeuron_MS x_print[{t}].abs().mean()",x_print[t].abs().mean(),x_print.dtype)

            x = self.after_attn_IF(x)

            x = x.transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            # print("after proj",x.abs().mean())
            x = self.proj_drop(x)
            # x = self.spikeBN_proj(x)
            # print("after spikeBN_proj",x.abs().mean())
            # x = self.proj_IF(x)
            # print("after proj_IF",x.abs().mean())

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
        self.attn_IF.prefire.data = torch.tensor(0.2)
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
        with nvtx_range("snn.layer.attention.SAttention_without_softmax.forward"):
            self.t = self.t + 1
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3 B self.num_heads N self.head_dim
            # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())
            # print("x.abs().mean()",x.abs().mean())

            q, k, v = qkv.unbind(0)
            q = self.q_IF(q)
            k = self.k_IF(k)
            v = self.v_IF(v)
            # print("q.abs().mean()",q.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
            # print("k.abs().mean()",k.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
            # print("v.abs().mean()",v.reshape(self.T,8,12,197,64).sum(dim=0).abs().mean())
            # print("q.abs().mean()",q.abs().mean())
            # print("k.abs().mean()",k.abs().mean())
            # print("v.abs().mean()",v.abs().mean())

            q = q * self.scale
            q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold

            attn = self.multi(q,k,q_acc - q.detach(),self.k_IF.acc_q*self.k_IF.q_threshold - k.detach())
            attn = self.attn_drop(attn)
            # print("attn.abs().mean() before",attn.reshape(self.T,8,12,197,197).sum(dim=0).abs().mean())
            # print("attn.abs().mean() before",attn.abs().mean())
            attn = self.attn_IF(attn/N)
            # print("attn.abs().mean()",attn.reshape(self.T,8,12,197,197).sum(dim=0).abs().mean())
            # print("attn.abs().mean()",attn.abs().mean())

            x = self.multi1(attn,v,(self.attn_IF.acc_q*self.attn_IF.q_threshold - attn.detach()),(self.v_IF.acc_q*self.v_IF.q_threshold - v.detach()))

            x = self.after_attn_IF(x)
            x = x.transpose(1, 2).reshape(B, N, C)
            # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())
            # print("x.abs().mean()",x.abs().mean())

            x = self.proj(x)
            x = self.proj_drop(x)
            # x = self.proj_IF(x)
            # print("x.abs().mean()",x.reshape(self.T,8,197,768).sum(dim=0).abs().mean())
            # print("x.abs().mean()",x.abs().mean())

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
        self.t = 0

    def forward(self, x):
        with nvtx_range("snn.layer.attention.SAttention_without_softmax_SS.forward"):
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
        self.DyHT_Softmax = DyHT_Softmax(H=num_heads)


    def forward(self, x):   
        with nvtx_range("snn.layer.attention.Attention_no_softmax.forward"):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn)
            attn = torch.clamp(attn/N, max=0.99, min=-0.01)
            # attn = self.DyHT_Softmax(attn)
            # attn = F.softmax(attn, dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
