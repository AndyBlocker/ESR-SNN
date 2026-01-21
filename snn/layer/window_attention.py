from typing import Optional

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from .quant import MyQuan
from .softmax import spiking_softmax
from .attention import AttentionMulti, AttentionMulti1
from .st_bifneuron_ms import ST_BIFNeuron_MS
from .st_bifneuron_ss import ST_BIFNeuron_SS
from .dyt import DyT


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
