
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import glo
from spike_quan_layer import (
    Addition,
    Attention_no_softmax,
    DyHT,
    DyHT_ReLU,
    DyT,
    IFNeuron,
    LLConv2d,
    LLConv2d_MS,
    LLLinear,
    LLLinear_MS,
    LN2BNorm,
    MLP_BN,
    MyBatchNorm1d,
    MyBatchNorm1d_SS,
    MyQuan,
    PatchEmbedConv,
    PatchMergingConv,
    QAttention,
    QAttention_without_softmax,
    QWindowAttention,
    QuanConv2d,
    QuanLinear,
    SAttention,
    SAttention_without_softmax,
    SAttention_without_softmax_SS,
    SDyHT,
    SDyHT_SS,
    SpikeMaxPooling,
    SpikeMaxPooling_SS,
    Spiking_LayerNorm,
    ST_BIFNeuron_MS,
    ST_BIFNeuron_SS,
    SWindowAttention,
    SWindowAttention_SS,
    WindowAttention_no_softmax,
    save_module_inout,
    spiking_dyt,
)
from timm.models.vision_transformer import Attention, Mlp, Block
from timm.models.swin_transformer import SwinTransformerBlock, WindowAttention, PatchEmbed, PatchMerging

from timm_vit_block import BlockWithAddition

def get_subtensors(tensor,mean,std,sample_grain=255,time_step=4):
    for i in range(int(time_step)):
        # output = (tensor).unsqueeze(0)
        output = (tensor/sample_grain).unsqueeze(0)
        if i == 0:
            accu = output
        elif i < sample_grain:
            accu = torch.cat((accu,output),dim=0)
        else:
            accu = torch.cat((accu,output*0.0),dim=0)
    return accu

def reset_model(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, IFNeuron) or isinstance(child, SpikeMaxPooling_SS) or \
            isinstance(child, MyBatchNorm1d_SS) or \
            isinstance(child, LLConv2d) or isinstance(child, LLLinear) or isinstance(child, SWindowAttention_SS) or \
            isinstance(child, SWindowAttention) or isinstance(child, Spiking_LayerNorm) or \
            isinstance(child, SpikeMaxPooling) or isinstance(child, SDyHT_SS) or \
            isinstance(child, ST_BIFNeuron_MS) or isinstance(child, ST_BIFNeuron_SS):
            model._modules[name].reset()
            is_need = True
        if not is_need:
            reset_model(child)

class Judger():
	def __init__(self):
		self.network_finish=True

	def judge_finish(self,model):
		children = list(model.named_children())
		for name, child in children:
			is_need = False
			if isinstance(child, IFNeuron) or isinstance(child, LLLinear) or isinstance(child, LLConv2d):
				self.network_finish = self.network_finish and (not model._modules[name].is_work)
				# print("child",child,"network_finish",self.network_finish,"model._modules[name].is_work",(model._modules[name].is_work))
				is_need = True
			if not is_need:
				self.judge_finish(child)

	def reset_network_finish_flag(self):
		self.network_finish = True

def attn_convert_QAttention(QAttn:QAttention,SAttn:SAttention,level,neuron_type, T,suppress_over_fire,step):
    if QAttn.qkv is not None:
        QAttn.qkv.bias.data = QAttn.qkv.bias.data/step
    SAttn.qkv = LLLinear_MS(linear = QAttn.qkv,neuron_type = "ST-BIF",time_step=T,level = level,step=step)
    
    if QAttn.proj is not None:
        QAttn.proj.bias.data = QAttn.proj.bias.data/step
    SAttn.proj = LLLinear_MS(linear = QAttn.proj,neuron_type = "ST-BIF",time_step=T,level = level,step=step)

    SAttn.q_IF.neuron_type= neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.T = T
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.bias_channel.data = QAttn.quan_q.bias_channel.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max_buf
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min_buf
    if isinstance(SAttn.q_IF,IFNeuron):
        SAttn.q_IF.is_init = False
    elif isinstance(SAttn.q_IF,ST_BIFNeuron_SS) or isinstance(SAttn.q_IF,ST_BIFNeuron_MS):
        SAttn.q_IF.init = True

    SAttn.k_IF.neuron_type= neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.T = T
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.bias_channel.data = QAttn.quan_k.bias_channel.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max_buf
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min_buf
    if isinstance(SAttn.k_IF,IFNeuron):
        SAttn.k_IF.is_init = False
    elif isinstance(SAttn.k_IF,ST_BIFNeuron_SS) or isinstance(SAttn.k_IF,ST_BIFNeuron_MS):
        SAttn.k_IF.init = True

    SAttn.v_IF.neuron_type= neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.T = T
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.bias_channel.data = QAttn.quan_v.bias_channel.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max_buf
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min_buf
    if isinstance(SAttn.v_IF,IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF,ST_BIFNeuron_SS) or isinstance(SAttn.v_IF,ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_IF.neuron_type= neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = QAttn.attn_quan.s.data
    SAttn.attn_IF.bias_channel.data = QAttn.attn_quan.bias_channel.data
    SAttn.attn_IF.T = T
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max_buf
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min_buf
    if isinstance(SAttn.attn_IF,IFNeuron):
        SAttn.attn_IF.is_init = False
    elif isinstance(SAttn.attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.attn_IF,ST_BIFNeuron_MS):
        SAttn.attn_IF.init = True
        # SAttn.attn_IF.q_threshold.data = torch.tensor(0.125)

    SAttn.attn_softmax_IF.neuron_type= neuron_type
    SAttn.attn_softmax_IF.level = level
    SAttn.attn_softmax_IF.q_threshold.data = QAttn.attn_softmax_quan.s.data
    SAttn.attn_softmax_IF.bias_channel.data = QAttn.attn_softmax_quan.bias_channel.data
    SAttn.attn_softmax_IF.T = T
    SAttn.attn_softmax_IF.pos_max = QAttn.attn_softmax_quan.pos_max_buf
    SAttn.attn_softmax_IF.neg_min = QAttn.attn_softmax_quan.neg_min_buf
    if isinstance(SAttn.attn_softmax_IF,IFNeuron):
        SAttn.attn_softmax_IF.is_init = False
    elif isinstance(SAttn.attn_softmax_IF,ST_BIFNeuron_SS) or isinstance(SAttn.attn_softmax_IF,ST_BIFNeuron_MS):
        SAttn.attn_softmax_IF.init = True

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.bias_channel.data = QAttn.after_attn_quan.bias_channel.data
    SAttn.after_attn_IF.T = T
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max_buf
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min_buf
    if isinstance(SAttn.after_attn_IF,IFNeuron):
        SAttn.after_attn_IF.is_init = False
    elif isinstance(SAttn.after_attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.after_attn_IF,ST_BIFNeuron_MS):
        SAttn.after_attn_IF.init = True
        # SAttn.after_attn_IF.q_threshold.data = torch.tensor(0.125)

    # SAttn.proj_IF.neuron_type= neuron_type
    # SAttn.proj_IF.level = level
    # SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    # SAttn.proj_IF.T = T
    # SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    # SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    # if isinstance(SAttn.proj_IF,IFNeuron):
    #     SAttn.proj_IF.is_init = False
    # elif isinstance(SAttn.proj_IF,ST_BIFNeuron_SS) or isinstance(SAttn.proj_IF,ST_BIFNeuron_MS):
    #     SAttn.proj_IF.init = True

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.proj_drop = QAttn.proj_drop

def attn_convert(QAttn:QAttention_without_softmax,SAttn:SAttention_without_softmax,level,neuron_type, T,suppress_over_fire,step):
    if QAttn.qkv is not None:
        QAttn.qkv.bias.data = QAttn.qkv.bias.data/step
    SAttn.qkv = LLLinear_MS(linear = QAttn.qkv,neuron_type = "ST-BIF",time_step=T,level = level,step=step)
    
    if QAttn.proj is not None:
        QAttn.proj.bias.data = QAttn.proj.bias.data/step
    SAttn.proj = LLLinear_MS(linear = QAttn.proj,neuron_type = "ST-BIF",time_step=T,level = level,step=step)

    SAttn.q_IF.neuron_type= neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.T = T
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.bias_channel.data = QAttn.quan_q.bias_channel.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max_buf
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min_buf
    SAttn.q_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.q_IF,IFNeuron):
        SAttn.q_IF.is_init = False
    elif isinstance(SAttn.q_IF,ST_BIFNeuron_SS) or isinstance(SAttn.q_IF,ST_BIFNeuron_MS):
        SAttn.q_IF.init = True

    SAttn.k_IF.neuron_type= neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.T = T
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.bias_channel.data = QAttn.quan_k.bias_channel.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max_buf
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min_buf
    SAttn.k_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.k_IF,IFNeuron):
        SAttn.k_IF.is_init = False
    elif isinstance(SAttn.k_IF,ST_BIFNeuron_SS) or isinstance(SAttn.k_IF,ST_BIFNeuron_MS):
        SAttn.k_IF.init = True

    SAttn.v_IF.neuron_type= neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.T = T
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.bias_channel.data = QAttn.quan_v.bias_channel.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max_buf
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min_buf
    SAttn.v_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.v_IF,IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF,ST_BIFNeuron_SS) or isinstance(SAttn.v_IF,ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_IF.neuron_type= neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = min(QAttn.attn_quan.s.data, QAttn.attn_quan.s_max.data)
    SAttn.attn_IF.bias_channel.data = QAttn.attn_quan.bias_channel.data
    SAttn.attn_IF.T = T
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max_buf
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min_buf
    SAttn.attn_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.attn_IF,IFNeuron):
        SAttn.attn_IF.is_init = False
    elif isinstance(SAttn.attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.attn_IF,ST_BIFNeuron_MS):
        SAttn.attn_IF.init = True
        # SAttn.attn_IF.q_threshold.data = torch.tensor(0.125)

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.bias_channel.data = QAttn.after_attn_quan.bias_channel.data
    SAttn.after_attn_IF.T = T
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max_buf
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min_buf
    SAttn.after_attn_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.after_attn_IF,IFNeuron):
        SAttn.after_attn_IF.is_init = False
    elif isinstance(SAttn.after_attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.after_attn_IF,ST_BIFNeuron_MS):
        SAttn.after_attn_IF.init = True
        # SAttn.after_attn_IF.q_threshold.data = torch.tensor(0.125)

    # SAttn.proj_IF.neuron_type= neuron_type
    # SAttn.proj_IF.level = level
    # SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    # SAttn.proj_IF.T = T
    # SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    # SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    # SAttn.proj_IF.suppress_over_fire = suppress_over_fire
    # if isinstance(SAttn.proj_IF,IFNeuron):
    #     SAttn.proj_IF.is_init = False
    # elif isinstance(SAttn.proj_IF,ST_BIFNeuron_SS) or isinstance(SAttn.proj_IF,ST_BIFNeuron_MS):
    #     SAttn.proj_IF.init = True

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.proj_drop = QAttn.proj_drop

def attn_convert_SS(QAttn:QAttention_without_softmax,SAttn:SAttention_without_softmax_SS,level,neuron_type, T):
    SAttn.qkv = LLLinear(linear = QAttn.qkv, neuron_type = "ST-BIF", time_step=T, level = level)
    SAttn.proj = LLLinear(linear = QAttn.proj, neuron_type = "ST-BIF", time_step=T, level = level)

    SAttn.q_IF.neuron_type= neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.T = T
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min
    if isinstance(SAttn.q_IF,IFNeuron):
        SAttn.q_IF.is_init = False
    elif isinstance(SAttn.q_IF,ST_BIFNeuron_SS) or isinstance(SAttn.q_IF,ST_BIFNeuron_MS):
        SAttn.q_IF.init = True

    SAttn.k_IF.neuron_type= neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.T = T
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min
    if isinstance(SAttn.k_IF,IFNeuron):
        SAttn.k_IF.is_init = False
    elif isinstance(SAttn.k_IF,ST_BIFNeuron_SS) or isinstance(SAttn.k_IF,ST_BIFNeuron_MS):
        SAttn.k_IF.init = True

    SAttn.v_IF.neuron_type= neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.T = T
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min
    if isinstance(SAttn.v_IF,IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF,ST_BIFNeuron_SS) or isinstance(SAttn.v_IF,ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_IF.neuron_type= neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = QAttn.attn_quan.s.data
    SAttn.attn_IF.T = T
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min
    if isinstance(SAttn.attn_IF,IFNeuron):
        SAttn.attn_IF.is_init = False
    elif isinstance(SAttn.attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.attn_IF,ST_BIFNeuron_MS):
        SAttn.attn_IF.init = True
        # SAttn.attn_IF.q_threshold.data = torch.tensor(0.125)

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.T = T
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    if isinstance(SAttn.after_attn_IF,IFNeuron):
        SAttn.after_attn_IF.is_init = False
    elif isinstance(SAttn.after_attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.after_attn_IF,ST_BIFNeuron_MS):
        SAttn.after_attn_IF.init = True
        # SAttn.after_attn_IF.q_threshold.data = torch.tensor(0.125)

    # SAttn.proj_IF.neuron_type= neuron_type
    # SAttn.proj_IF.level = level
    # SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    # SAttn.proj_IF.T = T
    # SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    # SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    # if isinstance(SAttn.proj_IF,IFNeuron):
    #     SAttn.proj_IF.is_init = False
    # elif isinstance(SAttn.proj_IF,ST_BIFNeuron_SS) or isinstance(SAttn.proj_IF,ST_BIFNeuron_MS):
    #     SAttn.proj_IF.init = True

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.proj_drop = QAttn.proj_drop


def attn_convert_Swin(QAttn:QWindowAttention,SAttn:SWindowAttention,level,neuron_type, T, suppress_over_fire,step):
    print("attn_convert_Swin:T=",T)
    if QAttn.qkv is not None:
        QAttn.qkv.bias.data = QAttn.qkv.bias.data/step
    SAttn.qkv = LLLinear_MS(linear = QAttn.qkv,neuron_type = "ST-BIF",time_step=T,level = level, step=step)
    
    if QAttn.proj is not None:
        QAttn.proj.bias.data = QAttn.proj.bias.data/step
    SAttn.proj = LLLinear_MS(linear = QAttn.proj,neuron_type = "ST-BIF",time_step=T,level = level, step=step)

    SAttn.relative_position_bias_table = QAttn.relative_position_bias_table
    SAttn.relative_position_index = QAttn.relative_position_index

    SAttn.q_IF.neuron_type= neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.T = T
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min
    SAttn.q_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.q_IF,IFNeuron):
        SAttn.q_IF.is_init = False
    elif isinstance(SAttn.q_IF,ST_BIFNeuron_SS) or isinstance(SAttn.q_IF,ST_BIFNeuron_MS):
        SAttn.q_IF.init = True

    SAttn.k_IF.neuron_type= neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.T = T
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min
    SAttn.k_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.k_IF,IFNeuron):
        SAttn.k_IF.is_init = False
    elif isinstance(SAttn.k_IF,ST_BIFNeuron_SS) or isinstance(SAttn.k_IF,ST_BIFNeuron_MS):
        SAttn.k_IF.init = True

    SAttn.v_IF.neuron_type= neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.T = T
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min
    SAttn.v_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.v_IF,IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF,ST_BIFNeuron_SS) or isinstance(SAttn.v_IF,ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_softmax_IF.neuron_type= neuron_type
    SAttn.attn_softmax_IF.level = level
    SAttn.attn_softmax_IF.q_threshold.data = min(QAttn.attn_softmax_quan.s.data,QAttn.attn_softmax_quan.s_max.data)
    SAttn.attn_softmax_IF.T = T
    SAttn.attn_softmax_IF.pos_max = QAttn.attn_softmax_quan.pos_max
    SAttn.attn_softmax_IF.neg_min = QAttn.attn_softmax_quan.neg_min
    SAttn.attn_softmax_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.attn_softmax_IF,IFNeuron):
        SAttn.attn_softmax_IF.is_init = False
    elif isinstance(SAttn.attn_softmax_IF,ST_BIFNeuron_SS) or isinstance(SAttn.attn_softmax_IF,ST_BIFNeuron_MS):
        SAttn.attn_softmax_IF.init = True
        

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.T = T
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    SAttn.after_attn_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.after_attn_IF,IFNeuron):
        SAttn.after_attn_IF.is_init = False
    elif isinstance(SAttn.after_attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.after_attn_IF,ST_BIFNeuron_MS):
        SAttn.after_attn_IF.init = True
        # SAttn.after_attn_IF.q_threshold.data = torch.tensor(0.125)

    # SAttn.proj_IF.neuron_type= neuron_type
    # SAttn.proj_IF.level = level
    # SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    # SAttn.proj_IF.T = T
    # SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    # SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    # SAttn.proj_IF.suppress_over_fire = suppress_over_fire
    # if isinstance(SAttn.proj_IF,IFNeuron):
    #     SAttn.proj_IF.is_init = False
    # elif isinstance(SAttn.proj_IF,ST_BIFNeuron_SS) or isinstance(SAttn.proj_IF,ST_BIFNeuron_MS):
    #     SAttn.proj_IF.init = True

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.attn_drop.p = 0.0
    SAttn.proj_drop = QAttn.proj_drop
    SAttn.proj_drop.p = 0.0


def attn_convert_Swin_SS(QAttn:QWindowAttention,SAttn:SWindowAttention_SS,level,neuron_type, T, suppress_over_fire,step):
    print("attn_convert_Swin:T=",T)
    SAttn.qkv = LLLinear(linear = QAttn.qkv,neuron_type = "ST-BIF",time_step=T,level = level, step=step)
    
    SAttn.proj = LLLinear(linear = QAttn.proj,neuron_type = "ST-BIF",time_step=T,level = level, step=step)

    SAttn.relative_position_bias_table = QAttn.relative_position_bias_table
    SAttn.relative_position_index = QAttn.relative_position_index

    SAttn.q_IF.neuron_type= neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.T = T
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min
    SAttn.q_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.q_IF,IFNeuron):
        SAttn.q_IF.is_init = False
    elif isinstance(SAttn.q_IF,ST_BIFNeuron_SS) or isinstance(SAttn.q_IF,ST_BIFNeuron_MS):
        SAttn.q_IF.init = True

    SAttn.k_IF.neuron_type= neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.T = T
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min
    SAttn.k_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.k_IF,IFNeuron):
        SAttn.k_IF.is_init = False
    elif isinstance(SAttn.k_IF,ST_BIFNeuron_SS) or isinstance(SAttn.k_IF,ST_BIFNeuron_MS):
        SAttn.k_IF.init = True

    SAttn.v_IF.neuron_type= neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.T = T
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min
    SAttn.v_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.v_IF,IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF,ST_BIFNeuron_SS) or isinstance(SAttn.v_IF,ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_softmax_IF.neuron_type= neuron_type
    SAttn.attn_softmax_IF.level = level
    SAttn.attn_softmax_IF.q_threshold.data = min(QAttn.attn_softmax_quan.s.data,QAttn.attn_softmax_quan.s_max.data)
    SAttn.attn_softmax_IF.T = T
    SAttn.attn_softmax_IF.pos_max = QAttn.attn_softmax_quan.pos_max
    SAttn.attn_softmax_IF.neg_min = QAttn.attn_softmax_quan.neg_min
    SAttn.attn_softmax_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.attn_softmax_IF,IFNeuron):
        SAttn.attn_softmax_IF.is_init = False
    elif isinstance(SAttn.attn_softmax_IF,ST_BIFNeuron_SS) or isinstance(SAttn.attn_softmax_IF,ST_BIFNeuron_MS):
        SAttn.attn_softmax_IF.init = True
        

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.T = T
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    SAttn.after_attn_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.after_attn_IF,IFNeuron):
        SAttn.after_attn_IF.is_init = False
    elif isinstance(SAttn.after_attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.after_attn_IF,ST_BIFNeuron_MS):
        SAttn.after_attn_IF.init = True
        # SAttn.after_attn_IF.q_threshold.data = torch.tensor(0.125)

    # SAttn.proj_IF.neuron_type= neuron_type
    # SAttn.proj_IF.level = level
    # SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    # SAttn.proj_IF.T = T
    # SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    # SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    # SAttn.proj_IF.suppress_over_fire = suppress_over_fire
    # if isinstance(SAttn.proj_IF,IFNeuron):
    #     SAttn.proj_IF.is_init = False
    # elif isinstance(SAttn.proj_IF,ST_BIFNeuron_SS) or isinstance(SAttn.proj_IF,ST_BIFNeuron_MS):
    #     SAttn.proj_IF.init = True

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.attn_drop.p = 0.0
    SAttn.proj_drop = QAttn.proj_drop
    SAttn.proj_drop.p = 0.0



def open_dropout(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, nn.Dropout):
            child.train()
            print(child)
            is_need = True
        if not is_need:
            open_dropout(child)



def cal_l1_loss(model):
    l1_loss = 0.0
    def _cal_l1_loss(model):
        nonlocal l1_loss
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, MyQuan) and child.cal_loss:
                l1_loss = l1_loss + child.l1_loss
                is_need = True
            if not is_need:
                _cal_l1_loss(child)
    _cal_l1_loss(model)
    return l1_loss

def adjust_LN2BN_Ratio(EndEpoch:int, curEpoch:int, model:nn.Module):
    for name,module in list(model.named_modules()):
        if isinstance(module, LN2BNorm):
            if curEpoch < EndEpoch:
                module.Lambda = 1 - (curEpoch+1)/(EndEpoch)
                print(f"adjust {name} Lambda = {module.Lambda}")

def add_bn_in_mlp(model,normLayer):
    children = list(model.named_children())
    for name, child in children:
        if name.count("decoder") > 0:
            continue
        is_need = False
        if isinstance(child, Mlp):
            mlp_bn = MLP_BN(in_features=child.fc1.in_features, hidden_features=child.fc1.out_features, act_layer=nn.ReLU, drop=child.drop1.p, norm_layer=normLayer)
            model._modules[name] = mlp_bn
            is_need = True
        # elif isinstance(child, nn.LayerNorm):
        #     LN = MyBatchNorm1d(num_features = child.normalized_shape[0])
        #     # LN.weight.data = child.weight
        #     # LN.bias.data = child.bias
        #     model._modules[name] = LN
        if not is_need:
            add_bn_in_mlp(child,normLayer)

class SNNWrapper(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="rate", learnable=False,**kwargs):
        super(SNNWrapper, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.step = self.level//2 - 1
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model
        self.model.patch_embed.spike = True
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.record_inout = kwargs["record_inout"]
        self.record_dir = kwargs["record_dir"]
        self.learnable = learnable
        self.max_T = 0
        self.visualize = False
        # self.model_reset = None
        if self.model_name.count("vit") > 0:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)
        self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)        
        if self.record_inout:
            self.calOrder = []
            self._record_inout(self.model)
            self.set_snn_save_name(self.model)
            local_rank = torch.distributed.get_rank()
            glo._init()
            if local_rank == 0:
                if not os.path.exists(self.record_dir):
                    os.mkdir(self.record_dir)
                glo.set_value("output_bin_snn_dir",self.record_dir)
                f = open(f"{self.record_dir}/calculationOrder.txt","w+")
                for order in self.calOrder:
                    f.write(order+"\n")
                f.close()
    
    def hook_mid_feature(self):
        self.feature_list = []
        self.input_feature_list = []
        def _hook_mid_feature(module, input, output):
            self.feature_list.append(output)
            self.input_feature_list.append(input[0])
        self.model.blocks[11].norm2[1].register_forward_hook(_hook_mid_feature)
        # self.model.blocks[11].attn.attn_IF.register_forward_hook(_hook_mid_feature)
    
    def get_mid_feature(self):
        self.feature_list = torch.stack(self.feature_list,dim=0)
        self.input_feature_list = torch.stack(self.input_feature_list,dim=0)
        print("self.feature_list",self.feature_list.shape) 
        print("self.input_feature_list",self.input_feature_list.shape) 
            
    def reset(self):
        # self.model = deepcopy(self.model_reset).cuda()
        if self.model_name.count("vit")>0:
            self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
            self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        # print(self.model.pos_embed)
        # print(self.model.cls_token)
        reset_model(self)
    
    def _record_inout(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, SAttention):
                model._modules[name].first = True
                model._modules[name].T = self.T
                is_need = True
            elif isinstance(child, nn.Sequential) and isinstance(child[1], IFNeuron):
                model._modules[name] = save_module_inout(m=child,T=self.T)
                model._modules[name].first = True
                is_need = True
            if not is_need:            
                self._record_inout(child)            

    def set_snn_save_name(self, model):
        children = list(model.named_modules())
        for name, child in children:
            if isinstance(child, save_module_inout):
                child.name = name
                self.calOrder.append(name)

            if isinstance(child, SAttention):
                child.name = name
                self.calOrder.append(name)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ST_BIFNeuron_SS,T=self.T)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QAttention_without_softmax):
                SAttn = SAttention_without_softmax_SS(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ST_BIFNeuron_SS,T=self.T)
                attn_convert_SS(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QWindowAttention):
                # self.blockNum = self.blockNum + 1/24
                SAttn = SWindowAttention_SS(dim=child.num_heads*child.head_dim, window_size=child.window_size,num_heads=child.num_heads,level=self.level,neuron_layer=ST_BIFNeuron_SS,T=self.T,step=self.step)
                attn_convert_Swin_SS(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T, suppress_over_fire=False, step=self.step)
                # SAttn.attn_softmax_IF.prefire.data = torch.tensor(self.blockNum*0.2)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d) or isinstance(child, QuanConv2d):
                model._modules[name] = LLConv2d(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear) or isinstance(child, QuanLinear):
                model._modules[name] = LLLinear(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling_SS(child, step=self.step,T=self.T)
                is_need = True
            elif isinstance(child, DyT):
                model._modules[name] = spiking_dyt(child,step=self.step,T=self.T)
                is_need = True
            elif isinstance(child, DyHT) or isinstance(child, DyHT_ReLU):
                sdyht = SDyHT_SS(C = 1)
                sdyht.step = self.step
                sdyht.T = self.T
                sdyht.beta.data = child.beta
                sdyht.gamma.data = child.gamma
                sdyht.alpha.data = child.alpha
                model._modules[name] = sdyht
                is_need = True
            elif isinstance(child,nn.BatchNorm2d) or isinstance(child,nn.BatchNorm1d) or isinstance(child, MyBatchNorm1d):
                # if self.learnable:
                #     model._modules[name] = MyBachNorm(bn=child,T=self.T)
                # else:
                # model._modules[name] = spiking_BatchNorm2d_MS(bn=child,level=self.level//2 - 1,input_allcate=False)
                bn = MyBatchNorm1d_SS(dim=1)
                bn.bias.data = model._modules[name].bias.data/self.step
                bn.running_mean = model._modules[name].running_mean/self.step
                bn.weight.data = model._modules[name].weight.data
                bn.running_var = model._modules[name].running_var
                bn.spike = True
                bn.T = self.T
                bn.step = self.step
                model._modules[name] = bn
                
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0],T=self.T)
                SNN_LN.layernorm = child
                if child.elementwise_affine:
                    SNN_LN.weight = child.weight.data
                    SNN_LN.bias = child.bias.data                
                model._modules[name] = SNN_LN
                # model._modules[name].register_full_backward_hook(modify_gradient_for_spiking_layernorm_softmax(self.T))
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = ST_BIFNeuron_SS(q_threshold = torch.tensor(1.0),sym=child.sym,level = child.pos_max)
                neurons.q_threshold.data = min(child.s.data, child.s_max.data)
                neurons.level = self.level
                neurons.pos_max = child.pos_max
                neurons.neg_min = child.neg_min
                neurons.init = True
                self.first_neuron = False
                neurons.cuda()
                model._modules[name] = neurons
                is_need = True
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
            if not is_need:
                self._replace_weight(child)

    def forward(self,x, verbose=False):
        accu = None
        count1 = 0
        accu_per_timestep = []
        output_per_timestep = []
        # print("self.bit",self.bit)
        # x = x*(2**self.bit-1)+0.0
        if self.visualize:
            self.hook_mid_feature()
        if self.Encoding_type == "rate":
            self.mean = 0.0
            self.std  = 0.0
            x = get_subtensors(x,self.mean,self.std,sample_grain=self.step)        
            if self.cfg.model.count("vit") > 0:        
                self.model.pos_embed.data = self.model.pos_embed/self.step
                self.model.cls_token.data = self.model.cls_token/self.step
            # print("x.shape",x.shape)
        while(1):
            self.finish_judger.reset_network_finish_flag()
            self.finish_judger.judge_finish(self)
            network_finish = self.finish_judger.network_finish
            # print(f"===================Timestep: {count1}===================")
            if (count1 > 0 and network_finish) or count1 >= self.T:
                self.max_T = max(count1, self.max_T)
                break
            # if self.neuron_type.count("QFFS") != -1 or self.neuron_type == 'ST-BIF':
            if (self.Encoding_type == "analog" and self.model_name.count("vit") > 0 and count1 > 0) or (self.Encoding_type == "rate" and self.model_name.count("vit") > 0 and count1 >= self.step):
                self.model.pos_embed.data = self.model.pos_embed*0.0
                self.model.cls_token.data = self.model.cls_token*0.0
            elif self.cfg.model.count("swin") > 0:
                self.model.pos_drop.p = 0
            if self.Encoding_type == "rate":
                if count1 < x.shape[0]:
                    input = x[count1]
                else:
                    input = torch.zeros(x[0].shape).to(x.device)            
            else:
                if count1 == 0:
                    input = x
                else:
                    input = torch.zeros(x.shape).to(x.device)
            # elif self.neuron_type == 'IF':
            #     input = x
            # else:
            #     print("No implementation of neuron type:",self.neuron_type)
            #     sys.exit(0)
            
            output = self.model(input)
            # print(count1,output[0,0:100])
            # print(count1,"output",torch.abs(output.sum()))
            
            if count1 == 0:
                accu = output + 0.0
            else:
                accu = accu + output
            if verbose:
                accu_per_timestep.append(accu)
                output_per_timestep.append(output)
            # print("accu",accu.sum(),"output",output.sum())
            count1 = count1 + 1
            if count1 % 100 == 0:
                print(count1)

        # print("verbose",verbose)
        # print("\nTime Step:",count1)
        if self.visualize:
            self.get_mid_feature()
            torch.save(self.feature_list,"model_blocks11_norm2.pth")
            torch.save(self.input_feature_list,"model_blocks11_norm2_input.pth")
        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep,dim=0)
            output_per_timestep = torch.stack(output_per_timestep,dim=0)
            return accu,count1,output_per_timestep,accu_per_timestep
        else:
            return accu,count1

def modify_gradient_for_spiking_layernorm_softmax(T):
    def _modify(module, grad_in, grad_out):
        nonlocal T
        # grad_out = grad_out[0].reshape(torch.Size([T,grad_out[0].shape[0]//T])+grad_out[0].shape[1:])
        # grad_in = grad_in[0].reshape(torch.Size([T,grad_in[0].shape[0]//T])+grad_in[0].shape[1:])
        # print(grad_out.abs().mean())

        # print(len(grad_in),len(grad_out))
        # print(grad_in[0].shape)
        # print("===========================================")
        # print("module",module)
        # print(grad_out.shape, grad_in[0].shape)
        # print(grad_in[0].abs().mean(),grad_in[1].abs().mean(),grad_in[2].abs().mean(),grad_in[3].abs().mean())
        # print(grad_out[0].abs().mean(),grad_out[1].abs().mean(),grad_out[2].abs().mean(),grad_out[3].abs().mean())
        # print(T,grad_in[0].shape, grad_out[0].shape)
        # grad1 = grad_in[0].reshape(torch.Size([T,grad_in[0].shape[0]//T])+grad_in[0].shape[1:])
        # print(grad_out.shape,grad_in.shape)
        # print(torch.cat([grad_in[0]*(T-i)/T for i in range(T)]).shape)
        # grad_new = tuple([torch.cat([grad_in[0]*(T-i)/T for i in range(T)])])
        # return grad_out
    return _modify

class SNNWrapper_MS(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="rate", learnable=False,**kwargs):
        super(SNNWrapper_MS, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.step = 2
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model

        self.model.spike = True
        self.model.T = time_step
        self.model.step = self.step

        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.record_inout = kwargs["record_inout"]
        self.record_dir = kwargs["record_dir"]
        self.suppress_over_fire = kwargs["suppress_over_fire"]
        self.learnable = learnable
        self.max_T = 0
        self.visualize = False
        self.first_neuron = True
        self.blockNum = 0
        self.confident_thr = [0.2,0.85]
        # self.model_reset = None
        if self.model_name.count("vit") > 0:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)

        self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)        
        if self.record_inout:
            self.calOrder = []
            self._record_inout(self.model)
            self.set_snn_save_name(self.model)
            local_rank = torch.distributed.get_rank()
            glo._init()
            if local_rank == 0:
                if not os.path.exists(self.record_dir):
                    os.mkdir(self.record_dir)
                glo.set_value("output_bin_snn_dir",self.record_dir)
                f = open(f"{self.record_dir}/calculationOrder.txt","w+")
                for order in self.calOrder:
                    f.write(order+"\n")
                f.close()
        
        # self.param_number = 4
        # init_list = []
        # for i in range(self.param_number - 1):
        #     if i < self.step - 1:
        #         init_list.append(1/(self.step))
        #     else:
        #         init_list.append(0.0)
        # if self.T > 1:
        #     self.biasAllocator = nn.Parameter(torch.tensor(init_list))
        # else:
        #     self.biasAllocator = 1.0
        
    def hook_mid_feature(self):
        self.feature_list = []
        self.input_feature_list = []
        def _hook_mid_feature(module, input, output):
            self.feature_list.append(output)
            self.input_feature_list.append(input[0])
        self.model.blocks[11].norm2[1].register_forward_hook(_hook_mid_feature)
        # self.model.blocks[11].attn.attn_IF.register_forward_hook(_hook_mid_feature)
    
    def get_mid_feature(self):
        self.feature_list = torch.stack(self.feature_list,dim=0)
        self.input_feature_list = torch.stack(self.input_feature_list,dim=0)
        print("self.feature_list",self.feature_list.shape) 
        print("self.input_feature_list",self.input_feature_list.shape) 
            
    def reset(self):
        # self.model = deepcopy(self.model_reset).cuda()
        if self.model_name.count("vit")>0:
            self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
            self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        # print(self.model.pos_embed)
        # print(self.model.cls_token)
        reset_model(self)
    

    def _record_inout(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, SAttention):
                model._modules[name].first = True
                model._modules[name].T = self.T
                is_need = True
            elif isinstance(child, nn.Sequential) and isinstance(child[1], IFNeuron):
                model._modules[name] = save_module_inout(m=child,T=self.T)
                model._modules[name].first = True
                is_need = True
            if not is_need:            
                self._record_inout(child)            

    def set_snn_save_name(self, model):
        children = list(model.named_modules())
        for name, child in children:
            if isinstance(child, save_module_inout):
                child.name = name
                self.calOrder.append(name)
            if isinstance(child, SAttention):
                child.name = name
                self.calOrder.append(name)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ST_BIFNeuron_MS,T=self.T)
                attn_convert_QAttention(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T, suppress_over_fire=self.suppress_over_fire, step=self.step)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QAttention_without_softmax):
                SAttn = SAttention_without_softmax(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ST_BIFNeuron_MS,T=self.T)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T, suppress_over_fire=self.suppress_over_fire, step=self.step)
                # self.blockNum = self.blockNum + 1/12
                # SAttn.attn_IF.prefire.data = torch.tensor(0.125)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QWindowAttention):
                # self.blockNum = self.blockNum + 1/24
                SAttn = SWindowAttention(dim=child.num_heads*child.head_dim, window_size=child.window_size,num_heads=child.num_heads,level=self.level,neuron_layer=ST_BIFNeuron_MS,T=self.T,step=self.step)
                attn_convert_Swin(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T, suppress_over_fire=self.suppress_over_fire, step=self.step)
                # SAttn.attn_softmax_IF.prefire.data = torch.tensor(self.blockNum*0.2)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d) or isinstance(child, QuanConv2d):
                if child.bias is not None:
                    model._modules[name].bias.data = model._modules[name].bias.data/self.step
                model._modules[name] = LLConv2d_MS(child,time_step=self.T,step=self.step,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear) or isinstance(child, QuanLinear):
                # if name.count("head") > 0:
                #     model._modules[name] = LLLinear(child,**self.kwargs)
                # else:
                if child.bias is not None:
                    model._modules[name].bias.data = model._modules[name].bias.data/self.step
                model._modules[name] = LLLinear_MS(child,time_step=self.T,step=self.step,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling(child, step=self.step,T=self.T)
                is_need = True
            elif isinstance(child, DyT):
                model._modules[name] = spiking_dyt(child,step=self.step,T=self.T)
                is_need = True
            elif isinstance(child, DyHT) or isinstance(child, DyHT_ReLU):
                sdyht = SDyHT(C = 1, step=self.step,T=self.T)
                sdyht.step = self.step
                sdyht.T = self.T
                sdyht.beta.data = child.beta
                # sdyht.beta.data = child.beta/sdyht.step
                sdyht.gamma.data = child.gamma
                sdyht.alpha.data = child.alpha
                model._modules[name] = sdyht
                is_need = True
            elif isinstance(child,nn.BatchNorm2d) or isinstance(child,nn.BatchNorm1d) or isinstance(child, MyBatchNorm1d):
                # if self.learnable:
                #     model._modules[name] = MyBachNorm(bn=child,T=self.T)
                # else:
                # model._modules[name] = spiking_BatchNorm2d_MS(bn=child,level=self.level//2 - 1,input_allcate=False)
                model._modules[name].bias.data = model._modules[name].bias.data/self.step
                model._modules[name].running_mean = model._modules[name].running_mean/self.step
                model._modules[name].spike = True
                model._modules[name].T = self.T
                model._modules[name].step = self.step
                
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0],T=self.T,step=self.step)
                SNN_LN.layernorm = child
                if child.elementwise_affine:
                    SNN_LN.weight = child.weight.data
                    SNN_LN.bias = child.bias.data
                model._modules[name] = SNN_LN
                # model._modules[name].register_full_backward_hook(modify_gradient_for_spiking_layernorm_softmax(self.T))
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = ST_BIFNeuron_MS(q_threshold = torch.tensor(1.0),sym=child.sym,level = self.level, first_neuron=self.first_neuron, T = self.T, C=child.channel_num)
                neurons.q_threshold.data = min(child.s.data, child.s_max.data)
                neurons.bias_channel.data = child.bias_channel
                neurons.level = self.level
                neurons.pos_max = child.pos_max_buf
                neurons.neg_min = child.neg_min_buf
                neurons.init = True
                self.first_neuron = False
                neurons.cuda()
                model._modules[name] = neurons
                is_need = True
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
            if not is_need:            
                self._replace_weight(child)

    def forward(self,x, verbose=False):
        if self.cfg.dataset == "dvs128":
            input = x.transpose(0,1)/self.T
        else:
            input = get_subtensors(x,0.0,0.0,sample_grain=self.step, time_step=self.T)
        # input = input * self.step
        if self.cfg.model.count("vit") > 0:
            self.model.pos_embed.data = self.model.pos_embed/self.step
            self.model.cls_token.data = self.model.cls_token/self.step
        elif self.cfg.model.count("swin") > 0:
            self.model.pos_drop.p = 0
        T,B,C,H,W = input.shape
        # biasAllocator = torch.cat([1 - torch.sum(self.biasAllocator,dim=0,keepdim=True), self.biasAllocator], dim=0)
        # input = torch.cat([input[0].unsqueeze(0) * biasAllocator.reshape(-1,1,1,1,1), input[self.param_number:]], dim=0)
        
        input = input.reshape(T*B,C,H,W)
        output = self.model(input)
        output = output.reshape(torch.Size([T,B]) + output.shape[1:])
        
        if not self.training:
            print(output.abs().sum(dim=[-1,-2]))
        
        accu_per_t = []
        accu = 0.0
        self.reset()
        if verbose == True:
            for t in range(T):
                accu = accu + output[t]
                prob = torch.max(F.softmax(accu, dim=1), dim=1)[0]
                # if t == 0:
                #     if prob > self.confident_thr[0]:
                #         for i in range(t,T):
                #             accu_per_t.append(accu + 0.0)                        
                #         break
                # else:
                if prob > self.confident_thr[1]:
                    for i in range(t,T):
                        accu_per_t.append(accu + 0.0)                        
                    break
                accu_per_t.append(accu + 0.0)
            return accu, t+1, output, torch.stack(accu_per_t,dim=0)
        return output.sum(dim=0), self.T


def swap_BN_MLP_MHSA(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, Block):
            child = BlockWithAddition.from_block(child)
            model._modules[name] = child
        if isinstance(child, BlockWithAddition):
            child.addition1 = Addition()
            child.addition2 = Addition()
            is_need = True

        if not is_need:
            swap_BN_MLP_MHSA(child)
            

def add_convEmbed(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, PatchMerging):
            patchergingConv = PatchMergingConv(child.input_resolution, child.dim, norm_layer=MyBatchNorm1d)
            is_need = True
            model._modules[name] = patchergingConv
        if isinstance(child,PatchEmbed):
            patchembedConv = PatchEmbedConv(img_size=child.img_size, patch_size=child.patch_size, in_chans=child.proj.in_channels, embed_dim=child.proj.out_channels, norm_layer=MyBatchNorm1d, flatten=True)
            patchembedConv.norm = DyHT(patchembedConv.embed_dim)
            is_need = True
            model._modules[name] = patchembedConv
        if not is_need:
            add_convEmbed(child)

def remove_softmax(model):
    children = list(model.named_children())
    for name, child in children:
        if name.count("decoder") > 0:
            continue        
        is_need = False
        if isinstance(child, Attention):
            reluattn = Attention_no_softmax(dim=child.num_heads*child.head_dim,num_heads=child.num_heads)
            reluattn.qkv = child.qkv
            reluattn.attn_drop = child.attn_drop
            reluattn.proj = child.proj
            reluattn.proj_drop = child.proj_drop
            is_need = True
            model._modules[name] = reluattn
        if isinstance(child,WindowAttention):
            reluattn = WindowAttention_no_softmax(dim=child.num_heads*child.head_dim, window_size=child.window_size,num_heads=child.num_heads)
            reluattn.qkv = child.qkv
            reluattn.attn_drop = child.attn_drop
            reluattn.proj = child.proj
            reluattn.proj_drop = child.proj_drop
            reluattn.relative_position_bias_table = child.relative_position_bias_table
            reluattn.relative_position_index = child.relative_position_index
            is_need = True
            model._modules[name] = reluattn
            
        # elif isinstance(child, nn.LayerNorm):
        #     LN = MyBatchNorm1d(num_features = child.normalized_shape[0])
        #     # LN.weight.data = child.weight
        #     # LN.bias.data = child.bias
        #     model._modules[name] = LN
        if not is_need:
            remove_softmax(child)

def hook_layernorm(module, input, output):
    print("layernorm input",input[0].abs().mean())    
    print("layernorm output",output.abs().mean())    



def myquan_replace(model,level,weight_bit=32, is_softmax = True, eval1=False):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention_without_softmax):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model,level):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, (Block, BlockWithAddition)):
                # print(children)
                if is_softmax:
                   qattn = QAttention(dim=child.attn.num_heads*child.attn.head_dim,num_heads=child.attn.num_heads,level=level,is_softmax=is_softmax)
                else:
                   qattn = QAttention_without_softmax(dim=child.attn.num_heads*child.attn.head_dim,num_heads=child.attn.num_heads,level=level,is_softmax=is_softmax)
                qattn.qkv = child.attn.qkv
                qattn.attn_drop = child.attn.attn_drop
                qattn.proj = child.attn.proj
                qattn.proj_drop = child.attn.proj_drop
                model._modules[name].attn = qattn

                if isinstance(child.norm1, DyHT_ReLU):
                    model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=False, cal_loss=True))
                else:
                    model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True, cal_loss=True))
                if not isinstance(child.norm1[0], nn.LayerNorm):
                    model._modules[name].norm1[1].s_max.data = model._modules[name].norm1[0].gamma/model._modules[name].norm1[1].pos_max
                # print("model._modules[name].norm1[0].gamma",model._modules[name].norm1[0].gamma,"model._modules[name].norm1[1].pos_max",model._modules[name].norm1[1].pos_max)
                if isinstance(child.norm2, DyHT_ReLU):
                    model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=False, cal_loss=True))
                else:
                    model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True, cal_loss=True))
                if not isinstance(child.norm2[0], nn.LayerNorm):
                    model._modules[name].norm2[1].s_max.data = model._modules[name].norm2[0].gamma/model._modules[name].norm2[1].pos_max
                # print("model._modules[name].norm2[0].gamma",model._modules[name].norm2[0].gamma,"model._modules[name].norm2[1].pos_max",model._modules[name].norm2[1].pos_max)

                model._modules[name].mlp.act = nn.Sequential(child.mlp.act,MyQuan(level, sym=False,channel_num=3072))
                model._modules[name].mlp.fc2 = nn.Sequential(child.mlp.fc2)
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, PatchEmbed):
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, PatchEmbedConv):
                if isinstance(model._modules[name].proj_conv, nn.Sequential):
                    model._modules[name].proj_conv[0] = nn.Sequential(child.proj_conv[0],MyQuan(level, sym=True))
                model._modules[name].proj_ReLU = nn.Sequential(child.proj_ReLU ,MyQuan(level, sym=False))
                model._modules[name].proj1_ReLU = nn.Sequential(child.proj1_ReLU ,MyQuan(level, sym=False))
                model._modules[name].proj_res_ReLU = nn.Sequential(child.proj_res_ReLU ,MyQuan(level, sym=False))
                model._modules[name].norm = nn.Sequential(child.norm ,MyQuan(level, sym=True))
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, PatchMergingConv):                
                model._modules[name].proj1_conv = nn.Sequential(MyQuan(level, sym=True), child.proj1_conv)
                model._modules[name].proj1_ReLU = nn.Sequential(child.proj1_ReLU , MyQuan(level, sym=False))
                model._modules[name].proj_res_conv = nn.Sequential(MyQuan(level, sym=True), child.proj_res_conv)
                model._modules[name].proj_res_ReLU = nn.Sequential(child.proj_res_ReLU ,MyQuan(level, sym=False))
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, SwinTransformerBlock):
                # print(children)
                qattn = QWindowAttention(dim=child.attn.num_heads*child.attn.head_dim, window_size=child.attn.window_size,num_heads=child.attn.num_heads,level=level)
                qattn.qkv = child.attn.qkv
                # qattn.q_norm = child.q_norm
                # qattn.k_norm = child.k_norm
                qattn.attn_drop = child.attn.attn_drop
                qattn.proj = child.attn.proj
                qattn.proj_drop = child.attn.proj_drop
                qattn.relative_position_bias_table = child.attn.relative_position_bias_table
                qattn.relative_position_index = child.attn.relative_position_index

                if isinstance(child.norm1, DyHT_ReLU):
                    model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=False, cal_loss=False))
                else:
                    model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True, cal_loss=False))
                model._modules[name].norm1[1].s_max.data = model._modules[name].norm1[0].gamma/model._modules[name].norm1[1].pos_max
                # print("model._modules[name].norm1[0].gamma",model._modules[name].norm1[0].gamma,"model._modules[name].norm1[1].pos_max",model._modules[name].norm1[1].pos_max)
                if isinstance(child.norm2, DyHT_ReLU):
                    model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=False, cal_loss=False))
                else:
                    model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True, cal_loss=False))
                model._modules[name].norm2[1].s_max.data = model._modules[name].norm2[0].gamma/model._modules[name].norm2[1].pos_max

                qattn.attn_softmax_quan.s_max.data = torch.tensor(1.0/qattn.attn_softmax_quan.pos_max)

                model._modules[name].attn = qattn
                # model._modules[name].act1 = MyQuan(level, sym=True)
                # model._modules[name].act2 = MyQuan(level, sym=True)
                # model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True))
                # model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True))
                # model._modules[name].mlp.fc1 = nn.Sequential(child.mlp.fc1,MyQuan(level, sym=False))
                model._modules[name].mlp.act = nn.Sequential(child.mlp.act,MyQuan(level, sym=False))
                model._modules[name].mlp.fc2 = nn.Sequential(child.mlp.fc2)
                # model._modules[name].addition1 = nn.Sequential(Addition(),MyQuan(level, sym=True))
                # model._modules[name].addition2 = nn.Sequential(Addition(),MyQuan(level, sym=True))
                # print("model._modules[name].addition1",model._modules[name].addition1)
                # print("index",cur_index,"myquan replace finish!!!!")
                cur_index = cur_index + 1
                is_need = True
            # if isinstance(child, Attention):
            #     # print(children)
            #     qattn = QAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=level)
            #     qattn.qkv = child.qkv
            #     # qattn.q_norm = child.q_norm
            #     # qattn.k_norm = child.k_norm
            #     qattn.attn_drop = child.attn_drop
            #     qattn.proj = child.proj
            #     qattn.proj_drop = child.proj_drop
            #     model._modules[name] = qattn
            #     print("index",cur_index,"myquan replace finish!!!!")
            #     cur_index = cur_index + 1
            #     is_need = True
            # elif isinstance(child,Mlp):
            #     model._modules[name].act = nn.Sequential(MyQuan(level,sym = False),child.act)
            #     model._modules[name].fc2 = nn.Sequential(child.fc2,MyQuan(level,sym = True))
            #     is_need = True
            elif isinstance(child, MyBatchNorm1d) or isinstance(child, DyT) or isinstance(child, DyHT) or isinstance(child, DyHT_ReLU):
                if isinstance(child, DyHT_ReLU):
                    model._modules[name] = nn.Sequential(child,MyQuan(level,sym = False))
                    model._modules[name][1].s_max.data = model._modules[name][0].gamma/model._modules[name][1].pos_max
                else:
                    model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                is_need = True
            # elif isinstance(child, DyHT):
            #     print("replace DyHT!!!")
            #     model._modules[name] = MyQuan(level,sym = True, alpha=1/(child.alpha.mean()*child.gamma.mean()))
            #     is_need = True
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True,first=True))
                is_need = True
            # elif isinstance(child, nn.Linear):
            #     model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
            #     is_need = True
            # elif isinstance(child, Block):
            #     model._modules[name].norm1 = nn.Sequential(child.norm1,MyQuan(level,sym = True))
            #     model._modules[name].norm2 = nn.Sequential(child.norm2,MyQuan(level,sym = True))
            #     is_need = False
            elif isinstance(child, nn.LayerNorm):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                # child.register_forward_hook(hook_layernorm)
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    
    def _weight_quantization(model,weight_bit):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            if not is_need:
                _weight_quantization(child,weight_bit)
                
    get_index(model)
    _myquan_replace(model,level)
    if weight_bit < 32:
        _weight_quantization(model,weight_bit)



def myquan_replace_resnet(model,level,weight_bit=32, is_softmax = True):
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

    def _myquan_replace(model,level):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.ReLU):
                model._modules[name] = MyQuan(level,sym = False)
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    
    def _weight_quantization(model,weight_bit):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            if not is_need:
                _weight_quantization(child,weight_bit)
                
    get_index(model)
    _myquan_replace(model,level)
    if weight_bit < 32:
        _weight_quantization(model,weight_bit)
