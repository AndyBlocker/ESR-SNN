import torch

from snn.layer import (
    IFNeuron,
    LLLinear,
    LLLinear_MS,
    QAttention,
    QAttention_without_softmax,
    QWindowAttention,
    SAttention,
    SAttention_without_softmax,
    SAttention_without_softmax_SS,
    ST_BIFNeuron_MS,
    ST_BIFNeuron_SS,
    SWindowAttention,
    SWindowAttention_SS,
)


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
    SAttn.attn_softmax_IF.suppress_over_fire = suppress_over_fire
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
    # if isinstance(SAttn.proj_IF,IFNeuron):
    #     SAttn.proj_IF.is_init = False
    # elif isinstance(SAttn.proj_IF,ST_BIFNeuron_SS) or isinstance(SAttn.proj_IF,ST_BIFNeuron_MS):
    #     SAttn.proj_IF.init = True

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.proj_drop = QAttn.proj_drop

def attn_convert_QAttention_SS(QAttn:QAttention,SAttn:SAttention,level,neuron_type, T):
    SAttn.qkv = LLLinear(linear=QAttn.qkv, neuron_type="ST-BIF", time_step=T, level=level)
    SAttn.proj = LLLinear(linear=QAttn.proj, neuron_type="ST-BIF", time_step=T, level=level)

    SAttn.q_IF.neuron_type = neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.T = T
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max_buf
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min_buf
    if isinstance(SAttn.q_IF, IFNeuron):
        SAttn.q_IF.is_init = False
    elif isinstance(SAttn.q_IF, ST_BIFNeuron_SS) or isinstance(SAttn.q_IF, ST_BIFNeuron_MS):
        SAttn.q_IF.init = True

    SAttn.k_IF.neuron_type = neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.T = T
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max_buf
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min_buf
    if isinstance(SAttn.k_IF, IFNeuron):
        SAttn.k_IF.is_init = False
    elif isinstance(SAttn.k_IF, ST_BIFNeuron_SS) or isinstance(SAttn.k_IF, ST_BIFNeuron_MS):
        SAttn.k_IF.init = True

    SAttn.v_IF.neuron_type = neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.T = T
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max_buf
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min_buf
    if isinstance(SAttn.v_IF, IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF, ST_BIFNeuron_SS) or isinstance(SAttn.v_IF, ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_IF.neuron_type = neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = QAttn.attn_quan.s.data
    SAttn.attn_IF.T = T
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max_buf
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min_buf
    if isinstance(SAttn.attn_IF, IFNeuron):
        SAttn.attn_IF.is_init = False
    elif isinstance(SAttn.attn_IF, ST_BIFNeuron_SS) or isinstance(SAttn.attn_IF, ST_BIFNeuron_MS):
        SAttn.attn_IF.init = True

    if hasattr(QAttn, "attn_softmax_quan"):
        SAttn.attn_softmax_IF.neuron_type = neuron_type
        SAttn.attn_softmax_IF.level = level
        SAttn.attn_softmax_IF.q_threshold.data = QAttn.attn_softmax_quan.s.data
        SAttn.attn_softmax_IF.T = T
        SAttn.attn_softmax_IF.pos_max = QAttn.attn_softmax_quan.pos_max_buf
        SAttn.attn_softmax_IF.neg_min = QAttn.attn_softmax_quan.neg_min_buf
        if isinstance(SAttn.attn_softmax_IF, IFNeuron):
            SAttn.attn_softmax_IF.is_init = False
        elif isinstance(SAttn.attn_softmax_IF, ST_BIFNeuron_SS) or isinstance(SAttn.attn_softmax_IF, ST_BIFNeuron_MS):
            SAttn.attn_softmax_IF.init = True

    SAttn.after_attn_IF.neuron_type = neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.T = T
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max_buf
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min_buf
    if isinstance(SAttn.after_attn_IF, IFNeuron):
        SAttn.after_attn_IF.is_init = False
    elif isinstance(SAttn.after_attn_IF, ST_BIFNeuron_SS) or isinstance(SAttn.after_attn_IF, ST_BIFNeuron_MS):
        SAttn.after_attn_IF.init = True

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
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max_buf
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min_buf
    if isinstance(SAttn.v_IF,IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF,ST_BIFNeuron_SS) or isinstance(SAttn.v_IF,ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_IF.neuron_type= neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = QAttn.attn_quan.s.data
    SAttn.attn_IF.T = T
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max_buf
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min_buf
    if isinstance(SAttn.attn_IF,IFNeuron):
        SAttn.attn_IF.is_init = False
    elif isinstance(SAttn.attn_IF,ST_BIFNeuron_SS) or isinstance(SAttn.attn_IF,ST_BIFNeuron_MS):
        SAttn.attn_IF.init = True
        # SAttn.attn_IF.q_threshold.data = torch.tensor(0.125)

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
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
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max_buf
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min_buf
    if isinstance(SAttn.v_IF,IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF,ST_BIFNeuron_SS) or isinstance(SAttn.v_IF,ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_softmax_IF.neuron_type= neuron_type
    SAttn.attn_softmax_IF.level = level
    SAttn.attn_softmax_IF.q_threshold.data = min(QAttn.attn_softmax_quan.s.data,QAttn.attn_softmax_quan.s_max.data)
    SAttn.attn_softmax_IF.T = T
    SAttn.attn_softmax_IF.pos_max = QAttn.attn_softmax_quan.pos_max_buf
    SAttn.attn_softmax_IF.neg_min = QAttn.attn_softmax_quan.neg_min_buf
    SAttn.attn_softmax_IF.suppress_over_fire = suppress_over_fire
    if isinstance(SAttn.attn_softmax_IF,IFNeuron):
        SAttn.attn_softmax_IF.is_init = False
    elif isinstance(SAttn.attn_softmax_IF,ST_BIFNeuron_SS) or isinstance(SAttn.attn_softmax_IF,ST_BIFNeuron_MS):
        SAttn.attn_softmax_IF.init = True
        

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
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
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max_buf
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min_buf
    if isinstance(SAttn.v_IF,IFNeuron):
        SAttn.v_IF.is_init = False
    elif isinstance(SAttn.v_IF,ST_BIFNeuron_SS) or isinstance(SAttn.v_IF,ST_BIFNeuron_MS):
        SAttn.v_IF.init = True

    SAttn.attn_softmax_IF.neuron_type= neuron_type
    SAttn.attn_softmax_IF.level = level
    SAttn.attn_softmax_IF.q_threshold.data = min(QAttn.attn_softmax_quan.s.data,QAttn.attn_softmax_quan.s_max.data)
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
    # SAttn.proj_IF.suppress_over_fire = suppress_over_fire
    # if isinstance(SAttn.proj_IF,IFNeuron):
    #     SAttn.proj_IF.is_init = False
    # elif isinstance(SAttn.proj_IF,ST_BIFNeuron_SS) or isinstance(SAttn.proj_IF,ST_BIFNeuron_MS):
    #     SAttn.proj_IF.init = True

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.attn_drop.p = 0.0
    SAttn.proj_drop = QAttn.proj_drop
    SAttn.proj_drop.p = 0.0
