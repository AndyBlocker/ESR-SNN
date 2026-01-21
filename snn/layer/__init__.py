"""SNN/QANN layers and utilities."""

from .attention import (
    AttentionMulti,
    AttentionMulti1,
    Attention_no_softmax,
    QAttention,
    QAttention_without_softmax,
    SAttention,
    SAttention_without_softmax,
    SAttention_without_softmax_SS,
    multi,
    multi1,
)
from .dyt import DyHT, DyHT_ReLU, DyHT_Softmax, DyT, SDyHT, SDyHT_SS, spiking_dyt
from .embed import PatchEmbedConv, PatchMergingConv
from .if_neuron import IFNeuron
from .linear import LLConv2d, LLConv2d_MS, LLLinear, LLLinear_MS
from .norm import (
    LN2BNorm,
    MLP_BN,
    MyBatchNorm1d,
    MyBatchNorm1d_SS,
    MyLayerNorm,
    Spiking_LayerNorm,
    Spiking_LayerNorm_SS,
)
from .pooling import SpikeMaxPooling, SpikeMaxPooling_SS
from .quant import MyQuan, MyQuanRound, QuanConv2d, QuanLinear
from .record import (
    Addition,
    save_fc_input_for_bin_snn,
    save_input_for_bin_snn_4dim,
    save_input_for_bin_snn_5dim,
    save_module_inout,
)
from .softmax import spiking_softmax, spiking_softmax_ss
from .st_bifneuron_ms import ST_BIFNeuron_MS
from .st_bifneuron_ss import ST_BIFNodeATGF_SS, ST_BIFNeuron_SS
from .utils import (
    cal_overfire_loss,
    clip,
    floor_pass,
    grad_scale,
    round_pass,
    set_init_false,
    theta,
    theta_backward,
    theta_eq,
)
from .window_attention import QWindowAttention, SWindowAttention, SWindowAttention_SS, WindowAttention_no_softmax

__all__ = [
    "Addition",
    "AttentionMulti",
    "AttentionMulti1",
    "Attention_no_softmax",
    "DyHT",
    "DyHT_ReLU",
    "DyHT_Softmax",
    "DyT",
    "IFNeuron",
    "LLConv2d",
    "LLConv2d_MS",
    "LLLinear",
    "LLLinear_MS",
    "LN2BNorm",
    "MLP_BN",
    "MyBatchNorm1d",
    "MyBatchNorm1d_SS",
    "MyLayerNorm",
    "MyQuan",
    "MyQuanRound",
    "PatchEmbedConv",
    "PatchMergingConv",
    "QAttention",
    "QAttention_without_softmax",
    "QWindowAttention",
    "QuanConv2d",
    "QuanLinear",
    "SAttention",
    "SAttention_without_softmax",
    "SAttention_without_softmax_SS",
    "SDyHT",
    "SDyHT_SS",
    "SpikeMaxPooling",
    "SpikeMaxPooling_SS",
    "Spiking_LayerNorm",
    "Spiking_LayerNorm_SS",
    "ST_BIFNodeATGF_SS",
    "ST_BIFNeuron_MS",
    "ST_BIFNeuron_SS",
    "SWindowAttention",
    "SWindowAttention_SS",
    "WindowAttention_no_softmax",
    "cal_overfire_loss",
    "clip",
    "floor_pass",
    "grad_scale",
    "multi",
    "multi1",
    "round_pass",
    "save_fc_input_for_bin_snn",
    "save_input_for_bin_snn_4dim",
    "save_input_for_bin_snn_5dim",
    "save_module_inout",
    "set_init_false",
    "spiking_dyt",
    "spiking_softmax",
    "spiking_softmax_ss",
    "theta",
    "theta_backward",
    "theta_eq",
]
