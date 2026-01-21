"""Model wrappers and replacement utilities for SNN/QANN."""

from .convert import (
    attn_convert,
    attn_convert_QAttention,
    attn_convert_SS,
    attn_convert_Swin,
    attn_convert_Swin_SS,
)
from .ms import SNNWrapper_MS
from .replace import (
    add_bn_in_mlp,
    add_convEmbed,
    adjust_LN2BN_Ratio,
    cal_l1_loss,
    myquan_replace,
    myquan_replace_resnet,
    open_dropout,
    remove_softmax,
    swap_BN_MLP_MHSA,
)
from .ss import SNNWrapper
from .utils import Judger, get_subtensors, reset_model

__all__ = [
    "Judger",
    "SNNWrapper",
    "SNNWrapper_MS",
    "add_bn_in_mlp",
    "add_convEmbed",
    "adjust_LN2BN_Ratio",
    "attn_convert",
    "attn_convert_QAttention",
    "attn_convert_SS",
    "attn_convert_Swin",
    "attn_convert_Swin_SS",
    "cal_l1_loss",
    "get_subtensors",
    "myquan_replace",
    "myquan_replace_resnet",
    "open_dropout",
    "remove_softmax",
    "reset_model",
    "swap_BN_MLP_MHSA",
]
