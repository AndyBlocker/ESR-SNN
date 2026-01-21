# st_bif_ss_cuda.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.cpp_extension import load
from torch.cuda.amp import custom_fwd, custom_bwd

_THIS_DIR = Path(__file__).resolve().parent
_SRC = str(_THIS_DIR / "st_bif_ss_cuda_kernel.cu")

_st = load(
    name="_st_bif_ss_cuda",
    sources=[_SRC],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

def _as_broadcastable_param(x_like: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if p.numel() == 1:
        return p
    if p.shape == x_like.shape:
        return p
    raise RuntimeError("Parameter must be scalar or same shape as x.")

class ST_BIFNodeATGF_SS_CUDA(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx,
                x_t: torch.Tensor,
                V_t_1: torch.Tensor,
                T_t_1: torch.Tensor,
                v_th: torch.Tensor,
                T_max: torch.Tensor,
                T_min: torch.Tensor,
                t: torch.Tensor = None):
        assert x_t.is_cuda, "x_t must be CUDA"
        x_t = x_t.contiguous()
        V_t_1 = V_t_1.contiguous()
        T_t_1 = T_t_1.contiguous()

        v_th = _as_broadcastable_param(x_t, v_th).contiguous()
        T_max = _as_broadcastable_param(x_t, T_max).contiguous()
        T_min = _as_broadcastable_param(x_t, T_min).contiguous()

        spike, V_t, T_t, H_t = _st.st_bifnode_forward(x_t, V_t_1, T_t_1, v_th, T_max, T_min)
        ctx.save_for_backward(H_t, T_t_1, v_th, T_max, T_min)
        return spike, V_t, T_t

    @staticmethod
    @custom_bwd
    def backward(ctx,
                 grad_spike_t: torch.Tensor,
                 grad_v_t: torch.Tensor,
                 grad_T_t: torch.Tensor):
        H_t, T_t_1, v_th, T_max, T_min = ctx.saved_tensors
        grad_x_t, grad_V_t_1, grad_T_t_1 = _st.st_bifnode_backward(
            grad_spike_t.contiguous(),
            grad_v_t.contiguous(),
            grad_T_t.contiguous(),
            H_t.contiguous(),
            T_t_1.contiguous(),
            v_th.contiguous(),
            T_max.contiguous(),
            T_min.contiguous()
        )
        return grad_x_t, grad_V_t_1, grad_T_t_1, None, None, None, None


def dtheta_gauss_norm_torch(x: torch.Tensor, vthr: torch.Tensor) -> torch.Tensor:
    sqrt_2pi = 2.506628253
    sigma = torch.clamp(vthr / sqrt_2pi, min=1e-3)
    a = 1.0 / (sigma * sqrt_2pi)
    u = -(x * x) / (2.0 * sigma * sigma)
    return a * torch.exp(u)

class ST_BIFNodeATGF_SS_Ref(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_t, V_t_1, T_t_1, v_th, T_max, T_min, t=None):
        H_t = V_t_1 + x_t
        pos = (H_t >= v_th) & ((T_t_1 - T_max) < 0)
        neg = (H_t < 0) & ((T_t_1 - T_min) > 0)
        spike = torch.where(pos, torch.ones_like(H_t),
                            torch.where(neg, -torch.ones_like(H_t),
                                        torch.zeros_like(H_t)))
        V_t = H_t - v_th * spike
        T_t = T_t_1 + spike
        ctx.save_for_backward(H_t, T_t_1, v_th, T_max, T_min)
        return spike, V_t, T_t

    @staticmethod
    def backward(ctx, grad_spike_t, grad_v_t, grad_T_t):
        H_t, T_t_1, v_th, T_max, T_min = ctx.saved_tensors
        th_Tmax = ((T_max - T_t_1) > 0).to(H_t.dtype)
        th_Tmin = ((T_t_1 - T_min) > 0).to(H_t.dtype)
        th_eq_H = ((H_t - v_th) >= 0).to(H_t.dtype)
        th_negH = ((-H_t) > 0).to(H_t.dtype)

        g1 = dtheta_gauss_norm_torch(H_t - v_th, v_th)
        g2 = dtheta_gauss_norm_torch(-H_t, v_th)
        dg_Tmax = dtheta_gauss_norm_torch(T_max - T_t_1, v_th)
        dg_Tmin = dtheta_gauss_norm_torch(T_t_1 - T_min, v_th)

        grad_Tt_to_H = g1 * th_Tmax + g2 * th_Tmin
        grad_Yt_to_Tprev = -(th_eq_H * dg_Tmax + th_negH * dg_Tmin)

        tmp = grad_spike_t - v_th * grad_v_t + grad_T_t
        grad_x_t = tmp * grad_Tt_to_H + grad_v_t
        grad_T_t_1 = tmp * grad_Yt_to_Tprev + grad_T_t
        grad_V_t_1 = grad_x_t
        return grad_x_t, grad_V_t_1, grad_T_t_1, None, None, None, None

class ST_BIFNodeATGF_SS_CUDAOp(nn.Module):
    def forward(self, x_t, V_t_1, T_t_1, v_th, T_max, T_min, t=None):
        return ST_BIFNodeATGF_SS_CUDA.apply(x_t, V_t_1, T_t_1, v_th, T_max, T_min, t)
