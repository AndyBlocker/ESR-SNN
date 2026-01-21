# -*- coding: utf-8 -*-
import os
from pathlib import Path
import importlib.resources as pkg_resources

import torch
from torch.utils.cpp_extension import load as _load_ext


class ST_BIFNodeATGF_MS_CUDA(torch.autograd.Function):

    _built = False
    _ext_mod = None

    @staticmethod
    def _find_cuda_source() -> Path:
        candidates = []
        here = Path(__file__).resolve().parent
        candidates.append(here / "cuda_snn_kernels_new.cu")
        candidates.append(here / "neuron_cupy" / "cuda_snn_kernels_new.cu")

        env_path = os.getenv("CUDA_SNN_KERNELS_PATH")
        if env_path:
            candidates.append(Path(env_path))

        try:
            resource_path = pkg_resources.files("neuron_cupy") / "cuda_snn_kernels_new.cu"
            candidates.append(resource_path)
        except ModuleNotFoundError:
            pass

        for p in candidates:
            try:
                if Path(p).is_file():
                    return Path(p)
            except Exception:
                continue

        tried = "\n  - ".join(str(p) for p in candidates)
        raise FileNotFoundError("cuda_snn_kernels_new.cu not found, tried: \n  - " + tried)

    @staticmethod
    def _ensure_built():
        if ST_BIFNodeATGF_MS_CUDA._built:
            return
        src = ST_BIFNodeATGF_MS_CUDA._find_cuda_source()

        mod = _load_ext(
            name="snn_cuda_ext",
            sources=[str(src)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
            ],
            verbose=False,
        )
        ST_BIFNodeATGF_MS_CUDA._ext_mod = mod
        ST_BIFNodeATGF_MS_CUDA._built = True

    @staticmethod
    def forward(ctx,
                x_seq: torch.Tensor,
                v_th: torch.Tensor,
                T_max: torch.Tensor,
                T_min: torch.Tensor,
                prefire: torch.Tensor):

        ST_BIFNodeATGF_MS_CUDA._ensure_built()

        if not x_seq.is_cuda:
            raise RuntimeError("x_seq must be a CUDA Tensor")

        dtype = x_seq.dtype
        device = x_seq.device
        v_th = v_th.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        T_max = T_max.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        T_min = T_min.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        prefire = prefire.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        x_seq = x_seq.contiguous()

        spike_all, v_out, T_all, H_all = torch.ops.snn.st_bif_forward(
            x_seq, v_th, T_max, T_min, prefire
        )

        ctx.save_for_backward(spike_all, T_all, H_all, v_th, T_max, T_min)
        return spike_all[1:], v_out, T_all[1:]

    @staticmethod
    def backward(ctx,
                 grad_spike_seq: torch.Tensor,
                 grad_v: torch.Tensor,
                 grad_T_seq: torch.Tensor):
        spike_all, T_all, H_all, v_th, T_max, T_min = ctx.saved_tensors

        grad_spike_seq = grad_spike_seq.contiguous()
        grad_v = grad_v.contiguous()
        grad_T_seq = grad_T_seq.contiguous()

        grad_x = torch.ops.snn.st_bif_backward(
            grad_spike_seq, grad_v, grad_T_seq,
            spike_all, T_all, H_all,
            v_th, T_max, T_min
        )
        return grad_x, None, None, None, None


def st_bifnode_atgf_ms(x_seq, v_th, T_max, T_min, prefire):
    return ST_BIFNodeATGF_MS_CUDA.apply(x_seq, v_th, T_max, T_min, prefire)


if __name__ == "__main__":
    x_seq = torch.randn(10, 10, 10).to("cuda")
    v_th = torch.randn(1).to("cuda")
    T_max = torch.randn(1).to("cuda")
    T_min = torch.randn(1).to("cuda")
    prefire = torch.randn(1).to("cuda") 
    spike_seq, v_out, T_seq = st_bifnode_atgf_ms(x_seq, v_th, T_max, T_min, prefire)
    print(spike_seq, v_out, T_seq)