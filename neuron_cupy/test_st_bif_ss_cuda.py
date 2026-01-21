# test_st_bif_ss_cuda.py
# -*- coding: utf-8 -*-
import torch
from torch.cuda.amp import autocast, GradScaler
from st_bif_ss_cuda import ST_BIFNodeATGF_SS_Ref, ST_BIFNodeATGF_SS_CUDA, ST_BIFNodeATGF_SS_CUDAOp

def run_sequence(op_apply, T=16, B=8, D=256, dtype=torch.float32, amp=False):
    device = "cuda"
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)

    V = torch.zeros(B, D, device=device, dtype=dtype)
    Tstate = torch.zeros(B, D, device=device, dtype=dtype)

    v_th  = torch.tensor(0.5, device=device, dtype=dtype)
    T_max = torch.tensor(7.0, device=device, dtype=dtype)
    T_min = torch.tensor(-7.0, device=device, dtype=dtype)

    spikes_out = []
    V_out = []
    T_out = []

    loss = torch.zeros((), device=device, dtype=torch.float32)
    scaler = GradScaler(enabled=amp)

    if amp:
        ctx = autocast(dtype=torch.float16)
    else:
        class _Null:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        ctx = _Null()

    with ctx:
        for t in range(T):
            s, V, Tstate = op_apply(x_seq[t], V, Tstate, v_th, T_max, T_min, torch.tensor(t+1, device=device))
            spikes_out.append(s)
            V_out.append(V)
            T_out.append(Tstate)

            loss = loss + (s.float().pow(2).mean() + 1e-3*V.float().abs().mean() + 1e-4*Tstate.float().abs().mean())

    scaler.scale(loss).backward() if amp else loss.backward()

    return {
        "loss": loss.detach().float(),
        "spike_last": spikes_out[-1].detach().float(),
        "V_last": V_out[-1].detach().float(),
        "T_last": T_out[-1].detach().float(),
        "x_grad": x_seq.grad.detach().float()
    }

def compare_all(T=16, B=8, D=256):
    device = "cuda"
    print(f"[shape] T={T}, B={B}, D={D}")

    ref_apply  = lambda *args: ST_BIFNodeATGF_SS_Ref.apply(*args)
    cuda_apply = lambda *args: ST_BIFNodeATGF_SS_CUDA.apply(*args)

    ref_fp32  = run_sequence(ref_apply,  T,B,D, dtype=torch.float32, amp=False)
    cuda_fp32 = run_sequence(cuda_apply, T,B,D, dtype=torch.float32, amp=False)

    def report(name, A, B, atol=5e-5, rtol=1e-4):
        ma = (A - B).abs().max().item()
        ok = torch.allclose(A, B, atol=atol, rtol=rtol)
        print(f"{name}: allclose={ok}, max_abs_diff={ma:.3e}")

    print("\n[FP32]")
    report("loss",        ref_fp32["loss"],        cuda_fp32["loss"])
    report("spike_last",  ref_fp32["spike_last"],  cuda_fp32["spike_last"])
    report("V_last",      ref_fp32["V_last"],      cuda_fp32["V_last"])
    report("T_last",      ref_fp32["T_last"],      cuda_fp32["T_last"])
    report("x_grad",      ref_fp32["x_grad"],      cuda_fp32["x_grad"])

    ref_fp16  = run_sequence(ref_apply,  T,B,D, dtype=torch.float16, amp=True)
    cuda_fp16 = run_sequence(cuda_apply, T,B,D, dtype=torch.float16, amp=True)

    print("\n[AMP FP16]")
    report("loss",        ref_fp16["loss"],        cuda_fp16["loss"], atol=2e-3, rtol=1e-2)
    report("spike_last",  ref_fp16["spike_last"],  cuda_fp16["spike_last"], atol=2e-3, rtol=1e-2)
    report("V_last",      ref_fp16["V_last"],      cuda_fp16["V_last"], atol=2e-3, rtol=1e-2)
    report("T_last",      ref_fp16["T_last"],      cuda_fp16["T_last"], atol=2e-3, rtol=1e-2)
    report("x_grad",      ref_fp16["x_grad"],      cuda_fp16["x_grad"], atol=5e-3, rtol=2e-2)

if __name__ == "__main__":
    assert torch.cuda.is_available()
    compare_all(T=16, B=8, D=512)
