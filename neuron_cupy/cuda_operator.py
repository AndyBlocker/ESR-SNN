
import os
import cupy as cp
import torch


class ST_BIFNodeATGF_MS_CUDA(torch.autograd.Function):
    """
    CuPy RawKernel wrapper for SNN kernels.

    Key fix:
      - Avoid cp.from_dlpack(to_dlpack(torch.bfloat16)) which can hard-crash in many CuPy builds.
      - For BF16 tensors, create a CuPy view with dtype=uint16 (2 bytes) via UnownedMemory and pass it
        to kernels that expect __nv_bfloat16* pointers. This works even if CuPy has no bfloat16 dtype.
      - Also avoid DLPack entirely (more robust + no extra copies).
    """

    cuda_source = None
    _module = None

    # kernel handles
    forward_kernel_fp32 = None
    backward_kernel_fp32 = None
    forward_kernel_fp16 = None
    backward_kernel_fp16 = None
    forward_kernel_bf16 = None
    backward_kernel_bf16 = None

    @staticmethod
    def _resolve_kernel_path() -> str:
        # Prefer env override
        p = os.getenv("SNN_CUDA_KERNEL_PATH", "")
        if p and os.path.exists(p):
            return p

        # Try colocated files
        here = os.path.dirname(__file__)
        for name in ("cuda_snn_kernels_bf16.cu", "cuda_snn_kernels.cu"):
            cand = os.path.join(here, name)
            if os.path.exists(cand):
                return cand

        # Last resort: keep original hardcoded path if it exists
        hard = "/home/kang_you/SpikeZIP_transformer_Hybrid_CVPR/neuron_cupy/cuda_snn_kernels.cu"
        if os.path.exists(hard):
            return hard

        raise FileNotFoundError(
            "Cannot find CUDA kernel source. Set SNN_CUDA_KERNEL_PATH to your .cu file."
        )

    @staticmethod
    def _load_cuda_kernels():
        if ST_BIFNodeATGF_MS_CUDA._module is not None:
            return

        path = ST_BIFNodeATGF_MS_CUDA._resolve_kernel_path()
        with open(path, "r") as f:
            ST_BIFNodeATGF_MS_CUDA.cuda_source = f.read()

        # Compile kernels (NVRTC). c++14 helps when including cuda_bf16.h.
        module = cp.RawModule(code=ST_BIFNodeATGF_MS_CUDA.cuda_source, options=("--std=c++14",))
        ST_BIFNodeATGF_MS_CUDA._module = module

        ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp32 = module.get_function("forward_kernel_fp32")
        ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp32 = module.get_function("backward_kernel_fp32")
        ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp16 = module.get_function("forward_kernel_fp16")
        ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp16 = module.get_function("backward_kernel_fp16")

        # Optional BF16 kernels (present in cuda_snn_kernels_bf16.cu)
        try:
            ST_BIFNodeATGF_MS_CUDA.forward_kernel_bf16 = module.get_function("forward_kernel_bf16")
            ST_BIFNodeATGF_MS_CUDA.backward_kernel_bf16 = module.get_function("backward_kernel_bf16")
        except Exception:
            ST_BIFNodeATGF_MS_CUDA.forward_kernel_bf16 = None
            ST_BIFNodeATGF_MS_CUDA.backward_kernel_bf16 = None

    @staticmethod
    def _torch_to_cupy_view(t: torch.Tensor):
        """
        Zero-copy view of a CUDA torch Tensor as a CuPy ndarray without DLPack.
        BF16 is exposed as uint16 (same storage width) to bypass lack of CuPy bf16 dtype.
        """
        if not t.is_cuda:
            raise ValueError("Expected CUDA tensor")
        t = t.contiguous()
        dev = t.device.index if t.device.index is not None else 0
        cp.cuda.Device(dev).use()

        nbytes = t.numel() * t.element_size()
        mem = cp.cuda.UnownedMemory(t.data_ptr(), nbytes, t)
        mp = cp.cuda.MemoryPointer(mem, 0)

        if t.dtype == torch.bfloat16:
            # CuPy may not support bf16 dtype; use uint16 view (2 bytes) and reinterpret in kernel.
            cpdtype = cp.uint16
        elif t.dtype == torch.float16:
            cpdtype = cp.float16
        elif t.dtype == torch.float32:
            cpdtype = cp.float32
        else:
            raise TypeError(f"Unsupported dtype for CuPy view: {t.dtype}")

        arr = cp.ndarray(t.shape, dtype=cpdtype, memptr=mp)
        return arr, dev

    @staticmethod
    def _cupy_stream_for_device(dev: int):
        # Ensure CuPy launches on the same CUDA stream as PyTorch for that device.
        ts = torch.cuda.current_stream(dev).cuda_stream
        return cp.cuda.ExternalStream(ts)

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, prefire: torch.Tensor):
        ST_BIFNodeATGF_MS_CUDA._load_cuda_kernels()

        # Dimensions
        time_steps, batch_size, *features = x_seq.shape
        features_flat = int(torch.tensor(features, device="cpu").prod().item())

        # Choose kernel by dtype
        if x_seq.dtype == torch.float16:
            forward_kernel = ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp16
            v_th_k = v_th.to(torch.float16)
            T_max_k = T_max.to(torch.float16)
            T_min_k = T_min.to(torch.float16)
            prefire_k = prefire.to(torch.float16)
        elif x_seq.dtype == torch.bfloat16:
            if ST_BIFNodeATGF_MS_CUDA.forward_kernel_bf16 is None:
                raise RuntimeError(
                    "BF16 input requested but BF16 kernels not found. "
                    "Point SNN_CUDA_KERNEL_PATH to cuda_snn_kernels_bf16.cu (with forward_kernel_bf16/backward_kernel_bf16)."
                )
            forward_kernel = ST_BIFNodeATGF_MS_CUDA.forward_kernel_bf16
            # For true BF16 path, parameters must also be bf16 so kernel reads correct type.
            v_th_k = v_th.to(torch.bfloat16)
            T_max_k = T_max.to(torch.bfloat16)
            T_min_k = T_min.to(torch.bfloat16)
            prefire_k = prefire.to(torch.bfloat16)
        else:
            forward_kernel = ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp32
            v_th_k = v_th.to(torch.float32)
            T_max_k = T_max.to(torch.float32)
            T_min_k = T_min.to(torch.float32)
            prefire_k = prefire.to(torch.float32)

        # Outputs (same dtype as x_seq)
        spike_seq_out = torch.zeros((time_steps + 1, batch_size, *features), device=x_seq.device, dtype=x_seq.dtype)
        T_seq_out = torch.zeros_like(spike_seq_out)
        H_seq_out = torch.zeros_like(spike_seq_out)
        v_out = torch.zeros((time_steps, batch_size, *features), device=x_seq.device, dtype=x_seq.dtype)

        # CuPy views (zero-copy)
        x_seq_cp, dev = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(x_seq)
        v_th_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(v_th_k)
        T_max_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(T_max_k)
        T_min_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(T_min_k)
        prefire_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(prefire_k)

        spike_seq_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(spike_seq_out)
        v_out_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(v_out)
        T_seq_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(T_seq_out)
        H_seq_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(H_seq_out)

        threads_per_block = 256
        blocks = (batch_size * features_flat + threads_per_block - 1) // threads_per_block

        stream = ST_BIFNodeATGF_MS_CUDA._cupy_stream_for_device(dev)
        with stream:
            forward_kernel(
                (blocks,), (threads_per_block,),
                (x_seq_cp, v_th_cp, T_max_cp, T_min_cp, prefire_cp,
                 spike_seq_cp, v_out_cp, T_seq_cp, H_seq_cp,
                 batch_size, time_steps, features_flat)
            )

        # Save for backward
        ctx.save_for_backward(spike_seq_out, T_seq_out, H_seq_out, v_th_k, T_max_k, T_min_k)
        ctx.x_dtype = x_seq.dtype
        ctx.v_th_in_dtype = v_th.dtype
        ctx.v_th_in_shape = tuple(v_th.shape)

        return spike_seq_out[1:], v_out, T_seq_out[1:]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_T_seq: torch.Tensor):
        spike_seq, T_seq, H_seq, v_th_k, T_max_k, T_min_k = ctx.saved_tensors
        x_dtype = ctx.x_dtype

        if x_dtype == torch.float16:
            backward_kernel = ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp16
        elif x_dtype == torch.bfloat16:
            backward_kernel = ST_BIFNodeATGF_MS_CUDA.backward_kernel_bf16
            if backward_kernel is None:
                raise RuntimeError("BF16 backward kernel not available.")
        else:
            backward_kernel = ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp32

        time_steps = grad_spike_seq.shape[0]
        batch_size = grad_spike_seq.shape[1]
        features_flat = grad_spike_seq[0].numel() // batch_size

        # Outputs
        grad_x_seq = torch.zeros_like(grad_spike_seq)
        grad_V_th = torch.zeros_like(grad_spike_seq)

        # CuPy views
        grad_spike_cp, dev = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(grad_spike_seq)
        grad_v_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(grad_v_seq)
        grad_T_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(grad_T_seq)

        spike_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(spike_seq)
        T_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(T_seq)
        H_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(H_seq)

        v_th_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(v_th_k)
        T_max_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(T_max_k)
        T_min_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(T_min_k)

        grad_x_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(grad_x_seq)
        grad_V_cp, _ = ST_BIFNodeATGF_MS_CUDA._torch_to_cupy_view(grad_V_th)

        threads_per_block = 256
        blocks = (batch_size * features_flat + threads_per_block - 1) // threads_per_block

        stream = ST_BIFNodeATGF_MS_CUDA._cupy_stream_for_device(dev)
        with stream:
            backward_kernel(
                (blocks,), (threads_per_block,),
                (grad_spike_cp, grad_v_cp, grad_T_cp,
                 spike_cp, T_cp, H_cp,
                 v_th_cp, T_max_cp, T_min_cp,
                 grad_x_cp, grad_V_cp,
                 batch_size, time_steps, features_flat)
            )

        # Reduce grad_V in FP32 and cast back to original v_th dtype/shape
        grad_V = grad_V_th.float().mean()
        v_th_dtype = ctx.v_th_in_dtype
        v_shape = ctx.v_th_in_shape
        grad_V = grad_V.to(v_th_dtype)
        if v_shape == ():
            grad_V = grad_V
        else:
            grad_V = grad_V.reshape(v_shape)

        return grad_x_seq, grad_V, None, None, None
