import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack


first = True
class ST_BIFNodeATGF_MS_CUDA(torch.autograd.Function):
    cuda_source = None  # Will store the CUDA source code
    count = 0
    v_thr1 = 0
    
    @staticmethod
    def _load_cuda_kernels():
        if ST_BIFNodeATGF_MS_CUDA.cuda_source is None:
            with open('/home/kang_you/SpikeZIP_transformer_Hybrid/neuron_cupy/cuda_snn_kernels_select.cu', 'r') as f:
                ST_BIFNodeATGF_MS_CUDA.cuda_source = f.read()
                
            # Load CUDA kernels into CuPy
            module = cp.RawModule(code=ST_BIFNodeATGF_MS_CUDA.cuda_source)
            ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp32 = module.get_function('forward_kernel_fp32')
            ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp32 = module.get_function('backward_kernel_fp32')
            ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp16 = module.get_function('forward_kernel_fp16')
            ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp16 = module.get_function('backward_kernel_fp16')

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, prefire: torch.Tensor):
        ST_BIFNodeATGF_MS_CUDA._load_cuda_kernels()
        Time = x_seq.shape[0]
        device = x_seq.device
        dtype = x_seq.dtype
        total_neurons = x_seq[0].numel()  # B * N * C
        max_spikes = max(1, int(total_neurons * 0.5))  # 至少允许 1 个
        is_half = x_seq.dtype == torch.float16
        if is_half:
            v_th = v_th.type(torch.float16)
            T_max = T_max.type(torch.float16)
            T_min = T_min.type(torch.float16)
        v_seq = []
        T_seq = []
        H_seq = []
        spike_seq = []
        keep_mask_seq = []
        v = x_seq[0]*0 + 0.5*v_th
        T = x_seq[0]*0
        spike = x_seq[0]*0
        T_seq.append(T)
        spike_seq.append(spike)
        H_seq.append(v)
        
        for t in range(Time):
            spike = spike * 0.0
            v = v + x_seq[t]
            H_seq.append(v)
            spike[torch.logical_and((torch.ge(v-v_th,0)), (torch.lt(T-T_max,0)))] = 1
            spike[torch.logical_and((torch.lt(v,0)), (torch.gt(T-T_min,0)))] = -1

            # ---------------- Gate机制：全局最多10%神经元发放 ----------------
            spike_mask = spike != 0
            num_spike = spike_mask.sum().item()
            if num_spike > max_spikes:
                # 扁平化
                v_flat = v.reshape(-1)            # (total_neurons,)
                mask_flat = spike_mask.reshape(-1)  # bool

                # 把非脉冲位置的 v 置为 -inf，保证 topk 只从脉冲位置中选
                # 使用 masked_fill 更安全（保持 dtype/device）
                v_masked = v_flat.clone()
                v_masked = v_masked.masked_fill(~mask_flat, float('-inf'))

                # 选出 top-k 的扁平索引
                topk = torch.topk(v_masked, k=int(max_spikes), largest=True)
                keep_idx_flat = topk.indices  # 长度 = max_spikes

                # 构造保留 mask（扁平）
                keep_mask_flat = torch.zeros_like(mask_flat, dtype=torch.bool, device=device)
                keep_mask_flat[keep_idx_flat] = True

                # reshape 回原始形状并应用到 spike（保留符号）
                keep_mask = keep_mask_flat.view_as(spike_mask)
                spike = spike * keep_mask
                keep_mask_seq.append(keep_mask.type(dtype))
            # ---------------------------------------------------------------
            else:
                keep_mask = torch.ones_like(spike, dtype=torch.bool, device=device)
                keep_mask_seq.append(keep_mask.type(dtype))

            if t < T_max:
                v = v - v_th * spike
            else:
                v = v - v_th * spike
            T = T + spike
            T_seq.append(T)
            spike_seq.append(spike)
            v_seq.append(v)

        H_seq = torch.stack(H_seq,dim=0)
        T_seq = torch.stack(T_seq,dim=0)
        v_seq = torch.stack(v_seq,dim=0)
        keep_mask_seq = torch.stack(keep_mask_seq,dim=0)
        spike_seq = torch.stack(spike_seq,dim=0)
        # print(is_half, spike_seq.dtype, v_seq.dtype, T_seq.dtype, x_seq.dtype, v_th.dtype, T_max.dtype, T_min.dtype)

        ctx.save_for_backward(spike_seq, keep_mask_seq, T_seq, H_seq, v_th, T_max, T_min)
        ctx.is_half = is_half
        
        return spike_seq[1:], v_seq, T_seq[1:]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_T_seq: torch.Tensor):
        # print(grad_v_seq.sum())
        spike_seq, keep_mask_seq, T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        is_half = ctx.is_half
        
        backward_kernel = ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp16 if is_half else ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp32
        
        time_steps = grad_spike_seq.shape[0]
        batch_size = grad_spike_seq.shape[1]
        features_flat = grad_spike_seq[0].numel() // batch_size
        
        # Prepare output tensor
        grad_x_seq = torch.zeros_like(grad_spike_seq)
        grad_V_th = torch.zeros_like(grad_spike_seq)
        
        # Convert tensors to CuPy
        grad_spike_seq_cp = cp.from_dlpack(to_dlpack(grad_spike_seq.contiguous()))
        grad_v_seq_cp = cp.from_dlpack(to_dlpack(grad_v_seq.contiguous()))
        grad_T_seq_cp = cp.from_dlpack(to_dlpack(grad_T_seq.contiguous()))
        spike_seq_cp = cp.from_dlpack(to_dlpack(spike_seq.contiguous()))
        keep_mask_seq_cp = cp.from_dlpack(to_dlpack(keep_mask_seq.contiguous()))
        T_seq_cp = cp.from_dlpack(to_dlpack(T_seq.contiguous()))
        H_seq_cp = cp.from_dlpack(to_dlpack(H_seq.contiguous()))
        v_th_cp = cp.from_dlpack(to_dlpack(v_th.contiguous()))
        T_max_cp = cp.from_dlpack(to_dlpack(T_max.contiguous()))
        T_min_cp = cp.from_dlpack(to_dlpack(T_min.contiguous()))
        grad_x_seq_cp = cp.from_dlpack(to_dlpack(grad_x_seq.contiguous()))
        grad_V_th_cp = cp.from_dlpack(to_dlpack(grad_V_th.contiguous()))
        
        # Launch kernel
        threads_per_block = 256
        blocks = (batch_size * features_flat + threads_per_block - 1) // threads_per_block
        
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_seq_cp, grad_T_seq_cp,
             spike_seq_cp, keep_mask_seq_cp, T_seq_cp, H_seq_cp,
             v_th_cp, T_max_cp, T_min_cp,
             grad_x_seq_cp,
             grad_V_th_cp,
             batch_size, time_steps, features_flat)
        )
        
        # Convert back to PyTorch
        grad_x = from_dlpack(grad_x_seq_cp.toDlpack())
        grad_V = from_dlpack(grad_V_th_cp.toDlpack()).mean().unsqueeze(0)

        # print(v_th_cp, grad_V)
        
        # print("ST_BIFNodeATGF_MS_CUDA Backward:", grad_x_seq.dtype, grad_v_cp.dtype, grad_T_seq_cp.dtype, spike_seq_cp.dtype, spike_seq.dtype, H_seq.dtype, T_seq.dtype, v_th.dtype, T_max.dtype, T_min.dtype)

        # global first
        # if first:
        #     print('=================================================================')
        #     # print("v_th",v_th)
        #     # ST_BIFNodeATGF_MS_CUDA.v_thr1 = ST_BIFNodeATGF_MS_CUDA.v_thr1 + v_th
        #     # ST_BIFNodeATGF_MS_CUDA.count = ST_BIFNodeATGF_MS_CUDA.count + 1
        #     # print("vthr Sum",ST_BIFNodeATGF_MS_CUDA.v_thr1, "count", ST_BIFNodeATGF_MS_CUDA.count)
            
        #     for t in range(4):
        #         print("t=",t+1)
        #         print("H_seq_cp",H_seq[t+1].abs().mean())
        #         print("grad_x",grad_x[t].abs().mean())
        #         print("grad_spike_seq",grad_spike_seq[t].abs().mean())
        #         print("grad_x/grad_spike_seq",(torch.abs(grad_x)/(torch.abs(grad_spike_seq)+1e-5))[t].abs().mean())
        #     first = False 
            
        return grad_x, grad_V, None, None, None