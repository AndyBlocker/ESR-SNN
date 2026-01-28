import warnings

import torch
import torch.nn as nn

from snn.nvtx import nvtx_range


try:
    from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA as _ST_BIFNodeATGF_MS_CUDA
except Exception:  # pragma: no cover
    _ST_BIFNodeATGF_MS_CUDA = None


_MS_FORCE_TORCH = False
_MS_FALLBACK_WARNED = False


def _theta_backward_ms(x: torch.Tensor, v_thr: torch.Tensor):
    sigmoid = torch.sigmoid(4.0 * x / v_thr)
    return (4.0 / v_thr) * sigmoid * (1.0 - sigmoid)


def _theta_ms(x: torch.Tensor):
    return 1.0 * torch.gt(x, 0)


def _theta_eq_ms(x: torch.Tensor):
    return 1.0 * torch.ge(x, 0)


class ST_BIFNodeATGF_MS_Torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, prefire: torch.Tensor, tmax_int: int):
        time_steps = x_seq.shape[0]

        # Ensure scalars are in the same dtype/device as x_seq for stable comparisons
        v_th = v_th.to(dtype=x_seq.dtype, device=x_seq.device)
        T_max = T_max.to(dtype=x_seq.dtype, device=x_seq.device)
        T_min = T_min.to(dtype=x_seq.dtype, device=x_seq.device)
        prefire = prefire.to(dtype=x_seq.dtype, device=x_seq.device)

        # Pre-allocate to avoid Python list growth + stack overhead.
        H_seq = x_seq.new_empty((time_steps + 1,) + x_seq.shape[1:])
        T_seq = x_seq.new_empty((time_steps + 1,) + x_seq.shape[1:])
        spike_seq = x_seq.new_empty((time_steps + 1,) + x_seq.shape[1:])

        v = x_seq[0] * 0.0 + 0.5 * v_th + prefire * v_th
        T = x_seq[0] * 0.0
        spike = x_seq[0] * 0.0

        H_seq[0] = v
        T_seq[0] = T
        spike_seq[0] = spike

        prefire_term = None
        if int(tmax_int) > 0:
            # Note: prefire is typically 0.0; computing it unconditionally avoids any host sync.
            prefire_term = prefire * v_th / T_max

        one = x_seq.new_ones(())
        neg_one = x_seq.new_full((), -1.0)
        zero = x_seq.new_zeros(())

        for t in range(time_steps):
            v = v + x_seq[t]
            H_seq[t + 1] = v

            pos_mask = torch.logical_and(torch.ge(v, v_th), torch.lt(T, T_max))
            neg_mask = torch.logical_and(torch.lt(v, 0), torch.gt(T, T_min))
            spike = torch.where(neg_mask, neg_one, torch.where(pos_mask, one, zero))

            v = v - v_th * spike
            if prefire_term is not None and t < int(tmax_int):
                v = v - prefire_term

            T = T + spike
            T_seq[t + 1] = T
            spike_seq[t + 1] = spike

        # NOTE: spike_seq is not needed for backward (only grad_spike_seq is).
        # Saving it roughly doubles the saved activation footprint for this op.
        ctx.save_for_backward(T_seq, H_seq, v_th, T_max, T_min)
        return spike_seq[1:], v, T_seq[1:]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_T_seq: torch.Tensor):
        T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        time_steps = H_seq.shape[0] - 1

        grad_x_seq = H_seq.new_empty((time_steps,) + H_seq.shape[1:])
        if grad_spike_seq is None:
            grad_Y_seq = H_seq.new_zeros((time_steps,) + H_seq.shape[1:])
        else:
            grad_Y_seq = grad_spike_seq

        grad_V = 0.0
        grad_T = 0.0
        for t in range(time_steps, 0, -1):
            grad_T_t_to_H_t = (
                _theta_backward_ms(H_seq[t] - v_th, v_th) * _theta_ms(T_max - T_seq[t - 1])
                + _theta_backward_ms(-H_seq[t], v_th) * _theta_ms(T_seq[t - 1] - T_min)
            )
            grad_Y_t_to_T_t_1 = -(
                _theta_eq_ms(H_seq[t] - v_th) * _theta_backward_ms(T_max - T_seq[t - 1], H_seq[t].new_ones(()))
                + _theta_ms(-H_seq[t]) * _theta_backward_ms(T_seq[t - 1] - T_min, H_seq[t].new_ones(()))
            )

            grad_X = (grad_Y_seq[t - 1] - v_th * grad_V + grad_T) * grad_T_t_to_H_t + grad_V
            grad_T = (grad_Y_seq[t - 1] - v_th * grad_V + grad_T) * grad_Y_t_to_T_t_1 + grad_T
            grad_V = grad_X + 0.0
            grad_x_seq[t - 1] = grad_X

        return grad_x_seq, None, None, None, None, None


class ST_BIFNeuron_MS(nn.Module):
    def __init__(self,q_threshold,level,sym=False, first_neuron=False, need_spike_tracer=False, T=8, C=768, neuron_impl: str = "auto"):
        super(ST_BIFNeuron_MS,self).__init__()
        # self.q = 0.0
        self.need_spike_tracer = need_spike_tracer
        if self.need_spike_tracer:
            self.acc_q = 0.0
        self.T = T
        self.step = level//2 - 1
        self.first_neuron = first_neuron
        self.suppress_over_fire = True
        self.overfireLoss = 0.0
        self.name = ""

        self.dim = 197
        print("self.T",self.T, "C",C)
        # self.time_allocator = nn.Parameter(torch.ones(self.T - 1, 1, 1, 1),requires_grad=True)
        self.bias_channel = nn.Parameter(torch.zeros(C), requires_grad=False).to(torch.float32)

        init_list = []
        self.param_number = self.step
        if self.param_number == 1:
            init_list.append(1 / self.step)
        else:
            for i in range(self.param_number-1):
                if i < self.step - 1:
                    init_list.append(1/(self.step))
                else:
                    init_list.append(0.0)

        self.biasAllocator = nn.Parameter(torch.tensor(init_list),requires_grad=True)


        self.q_threshold = nn.Parameter(torch.tensor(q_threshold),requires_grad=False)
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.pos_max_int = int(level // 2 - 1)
            self.neg_min_int = int(-level // 2 - 1)
            self.register_buffer("pos_max",torch.tensor(level//2 - 1))
            self.register_buffer("neg_min",torch.tensor(-level//2 - 1))
            # self.pos_max = torch.tensor(level//2 - 1)
            # self.neg_min = torch.tensor(-level//2)
        else:
            self.pos_max_int = int(level - 1)
            self.neg_min_int = int(0)
            self.register_buffer("pos_max",torch.tensor(level - 1))
            self.register_buffer("neg_min",torch.tensor(0))
            # self.pos_max = torch.tensor(level - 1)
            # self.neg_min = torch.tensor(0)
        self.register_buffer("prefire",torch.tensor(0.0))
        self.init = True
        self.eps = 0
        self.neuron_impl = neuron_impl if neuron_impl in {"auto", "torch"} else "auto"
        if neuron_impl not in {"auto", "torch"}:
            warnings.warn(f"Unknown neuron_impl='{neuron_impl}', falling back to 'auto'.", stacklevel=2)

    def __repr__(self):
        return f"ST_BIFNeuron_MS(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        # self.q = 0.0
        if self.need_spike_tracer:
            self.acc_q = 0.0

    def forward(self,input):
        with nvtx_range("snn.layer.st_bifneuron_ms.ST_BIFNeuron_MS.forward"):
            global _MS_FORCE_TORCH, _MS_FALLBACK_WARNED
            N = input.shape[0]
            ori_shape = input.shape
            # print("self.q_threshold",self.q_threshold.data.item())

            input = input.reshape(torch.Size([int((self.T)),N//int((self.T))]) + input.shape[1:])
            # print("ST_BIFNeuron_MS input.sum(dim=0).abs().mean()",input.sum(dim=0).abs().mean(),input.dtype)
            # print("ST_BIFNeuron_MS input.abs().mean()",input.abs().mean(),input.dtype)

            effect_T = min(self.T, self.param_number)
            biasAllocator = torch.cat([1 - torch.sum(self.biasAllocator,dim=0,keepdim=True), self.biasAllocator], dim=0)[:effect_T]
            # print(biasAllocator)

            if len(input.shape) == 3:
                bias_term = (
                    biasAllocator.to(dtype=input.dtype, device=input.device).view(-1, 1, 1)
                    * self.bias_channel.to(dtype=input.dtype, device=input.device).view(1, 1, -1)
                )
                if effect_T >= input.shape[0]:
                    input = input + bias_term
                else:
                    input[:effect_T] = input[:effect_T] + bias_term
            elif len(input.shape) == 4:
                bias_term = (
                    biasAllocator.to(dtype=input.dtype, device=input.device).view(-1, 1, 1, 1)
                    * self.bias_channel.to(dtype=input.dtype, device=input.device).view(1, 1, 1, -1)
                )
                # print(self.biasAllocator.shape, biasAllocator.shape, bias_term.shape)
                if effect_T >= input.shape[0]:
                    input = input + bias_term
                else:
                    input[:effect_T] = input[:effect_T] + bias_term
            elif len(input.shape) == 5:
                if input.shape[-1] != input.shape[-2]:
                    T1,B1,Head1,N1,C1 = input.shape
                    bias_term = (
                        biasAllocator.to(dtype=input.dtype, device=input.device).view(-1, 1, 1, 1)
                        * self.bias_channel.to(dtype=input.dtype, device=input.device).view(1, 1, 1, -1)
                    )
                    input = input.transpose(2,3).reshape(T1,B1,N1,C1*Head1)
                    if effect_T >= input.shape[0]:
                        input = input + bias_term
                    else:
                        input[:effect_T] = input[:effect_T] + bias_term
                    input = input.reshape(T1,B1,N1,Head1,C1).transpose(2,3)
                else:
                    pass

            q_threshold = self.q_threshold.to(dtype=input.dtype, device=input.device)
            input = input / q_threshold
            x_seq = input.flatten(2)
            v_th = x_seq.new_tensor(1.0)
            T_max = self.pos_max.to(dtype=x_seq.dtype, device=x_seq.device)
            T_min = self.neg_min.to(dtype=x_seq.dtype, device=x_seq.device)
            prefire = self.prefire.to(dtype=x_seq.dtype, device=x_seq.device)

            use_torch = self.neuron_impl == "torch" or _MS_FORCE_TORCH or _ST_BIFNodeATGF_MS_CUDA is None
            if use_torch:
                spike_seq, v_seq, T_seq = ST_BIFNodeATGF_MS_Torch.apply(
                    x_seq, v_th, T_max, T_min, prefire, int(self.pos_max_int)
                )
            else:
                try:
                    spike_seq, v_seq, T_seq = _ST_BIFNodeATGF_MS_CUDA.apply(x_seq, v_th, T_max, T_min, prefire)
                except Exception as exc:
                    if self.neuron_impl == "auto":
                        _MS_FORCE_TORCH = True
                        if not _MS_FALLBACK_WARNED:
                            _MS_FALLBACK_WARNED = True
                            warnings.warn(
                                f"ST_BIFNodeATGF_MS_CUDA unavailable ({exc}). Falling back to torch MS neuron op.",
                                stacklevel=2,
                            )
                        spike_seq, v_seq, T_seq = ST_BIFNodeATGF_MS_Torch.apply(
                            x_seq, v_th, T_max, T_min, prefire, int(self.pos_max_int)
                        )
                    else:
                        raise
            # self.q = v
            # print(self.q[self.q>0].mean())
            if self.need_spike_tracer:
                self.acc_q = T_seq.reshape(ori_shape)
            # print("ST_BIFNeuron_MS output.abs().mean()",(spike_seq*self.q_threshold).abs().mean(),input.dtype)
            # print("ST_BIFNeuron_MS output.sum(dim=0).abs().mean()",(spike_seq*self.q_threshold).sum(dim=0).abs().mean(),spike_seq.dtype)
            spike_seq = spike_seq.reshape(ori_shape)
            if len(input.shape) == 4 and self.suppress_over_fire:
                spike_seq_1 = spike_seq.reshape(torch.Size([int((self.T)),N//int((self.T))]) + spike_seq.shape[1:])
                # print(spike_seq_1.shape)
                # self.overfireLoss = (spike_seq_1.abs().sum(dim=0) - spike_seq_1.sum(dim=0).abs()).sum() * 1e-5
                # if len(spike_seq.shape) == 3:
                #     # print(spike_seq.shape)
                #     channel_dist = (spike_seq).abs().mean(dim=1)
                #     mask = channel_dist < 1.0 * channel_dist.mean(dim=1,keepdim=True)
                #     # print(channel_dist.max(), channel_dist.min(), channel_dist.mean())
                #     self.overfireLoss = (channel_dist * mask).sum() * 2e-5
                self.overfireLoss = spike_seq.abs().sum() / spike_seq.numel()
                # self.overfireLoss = spike_seq[6:].abs().sum() * 1e-6
            return spike_seq * q_threshold
