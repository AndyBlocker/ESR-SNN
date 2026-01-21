import torch
import torch.nn as nn

from .utils import theta_backward, theta, theta_eq
from snn.nvtx import nvtx_range

try:
    from neuron_cupy.st_bif_ss_cuda import ST_BIFNodeATGF_SS_CUDA
except Exception:
    ST_BIFNodeATGF_SS_CUDA = None


class ST_BIFNodeATGF_SS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_t: torch.Tensor, V_t_1: torch.Tensor, T_t_1: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, t: torch.Tensor):

        with nvtx_range("snn.layer.st_bifneuron_ss.ST_BIFNodeATGF_SS.forward"):
            spike = x_t * 0.0
            H_t = V_t_1 + x_t

            # Group_size = 4
            # if H_t.shape[-1] == 3072:
            #     B,H,C = H_t.shape
            #     H_t_group = H_t.reshape(B,H,C//Group_size,Group_size).abs()
            #     H_t_group = H_t_group.mean(-1)
            #     mid_value = torch.quantile(H_t_group,0.5,interpolation='nearest')
            #     mask = (H_t_group >= mid_value).unsqueeze(-1)
            #     spike_condition = ((H_t >= v_th) & (T_t_1-T_max < 0)).reshape(B,H,C//Group_size,Group_size) & mask
            #     neg_spike_condition = ((H_t < 0) & (T_t_1-T_min > 0)).reshape(B,H,C//Group_size,Group_size) & mask
            #     spike_condition = spike_condition.reshape(B,H,C)
            #     neg_spike_condition = neg_spike_condition.reshape(B,H,C)
            # else:
            spike_condition = (H_t >= v_th) & (T_t_1-T_max < 0)
            neg_spike_condition = (H_t < 0) & (T_t_1-T_min > 0)

            # spike[torch.logical_and((torch.ge(H_t-v_th,0)), (torch.lt(T_t_1-T_max,0)))] = 1
            # spike[torch.logical_and((torch.lt(H_t,0)), (torch.gt(T_t_1-T_min,0)))] = -1

            spike = torch.where(spike_condition, torch.ones_like(H_t),
                                          torch.where(neg_spike_condition, -torch.ones_like(H_t),
                                                      torch.zeros_like(H_t)))

            V_t = H_t - v_th * spike
            T_t = T_t_1 + spike

            ctx.save_for_backward(T_t_1,H_t,v_th,T_max,T_min,t)

            return spike, V_t, T_t

    @staticmethod
    def backward(ctx, grad_spike_t: torch.Tensor, grad_v_t: torch.Tensor, grad_T_t: torch.Tensor):

        with nvtx_range("snn.layer.st_bifneuron_ss.ST_BIFNodeATGF_SS.backward"):
            T_t_1,H_t,v_th,T_max,T_min,t = ctx.saved_tensors

            grad_T_t_to_H_t = (theta_backward(H_t - v_th)*theta(T_max - T_t_1)+theta_backward(-H_t)*theta(T_t_1 - T_min))
            grad_Y_t_to_T_t_1 = -(theta_eq(H_t-v_th)*theta_backward(T_max - T_t_1)+theta(-H_t)*theta_backward(T_t_1 - T_min))

            tmp = grad_spike_t - v_th*grad_v_t + grad_T_t
            grad_X_t = tmp*grad_T_t_to_H_t + grad_v_t
            grad_T_t_1 = tmp*grad_Y_t_to_T_t_1 + grad_T_t
            grad_V_t_1 = grad_X_t + 0.0
            # print("t:",t,"grad_V_t_1",grad_X_t.mean().item(),"grad_T_t_1",grad_T_t_1.mean().item(),"grad_v_t",grad_v_t.mean().item(),"grad_T_t",grad_T_t.mean().item())
            return grad_X_t, grad_V_t_1, grad_T_t_1, None, None, None, None

class ST_BIFNeuron_SS(nn.Module):
    def __init__(self, q_threshold, level, sym=False, need_spike_tracer=False, T=None, C=None):
        super(ST_BIFNeuron_SS, self).__init__()
        self.need_spike_tracer = need_spike_tracer
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = nn.Parameter(torch.tensor(q_threshold), requires_grad=False)
        self.level = torch.tensor(level)
        self.T = T if T is not None else self.level // 2 - 1
        self.sym = sym
        if sym:
            self.register_buffer("pos_max",torch.tensor(level//2 - 1))
            self.register_buffer("neg_min",torch.tensor(-level//2 + 1))
            # self.pos_max = torch.tensor(level//2 - 1)
            # self.neg_min = torch.tensor(-level//2)
        else:
            self.register_buffer("pos_max",torch.tensor(level - 1))
            self.register_buffer("neg_min",torch.tensor(0))
            # self.pos_max = torch.tensor(level - 1)
            # self.neg_min = torch.tensor(0)
        self.register_buffer("prefire", torch.tensor(0.0))
        self.init = True
        # self.steps = max(self.pos_max,torch.abs(self.neg_min))
        self.eps = 0
        self.t = 0
        self.init_state = 0
        self.init_batch = 20

    # def __repr__(self):
    #         return f"ST_BIFNeuron_SS(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.acc_q = 0.0
        self.t = 0
        self.init_state = 0

    def forward(self,input):
        with nvtx_range("snn.layer.st_bifneuron_ss.ST_BIFNeuron_SS.forward"):
            # print("input.mean()",input.abs().mean().item(),"self.q_threshold",self.q_threshold.data.item())

            # s_grad_scale = 1.0 / (((input).detach().abs().mean() * input.numel()) ** 0.5)
            # if self.init:
            #     if self.init_state == 0:
            #         self.q_threshold.data = (((input).detach().abs().mean() * 2) / (self.pos_max.detach().abs() ** 0.5))
            #         self.init_state += 1
            #     elif self.init_state < self.init_batch:
            #         self.q_threshold.data = 0.1*self.q_threshold.data + 0.9*(((input).detach().abs().mean() * 2) / (self.pos_max.detach().abs() ** 0.5))
            #         self.init_state += 1
            #     else:
            #         self.init = False

            if not torch.is_tensor(self.q) and not torch.is_tensor(self.acc_q):
                self.q = input * 0.0 + 0.5*self.q_threshold
                # self.q = input * 0.0
                self.acc_q = input * 0.0

            # if self.steps > 0:
            #     self.q = self.q + 0.5*self.q_threshold/self.steps
            #     self.steps = self.steps - 1

            # s_scale = grad_scale(self.q_threshold, s_grad_scale)
            self.t = self.t + 1
            if ST_BIFNodeATGF_SS_CUDA is not None and input.is_cuda:
                spikes, self.q, self.acc_q = ST_BIFNodeATGF_SS_CUDA.apply(
                    input,
                    self.q,
                    self.acc_q,
                    self.q_threshold,
                    self.pos_max,
                    self.neg_min,
                    torch.tensor(self.t),
                )
            else:
                spikes, self.q, self.acc_q = ST_BIFNodeATGF_SS.apply(
                    input,
                    self.q,
                    self.acc_q,
                    self.q_threshold,
                    self.pos_max,
                    self.neg_min,
                    torch.tensor(self.t),
                )

            return spikes * self.q_threshold
