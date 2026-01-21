import torch
import torch.nn as nn

from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA
from snn.nvtx import nvtx_range


class ST_BIFNeuron_MS(nn.Module):
    def __init__(self,q_threshold,level,sym=False, first_neuron=False, need_spike_tracer=False, T=8, C=768):
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
            self.register_buffer("pos_max",torch.tensor(level//2 - 1))
            self.register_buffer("neg_min",torch.tensor(-level//2 - 1))
            # self.pos_max = torch.tensor(level//2 - 1)
            # self.neg_min = torch.tensor(-level//2)
        else:
            self.register_buffer("pos_max",torch.tensor(level - 1))
            self.register_buffer("neg_min",torch.tensor(0))
            # self.pos_max = torch.tensor(level - 1)
            # self.neg_min = torch.tensor(0)
        self.register_buffer("prefire",torch.tensor(0.0))
        self.init = True
        self.eps = 0

    def __repr__(self):
        return f"ST_BIFNeuron_MS(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        # self.q = 0.0
        if self.need_spike_tracer:
            self.acc_q = 0.0

    def forward(self,input):
        with nvtx_range("snn.layer.st_bifneuron_ms.ST_BIFNeuron_MS.forward"):
            N = input.shape[0]
            ori_shape = input.shape
            # print("self.q_threshold",self.q_threshold.data.item())

            input = input.reshape(torch.Size([int((self.T)),N//int((self.T))]) + input.shape[1:])
            # print("ST_BIFNeuron_MS input.sum(dim=0).abs().mean()",input.sum(dim=0).abs().mean(),input.dtype)
            # print("ST_BIFNeuron_MS input.abs().mean()",input.abs().mean(),input.dtype)

            effect_T = min(self.T, self.param_number)
            biasAllocator = torch.cat([1 - torch.sum(self.biasAllocator,dim=0,keepdim=True), self.biasAllocator], dim=0)[:effect_T]
            # print(biasAllocator)

            if len(input.shape) == 4:
                bias_term = biasAllocator.view(-1, 1, 1, 1) * self.bias_channel.view(1, 1, 1, -1)
                # print(self.biasAllocator.shape, biasAllocator.shape, bias_term.shape)
                input = torch.cat([
                    input[:effect_T] + bias_term,
                    input[effect_T:]
                ], dim=0)
            elif len(input.shape) == 5:
                if input.shape[-1] != input.shape[-2]:
                    T1,B1,Head1,N1,C1 = input.shape
                    bias_term = biasAllocator.view(-1, 1, 1, 1) * self.bias_channel.view(1, 1, 1, -1)
                    input = input.transpose(2,3).reshape(T1,B1,N1,C1*Head1)
                    input = torch.cat([
                        input[:effect_T] + bias_term,
                        input[effect_T:]
                    ], dim=0)
                    input = input.reshape(T1,B1,N1,Head1,C1).transpose(2,3)

            input = input / self.q_threshold
            spike_seq, v_seq, T_seq = ST_BIFNodeATGF_MS_CUDA.apply(input.flatten(2), torch.tensor(1.0).to(input.device), self.pos_max, self.neg_min, self.prefire)
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

            return spike_seq*self.q_threshold
