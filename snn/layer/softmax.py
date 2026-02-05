import torch
import torch.nn as nn
import torch.nn.functional as F

from snn.nvtx import nvtx_range


class spiking_softmax(nn.Module):
    def __init__(self,step,T):
        super(spiking_softmax, self).__init__()
        self.X = 0.0
        self.Y_pre = None
        self.step = step
        self.t = 0
        self.T = T
        # print(self.divide)

        init_list = []
        self.param_number = self.step
        if self.param_number == 1:
            init_list.append(self.step)
        else:
            for i in range(self.param_number-1):
                if i < self.step - 1:
                    init_list.append((self.step)/(i+1))
                else:
                    init_list.append(1.0)
        
        init_list.append(1.0)
        self.biasAllocator = nn.Parameter(torch.tensor(init_list),requires_grad=False)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.Y_pre = None       
        self.t = 0
    
    def forward(self, input):
        with nvtx_range("snn.layer.softmax.spiking_softmax.forward"):
            ori_shape = input.shape
            # 维度重塑
            input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])

            # 累加操作
            input = torch.cumsum(input, dim=0)

            limit = min(self.step, self.T, int(self.biasAllocator.numel()))
            if limit > 0:
                weights = input.new_ones((self.T,))
                weights[:limit] = self.biasAllocator[:limit].to(dtype=input.dtype, device=input.device)
                view_shape = (self.T,) + (1,) * (input.dim() - 1)
                input = input * weights.view(view_shape)

            output = F.softmax(input, dim=-1)

            # 差分操作 (这里保持原样，prepend的处理是安全的)
            output = torch.diff(output, dim=0, prepend=(output[0:1] * 0.0))

            return output.reshape(ori_shape)


class spiking_softmax_ss(nn.Module):
    def __init__(self, step, T):
        super(spiking_softmax_ss, self).__init__()
        self.X = None
        self.Y_pre = None
        self.step = step
        self.register_buffer("t", torch.zeros((), dtype=torch.int64))
        self.T = T

        init_list = []
        self.param_number = self.step
        if self.param_number == 1:
            init_list.append(self.step)
        else:
            for i in range(self.param_number - 1):
                if i < self.step - 1:
                    init_list.append((self.step) / (i + 1))
                else:
                    init_list.append(1.0)

        init_list.append(1.0)
        self.biasAllocator = nn.Parameter(torch.tensor(init_list), requires_grad=False)

    def reset(self):
        self.X = None
        self.Y_pre = None
        self.t.zero_()

    def forward(self, input):
        with nvtx_range("snn.layer.softmax.spiking_softmax_ss.forward"):
            self.t.add_(1)
            if self.X is None:
                self.X = input * 0.0
            self.X = self.X + input

            limit = min(self.step, self.T)
            if limit > 0:
                t_idx = torch.clamp(self.t - 1, 0, limit - 1)
                bias = self.biasAllocator[t_idx]
                scale = torch.where(self.t <= limit, bias, bias.new_ones(()))
            else:
                scale = self.biasAllocator.new_ones(())
            output = F.softmax(self.X * scale, dim=-1)

            if self.Y_pre is None:
                delta = output
            else:
                delta = output - self.Y_pre
            self.Y_pre = output
            return delta
