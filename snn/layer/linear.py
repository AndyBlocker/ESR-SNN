import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from snn.nvtx import nvtx_range

class LLConv2d(nn.Module):
    def __init__(self,conv:nn.Conv2d,**kwargs):
        super(LLConv2d,self).__init__()
        self.conv = conv
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level//2 - 1
        self.realize_time = self.steps
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self._zero_output_meta = None
        # self.quan_w_fn = self.conv.quan_w_fn
        
    def reset(self):
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps
        self._zero_output_meta = None

    def forward(self,input):
        with nvtx_range("snn.layer.linear.LLConv2d.forward"):
            # print("LLConv2d.steps",self.steps)
            x = input
            if not torch.is_tensor(x):
                if x == 0.0:
                    self.is_work = False
                    return self.zero_output if self.zero_output is not None else x
                return x
            N,C,H,W = x.shape
            F_h,F_w = self.conv.kernel_size
            S_h,S_w = self.conv.stride
            P_h,P_w = self.conv.padding
            C = self.conv.out_channels
            H = math.floor((H - F_h + 2*P_h)/S_h)+1
            W = math.floor((W - F_w + 2*P_w)/S_w)+1

            out_shape = (N, C, H, W)
            meta = (out_shape, x.device, x.dtype)
            if self.zero_output is None or self._zero_output_meta != meta:
                # self.zero_output = 0.0
                self.zero_output = torch.zeros(size=out_shape, device=x.device, dtype=x.dtype)
                self._zero_output_meta = meta

            if not torch.any(x):
                self.is_work = False
                if self.realize_time > 0:
                    if self.conv.bias is None:
                        output = self.zero_output
                    else:
                        bias = self.conv.bias.detach() / self.steps
                        output = bias.view(1, -1, 1, 1).expand(out_shape)
                    self.realize_time = self.realize_time - 1
                    self.is_work = True
                    return output
                return self.zero_output

            # output = self.conv(x)
            if self.realize_time > 0:
                output = torch.nn.functional.conv2d(input, self.conv.weight, (self.conv.bias/self.steps if self.conv.bias is not None else 0.0), stride=self.conv.stride, \
                    padding=self.conv.padding, dilation=self.conv.dilation,groups=self.conv.groups)
                self.realize_time = self.realize_time - 1
            else:
                output = torch.nn.functional.conv2d(input, self.conv.weight, None, stride=self.conv.stride, \
                    padding=self.conv.padding, dilation=self.conv.dilation,groups=self.conv.groups)
            # if self.neuron_type == 'IF':
            #     pass
            # else:
            #     if self.conv.bias is None:
            #         pass
            #     else:
            #         # if not self.first:
            #         #     output = output - self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            #         output = output - (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if self.conv.bias is not None else 0.0)
            #         if self.realize_time > 0:
            #             output = output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
            #             self.realize_time = self.realize_time - 1
            #             # print("conv2d self.realize_time",self.realize_time)

            self.is_work = True
            self.first = False

            return output

class LLLinear_MS(nn.Module):
    def __init__(self,linear:nn.Linear,**kwargs):
        super(LLLinear_MS,self).__init__()
        self.linear = linear
        self.level = kwargs["level"]
        self.T = kwargs["time_step"]
        self.steps = kwargs["step"]
        self.linear.bias.data = self.linear.bias * self.steps
        # 冻结原始参数
        # self.linear.weight.requires_grad_(False)
        # self.linear.bias.requires_grad_(False)
        self.group_size = 4
        self.param_number = self.steps
        init_list = []

        if self.param_number == 1:
            init_list.append(1 / self.steps)
        else:
            for i in range(self.param_number-1):
                if i < self.steps - 1:
                    init_list.append(1/(self.steps))
                else:
                    init_list.append(0.0)

        self.biasAllocator = nn.Parameter(torch.tensor(init_list),requires_grad=True)
        self.overfireLoss = torch.tensor(0.0)
    
    def forward(self, input):
        with nvtx_range("snn.layer.linear.LLLinear_MS.forward"):
            # 输入可能是 [B*T, C] 或 [B*T, N, C]
            if input.dim() == 3:
                BT, N, C = input.shape
                B = BT // self.T
                input = input.view(self.T, B, N, C)  # [T, B, N, C]
            elif input.dim() == 2:
                BT, C = input.shape
                B = BT // self.T
                N = 1
                input = input.view(self.T, B, N, C)  # [T, B, 1, C]
            else:
                raise ValueError("Input must be [B*T, C] or [B*T, N, C]")

            # biasAllocator 计算
            effect_T = min(self.T, self.param_number)
            biasAllocator = torch.cat([
                1 - torch.sum(self.biasAllocator, dim=0, keepdim=True),
                self.biasAllocator
            ], dim=0)[:effect_T]  # [T, out_features]

            # 线性变换 (向量化)
            weight = self.linear.weight  # [out_features, in_features]
            output = torch.einsum('tbnc,oc->tbno', input, weight)  # [T, B, N, out_features]

            # 加偏置
            if self.linear.bias is not None:
                # print(self.linear.bias.shape, biasAllocator.shape)
                bias_all = self.linear.bias.unsqueeze(0) * biasAllocator.unsqueeze(-1)  # [T, out_features]
                # print(output[:self.param_number].shape, bias_all.view(self.param_number, 1, 1, -1).shape)
                output[:effect_T] = output[:effect_T] + bias_all.view(effect_T, 1, 1, -1)

            # reshape 回原始形状
            if N == 1:
                output = output.view(self.T * B, -1)  # [B*T, out_features]
            else:
                output = output.view(self.T * B, N, -1)  # [B*T, N, out_features]

            # print("LLIENAR: output.shape",output.shape)
            return output

class LLConv2d_MS(nn.Module):
    def __init__(self,conv:nn.Conv2d,**kwargs):
        super(LLConv2d_MS,self).__init__()
        self.conv = conv
        self.level = kwargs["level"]
        self.T = kwargs["time_step"]
        self.steps = kwargs["step"]
    
    def forward(self,input):
        with nvtx_range("snn.layer.linear.LLConv2d_MS.forward"):
            B = input.shape[0]//self.T
            # print("LLConv2d_MS.input",input.reshape(torch.Size([self.T,B])+input.shape[1:]).sum(dim=0).abs().mean())
            output = torch.cat([nn.functional.conv2d(input[:B*self.steps], self.conv.weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,groups=self.conv.groups),\
                                nn.functional.conv2d(input[B*self.steps:], self.conv.weight, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,groups=self.conv.groups)],dim=0)
            # print("LLConv2d_MS.output",output.reshape(torch.Size([self.T,B])+output.shape[1:]).sum(dim=0).abs().mean())
            return output

class LLLinear(nn.Module):
    def __init__(self,linear,**kwargs):
        super(LLLinear,self).__init__()
        self.linear = linear
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level//2 - 1
        self.realize_time = self.steps
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self._zero_output_meta = None
        # self.quan_w_fn = self.linear.quan_w_fn
        
    def reset(self):
        # print("LLLinear reset")
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps
        self._zero_output_meta = None

    def forward(self,input):
        with nvtx_range("snn.layer.linear.LLLinear.forward"):
            # print("LLLinear", input.mean())
            # print("LLLinear.steps",self.steps)
            x = input
            if x.dim() == 3:
                B, N, _ = x.shape
                shape_new = (B, N, self.linear.out_features)
            elif x.dim() == 2:
                B, _ = x.shape
                shape_new = (B, self.linear.out_features)
            else:
                raise ValueError("Input must be 2D or 3D tensor")
            meta = (shape_new, x.device, x.dtype)
            if self.zero_output is None or self._zero_output_meta != meta:
                self.zero_output = torch.zeros(size=shape_new, device=x.device, dtype=x.dtype)
                self._zero_output_meta = meta

            if not torch.any(x):
                self.is_work = False
                return self.zero_output

            if self.realize_time > 0:
                output = torch.nn.functional.linear(x,self.linear.weight,self.linear.bias/self.steps)
                self.realize_time = self.realize_time - 1
            else:
                output = torch.nn.functional.linear(x,self.linear.weight,None)

            # if self.neuron_type == 'IF':
            #     pass
            # else:
            #     if self.linear.bias is None:
            #         pass
            #     else:
            #         if self.realize_time > 0:
            #             output = output - (self.linear.bias.data.unsqueeze(0) if self.linear.bias is not None else 0.0) * (1 - 1/(self.steps)) 
            #             self.realize_time = self.realize_time - 1
            #         else:
            #             output = output - (self.linear.bias.data.unsqueeze(0) if self.linear.bias is not None else 0.0)

            self.is_work = True
            self.first = False

            return output
