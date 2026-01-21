import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple

from snn.nvtx import nvtx_range

class Spiking_LayerNorm(nn.Module):
    def __init__(self,dim,T,step):
        super(Spiking_LayerNorm, self).__init__()
        self.layernorm = None
        self.X = 0.0
        self.Y_pre = None
        self.weight = None
        self.bias = None
        self.T = T
        self.t = 0
        self.step = step

        init_list = []
        self.param_number = self.step
        if self.param_number == 1:
            init_list.append(1 / self.step)
        else:
            for i in range(self.param_number-1):
                if i < self.step - 1:
                    init_list.append((i+1)/(self.step))
                else:
                    init_list.append(1.0)
        
        init_list.append(1.0)
        self.biasAllocator = nn.Parameter(torch.tensor(init_list),requires_grad=False)


    def reset(self):
        # print("Spiking_LayerNorm reset")
        self.X = 0.0
        self.Y_pre = None
        self.t = 0
        
    def forward(self,input):
        with nvtx_range("snn.layer.norm.Spiking_LayerNorm.forward"):
            ori_shape = input.shape
            input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
            # print("input.sum(dim=0).abs().mean()",input.sum(dim=0).abs().mean(), "after layernorm:", self.layernorm(input.sum(dim=0)).abs().mean())
            cum = torch.cumsum(input, dim=0)
            if torch.is_tensor(self.X):
                cum = cum + self.X

            limit = min(self.step, self.T, int(self.biasAllocator.numel()))
            if limit > 0:
                weights = cum.new_ones((self.T,))
                weights[:limit] = self.biasAllocator[:limit].to(dtype=cum.dtype, device=cum.device)
                view_shape = (self.T,) + (1,) * (cum.dim() - 1)
                Y = self.layernorm(cum) * weights.view(view_shape)
            else:
                Y = self.layernorm(cum)

            Y_pre = torch.cat([Y[0:1] * 0.0, Y[:-1]], dim=0)
            output = Y - Y_pre

            self.X = cum[-1].detach()
            self.Y_pre = Y[-1].detach()
            return output.reshape(ori_shape)


class Spiking_LayerNorm_SS(nn.Module):
    def __init__(self, dim, T, step):
        super(Spiking_LayerNorm_SS, self).__init__()
        self.layernorm = None
        self.X = None
        self.Y_pre = None
        self.weight = None
        self.bias = None
        self.T = T
        self.t = 0
        self.step = step

        init_list = []
        self.param_number = self.step
        if self.param_number == 1:
            init_list.append(1 / self.step)
        else:
            for i in range(self.param_number - 1):
                if i < self.step - 1:
                    init_list.append((i + 1) / (self.step))
                else:
                    init_list.append(1.0)

        init_list.append(1.0)
        self.biasAllocator = nn.Parameter(torch.tensor(init_list), requires_grad=False)

    def reset(self):
        self.X = None
        self.Y_pre = None
        self.t = 0

    def forward(self, input):
        with nvtx_range("snn.layer.norm.Spiking_LayerNorm_SS.forward"):
            self.t += 1
            if self.X is None:
                self.X = input * 0.0
            self.X = self.X + input

            limit = min(self.step, self.T)
            if self.t <= limit:
                Y = self.layernorm(self.X) * self.biasAllocator[self.t - 1]
            else:
                Y = self.layernorm(self.X)

            Y_pre = self.Y_pre if self.Y_pre is not None else 0.0
            self.Y_pre = Y.detach()
            return Y - Y_pre

class MyBatchNorm1d_SS(nn.BatchNorm1d):
    def __init__(self, dim, **kwargs):
        super(MyBatchNorm1d_SS, self).__init__(dim, **kwargs)
        self.spike = False
        self.T = 0
        self.step = 0
        self.momentum = 0.1
        self.eps = 1e-5
        self.t = 0
        self._zero_mean = None
        self._zero_bias = None
        
    def reset(self):
        self.t = 0

    def _get_zero_buffers(self):
        if self._zero_mean is None or self._zero_mean.device != self.running_mean.device or self._zero_mean.dtype != self.running_mean.dtype:
            self._zero_mean = torch.zeros_like(self.running_mean)
        if self.bias is not None:
            if self._zero_bias is None or self._zero_bias.device != self.bias.device or self._zero_bias.dtype != self.bias.dtype:
                self._zero_bias = torch.zeros_like(self.bias)
        else:
            self._zero_bias = None
        return self._zero_mean, self._zero_bias
    
    def forward(self,x):
        with nvtx_range("snn.layer.norm.MyBatchNorm1d_SS.forward"):
            # self.training = False
            input_shape = len(x.shape)
            if input_shape == 4:
                B,H,N,C = x.shape
                x = x.reshape(B*H,N,C)
            if input_shape == 2:
                x = x.unsqueeze(1)
            x = x.transpose(1,2)
            # if self.spike:
            #     print("before mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
            # else:
            #     print("before mybatchnorm1d:",x.abs().mean())
            self.t = self.t + 1
            if self.t <= self.step:
                x = F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
            else:
                zero_mean, zero_bias = self._get_zero_buffers()
                x = F.batch_norm(x,zero_mean,self.running_var,self.weight,zero_bias,self.training,self.momentum,self.eps)
            # if self.spike:
            #     print("after mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
            # else:
            #     print("after mybatchnorm1d:",x.abs().mean())
            x = x.transpose(1,2)
            if input_shape == 2:
                x = x.squeeze(1)
            if input_shape == 4:
                x = x.reshape(B,H,N,C)
            # print("self.running_mean",self.running_mean.abs().mean())
            return x

class MyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, dim, **kwargs):
        super(MyBatchNorm1d, self).__init__(dim, **kwargs)
        self.spike = False
        self.T = 0
        self.step = 0
        self.momentum = 0.1
        self.eps = 1e-5
        self._zero_mean = None
        self._zero_bias = None

    def _get_zero_buffers(self):
        if self._zero_mean is None or self._zero_mean.device != self.running_mean.device or self._zero_mean.dtype != self.running_mean.dtype:
            self._zero_mean = torch.zeros_like(self.running_mean)
        if self.bias is not None:
            if self._zero_bias is None or self._zero_bias.device != self.bias.device or self._zero_bias.dtype != self.bias.dtype:
                self._zero_bias = torch.zeros_like(self.bias)
        else:
            self._zero_bias = None
        return self._zero_mean, self._zero_bias
    
    def forward(self,x):
        with nvtx_range("snn.layer.norm.MyBatchNorm1d.forward"):
            # self.training = False
            input_shape = len(x.shape)
            if input_shape == 4:
                B,H,N,C = x.shape
                x = x.reshape(B*H,N,C)
            if input_shape == 2:
                x = x.unsqueeze(1)
            x = x.transpose(1,2)
            # if self.spike:
            #     print("before mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
            # else:
            #     print("before mybatchnorm1d:",x.abs().mean())
            if not self.spike:
                x = F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
            else:
                Fd = x.shape[0]
                if self.step >= self.T:
                    x = F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,False,self.momentum,self.eps)
                else:
                    zero_mean, zero_bias = self._get_zero_buffers()
                    x = torch.cat([F.batch_norm(x[:int(Fd*(self.step/self.T))],self.running_mean,self.running_var,self.weight,self.bias,False,self.momentum,self.eps), \
                                F.batch_norm(x[int(Fd*(self.step/self.T)):],zero_mean,self.running_var,self.weight,zero_bias,False,self.momentum,self.eps)])
            # if self.spike:
            #     print("after mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
            # else:
            #     print("after mybatchnorm1d:",x.abs().mean())
            x = x.transpose(1,2)
            if input_shape == 2:
                x = x.squeeze(1)
            if input_shape == 4:
                x = x.reshape(B,H,N,C)
            # print("self.running_mean",self.running_mean.abs().mean())
            return x

class LN2BNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LN2BNorm,self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        self.Eta = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("running_mean",torch.zeros(self.dim))
        self.register_buffer("running_var",torch.ones(self.dim))
        self.Lambda = 1.0
        self.momentum = 0.1
        self.eps = eps
        
    def forward(self,x):
        with nvtx_range("snn.layer.norm.LN2BNorm.forward"):
            out_LN = F.layer_norm(x, (self.dim,), self.weight, self.bias)
            out_Identity = x + 0.0
            input_shape = len(x.shape)
            if input_shape == 4:
                B,H,N,C = x.shape
                x = x.reshape(B*H,N,C)
            if input_shape == 2:
                x = x.unsqueeze(1)
            x = x.transpose(1,2)
            out_BN = F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
            out_BN = out_BN.transpose(1,2)
            if input_shape == 2:
                out_BN = out_BN.squeeze(1)
            if input_shape == 4:
                out_BN = out_BN.reshape(B,H,N,C)
            return out_LN*self.Lambda + out_BN*(1 - self.Lambda)

class MLP_BN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=MyBatchNorm1d, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])
        self.name = "MLP"

    def forward(self, x):
        with nvtx_range("snn.layer.norm.MLP_BN.forward"):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x

class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)
        self.running_mean = None
        self.running_var = None
        self.momentum = 0.9
        self.eps = 1e-6
    
    def forward(self,x):        
        with nvtx_range("snn.layer.norm.MyLayerNorm.forward"):
            if self.training:
                if self.running_mean is None:
                    self.running_mean = nn.Parameter((1-self.momentum) * x.mean([-1], keepdim=True),requires_grad=False)
                    self.running_var = nn.Parameter((1-self.momentum) * x.var([-1], keepdim=True),requires_grad=False)
                else:
                    self.running_mean.data = (1-self.momentum) * x.mean([-1], keepdim=True) + self.momentum * self.running_mean # mean: [1, max_len, 1]
                    self.running_var.data = (1-self.momentum) * x.var([-1], keepdim=True) + self.momentum * self.running_var # std: [1, max_len, 1]
                return self.weight * (x - self.running_mean) / (self.running_var + self.eps) + self.bias
            else:
                # if self.running_mean is None:
                self.running_mean = nn.Parameter(x.mean([-1], keepdim=True),requires_grad=False)
                self.running_var = nn.Parameter(x.var([-1], keepdim=True),requires_grad=False)
                running_mean = self.running_mean
                running_var = self.running_var
                return self.weight * (x) / (running_var + self.eps).sqrt() + self.bias
