import torch
import torch.nn as nn

from snn.nvtx import nvtx_range

class SpikeMaxPooling_SS(nn.Module):
    def __init__(self,maxpool:nn.MaxPool2d,step,T):
        # take from 
        super(SpikeMaxPooling_SS, self).__init__()
        self.X = None
        self.Y_pre = None
        self.step = step
        self.maxpool = maxpool
        self.stride = maxpool.stride
        self.t = 0
        self.T = T
        # print(self.divide)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = None
        self.Y_pre = None       
        self.t = 0
    
    def forward(self,input):
        with nvtx_range("snn.layer.pooling.SpikeMaxPooling_SS.forward"):
            if self.X is None:
                self.X = input
                self.Y_pre = self.maxpool(input)
                return self.Y_pre
            self.X = self.X + input
            output = self.maxpool(self.X)
            delta = output - self.Y_pre
            self.Y_pre = output
            return delta

class SpikeMaxPooling(nn.Module):
    def __init__(self,maxpool:nn.MaxPool2d,step,T):
        # take from 
        super(SpikeMaxPooling, self).__init__()
        self.X = 0.0
        self.Y_pre = None
        self.step = step
        self.maxpool = maxpool
        self.stride = maxpool.stride
        self.t = 0
        self.T = T
        # print(self.divide)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.Y_pre = None       
        self.t = 0
    
    def forward(self,input):
        with nvtx_range("snn.layer.pooling.SpikeMaxPooling.forward"):
            ori_shape = input.shape
            input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
            input = torch.cumsum(input, dim=0).reshape(ori_shape)
            output = self.maxpool(input)
            ori_shape = output.shape
            output = output.reshape(torch.Size([self.T, output.shape[0]//self.T]) + output.shape[1:])
            output = torch.diff(output,dim=0,prepend=(output[0]*0.0).unsqueeze(0))
            return output.reshape(ori_shape)
