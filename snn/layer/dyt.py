import torch
import torch.nn as nn
import torch.nn.functional as F


class spiking_dyt(nn.Module):
    def __init__(self,dyt,step,T):
        super(spiking_dyt, self).__init__()
        self.X = 0.0
        # self.gamma = dyt.gamma
        # self.alpha = dyt.alpha
        self.beta = nn.Parameter(dyt.beta.data, requires_grad=True)
        self.dyt = dyt
        self.dyt.beta.data = self.dyt.beta * 0.0
        self.dyt.beta.requires_grad = False
        # self.dyt.gamma.requires_grad = False
        # self.dyt.alpha.requires_grad = False
        self.step = step
        self.t = 0
        self.T = T
        # self.divide = torch.tensor([min(1.0,1.0*(i+1)/self.step) for i in range(self.T)])
        # print(self.divide)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.t = 0
    
    def forward(self,input):
        # ori_shape = input.shape
        # input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        # self.X = torch.cumsum(input, dim=0) - input
        # Y = self.gamma * torch.sinh(self.alpha * input) / (torch.cosh(self.alpha*self.X) * torch.cosh(self.alpha*(input + self.X))) + self.beta/self.T
        # return Y.reshape(ori_shape)
        ori_shape = input.shape
        input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        input = torch.cumsum(input, dim=0)
        output = self.dyt(input)
        output = torch.diff(output,dim=0,prepend=(output[0]*0.0).unsqueeze(0))
        output[:self.step] = output[:self.step] + self.beta/self.step
        return output.reshape(ori_shape)

class DyT(nn.Module):
    def __init__(self, C, init_alpha=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(torch.ones(1) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(C)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))

    def forward(self,x):
        x = torch.tanh(self.alpha*x)
        return self.gamma * x + self.beta

class DyHT_ReLU(nn.Module):
    def __init__(self, C, init_alpha=0.5):
        super(DyHT_ReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(torch.ones(C) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(1)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))

    def forward(self,x):
        # B = x.shape[0]
        # print("QANN INPUT DyHT.abs().mean()",x.abs().mean())
        x = (torch.nn.functional.hardtanh(self.alpha*x + self.beta, min_val=0.0, max_val=1.0))
        # print("QANN DyHT.abs().mean()",x.abs().mean())
        # print("self.gamma min",self.gamma.min(),"self.gamma max",self.gamma.max())
        
        return x * self.gamma

class DyHT_Softmax(nn.Module):
    def __init__(self, H, init_alpha=1.0):
        super(DyHT_Softmax, self).__init__()
        self.H = H
        self.alpha_init_value = init_alpha
        self.alpha = nn.Parameter(torch.tensor(torch.ones(H) * init_alpha), requires_grad=True)

    def reset_parameters(self):
        self.alpha.data.fill_(self.alpha_init_value)

    def extra_repr(self):
        return f"DyHT_Softmax(self.H={self.H}, alpha_init_value={self.alpha_init_value})"

    def forward(self,x):
        # x = x.permute(0,2,3,1) # B,N,N,H
        N = x.shape[-1]

        # print(self.alpha.mean().item())
        alpha = self.alpha.view(1,self.H,1,1)
        x = torch.clip(torch.relu(x*alpha/N), 0.0, 1.0)
        # x = torch.clip(x, 0.0, 1.0)
                
        # x = x.permute(0,3,1,2) # B,H,N,N
        
        return x

class DyHT(nn.Module):
    def __init__(self, C, init_alpha=0.5):
        super(DyHT, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(torch.ones(C) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(1)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))

    def forward(self,x):
        # B = x.shape[0]
        # print("QANN INPUT DyHT.abs().mean()",x.abs().mean())
        x = torch.nn.functional.hardtanh(self.alpha*x + self.beta, min_val=-1.0, max_val=1.0)
        # print("QANN DyHT.abs().mean()",x.abs().mean())
        # print("self.gamma min",self.gamma.min(),"self.gamma max",self.gamma.max())
        
        return x * self.gamma

class SDyHT_SS(nn.Module):
    def __init__(self, C, init_alpha=0.25):
        super(SDyHT_SS, self).__init__()
        self.step = 1
        self.T = 1
        self.t = 0
        self.alpha = nn.Parameter(torch.tensor(torch.ones(C) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(1)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))
        self.accu1 = 0.0
        self.accu2 = 0.0
    
    def reset(self):
        self.t = 0
    
    def forward(self,x):
        self.t = self.t + 1
        x = self.alpha * self.gamma * x
        if self.t <= self.step:
            x = x + self.beta * self.gamma/self.step
        return x

class SDyHT(nn.Module):
    def __init__(self, C, init_alpha=0.25, step=3, T=32):
        super(SDyHT, self).__init__()
        self.step = step
        self.T = T
        self.alpha = nn.Parameter(torch.tensor(torch.ones(C) * init_alpha))
        self.gamma = nn.Parameter(torch.tensor(torch.ones(1)))
        self.beta = nn.Parameter(torch.tensor(torch.zeros(C)))


        self.param_number = self.step
        init_list = []

        if self.param_number == 1:
            init_list.append(1 / self.step)
        else:
            for i in range(self.param_number-1):
                if i < self.step - 1:
                    init_list.append(1/(self.step))
                else:
                    init_list.append(0.0)

        self.biasAllocator = nn.Parameter(torch.tensor(init_list),requires_grad=True)


    def __repr__(self):
        return f"SDyHT(self.gamma={self.gamma.data[0]})"

    def forward(self,x):
        effect_T = min(self.T, self.param_number)
        biasAllocator = torch.cat([1 - torch.sum(self.biasAllocator,dim=0,keepdim=True), self.biasAllocator], dim=0)[:effect_T]
        if x.dim() == 3:
            B,N,C = x.shape
            # print("SNN INPUT DyHT.abs().mean()",x.reshape(32,8,-1).sum(dim=0).abs().mean())
            x = x.reshape(self.T,B//self.T,N,C)
            x = self.alpha * self.gamma * x
            bias_term = biasAllocator.view(-1, 1, 1, 1) * self.gamma * self.beta.view(1, 1, 1, -1)
            x = torch.cat([
                x[:effect_T] + bias_term,
                x[effect_T:]
            ], dim=0)
            x = x.reshape(B,N,C)
            # print("SNN DyHT.abs().mean()",x.reshape(32,8,-1).sum(dim=0).abs().mean())
        else:
            B,C = x.shape
            # print("SNN INPUT DyHT.abs().mean()",x.reshape(32,8,-1).sum(dim=0).abs().mean())
            x = x.reshape(self.T,B//self.T,C)
            x = self.alpha * self.gamma * x
            bias_term = biasAllocator.view(-1, 1, 1) * self.gamma * self.beta.view(1, 1, -1)
            x = torch.cat([
                x[:effect_T] + bias_term,
                x[effect_T:]
            ], dim=0)
            x = x.reshape(B,C)
        
        # print("self.beta",self.beta.data)        
        
        return x
