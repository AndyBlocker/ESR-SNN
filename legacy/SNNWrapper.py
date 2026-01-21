import torch
import torch.nn.functional as F
import torch.nn as nn
from spikingjelly.clock_driven import layer

from spikingjelly.clock_driven import neuron as cext_neuron
import torch.nn.functional
from spike_quan_layer_snn import MyQuan,QAttention,SAttention,Spiking_LayerNorm, spiking_softmax
from spike_quan_wrapper import attn_convert
from copy import deepcopy

def reset_model(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, SAttention) or isinstance(child, Spiking_LayerNorm) or isinstance(child, spiking_softmax) or isinstance(child, ST_BIFNeuron):
            model._modules[name].reset()
            is_need = True
        if not is_need:
            reset_model(child)

def fuse_backward1(H_seq_t,v_th,T_max,T_seq_t_1,T_min):
    return (4.0 * F.sigmoid(4*(H_seq_t-v_th))*(1-F.sigmoid(4*(H_seq_t-v_th))))* \
        ((T_max-T_seq_t_1) > 0).int() + \
        (4.0 * F.sigmoid(4*(-H_seq_t))*(1-F.sigmoid(4*(-H_seq_t))))* \
        ((T_seq_t_1-T_min) > 0).int()

def theta_backward(x):
    sigmoid = torch.sigmoid(4*x)
    return 2*sigmoid*(1-sigmoid)
    # return 1 - F.tanh(2*x)*F.tanh(2*x)

def ReLU_backward(x):
    # return 2.0*(x <= 0).int()
    return 1.0*(torch.le(x,0))

def theta(x):
    # return (x > 0).int()
    return 1.0*(torch.gt(x,0))
 
def theta_eq(x):
    # return (x >= 0).int()
    return 1.0*(torch.ge(x,0))

class ST_BIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor):

        Time = x_seq.shape[0]
        # v_seq = []
        T_seq = []
        H_seq = []
        spike_seq = []
        # v = torch.zeros(x_seq[0].shape).to(x_seq.device) + v_th*0.5
        # T = torch.zeros(x_seq[0].shape).to(x_seq.device)
        # spike = torch.zeros(x_seq[0].shape).to(x_seq.device)
        v = x_seq[0]*0 + 0.5*v_th
        T = x_seq[0]*0
        spike = x_seq[0]*0
        # v = v*0 + 0.5*v_th
        # T = T*0
        # spike = spike*0

        # v_seq.append(v)
        T_seq.append(T)
        spike_seq.append(spike)
        H_seq.append(v)
        
        for t in range(Time):
            spike = spike * 0.0
            v = v + x_seq[t]
            H_seq.append(v)
            # pos_spike = (v >= v_th) & (T < T_max)
            # neg_spike = (v < 0) & (T > T_min)
            spike[torch.logical_and((torch.ge(v-v_th,0)), (torch.lt(T-T_max,0)))] = 1
            spike[torch.logical_and((torch.lt(v,0)), (torch.gt(T-T_min,0)))] = -1
            v = v - v_th * spike
            T = T + spike
            # v_seq.append(v)
            T_seq.append(T)
            spike_seq.append(spike)

        # v_seq = torch.stack(v_seq,dim=0)
        H_seq = torch.stack(H_seq,dim=0)
        T_seq = torch.stack(T_seq,dim=0)
        spike_seq = torch.stack(spike_seq,dim=0)
        
        ctx.save_for_backward(T_seq,H_seq,v_th,T_max,T_min)
        
        return spike_seq[1:,], v, T_seq[1:,]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_T_seq: torch.Tensor):

        T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        Time = T_seq.shape[0] - 1
        grad_x_seq = []

        grad_spike_seq = torch.nn.functional.pad(grad_spike_seq,[0,0,1,0])
        # grad_v_seq = torch.nn.functional.pad(grad_v_seq,[0,0,1,0])
        # grad_T_seq = torch.nn.functional.pad(grad_T_seq,[0,0,1,0])
        # print(grad_spike_seq.mean(),grad_T_seq.mean(),grad_v_seq.mean())
        # print(grad_T_seq.shape,grad_v_seq.shape,grad_spike_seq.shape)
        # print(torch.abs(grad_v_seq).sum())
        
        grad_H = 0.0 # t + 1
        grad_next_S = 0.0 # t + 1 
        for t in range(Time, 0, -1):
            # grad_T = grad_T_seq[t]
            grad_T_to_S = 1
            grad_S_to_H = (theta_backward(H_seq[t] - v_th)*theta(T_max - T_seq[t-1])+theta_backward(-H_seq[t])*theta(T_seq[t-1] - T_min))
            # grad_S_to_H = fuse_backward1(H_seq[t],v_th,T_max,T_seq[t-1],T_min)
            if t < Time:
                grad_next_S = grad_spike_seq[t + 1]
            grad_S = grad_spike_seq[t]
            grad_next_h_to_v = 1
            grad_v_to_H = 1 - v_th*grad_S_to_H
            # grad_v = grad_v_seq[t]
            grad_v = 0
            grad_H_to_x = 1
            
            if t == Time:
                grad_next_S_to_T = -(theta_eq(H_seq[t]-v_th)*theta_backward(T_max - T_seq[t])+theta(-H_seq[t])*theta_backward(T_seq[t] - T_min))
            else:
                grad_next_S_to_T = -(theta_eq(H_seq[t+1]-v_th)*theta_backward(T_max - T_seq[t])+theta(-H_seq[t+1])*theta_backward(T_seq[t] - T_min))

            grad_T = grad_next_S*grad_next_S_to_T
            grad_H = grad_T*grad_T_to_S*grad_S_to_H + grad_S*grad_S_to_H + grad_H*grad_next_h_to_v*grad_v_to_H + grad_v*grad_v_to_H
            
            grad_x = grad_H_to_x*grad_H
            grad_x_seq.append(grad_x)
        
        grad_x_seq = torch.stack(grad_x_seq,dim=0)
        # print("grad_spike_seq.max()",grad_spike_seq.max().item(),"grad_x_seq.max()",grad_x_seq.max().item())
        # grad_x_seq = torch.zeros(spike_seq.shape)
        # print(grad_x_seq.mean())

        return grad_x_seq, None, None, None

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

class ST_BIFNeuron(nn.Module):
    def __init__(self,q_threshold,level,sym=False):
        super(ST_BIFNeuron,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = nn.Parameter(torch.tensor(q_threshold),requires_grad=False)
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.register_buffer("pos_max",torch.tensor(level//2 - 1))
            self.register_buffer("neg_min",torch.tensor(-level//2))
            # self.pos_max = torch.tensor(level//2 - 1)
            # self.neg_min = torch.tensor(-level//2)
        else:
            self.register_buffer("pos_max",torch.tensor(level - 1))
            self.register_buffer("neg_min",torch.tensor(0))
            # self.pos_max = torch.tensor(level - 1)
            # self.neg_min = torch.tensor(0)
        self.init = True
        self.eps = 0

    def __repr__(self):
            return f"IFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.acc_q = 0.0

    def forward(self,input):
        N = input.shape[0]
        ori_shape = input.shape
        # print("ST_BIF Neuron before:",torch.abs(input).mean())

        self.level = torch.tensor(self.level)
        input = input.reshape(torch.Size([int((self.level).item()),N//int((self.level).item())]) + input.shape[1:])
 
        s_grad_scale = 1.0 / (((input.sum(dim=0)).detach().abs().mean() * input.numel()) ** 0.5)
    
        if self.init:
            self.q_threshold.data = (((input.sum(dim=0)).detach().abs().mean() * 2) / (self.pos_max.detach().abs() ** 0.5))
            self.init = False

        s_scale = grad_scale(self.q_threshold, s_grad_scale)
        # print("self.q_threshold",self.q_threshold.item())
        spike_seq, v_seq, T_seq = ST_BIFNodeATGF.apply(input.flatten(1), s_scale,self.pos_max, self.neg_min)
        self.q = v_seq.view(input.shape[1:])
        # print(self.q[self.q>0].mean())
        self.acc_q = T_seq.view(ori_shape)
        # print("ST_BIF Neuron after:",torch.abs(spike_seq).mean())
        return spike_seq.reshape(ori_shape)*s_scale


def replace_relu_to_stbif(model,level):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, nn.ReLU):
            # model._modules[name] = cext_neuron.MultiStepIFNode(detach_reset=True)
            # model._modules[name] = ST_BIFNeuron(q_threshold=1.0,level=level,sym=False)            
            # print("nn.ReLU!!!!")
            model._modules[name] = nn.Identity()
            is_need = True
        if isinstance(child,MyQuan):
            # print("MyQuan!!!!")
            neurons = ST_BIFNeuron(q_threshold = torch.tensor(1.0),sym=child.sym,level = (child.pos_max - child.neg_min + 1))
            neurons.init = False
            neurons.q_threshold=child.s
            neurons.level = level
            neurons.pos_max = child.pos_max
            neurons.neg_min = child.neg_min
            model._modules[name] = neurons             
            
            # model._modules[name] = ST_BIFNeuron(q_threshold=1.0,level=level,sym=False)            
            # model._modules[name].init = False
            # model._modules[name].q_threshold.data = child.s
            is_need = True
        if isinstance(child, QAttention):
            # print("QAttention")
            SAttn = SAttention(dim=child.num_heads*child.head_dim, num_heads=child.num_heads, level=level, is_softmax=True, neuron_layer=ST_BIFNeuron)
            attn_convert(QAttn=child,SAttn=SAttn,level=level,neuron_type = "ST-BIF")
            model._modules[name] = SAttn
            is_need = True
        # elif isinstance(child, nn.Conv2d) or isinstance(child, nn.BatchNorm2d) or isinstance(child,nn.AvgPool2d) or isinstance(child,nn.MaxPool2d) or isinstance(child,nn.AdaptiveAvgPool2d):
        #     print(child)
        #     model._modules[name] = layer.SeqToANNContainer(child)
        #     is_need = True
        if not is_need:
            replace_relu_to_stbif(child,level)

# def hook_func(model,grad_input,grad_output):
#     print(grad_input[0].max())

# def hook_relu(model):
#     children = list(model.named_children())
#     for name, child in children:
#         is_need = False
#         if isinstance(child, nn.ReLU):
#             child.register_backward_hook(hook_func)
#             is_need = True
#         if not is_need:
#             hook_relu(child)


class MyBachNorm(nn.Module):
    def __init__(self,bn,T):
        super(MyBachNorm,self).__init__()
        bn.bias.data = bn.bias/T
        bn.running_mean = bn.running_mean/T
        # self.bn = bn
        self.bn_list = [deepcopy(bn).cuda() for i in range(T)]
        self.bn_list = nn.ModuleList(self.bn_list)
        self.T = T
    
    def forward(self,x):
        # self.bn.eval()
        # x = self.bn(x)
        N = x.shape[0]

        x = x.reshape(torch.Size([int((self.T)),N//int((self.T))]) + x.shape[1:])

        x_seq = []
        for t in range(self.T):
            x_seq.append(self.bn_list[t](x[t]))
        x = torch.cat(x_seq,dim=0)
        # print(x.shape)
        # for t in range(self.T):
        #     x[t] = self.bn_list[t](x[t])
        
        # x = x.reshape(torch.Size([N]) + x.shape[2:])
        # x = x.sum(dim=0)
        
        # x = self.bn(x)
        
        # x = x.repeat(self.T,1,1,1)/self.T
        # x_seq = []
        # for t in range(self.T):
        #     x_seq.append(x/self.T)
        # x_seq = torch.cat(x_seq,dim=0)

        return x


        
        


def calibrate(model,level):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
            if hasattr(model._modules[name],"bias") and model._modules[name].bias is not None:
                model._modules[name].bias.data = model._modules[name].bias.data/level
        elif isinstance(child,nn.BatchNorm2d):
            # model._modules[name] = MyBachNorm(bn=child,T=int(level))
            model._modules[name].bias.data = model._modules[name].bias.data/level
            model._modules[name].running_mean = model._modules[name].running_mean/level
            is_need = True
        elif isinstance(child,nn.LayerNorm):
            # model._modules[name].bias.data = model._modules[name].bias.data/level
            SNN_LN = Spiking_LayerNorm(child.normalized_shape[0])
            if child.elementwise_affine:
                SNN_LN.layernorm.weight.data = child.weight.data
                SNN_LN.layernorm.bias.data = child.bias.data                
            model._modules[name] = SNN_LN
            is_need = True
        if not is_need:
            calibrate(child,level)


# class SNNWrapper(nn.Module):
#     def __init__(self,model,level):
#         super(SNNWrapper,self).__init__()
#         self.model = model
#         self.level = level + 1
#         self.T = level
#         print("convert to SNN......")
#         replace_relu_to_stbif(self.model,self.level)
#         calibrate(self.model,self.level)
    
#     def forward(self,x):
#         # x_seq = []
#         # for t in range(self.T):
#         #     x_seq.append(x)
#         # x_seq = torch.cat(x_seq,dim=0).cuda()
#         B, C, H, W = x.shape
#         x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1).reshape(-1, C, H, W)
#         output = self.model(x_seq).reshape(self.T, B, -1)
#         return output.mean(dim=0)

class SNNWrapper(nn.Module):
    def __init__(self,model,level,time_step):
        super(SNNWrapper,self).__init__()
        self.model = model
        self.level = int(level)
        self.T = int(time_step)
        self.model.pos_embed.data = self.model.pos_embed/(self.T)
        self.model.cls_token.data = self.model.cls_token/(self.T)
        print("convert to SNN......")
        replace_relu_to_stbif(self.model,self.level)
        calibrate(self.model,self.T)
        # print(self.model)
        
    def reset(self):
        reset_model(self.model)
    
    # def forward(self,x):
    #     x_seq = []
    #     for t in range(self.T):
    #         x_seq.append(x/self.T)
    #     x_seq = torch.cat(x_seq,dim=0)

    #     output = self.model(x_seq)
    #     # print("in SNN wrapper:",output.mean())
    #     N,L = output.shape
    #     output = output.reshape(self.T,N//self.T,L)
    #     # print("out SNN wrapper:",output.mean())
    #     return output.sum(dim=0)

    def forward(self,x):
        x_seq = []
        for t in range(self.T):
            if t == 0:
                x_seq.append(x)
            else:
                x_seq.append(torch.zeros(x.shape,device=x.device))
        x_seq = torch.cat(x_seq,dim=0)
        output = self.model(x_seq)
        N,L = output.shape
        output = output.reshape(self.T,N//self.T,L)
        return output.sum(dim=0)
