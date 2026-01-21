import torch

def theta_backward(x, V_thr):
    sigmoid = torch.sigmoid(4.0*x/V_thr)
    return (4.0/V_thr*sigmoid*(1-sigmoid))
#     # tanh = F.tanh(2*x)
#     # return 1/(1+(2*x)*(2*x))
#     # return 1 - F.tanh(2*x)*F.tanh(2*x)
# def theta_backward(x, V_thr):
#     mu = 0
#     sigma = 0.4*V_thr
#     a = 1.0/V_thr
#     upper_x = (-(x-mu)**2)/(2*sigma**2)
#     x = a * torch.exp(upper_x)
#     return x

# def theta_backward(x, V_thr):
#     A = 3.5
#     B = 3
#     x = A * torch.exp(-B*torch.abs(x)**(1/3))
#     return x

def theta(x):
    # return (x > 0).int()
    return (1.0*(torch.gt(x,0)))
 
def theta_eq(x):
    # return (x >= 0).int()
    return (1.0*(torch.ge(x,0)))

class ST_BIFNodeATGF_MS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, prefire: torch.Tensor):

        Time = x_seq.shape[0]
        v_seq = []
        T_seq = []
        H_seq = []
        spike_seq = []
        v = x_seq[0]*0 + 0.5*v_th + prefire*v_th
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
            if t < T_max:
                v = v - v_th * spike - prefire*v_th/T_max
            else:
                v = v - v_th * spike
            T = T + spike
            T_seq.append(T)
            spike_seq.append(spike)

        H_seq = torch.stack(H_seq,dim=0)
        T_seq = torch.stack(T_seq,dim=0)
        spike_seq = torch.stack(spike_seq,dim=0)
        
        ctx.save_for_backward(spike_seq,T_seq,H_seq,v_th,T_max,T_min)
        
        return spike_seq[1:,], v, T_seq[1:,]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_T_seq: torch.Tensor):
        
        spike_seq, T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        Time = spike_seq.shape[0] - 1
        grad_x_seq = []

        grad_Y_seq = grad_spike_seq
        
        grad_V = 0.0 # t
        grad_T = 0.0 # t
        for t in range(Time, 0, -1):
            grad_T_t_to_H_t = (theta_backward(H_seq[t] - v_th, v_th)*theta(T_max - T_seq[t-1])+theta_backward(-H_seq[t], v_th)*theta(T_seq[t-1] - T_min))
            grad_Y_t_to_T_t_1 = -(theta_eq(H_seq[t]-v_th)*theta_backward(T_max - T_seq[t-1], 1.0)+theta(-H_seq[t])*theta_backward(T_seq[t-1] - T_min, 1.0))
            
            grad_X = (grad_Y_seq[t-1] - v_th*grad_V + grad_T)*grad_T_t_to_H_t + grad_V
            grad_T = (grad_Y_seq[t-1] - v_th*grad_V + grad_T)*grad_Y_t_to_T_t_1 + grad_T
            grad_V = grad_X + 0.0
            grad_x_seq.append(grad_X)
        grad_x_seq = torch.flip(torch.stack(grad_x_seq,dim=0),dims=[0])
        # print(spike_seq.dtype, T_seq.dtype, H_seq.dtype, v_th.dtype, T_max.dtype, T_min.dtype, grad_spike_seq.dtype, grad_x_seq.dtype, grad_v_seq.dtype, grad_T_seq.dtype)
        return grad_x_seq, None, None, None, None
