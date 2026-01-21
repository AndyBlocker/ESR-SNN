import torch
from spike_quan_layer import Spiking_LayerNorm, DyHT, SDyHT, MyQuan, ST_BIFNeuron_MS

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True

# torch.set_printoptions(10)
# torch.set_default_dtype(torch.double)
# torch.set_default_tensor_type(torch.DoubleTensor)

B = 1
T = 32
N = 4
C = 4
dyht = DyHT(C=C).cuda()
dyht.gamma.data = dyht.gamma * 1.1
sdyht = SDyHT(C=C).cuda()
sdyht.alpha.data = dyht.alpha
sdyht.gamma.data = dyht.gamma
myquan = MyQuan(level=10, sym=True).cuda()
myquan.s.data = torch.tensor(0.125)
myquan.init_state= myquan.batch_init
neurons = ST_BIFNeuron_MS(q_threshold = torch.tensor(1.0), sym=myquan.sym, level = myquan.level, first_neuron=False)
neurons.q_threshold.data = myquan.s.data
neurons.level = myquan.level
neurons.pos_max = myquan.pos_max
neurons.neg_min = myquan.neg_min
neurons.init = True
neurons.T = T
neurons.cuda()

setup_seed(42)
x = (torch.rand(1,N,C).cuda() - 0.5)*128
x1 = dyht(x)
x2 = myquan(x1)

x_spike = torch.cat([x/T for t in range(T)], dim=0)
x1_spike = sdyht(x_spike)
x2_spike = neurons(x1_spike)

print(x)
print(x_spike.sum(dim=0,keepdim=True))
print("===============================================")
print(x1)
print(x1_spike.sum(dim=0,keepdim=True))
print("===============================================")
print(x2)
print(x2_spike.sum(dim=0,keepdim=True))
print(torch.abs(x2 - x2_spike.reshape(T, B, N, C).sum(dim=0)).max()<1e-3)

