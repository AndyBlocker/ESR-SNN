import torch
import torch.nn as nn
import torch.nn.functional as F


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
        ori_shape = input.shape
        # 维度重塑
        input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        
        # 累加操作
        input = torch.cumsum(input, dim=0)
        
        # 【修改部分开始】
        # 使用列表收集每一时刻的结果，避免原位修改(in-place)导致的梯度报错
        processed_input = []
        limit = min(self.step, self.T)
        
        for i in range(self.T):
            if i < limit:
                # 对前 step 个时间步进行加权（非原位操作）
                processed_input.append(input[i] * self.biasAllocator[i])
            else:
                # 其他时间步保持不变
                processed_input.append(input[i])
        
        # 将列表重新堆叠回张量
        input = torch.stack(processed_input, dim=0)
        # 【修改部分结束】

        output = F.softmax(input, dim=-1)  
        
        # 差分操作 (这里保持原样，prepend的处理是安全的)
        output = torch.diff(output, dim=0, prepend=(output[0]*0.0).unsqueeze(0))
        
        return output.reshape(ori_shape)
