import torch
import torch.nn as nn

import glo
from snn.nvtx import nvtx_range


class Addition(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(self,x):
        return x[0]+x[1]

class save_module_inout(nn.Module):
    def __init__(self, m, T):
        super().__init__()
        self.m = m
        self.T = T
        self.name = ""
        self.t = 0
        self.accu = []
        self.accu2 = []
        self.accu1 = []
        self.first = False
    
    def forward(self,x):
        with nvtx_range("snn.layer.record.save_module_inout.forward"):
            if isinstance(self.m[0],Addition):
                dimNum = len(x[0].shape) + 1
            else:
                dimNum = len(x.shape) + 1
            self.t = self.t + 1
            if self.first:
                if isinstance(self.m[0],Addition):
                    self.accu.append(x[0][0].unsqueeze(0)+0)
                else:
                    self.accu.append(x[0].unsqueeze(0)+0)
                if self.t == self.T:
                    if dimNum == 3:
                        save_fc_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                    if dimNum == 4:
                        save_input_for_bin_snn_4dim(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                    if dimNum == 5:
                        save_input_for_bin_snn_5dim(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                if isinstance(self.m[0],Addition):
                    self.accu2.append(x[1][0].unsqueeze(0)+0)
                    if self.t == self.T:
                        if dimNum == 3:
                            save_fc_input_for_bin_snn(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                        if dimNum == 4:
                            save_input_for_bin_snn_4dim(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                        if dimNum == 5:
                            save_input_for_bin_snn_5dim(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                        del self.accu2

            x = self.m(x)
            if self.first:
                self.accu1.append(x[0].unsqueeze(0)+0)
                if self.t == self.T:
                    if dimNum == 3:
                        save_fc_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                    if dimNum == 4:
                        save_input_for_bin_snn_4dim(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                    if dimNum == 5:
                        save_input_for_bin_snn_5dim(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                    self.first = False

                    # saving weight and bias
                    local_rank = torch.distributed.get_rank()
                    if local_rank == 0 and not isinstance(self.m[0],Addition):
                        if hasattr(self.m[0], "quan_w_fn"):
                            torch.save(self.m[0].quan_w_fn(self.m[0].weight),f'{glo.get_value("output_bin_snn_dir")}/{self.name}_weight.pth')
                        else:
                            torch.save(self.m[0].weight,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_weight.pth')

                        if self.m[0].bias is not None:
                            torch.save(self.m[0].bias,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_bias.pth')

                    del self.accu
                    del self.accu1
            return x

def save_input_for_bin_snn_5dim(input,dir,name):
    T,B,L1,L2,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_L1={L1}_L2={L2}_N={N}.pth')

def save_input_for_bin_snn_4dim(input,dir,name):
    T,B,L,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_L={L}_N={N}.pth')

def save_fc_input_for_bin_snn(input,dir,name):
    T,B,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_N={N}.pth')
