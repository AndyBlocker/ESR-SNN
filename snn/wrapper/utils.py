import torch

from snn.layer import (
    IFNeuron,
    LLConv2d,
    LLLinear,
    MyBatchNorm1d_SS,
    SDyHT_SS,
    SpikeMaxPooling,
    SpikeMaxPooling_SS,
    Spiking_LayerNorm,
    Spiking_LayerNorm_SS,
    ST_BIFNeuron_MS,
    ST_BIFNeuron_SS,
    SWindowAttention,
    SWindowAttention_SS,
    spiking_softmax_ss,
)


def get_subtensors(tensor,mean,std,sample_grain=255,time_step=4):
    for i in range(int(time_step)):
        # output = (tensor).unsqueeze(0)
        output = (tensor/sample_grain).unsqueeze(0)
        if i == 0:
            accu = output
        elif i < sample_grain:
            accu = torch.cat((accu,output),dim=0)
        else:
            accu = torch.cat((accu,output*0.0),dim=0)
    return accu

def reset_model(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, IFNeuron) or isinstance(child, SpikeMaxPooling_SS) or \
            isinstance(child, MyBatchNorm1d_SS) or \
            isinstance(child, LLConv2d) or isinstance(child, LLLinear) or isinstance(child, SWindowAttention_SS) or \
            isinstance(child, SWindowAttention) or isinstance(child, Spiking_LayerNorm) or \
            isinstance(child, Spiking_LayerNorm_SS) or \
            isinstance(child, SpikeMaxPooling) or isinstance(child, SDyHT_SS) or \
            isinstance(child, ST_BIFNeuron_MS) or isinstance(child, ST_BIFNeuron_SS) or \
            isinstance(child, spiking_softmax_ss):
            model._modules[name].reset()
            is_need = True
        if not is_need:
            reset_model(child)

class Judger():
	def __init__(self):
		self.network_finish=True

	def judge_finish(self,model):
		children = list(model.named_children())
		for name, child in children:
			is_need = False
			if isinstance(child, IFNeuron) or isinstance(child, LLLinear) or isinstance(child, LLConv2d):
				self.network_finish = self.network_finish and (not model._modules[name].is_work)
				# print("child",child,"network_finish",self.network_finish,"model._modules[name].is_work",(model._modules[name].is_work))
				is_need = True
			if not is_need:
				self.judge_finish(child)

	def reset_network_finish_flag(self):
		self.network_finish = True
