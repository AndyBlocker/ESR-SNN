import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import glo
from snn.nvtx import nvtx_range
from snn.layer import (
    DyHT,
    DyHT_ReLU,
    DyT,
    IFNeuron,
    LLConv2d_MS,
    LLLinear_MS,
    MyBatchNorm1d,
    MyQuan,
    QAttention,
    QAttention_without_softmax,
    QWindowAttention,
    QuanConv2d,
    QuanLinear,
    SDyHT,
    SAttention,
    SAttention_without_softmax,
    SpikeMaxPooling,
    Spiking_LayerNorm,
    ST_BIFNeuron_MS,
    SWindowAttention,
    save_module_inout,
    spiking_dyt,
)
from .convert import attn_convert, attn_convert_QAttention, attn_convert_Swin
from .utils import Judger, get_subtensors, reset_model


class SNNWrapper_MS(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="rate", learnable=False,**kwargs):
        super(SNNWrapper_MS, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.step = 2
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model

        self.model.spike = True
        self.model.T = time_step
        self.model.step = self.step

        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.record_inout = kwargs["record_inout"]
        self.record_dir = kwargs["record_dir"]
        self.suppress_over_fire = kwargs["suppress_over_fire"]
        self.learnable = learnable
        self.max_T = 0
        self.visualize = False
        self.first_neuron = True
        self.blockNum = 0
        self.confident_thr = [0.2,0.85]
        # self.model_reset = None
        if self.model_name.count("vit") > 0:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)

        self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)        
        if self.record_inout:
            self.calOrder = []
            self._record_inout(self.model)
            self.set_snn_save_name(self.model)
            local_rank = torch.distributed.get_rank()
            glo._init()
            if local_rank == 0:
                if not os.path.exists(self.record_dir):
                    os.mkdir(self.record_dir)
                glo.set_value("output_bin_snn_dir",self.record_dir)
                f = open(f"{self.record_dir}/calculationOrder.txt","w+")
                for order in self.calOrder:
                    f.write(order+"\n")
                f.close()
        
        # self.param_number = 4
        # init_list = []
        # for i in range(self.param_number - 1):
        #     if i < self.step - 1:
        #         init_list.append(1/(self.step))
        #     else:
        #         init_list.append(0.0)
        # if self.T > 1:
        #     self.biasAllocator = nn.Parameter(torch.tensor(init_list))
        # else:
        #     self.biasAllocator = 1.0
        
    def hook_mid_feature(self):
        self.feature_list = []
        self.input_feature_list = []
        def _hook_mid_feature(module, input, output):
            self.feature_list.append(output)
            self.input_feature_list.append(input[0])
        self.model.blocks[11].norm2[1].register_forward_hook(_hook_mid_feature)
        # self.model.blocks[11].attn.attn_IF.register_forward_hook(_hook_mid_feature)
    
    def get_mid_feature(self):
        self.feature_list = torch.stack(self.feature_list,dim=0)
        self.input_feature_list = torch.stack(self.input_feature_list,dim=0)
        print("self.feature_list",self.feature_list.shape) 
        print("self.input_feature_list",self.input_feature_list.shape) 
            
    def reset(self):
        # self.model = deepcopy(self.model_reset).cuda()
        if self.model_name.count("vit")>0:
            self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
            self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        # print(self.model.pos_embed)
        # print(self.model.cls_token)
        reset_model(self)
    

    def _record_inout(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, SAttention):
                model._modules[name].first = True
                model._modules[name].T = self.T
                is_need = True
            elif isinstance(child, nn.Sequential) and isinstance(child[1], IFNeuron):
                model._modules[name] = save_module_inout(m=child,T=self.T)
                model._modules[name].first = True
                is_need = True
            if not is_need:            
                self._record_inout(child)            

    def set_snn_save_name(self, model):
        children = list(model.named_modules())
        for name, child in children:
            if isinstance(child, save_module_inout):
                child.name = name
                self.calOrder.append(name)
            if isinstance(child, SAttention):
                child.name = name
                self.calOrder.append(name)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ST_BIFNeuron_MS,T=self.T)
                attn_convert_QAttention(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T, suppress_over_fire=self.suppress_over_fire, step=self.step)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QAttention_without_softmax):
                SAttn = SAttention_without_softmax(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ST_BIFNeuron_MS,T=self.T)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T, suppress_over_fire=self.suppress_over_fire, step=self.step)
                # self.blockNum = self.blockNum + 1/12
                # SAttn.attn_IF.prefire.data = torch.tensor(0.125)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QWindowAttention):
                # self.blockNum = self.blockNum + 1/24
                SAttn = SWindowAttention(dim=child.num_heads*child.head_dim, window_size=child.window_size,num_heads=child.num_heads,level=self.level,neuron_layer=ST_BIFNeuron_MS,T=self.T,step=self.step)
                attn_convert_Swin(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T, suppress_over_fire=self.suppress_over_fire, step=self.step)
                # SAttn.attn_softmax_IF.prefire.data = torch.tensor(self.blockNum*0.2)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d) or isinstance(child, QuanConv2d):
                if child.bias is not None:
                    model._modules[name].bias.data = model._modules[name].bias.data/self.step
                model._modules[name] = LLConv2d_MS(child,time_step=self.T,step=self.step,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear) or isinstance(child, QuanLinear):
                # if name.count("head") > 0:
                #     model._modules[name] = LLLinear(child,**self.kwargs)
                # else:
                if child.bias is not None:
                    model._modules[name].bias.data = model._modules[name].bias.data/self.step
                model._modules[name] = LLLinear_MS(child,time_step=self.T,step=self.step,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling(child, step=self.step,T=self.T)
                is_need = True
            elif isinstance(child, DyT):
                model._modules[name] = spiking_dyt(child,step=self.step,T=self.T)
                is_need = True
            elif isinstance(child, DyHT) or isinstance(child, DyHT_ReLU):
                sdyht = SDyHT(C = 1, step=self.step,T=self.T)
                sdyht.step = self.step
                sdyht.T = self.T
                sdyht.beta.data = child.beta
                # sdyht.beta.data = child.beta/sdyht.step
                sdyht.gamma.data = child.gamma
                sdyht.alpha.data = child.alpha
                model._modules[name] = sdyht
                is_need = True
            elif isinstance(child,nn.BatchNorm2d) or isinstance(child,nn.BatchNorm1d) or isinstance(child, MyBatchNorm1d):
                # if self.learnable:
                #     model._modules[name] = MyBachNorm(bn=child,T=self.T)
                # else:
                # model._modules[name] = spiking_BatchNorm2d_MS(bn=child,level=self.level//2 - 1,input_allcate=False)
                model._modules[name].bias.data = model._modules[name].bias.data/self.step
                model._modules[name].running_mean = model._modules[name].running_mean/self.step
                model._modules[name].spike = True
                model._modules[name].T = self.T
                model._modules[name].step = self.step
                
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0],T=self.T,step=self.step)
                SNN_LN.layernorm = child
                if child.elementwise_affine:
                    SNN_LN.weight = child.weight.data
                    SNN_LN.bias = child.bias.data
                model._modules[name] = SNN_LN
                # model._modules[name].register_full_backward_hook(modify_gradient_for_spiking_layernorm_softmax(self.T))
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = ST_BIFNeuron_MS(q_threshold = torch.tensor(1.0),sym=child.sym,level = self.level, first_neuron=self.first_neuron, T = self.T, C=child.channel_num)
                neurons.q_threshold.data = min(child.s.data, child.s_max.data)
                neurons.bias_channel.data = child.bias_channel
                neurons.level = self.level
                neurons.pos_max = child.pos_max_buf
                neurons.neg_min = child.neg_min_buf
                neurons.init = True
                self.first_neuron = False
                neurons.cuda()
                model._modules[name] = neurons
                is_need = True
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
            if not is_need:            
                self._replace_weight(child)

    def forward(self,x, verbose=False):
        with nvtx_range("snn.wrapper.ms.SNNWrapper_MS.forward"):
            if self.cfg.dataset == "dvs128":
                input = x.transpose(0,1)/self.T
            else:
                input = get_subtensors(x,0.0,0.0,sample_grain=self.step, time_step=self.T)
            # input = input * self.step
            if self.cfg.model.count("vit") > 0:
                self.model.pos_embed.data = self.model.pos_embed/self.step
                self.model.cls_token.data = self.model.cls_token/self.step
            elif self.cfg.model.count("swin") > 0:
                self.model.pos_drop.p = 0
            T,B,C,H,W = input.shape
            # biasAllocator = torch.cat([1 - torch.sum(self.biasAllocator,dim=0,keepdim=True), self.biasAllocator], dim=0)
            # input = torch.cat([input[0].unsqueeze(0) * biasAllocator.reshape(-1,1,1,1,1), input[self.param_number:]], dim=0)

            input = input.reshape(T*B,C,H,W)
            with nvtx_range("snn.wrapper.ms.SNNWrapper_MS.model"):
                output = self.model(input)
            output = output.reshape(torch.Size([T,B]) + output.shape[1:])

            if not self.training:
                print(output.abs().sum(dim=[-1,-2]))

            accu_per_t = []
            accu = 0.0
            self.reset()
            if verbose == True:
                for t in range(T):
                    accu = accu + output[t]
                    prob = torch.max(F.softmax(accu, dim=1), dim=1)[0]
                    # if t == 0:
                    #     if prob > self.confident_thr[0]:
                    #         for i in range(t,T):
                    #             accu_per_t.append(accu + 0.0)                        
                    #         break
                    # else:
                    if prob > self.confident_thr[1]:
                        for i in range(t,T):
                            accu_per_t.append(accu + 0.0)
                        break
                    accu_per_t.append(accu + 0.0)
                return accu, t+1, output, torch.stack(accu_per_t,dim=0)
            return output.sum(dim=0), self.T
