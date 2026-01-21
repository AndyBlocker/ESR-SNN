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
    LLConv2d,
    LLLinear,
    MyBatchNorm1d,
    MyBatchNorm1d_SS,
    MyQuan,
    QAttention,
    QAttention_without_softmax,
    QWindowAttention,
    QuanConv2d,
    QuanLinear,
    SDyHT_SS,
    SAttention,
    SAttention_without_softmax_SS,
    SpikeMaxPooling_SS,
    Spiking_LayerNorm,
    ST_BIFNeuron_SS,
    SWindowAttention_SS,
    save_module_inout,
    spiking_dyt,
)
from .convert import (
    attn_convert,
    attn_convert_QAttention_SS,
    attn_convert_SS,
    attn_convert_Swin_SS,
)
from .utils import Judger, get_subtensors, reset_model


class SNNWrapper(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="rate", learnable=False,**kwargs):
        super(SNNWrapper, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.step = self.level//2 - 1
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model
        self.model.patch_embed.spike = True
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.record_inout = kwargs["record_inout"]
        self.record_dir = kwargs["record_dir"]
        self.learnable = learnable
        self.max_T = 0
        self.visualize = False
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
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ST_BIFNeuron_SS,T=self.T)
                attn_convert_QAttention_SS(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QAttention_without_softmax):
                SAttn = SAttention_without_softmax_SS(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ST_BIFNeuron_SS,T=self.T)
                attn_convert_SS(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QWindowAttention):
                # self.blockNum = self.blockNum + 1/24
                SAttn = SWindowAttention_SS(dim=child.num_heads*child.head_dim, window_size=child.window_size,num_heads=child.num_heads,level=self.level,neuron_layer=ST_BIFNeuron_SS,T=self.T,step=self.step)
                attn_convert_Swin_SS(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T, suppress_over_fire=False, step=self.step)
                # SAttn.attn_softmax_IF.prefire.data = torch.tensor(self.blockNum*0.2)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d) or isinstance(child, QuanConv2d):
                model._modules[name] = LLConv2d(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear) or isinstance(child, QuanLinear):
                model._modules[name] = LLLinear(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling_SS(child, step=self.step,T=self.T)
                is_need = True
            elif isinstance(child, DyT):
                model._modules[name] = spiking_dyt(child,step=self.step,T=self.T)
                is_need = True
            elif isinstance(child, DyHT) or isinstance(child, DyHT_ReLU):
                sdyht = SDyHT_SS(C = 1)
                sdyht.step = self.step
                sdyht.T = self.T
                sdyht.beta.data = child.beta
                sdyht.gamma.data = child.gamma
                sdyht.alpha.data = child.alpha
                model._modules[name] = sdyht
                is_need = True
            elif isinstance(child,nn.BatchNorm2d) or isinstance(child,nn.BatchNorm1d) or isinstance(child, MyBatchNorm1d):
                # if self.learnable:
                #     model._modules[name] = MyBachNorm(bn=child,T=self.T)
                # else:
                # model._modules[name] = spiking_BatchNorm2d_MS(bn=child,level=self.level//2 - 1,input_allcate=False)
                bn = MyBatchNorm1d_SS(dim=1)
                bn.bias.data = model._modules[name].bias.data/self.step
                bn.running_mean = model._modules[name].running_mean/self.step
                bn.weight.data = model._modules[name].weight.data
                bn.running_var = model._modules[name].running_var
                bn.spike = True
                bn.T = self.T
                bn.step = self.step
                model._modules[name] = bn
                
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
                neurons = ST_BIFNeuron_SS(q_threshold = torch.tensor(1.0),sym=child.sym,level = self.level, T=self.T)
                neurons.q_threshold.data = min(child.s.data, child.s_max.data)
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
        with nvtx_range("snn.wrapper.ss.SNNWrapper.forward"):
            accu = None
            count1 = 0
            accu_per_timestep = []
            output_per_timestep = []
            # print("self.bit",self.bit)
            # x = x*(2**self.bit-1)+0.0
            if self.visualize:
                self.hook_mid_feature()
            if self.Encoding_type == "rate":
                self.mean = 0.0
                self.std  = 0.0
                x = get_subtensors(x,self.mean,self.std,sample_grain=self.step)
                if self.cfg.model.count("vit") > 0:
                    self.model.pos_embed.data = self.model.pos_embed/self.step
                    self.model.cls_token.data = self.model.cls_token/self.step
                # print("x.shape",x.shape)
            while(1):
                self.finish_judger.reset_network_finish_flag()
                self.finish_judger.judge_finish(self)
                network_finish = self.finish_judger.network_finish
                # print(f"===================Timestep: {count1}===================")
                if (count1 > 0 and network_finish) or count1 >= self.T:
                    self.max_T = max(count1, self.max_T)
                    break
                # if self.neuron_type.count("QFFS") != -1 or self.neuron_type == 'ST-BIF':
                if (self.Encoding_type == "analog" and self.model_name.count("vit") > 0 and count1 > 0) or (self.Encoding_type == "rate" and self.model_name.count("vit") > 0 and count1 >= self.step):
                    self.model.pos_embed.data = self.model.pos_embed*0.0
                    self.model.cls_token.data = self.model.cls_token*0.0
                elif self.cfg.model.count("swin") > 0:
                    self.model.pos_drop.p = 0
                if self.Encoding_type == "rate":
                    if count1 < x.shape[0]:
                        input = x[count1]
                    else:
                        input = torch.zeros(x[0].shape).to(x.device)
                else:
                    if count1 == 0:
                        input = x
                    else:
                        input = torch.zeros(x.shape).to(x.device)
                # elif self.neuron_type == 'IF':
                #     input = x
                # else:
                #     print("No implementation of neuron type:",self.neuron_type)
                #     sys.exit(0)

                with nvtx_range("snn.wrapper.ss.SNNWrapper.step"):
                    output = self.model(input)
                # print(count1,output[0,0:100])
                # print(count1,"output",torch.abs(output.sum()))

                if count1 == 0:
                    accu = output + 0.0
                else:
                    accu = accu + output
                if verbose:
                    accu_per_timestep.append(accu)
                    output_per_timestep.append(output)
                # print("accu",accu.sum(),"output",output.sum())
                count1 = count1 + 1
                if count1 % 100 == 0:
                    print(count1)

            # print("verbose",verbose)
            # print("\nTime Step:",count1)
            if self.visualize:
                self.get_mid_feature()
                torch.save(self.feature_list,"model_blocks11_norm2.pth")
                torch.save(self.input_feature_list,"model_blocks11_norm2_input.pth")
            if verbose:
                accu_per_timestep = torch.stack(accu_per_timestep,dim=0)
                output_per_timestep = torch.stack(output_per_timestep,dim=0)
                return accu,count1,output_per_timestep,accu_per_timestep
            else:
                return accu,count1
