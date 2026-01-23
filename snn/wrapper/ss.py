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
    Spiking_LayerNorm_SS,
    ST_BIFNeuron_SS,
    ST_BIFNeuron_SS_Torch,
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
        self.neuron_impl = kwargs.get("neuron_impl", "torch")
        self.neuron_layer = ST_BIFNeuron_SS_Torch if self.neuron_impl == "torch" else ST_BIFNeuron_SS
        self.model = ann_model
        self.model_compiled = None
        self._pos_embed_zeroed = False
        self._swin_pos_drop_zeroed = False
        self.model.patch_embed.spike = True
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.record_inout = kwargs["record_inout"]
        self.record_dir = kwargs["record_dir"]
        self.learnable = learnable
        self.max_T = 0
        self.visualize = False
        self.encoding_time_step = max(1, int(getattr(cfg, "encoding_time_step", 4)))
        self.print_timestep = getattr(cfg, "print_timestep", False)
        self.enable_early_exit = getattr(cfg, "early_exit", True)
        self._zero_input_cache = None
        self._zero_input_meta = None
        self._x_seq_cache = None
        self._x_seq_meta = None
        # self.model_reset = None
        if self.model_name.count("vit") > 0:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)
        self._replace_weight(self.model)
        self._compile_requested = False
        self._compile_cfg = None
        self._drop_path_disabled = False
        if self.cfg is not None and getattr(self.cfg, "compile", False):
            self._compile_requested = True
            self._compile_cfg = {
                "backend": getattr(self.cfg, "compile_backend", "inductor"),
                "mode": getattr(self.cfg, "compile_mode", None),
                "fullgraph": getattr(self.cfg, "compile_fullgraph", False),
                "dynamic": getattr(self.cfg, "compile_dynamic", False),
            }
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

    def _get_zero_input(self, ref):
        meta = (ref.shape, ref.device, ref.dtype)
        if self._zero_input_cache is None or self._zero_input_meta != meta:
            self._zero_input_cache = torch.zeros_like(ref)
            self._zero_input_meta = meta
        return self._zero_input_cache

    def _get_rate_encoded(self, x):
        time_step = int(self.encoding_time_step)
        if time_step <= 0:
            return x.new_zeros((0,) + x.shape)
        out = None
        if not torch.is_grad_enabled():
            meta = (time_step, x.shape, x.device, x.dtype)
            if self._x_seq_cache is None or self._x_seq_meta != meta:
                self._x_seq_cache = x.new_zeros((time_step,) + x.shape)
                self._x_seq_meta = meta
            out = self._x_seq_cache
        return get_subtensors(x, self.mean, self.std, sample_grain=self.step, time_step=time_step, out=out)

    def _disable_drop_path_for_eval(self):
        if self._drop_path_disabled:
            return
        for module in self.model.modules():
            if hasattr(module, "drop_path"):
                drop_path = getattr(module, "drop_path")
                if isinstance(drop_path, nn.Module):
                    module.drop_path = nn.Identity()
        self._drop_path_disabled = True
    
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
            with torch.no_grad():
                self.model.pos_embed.copy_(self.pos_embed.to(self.model.pos_embed.device))
                self.model.cls_token.copy_(self.cls_token.to(self.model.cls_token.device))
        self._pos_embed_zeroed = False
        self._swin_pos_drop_zeroed = False
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
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=self.neuron_layer,T=self.T)
                attn_convert_QAttention_SS(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QAttention_without_softmax):
                SAttn = SAttention_without_softmax_SS(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=self.neuron_layer,T=self.T)
                attn_convert_SS(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type, T=self.T)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, QWindowAttention):
                # self.blockNum = self.blockNum + 1/24
                SAttn = SWindowAttention_SS(dim=child.num_heads*child.head_dim, window_size=child.window_size,num_heads=child.num_heads,level=self.level,neuron_layer=self.neuron_layer,T=self.T,step=self.step)
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
                SNN_LN = Spiking_LayerNorm_SS(child.normalized_shape[0],T=self.T,step=self.step)
                SNN_LN.layernorm = child
                if child.elementwise_affine:
                    SNN_LN.weight = child.weight.data
                    SNN_LN.bias = child.bias.data                
                model._modules[name] = SNN_LN
                # model._modules[name].register_full_backward_hook(modify_gradient_for_spiking_layernorm_softmax(self.T))
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = self.neuron_layer(q_threshold = torch.tensor(1.0),sym=child.sym,level = self.level, T=self.T)
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

    def forward(self, x, verbose=False, return_output_per_timestep=True):
        with nvtx_range("snn.wrapper.ss.SNNWrapper.forward"):
            if self._compile_requested and self.model_compiled is None:
                if not self.model.training and not self._drop_path_disabled:
                    self._disable_drop_path_for_eval()
                if not hasattr(torch, "compile"):
                    print("[SNNWrapper][WARN] torch.compile not available in this torch version.")
                    self._compile_requested = False
                else:
                    cfg = self._compile_cfg or {}
                    try:
                        self.__dict__["model_compiled"] = torch.compile(
                            self.model,
                            backend=cfg.get("backend", "inductor"),
                            mode=cfg.get("mode", None),
                            fullgraph=cfg.get("fullgraph", False),
                            dynamic=cfg.get("dynamic", False),
                        )
                        print(
                            "[SNNWrapper] torch.compile enabled "
                            f"(backend={cfg.get('backend', 'inductor')}, "
                            f"mode={cfg.get('mode', None)}, "
                            f"fullgraph={cfg.get('fullgraph', False)}, "
                            f"dynamic={cfg.get('dynamic', False)})"
                        )
                    except Exception as exc:
                        print(f"[SNNWrapper][WARN] torch.compile failed, fallback to eager: {exc}")
                        self.__dict__["model_compiled"] = None
                        self._compile_requested = False

            model_fn = self.model_compiled if self.model_compiled is not None else self.model
            accu = None
            accu_per_timestep = None
            output_per_timestep = None
            preallocate_verbose = verbose and not torch.is_grad_enabled()
            steps_done = 0
            # print("self.bit",self.bit)
            # x = x*(2**self.bit-1)+0.0
            if self.visualize:
                self.hook_mid_feature()
            if self.Encoding_type == "rate":
                self.mean = 0.0
                self.std  = 0.0
                x_seq = self._get_rate_encoded(x)
                if self.cfg.model.count("vit") > 0:
                    with torch.no_grad():
                        self.model.pos_embed.div_(self.step)
                        self.model.cls_token.div_(self.step)
                zero_input = self._get_zero_input(x_seq[0])
            else:
                x_seq = x
                zero_input = self._get_zero_input(x)

            if self.cfg.model.count("swin") > 0 and not self._swin_pos_drop_zeroed:
                self.model.pos_drop.p = 0
                self._swin_pos_drop_zeroed = True

            zero_at = None
            if self.model_name.count("vit") > 0:
                if self.Encoding_type == "analog":
                    zero_at = 1
                elif self.Encoding_type == "rate":
                    zero_at = self.step

            for count1 in range(self.T):
                if self.enable_early_exit:
                    self.finish_judger.reset_network_finish_flag()
                    self.finish_judger.judge_finish(self)
                    if count1 > 0 and self.finish_judger.network_finish:
                        break
                if zero_at is not None and (count1 >= zero_at) and not self._pos_embed_zeroed:
                    with torch.no_grad():
                        self.model.pos_embed.mul_(0)
                        self.model.cls_token.mul_(0)
                    self._pos_embed_zeroed = True

                if self.Encoding_type == "rate":
                    input = x_seq[count1] if count1 < x_seq.shape[0] else zero_input
                else:
                    input = x_seq if count1 == 0 else zero_input
                # elif self.neuron_type == 'IF':
                #     input = x
                # else:
                #     print("No implementation of neuron type:",self.neuron_type)
                #     sys.exit(0)

                with nvtx_range("snn.wrapper.ss.SNNWrapper.step"):
                    output = model_fn(input)
                # print(count1,output[0,0:100])
                # print(count1,"output",torch.abs(output.sum()))

                if count1 == 0:
                    accu = output
                    if verbose:
                        if preallocate_verbose:
                            accu_per_timestep = output.new_empty((self.T,) + output.shape)
                            if return_output_per_timestep:
                                output_per_timestep = output.new_empty((self.T,) + output.shape)
                        else:
                            accu_per_timestep = []
                            output_per_timestep = [] if return_output_per_timestep else None
                else:
                    accu = accu + output
                if verbose:
                    if preallocate_verbose:
                        accu_per_timestep[count1] = accu
                        if return_output_per_timestep:
                            output_per_timestep[count1] = output
                    else:
                        accu_per_timestep.append(accu)
                        if return_output_per_timestep:
                            output_per_timestep.append(output)
                # print("accu",accu.sum(),"output",output.sum())
                if self.print_timestep and (count1 + 1) % 100 == 0:
                    print(count1 + 1)
                steps_done = count1 + 1

            self.max_T = max(steps_done, self.max_T)
            # print("verbose",verbose)
            # print("\nTime Step:",count1)
            if self.visualize:
                self.get_mid_feature()
                torch.save(self.feature_list,"model_blocks11_norm2.pth")
                torch.save(self.input_feature_list,"model_blocks11_norm2_input.pth")
            if verbose:
                if not preallocate_verbose:
                    accu_per_timestep = torch.stack(accu_per_timestep, dim=0)
                    if return_output_per_timestep:
                        output_per_timestep = torch.stack(output_per_timestep, dim=0)
                if preallocate_verbose:
                    accu_per_timestep = accu_per_timestep[:steps_done]
                    if return_output_per_timestep:
                        output_per_timestep = output_per_timestep[:steps_done]
                return accu, steps_done, output_per_timestep, accu_per_timestep
            else:
                return accu, steps_done
