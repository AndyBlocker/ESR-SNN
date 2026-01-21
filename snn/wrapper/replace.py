import torch
import torch.nn as nn

from timm.models.swin_transformer import PatchEmbed, PatchMerging, SwinTransformerBlock, WindowAttention
from timm.models.vision_transformer import Attention, Block, Mlp

from snn.compat.timm_vit_block import BlockWithAddition
from snn.layer import (
    Addition,
    Attention_no_softmax,
    DyHT,
    DyHT_ReLU,
    LN2BNorm,
    MLP_BN,
    MyBatchNorm1d,
    MyQuan,
    PatchEmbedConv,
    PatchMergingConv,
    QAttention,
    QAttention_without_softmax,
    QWindowAttention,
    QuanConv2d,
    QuanLinear,
    WindowAttention_no_softmax,
)


def open_dropout(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, nn.Dropout):
            child.train()
            print(child)
            is_need = True
        if not is_need:
            open_dropout(child)

def cal_l1_loss(model):
    l1_loss = 0.0
    def _cal_l1_loss(model):
        nonlocal l1_loss
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, MyQuan) and child.cal_loss:
                l1_loss = l1_loss + child.l1_loss
                is_need = True
            if not is_need:
                _cal_l1_loss(child)
    _cal_l1_loss(model)
    return l1_loss

def adjust_LN2BN_Ratio(EndEpoch:int, curEpoch:int, model:nn.Module):
    for name,module in list(model.named_modules()):
        if isinstance(module, LN2BNorm):
            if curEpoch < EndEpoch:
                module.Lambda = 1 - (curEpoch+1)/(EndEpoch)
                print(f"adjust {name} Lambda = {module.Lambda}")

def add_bn_in_mlp(model,normLayer):
    children = list(model.named_children())
    for name, child in children:
        if name.count("decoder") > 0:
            continue
        is_need = False
        if isinstance(child, Mlp):
            mlp_bn = MLP_BN(in_features=child.fc1.in_features, hidden_features=child.fc1.out_features, act_layer=nn.ReLU, drop=child.drop1.p, norm_layer=normLayer)
            model._modules[name] = mlp_bn
            is_need = True
        # elif isinstance(child, nn.LayerNorm):
        #     LN = MyBatchNorm1d(num_features = child.normalized_shape[0])
        #     # LN.weight.data = child.weight
        #     # LN.bias.data = child.bias
        #     model._modules[name] = LN
        if not is_need:
            add_bn_in_mlp(child,normLayer)

def modify_gradient_for_spiking_layernorm_softmax(T):
    def _modify(module, grad_in, grad_out):
        nonlocal T
        # grad_out = grad_out[0].reshape(torch.Size([T,grad_out[0].shape[0]//T])+grad_out[0].shape[1:])
        # grad_in = grad_in[0].reshape(torch.Size([T,grad_in[0].shape[0]//T])+grad_in[0].shape[1:])
        # print(grad_out.abs().mean())

        # print(len(grad_in),len(grad_out))
        # print(grad_in[0].shape)
        # print("===========================================")
        # print("module",module)
        # print(grad_out.shape, grad_in[0].shape)
        # print(grad_in[0].abs().mean(),grad_in[1].abs().mean(),grad_in[2].abs().mean(),grad_in[3].abs().mean())
        # print(grad_out[0].abs().mean(),grad_out[1].abs().mean(),grad_out[2].abs().mean(),grad_out[3].abs().mean())
        # print(T,grad_in[0].shape, grad_out[0].shape)
        # grad1 = grad_in[0].reshape(torch.Size([T,grad_in[0].shape[0]//T])+grad_in[0].shape[1:])
        # print(grad_out.shape,grad_in.shape)
        # print(torch.cat([grad_in[0]*(T-i)/T for i in range(T)]).shape)
        # grad_new = tuple([torch.cat([grad_in[0]*(T-i)/T for i in range(T)])])
        # return grad_out
    return _modify

def swap_BN_MLP_MHSA(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, Block):
            child = BlockWithAddition.from_block(child)
            model._modules[name] = child
        if isinstance(child, BlockWithAddition):
            child.addition1 = Addition()
            child.addition2 = Addition()
            is_need = True

        if not is_need:
            swap_BN_MLP_MHSA(child)

def add_convEmbed(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, PatchMerging):
            patchergingConv = PatchMergingConv(child.input_resolution, child.dim, norm_layer=MyBatchNorm1d)
            is_need = True
            model._modules[name] = patchergingConv
        if isinstance(child,PatchEmbed):
            patchembedConv = PatchEmbedConv(img_size=child.img_size, patch_size=child.patch_size, in_chans=child.proj.in_channels, embed_dim=child.proj.out_channels, norm_layer=MyBatchNorm1d, flatten=True)
            patchembedConv.norm = DyHT(patchembedConv.embed_dim)
            is_need = True
            model._modules[name] = patchembedConv
        if not is_need:
            add_convEmbed(child)

def remove_softmax(model):
    children = list(model.named_children())
    for name, child in children:
        if name.count("decoder") > 0:
            continue        
        is_need = False
        if isinstance(child, Attention):
            reluattn = Attention_no_softmax(dim=child.num_heads*child.head_dim,num_heads=child.num_heads)
            reluattn.qkv = child.qkv
            reluattn.attn_drop = child.attn_drop
            reluattn.proj = child.proj
            reluattn.proj_drop = child.proj_drop
            is_need = True
            model._modules[name] = reluattn
        if isinstance(child,WindowAttention):
            reluattn = WindowAttention_no_softmax(dim=child.num_heads*child.head_dim, window_size=child.window_size,num_heads=child.num_heads)
            reluattn.qkv = child.qkv
            reluattn.attn_drop = child.attn_drop
            reluattn.proj = child.proj
            reluattn.proj_drop = child.proj_drop
            reluattn.relative_position_bias_table = child.relative_position_bias_table
            reluattn.relative_position_index = child.relative_position_index
            is_need = True
            model._modules[name] = reluattn
            
        # elif isinstance(child, nn.LayerNorm):
        #     LN = MyBatchNorm1d(num_features = child.normalized_shape[0])
        #     # LN.weight.data = child.weight
        #     # LN.bias.data = child.bias
        #     model._modules[name] = LN
        if not is_need:
            remove_softmax(child)

def hook_layernorm(module, input, output):
    print("layernorm input",input[0].abs().mean())    
    print("layernorm output",output.abs().mean())    

def myquan_replace(model,level,weight_bit=32, is_softmax = True, eval1=False):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention_without_softmax):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model,level):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, (Block, BlockWithAddition)):
                # print(children)
                if is_softmax:
                   qattn = QAttention(dim=child.attn.num_heads*child.attn.head_dim,num_heads=child.attn.num_heads,level=level,is_softmax=is_softmax)
                else:
                   qattn = QAttention_without_softmax(dim=child.attn.num_heads*child.attn.head_dim,num_heads=child.attn.num_heads,level=level,is_softmax=is_softmax)
                qattn.qkv = child.attn.qkv
                qattn.attn_drop = child.attn.attn_drop
                qattn.proj = child.attn.proj
                qattn.proj_drop = child.attn.proj_drop
                model._modules[name].attn = qattn

                if isinstance(child.norm1, DyHT_ReLU):
                    model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=False, cal_loss=True))
                else:
                    model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True, cal_loss=True))
                if not isinstance(child.norm1[0], nn.LayerNorm):
                    model._modules[name].norm1[1].s_max.data = model._modules[name].norm1[0].gamma/model._modules[name].norm1[1].pos_max
                # print("model._modules[name].norm1[0].gamma",model._modules[name].norm1[0].gamma,"model._modules[name].norm1[1].pos_max",model._modules[name].norm1[1].pos_max)
                if isinstance(child.norm2, DyHT_ReLU):
                    model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=False, cal_loss=True))
                else:
                    model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True, cal_loss=True))
                if not isinstance(child.norm2[0], nn.LayerNorm):
                    model._modules[name].norm2[1].s_max.data = model._modules[name].norm2[0].gamma/model._modules[name].norm2[1].pos_max
                # print("model._modules[name].norm2[0].gamma",model._modules[name].norm2[0].gamma,"model._modules[name].norm2[1].pos_max",model._modules[name].norm2[1].pos_max)

                model._modules[name].mlp.act = nn.Sequential(child.mlp.act,MyQuan(level, sym=False,channel_num=3072))
                model._modules[name].mlp.fc2 = nn.Sequential(child.mlp.fc2)
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, PatchEmbed):
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, PatchEmbedConv):
                if isinstance(model._modules[name].proj_conv, nn.Sequential):
                    model._modules[name].proj_conv[0] = nn.Sequential(child.proj_conv[0],MyQuan(level, sym=True))
                model._modules[name].proj_ReLU = nn.Sequential(child.proj_ReLU ,MyQuan(level, sym=False))
                model._modules[name].proj1_ReLU = nn.Sequential(child.proj1_ReLU ,MyQuan(level, sym=False))
                model._modules[name].proj_res_ReLU = nn.Sequential(child.proj_res_ReLU ,MyQuan(level, sym=False))
                model._modules[name].norm = nn.Sequential(child.norm ,MyQuan(level, sym=True))
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, PatchMergingConv):                
                model._modules[name].proj1_conv = nn.Sequential(MyQuan(level, sym=True), child.proj1_conv)
                model._modules[name].proj1_ReLU = nn.Sequential(child.proj1_ReLU , MyQuan(level, sym=False))
                model._modules[name].proj_res_conv = nn.Sequential(MyQuan(level, sym=True), child.proj_res_conv)
                model._modules[name].proj_res_ReLU = nn.Sequential(child.proj_res_ReLU ,MyQuan(level, sym=False))
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child, SwinTransformerBlock):
                # print(children)
                qattn = QWindowAttention(dim=child.attn.num_heads*child.attn.head_dim, window_size=child.attn.window_size,num_heads=child.attn.num_heads,level=level)
                qattn.qkv = child.attn.qkv
                # qattn.q_norm = child.q_norm
                # qattn.k_norm = child.k_norm
                qattn.attn_drop = child.attn.attn_drop
                qattn.proj = child.attn.proj
                qattn.proj_drop = child.attn.proj_drop
                qattn.relative_position_bias_table = child.attn.relative_position_bias_table
                qattn.relative_position_index = child.attn.relative_position_index

                if isinstance(child.norm1, DyHT_ReLU):
                    model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=False, cal_loss=False))
                else:
                    model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True, cal_loss=False))
                model._modules[name].norm1[1].s_max.data = model._modules[name].norm1[0].gamma/model._modules[name].norm1[1].pos_max
                # print("model._modules[name].norm1[0].gamma",model._modules[name].norm1[0].gamma,"model._modules[name].norm1[1].pos_max",model._modules[name].norm1[1].pos_max)
                if isinstance(child.norm2, DyHT_ReLU):
                    model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=False, cal_loss=False))
                else:
                    model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True, cal_loss=False))
                model._modules[name].norm2[1].s_max.data = model._modules[name].norm2[0].gamma/model._modules[name].norm2[1].pos_max

                qattn.attn_softmax_quan.s_max.data = torch.tensor(1.0/qattn.attn_softmax_quan.pos_max)

                model._modules[name].attn = qattn
                # model._modules[name].act1 = MyQuan(level, sym=True)
                # model._modules[name].act2 = MyQuan(level, sym=True)
                # model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True))
                # model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True))
                # model._modules[name].mlp.fc1 = nn.Sequential(child.mlp.fc1,MyQuan(level, sym=False))
                model._modules[name].mlp.act = nn.Sequential(child.mlp.act,MyQuan(level, sym=False))
                model._modules[name].mlp.fc2 = nn.Sequential(child.mlp.fc2)
                # model._modules[name].addition1 = nn.Sequential(Addition(),MyQuan(level, sym=True))
                # model._modules[name].addition2 = nn.Sequential(Addition(),MyQuan(level, sym=True))
                # print("model._modules[name].addition1",model._modules[name].addition1)
                # print("index",cur_index,"myquan replace finish!!!!")
                cur_index = cur_index + 1
                is_need = True
            # if isinstance(child, Attention):
            #     # print(children)
            #     qattn = QAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=level)
            #     qattn.qkv = child.qkv
            #     # qattn.q_norm = child.q_norm
            #     # qattn.k_norm = child.k_norm
            #     qattn.attn_drop = child.attn_drop
            #     qattn.proj = child.proj
            #     qattn.proj_drop = child.proj_drop
            #     model._modules[name] = qattn
            #     print("index",cur_index,"myquan replace finish!!!!")
            #     cur_index = cur_index + 1
            #     is_need = True
            # elif isinstance(child,Mlp):
            #     model._modules[name].act = nn.Sequential(MyQuan(level,sym = False),child.act)
            #     model._modules[name].fc2 = nn.Sequential(child.fc2,MyQuan(level,sym = True))
            #     is_need = True
            elif isinstance(child, MyBatchNorm1d):
                if isinstance(child, DyHT_ReLU):
                    model._modules[name] = nn.Sequential(child,MyQuan(level,sym = False))
                    model._modules[name][1].s_max.data = model._modules[name][0].gamma/model._modules[name][1].pos_max
                else:
                    model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                is_need = True
            # elif isinstance(child, DyHT):
            #     print("replace DyHT!!!")
            #     model._modules[name] = MyQuan(level,sym = True, alpha=1/(child.alpha.mean()*child.gamma.mean()))
            #     is_need = True
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True,first=True))
                is_need = True
            # elif isinstance(child, nn.Linear):
            #     model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
            #     is_need = True
            # elif isinstance(child, Block):
            #     model._modules[name].norm1 = nn.Sequential(child.norm1,MyQuan(level,sym = True))
            #     model._modules[name].norm2 = nn.Sequential(child.norm2,MyQuan(level,sym = True))
            #     is_need = False
            elif isinstance(child, nn.LayerNorm):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                # child.register_forward_hook(hook_layernorm)
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    
    def _weight_quantization(model,weight_bit):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            if not is_need:
                _weight_quantization(child,weight_bit)
                
    get_index(model)
    _myquan_replace(model,level)
    if weight_bit < 32:
        _weight_quantization(model,weight_bit)

def myquan_replace_resnet(model,level,weight_bit=32, is_softmax = True):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model,level):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.ReLU):
                model._modules[name] = MyQuan(level,sym = False)
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    
    def _weight_quantization(model,weight_bit):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            if not is_need:
                _weight_quantization(child,weight_bit)
                
    get_index(model)
    _myquan_replace(model,level)
    if weight_bit < 32:
        _weight_quantization(model,weight_bit)
