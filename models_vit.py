# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from torchvision import transforms
import timm.models.vision_transformer

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.spike = False
        self.T = 0
        self.step = 0
        self.grad_checkpointing = False
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = bool(enable)

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    
    def forward_features(self, x):
        # print((x == 0).all())
        B = x.shape[0]
        # print("B",B,"self.T",self.T)
        # if B == self.T*8:
        #     print("SNN: x",x.reshape(self.T,8,3,224,224).sum(dim = 0).abs().mean())
        # else:
        #     print("x",x.abs().mean())
        x = self.patch_embed(x)
        # if B == self.T*8:
        #     print("SNN: patch_embed",x.reshape(self.T,8,196,768).sum(dim = 0).abs().mean())
        # else:
        #     print("patch_embed",x.abs().mean())
        # print((x == 0).all())

        # print("x.shape",x.shape)
        if self.spike:
            B1 = B//self.T
            step = min(self.T, self.step)
            cls_tokens = torch.cat([self.cls_token.expand(B1*step, -1, -1), torch.zeros_like(self.cls_token.expand(B1*(self.T-step), -1, -1))],dim=0)  # stole cls_tokens impl from Phil Wang, thanks
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        # if B == self.T*8:
        #     print("SNN: cls_tokens",x.reshape(self.T,8,197,768).sum(dim = 0).abs().mean())
        # else:
        #     print("cls_tokens",x.abs().mean())
        # print((x == 0).all())
        if self.spike:
            step = min(self.T, self.step)
            pos_embed = torch.cat([self.pos_embed.expand(B1*step, -1, -1), torch.zeros_like(self.pos_embed.expand(B1*(self.T-step), -1, -1))],dim=0)  # stole cls_tokens impl from Phil Wang, thanks
        else:
            pos_embed = self.pos_embed.expand(B, -1, -1)
        x = x + pos_embed
        # if B == self.T*8:
        #     print("SNN: self.pos_embed",(self.pos_embed*self.step).abs().mean())
        #     print("SNN: after pos_embed",x.reshape(self.T,8,197,768).sum(dim = 0).abs().mean())
        # else:
        #     print("self.pos_embed",(self.pos_embed).abs().mean())
        #     print("after pos_embed",x.abs().mean())
        # print(self.pos_embed)
        x = self.pos_drop(x)

        use_checkpoint = bool(getattr(self, "grad_checkpointing", False)) and self.training
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            for blk in self.blocks:
                try:
                    x = checkpoint(blk, x, use_reentrant=False)
                except TypeError:  # older torch
                    x = checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            distill_token = x[:, 0]
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        # print("outcome", outcome)
        return outcome



class VisionTransformerDVS(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, in_channels_dvs=18, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), **kwargs):
        super(VisionTransformerDVS, self).__init__(**kwargs)

        self.align = nn.Conv2d(in_channels=in_channels_dvs, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.global_pool = global_pool
        self.mean = mean
        self.std = std
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        # print((x == 0).all())
        B = x.shape[0]
        x = self.align(x)
        # x = transforms.functional.normalize(x, self.mean, self.std)
        x = self.patch_embed(x)
        # print((x == 0).all())

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # # print(cls_tokens)
        # # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        # # print((x == 0).all())
        x = x + self.pos_embed
        # print(self.pos_embed)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome



def vit_test_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4, qkv_bias=True,
         **kwargs) # remember the modify
    return model

def vit_test_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
         **kwargs) # remember the modify
    return model

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
         **kwargs) # remember the modify
    return model

def vit_small_patch16_dvs(**kwargs):
    model = VisionTransformerDVS(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == '__main__':
    model = vit_small_patch16(act_layer=nn.ReLU)
    d = torch.load("pretrained/deit-small-pretrained.pth")["model"]
    model.load_state_dict(d)
    # import torch
    # # check you have the right version of timm
    # import timm
    #
    # assert timm.__version__ == "0.3.2"
    #
    # # now load it with torchhub
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
    # print(model.blocks[0].attn.num_heads)
