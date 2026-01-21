import torch
import torch.nn as nn

from timm.models.layers.helpers import to_2tuple
from timm.models.layers.trace_utils import _assert


class PatchEmbedConv(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.proj_conv = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = norm_layer(embed_dim // 2)
        self.proj_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_ReLU = nn.ReLU(inplace=True)

        self.proj1_conv = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = norm_layer(embed_dim)
        self.proj1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_ReLU = nn.ReLU(inplace=True)

        self.proj2_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = norm_layer(embed_dim)

        self.proj_res_conv = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = norm_layer(embed_dim)
        self.proj_res_ReLU = nn.ReLU(inplace=True)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj_conv(x).reshape(B, self.embed_dim//2, H * W).transpose(1,2).contiguous()
        x = self.proj_bn(x).transpose(1,2).reshape(B, self.embed_dim//2, H, W).contiguous()
        x = self.proj_maxpool(x)
        x = self.proj_ReLU(x)
        
        x_feat = x + 0.0
        x = self.proj1_conv(x).reshape(B, self.embed_dim, (H//2) * (W//2)).transpose(1,2).contiguous()
        x = self.proj1_bn(x).transpose(1,2).reshape(B, self.embed_dim, H//2, W//2).contiguous()
        x = self.proj1_maxpool(x)
        x = self.proj1_ReLU(x)

        x = self.proj2_conv(x).reshape(B, self.embed_dim, (H//4) * (W//4)).transpose(1,2).contiguous()
        x = self.proj2_bn(x).transpose(1,2).reshape(B, self.embed_dim, H//4, W//4).contiguous()

        x_feat = self.proj_res_conv(x_feat).reshape(B, self.embed_dim, (H//4) * (W//4)).transpose(1,2).contiguous()
        x_feat = self.proj_res_bn(x_feat).transpose(1,2).reshape(B, self.embed_dim, H//4, W//4).contiguous()

        x = self.proj_res_ReLU(x + x_feat)    
    
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)
        return x

class PatchMergingConv(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm): 
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.proj1_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = norm_layer(dim)
        self.proj1_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_ReLU = nn.ReLU(inplace=True)

        self.proj2_conv = nn.Conv2d(dim, 2*dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = norm_layer(2*dim)

        self.proj_res_conv = nn.Conv2d(dim, 2*dim, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = norm_layer(2*dim)
        self.proj_res_ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # B C H W
        x_feat = x + 0.0

        x = self.proj1_conv(x).reshape(B, C, H * W).transpose(1,2).contiguous()
        x = self.proj1_bn(x).transpose(1,2).reshape(B, C, H, W).contiguous()
        x = self.proj1_maxpool(x)
        x = self.proj1_ReLU(x)

        x = self.proj2_conv(x).reshape(B, 2*C, (H//2) * (W//2)).transpose(1,2).contiguous()
        x = self.proj2_bn(x).transpose(1,2).reshape(B, 2*C, H//2, W//2).contiguous()

        x_feat = self.proj_res_conv(x_feat).reshape(B, 2*C, (H//2) * (W//2)).transpose(1,2).contiguous()
        x_feat = self.proj_res_bn(x_feat).transpose(1,2).reshape(B, 2*C, H//2, W//2).contiguous()

        x = self.proj_res_ReLU(x_feat + x)
        return x.reshape(B, 2*C, (H//2) * (W//2)).transpose(1,2).contiguous()
