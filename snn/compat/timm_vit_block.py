import torch.nn as nn

from snn.nvtx import nvtx_range

class BlockWithAddition(nn.Module):
    def __init__(self, norm1, attn, drop_path, norm2, mlp):
        super().__init__()
        self.norm1 = norm1
        self.attn = attn
        self.drop_path = drop_path
        self.norm2 = norm2
        self.mlp = mlp
        self.addition1 = None
        self.addition2 = None

    @classmethod
    def from_block(cls, block):
        return cls(block.norm1, block.attn, block.drop_path, block.norm2, block.mlp)

    def forward(self, x):
        with nvtx_range("snn.compat.timm_vit_block.BlockWithAddition.forward"):
            if self.addition1 is not None:
                x = self.addition1((x, self.drop_path(self.attn(self.norm1(x)))))
                x = self.addition2((x, self.drop_path(self.mlp(self.norm2(x)))))
                return x
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
