from torch import nn


class StandardResidual(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()

    def forward(self, x, blocks):
        for block in blocks:
            x = x + block.attn(block.norm1(x))
            x = x + block.mlp(block.norm2(x))
        return x
