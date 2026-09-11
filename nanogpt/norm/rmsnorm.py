import torch
from torch import nn
import torch.nn.functional as F


#rms norm
#input hidden state B T D
class RMSnorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))
