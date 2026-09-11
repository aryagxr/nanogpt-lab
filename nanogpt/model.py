import torch
from torch import nn
import torch.nn.functional as F

from nanogpt.attention import build_attention


#rms norm
#input hidden state B T D
class RMSnorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))



#Linear
class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


#mlp
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


#block
class Block(nn.Module):
    def __init__(self, dim, attention):
        super().__init__()
        self.norm1 = RMSnorm(dim)
        self.norm2 = RMSnorm(dim)
        self.attn = build_attention(dim, attention, Linear)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

#gpt
class GPT(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers, attention):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, attention) for _ in range(num_layers)])
        self.norm1 = RMSnorm(model_dim)
        self.norm2 = RMSnorm(model_dim)
        self.proj = Linear(model_dim, vocab_size)


    def forward(self, inputs, targets):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        #block output -> norm2 -> linear
        logits = self.proj(self.norm2(x)).float()
        #logit softcap
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        #summed locally per step and averaging at the end
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
