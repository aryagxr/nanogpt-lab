import torch
from torch import nn
import torch.nn.functional as F

from nanogpt.position import build_position


#Causal self attention
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, linear, position, head_dim=128, scale=0.12):
        super().__init__()
        assert dim % head_dim == 0 and head_dim % 4 == 0
        self.scale = scale
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        #Wq, Wk, Wv
        self.q_proj = linear(dim, dim)
        self.k_proj = linear(dim, dim)
        self.v_proj = linear(dim, dim)
        self.out_proj = linear(dim, dim)
        self.rope = build_position(head_dim, position)

    def forward(self, x):
        B, T, D = x.shape

        #B T D -> B T H D_h
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        #QK norm, and rope
        q, k = F.rms_norm(Q, (Q.size(-1),)), F.rms_norm(K, (K.size(-1),))
        q, k = self.rope(q), self.rope(k)

        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), V.transpose(1, 2), scale=self.scale, is_causal=True).transpose(1, 2)
        out = out.contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out
