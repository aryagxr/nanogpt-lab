import torch
from torch import nn
import torch.nn.functional as F


#RoPE
class RoPE(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        #half-truncated RoPE (with base frequency tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(head_dim//4)]))

    def forward(self, x_BTHD):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

#Causal self attention
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, linear, head_dim=128, scale=0.12):
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
        self.rope = RoPE(head_dim)

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
