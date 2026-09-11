import torch
from torch import nn


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
