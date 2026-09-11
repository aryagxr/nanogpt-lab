import torch
import torch.distributed as dist
from torch.optim import AdamW


def zeropower_via_newtonschulz5(G):
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

def scale_invariant_update_(param, update, lr, eps = 1e-10):
    """Hyperball-constrained step: take a Muon-orthogonalised update of size lr * ||param||,
    then renormalise back onto the Frobenius sphere of the parameter's initial radius. Preserves
    ||param|| exactly across training; the invariant lets us drop weight decay on hidden
    matrices entirely (the constraint already prevents norm growth)."""
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)

class MuonH(torch.optim.Optimizer):
    """MuonH: same Newton-Schulz orthogonalised direction as Muon, applied via a Frobenius-
    norm-preserving hyperball projection. Used here for ALL hidden 2D weight matrices —
    q, k, v, mlp.fc, attn.proj, mlp.proj — under non-zero (Kaiming-derived) init."""
    def __init__(self, params, lr=0.014, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    scale_invariant_update_(p, update, group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

 
def build_optimizers(model, config):
    optimizer1 = AdamW([
        dict(params=[model.embed.weight], lr=config["embed_lr"], name="embed"),
        dict(params=[model.proj.weight], lr=config["head_lr"], name="head"),
        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=config["scalar_lr"], name="scalar"),
    ], betas=tuple(config["betas"]), eps=config["eps"],
        weight_decay=config["weight_decay"], fused=True)
    optimizer2 = MuonH([p for p in model.blocks.parameters() if p.ndim == 2],
                       lr=config["lr"], mu=config["momentum"])
    for group in optimizer2.param_groups:
        group["name"] = "muonh"
    return [optimizer1, optimizer2]
