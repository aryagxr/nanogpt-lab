"""Baseline from modded NanoGPT track 3"""


import os
import sys
import time
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW

from pathlib import Path


with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging



########################################
#              Dataloader              #
########################################


#return all tokens as a 1D tensor in a single .bin input file
def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    print(header[0])
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4) #skip the header
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        #tokens stored as unit16 (2 bytes each)
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(filename_pattern, batch_size, seq_len=1024):
    #convert the file pattern into a path
    #go through .bin shard files one by one
    #take batch size + 1 amount of of tokens
    #split those 524289 tokens across GPUs
    #each gpu takes its assigned slice and creates input and target pair (65,536 tokens each)
    #reshape the tokens in each gpu to shape 64,1024
    #advance the pos pointer by batch size and load the next batch
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        #check if reading the next batch of tokens will exceed the current 100M tok file
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        #distribute the data across ranks
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)





########################################
#             Architecture             #
########################################

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
    def __init__(self, dim, head_dim=128):
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        #Wq, Wk, Wv
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.out_proj = Linear(dim, dim)
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

        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), V.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        out = out.contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out



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
    def __init__(self, dim):
        super().__init__()
        self.norm1 = RMSnorm(dim)
        self.norm2 = RMSnorm(dim)
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

#gpt
class GPT(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
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



########################################
#              Optimizer               #
########################################


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



########################################
#                Setup                 #
########################################

#torchrun sets these environment variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
#the code can run equivalently with 1, 2, 4, or 8 GPUs
assert 8 % dist.get_world_size() == 0

#logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)

def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

#log this exact source file and the environment used for the run
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

train_steps = 3325
batch_size = 8 * 64 * 1024 #8 ranks, shape 64,1024
mbs = 64
val_tokens = 20 * 524288

val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))
model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
print0(f"num_params:{num_params}", console=True)



#initialize model params 
for name, p in model.named_parameters():
    w = p.data
    if name.endswith("weight"):
        if "embed" in name:
            w.normal_()
        else:
            w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
    elif name.endswith("bias"):
        w.zero_()
    elif name.endswith("gains"):
        #1 for evert param
        w.normal_(mean=1, std=0)
    else:
        raise Exception(f"Uninitialized parameter: {name}")
    #layer specific multipliers
    if name.endswith(".attn.out_proj.weight"):
        w.mul_(1.25)
    elif name.endswith(".mlp.proj.weight"):
        w.mul_(3.0)
    elif name.endswith(".mlp.fc.weight"):
        w.mul_(1.5)


#3 param groups for adamW
#each dict is a param group
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                    betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)

optimizer2 = MuonH([p for p in model.blocks.parameters() if p.ndim == 2], lr=0.018)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for grp in opt.param_groups:
        grp["initial_lr"] = grp["lr"]

for group in optimizer1.param_groups:
    group["cooldown_frac"] = 0.4

#muon needs linear cooldown throughout entire run
for group in optimizer2.param_groups:
    group["cooldown_frac"] = 1.0


#LR scheduler
#eta is lr multiplier η
def set_hparams(step):
    progress = step / train_steps
    assert 0 <= progress < 1
    for opt in optimizers:
        for grp in opt.param_groups:
            #if starting phase
            if progress < 1 - grp["cooldown_frac"]:
                eta = 1.0
            else:
                #cooldown phase, linear decrease
                eta = (1 - progress) / grp["cooldown_frac"]
            grp["lr"] = grp["initial_lr"] * eta

            


#train loop

train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
#make every rank start with rank 0's parameters
for p in model.parameters():
    dist.broadcast(p.detach(), 0)

training_time = 0
last_val_step = 0
best_val_loss = float("inf")
torch.cuda.reset_peak_memory_stats(device)
dist.barrier()
t0 = time.perf_counter()

for step in range(train_steps + 1):
    #validation
    if step == train_steps or step % 125 == 0:
        dist.barrier()
        time_since_last_val = time.perf_counter() - t0
        step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
        last_val_step = step
        training_time += time_since_last_val

        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        best_val_loss = min(best_val_loss, val_loss.item())
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} best_val_loss:{best_val_loss:.5f}"
               + f" train_time:{training_time:.3f}s step_avg:{1000*step_avg:.2f}ms", console=True)
        model.train()
        dist.barrier()
        t0 = time.perf_counter()

    #the extra loop iteration is only for final validation
    if step == train_steps:
        break

    #training
    step_start = time.perf_counter()
    inputs, targets = next(train_loader)
    #gradient accumulation in microbatches
    assert len(inputs) % mbs == 0
    num_mbs = len(inputs) // mbs
    train_loss = 0
    for i in range(num_mbs):
        #run fwd and bwd in microbatches
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        train_loss += loss.detach()
        loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        #sum up the gradients for every param across the gpu ranks
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    train_loss /= batch_size
    grad_norm = torch.sqrt(sum(torch.linalg.vector_norm(param.grad, dtype=torch.float32).square()
                               for param in model.parameters()))

    #set optimization hyperparams and take step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    param_norm = torch.sqrt(sum(torch.linalg.vector_norm(param, dtype=torch.float32).square()
                                for param in model.parameters()))
    torch.cuda.synchronize()
    step_time = torch.tensor(time.perf_counter() - step_start, dtype=torch.float64, device=device)
    peak_gpu_memory = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.float64, device=device)
    dist.all_reduce(step_time, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_gpu_memory, op=dist.ReduceOp.MAX)

    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.5f} tokens:{(step+1)*batch_size}"
           + f" step_time:{step_time.item():.4f}s grad_norm:{grad_norm:.6e} param_norm:{param_norm:.6e}"
           + f" peak_gpu_memory_bytes:{int(peak_gpu_memory.item())}"
           + f" lr_embed:{optimizer1.param_groups[0]['lr']:.6e}"
           + f" lr_head:{optimizer1.param_groups[1]['lr']:.6e}"
           + f" lr_scalar:{optimizer1.param_groups[2]['lr']:.6e}"
           + f" lr_muonh:{optimizer2.param_groups[0]['lr']:.6e}"
           + f" train_time:{approx_training_time:.3f}s", console=True, log=False)


dist.destroy_process_group()
