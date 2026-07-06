#code adapted from:
#https://github.com/tyler-romero/nanogpt-speedrun
#https://github.com/karpathy/llm.c


import contextlib
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import math
import glob
from dataclasses import dataclass
import subprocess
import time

import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb


#power iter
@torch.no_grad()
def power_iteration_spectral_norm(X, state, state_key="power_v", min_iters=3, max_iters=60, rtol=1e-3, eps=1e-7):
    Xf = X.float()
    m = Xf.size(-2)
    fro = Xf.norm().clamp_min(eps)
    sigma_lo = fro / (m ** 0.5)

    u = state.get(state_key)
    if u is None or u.shape[0] != m or u.device != X.device or not torch.isfinite(u).all():
        u = torch.randn(m, device=X.device, dtype=torch.float32)
        u = u / u.norm().clamp_min(eps)
    else:
        u = u.float()

    #doing XX^T because matrix is not square all the time
    #XX^T finds top left SV
    prev_est = None
    for i in range(max_iters):
        u = Xf @ (Xf.mT @ u)
        u_norm = u.norm()
        if not torch.isfinite(u_norm) or u_norm < eps:
            state.pop(state_key, None)
            return fro.to(dtype=X.dtype)
        u = u / u_norm
        est = u_norm.sqrt()
        if prev_est is not None and i + 1 >= min_iters and abs(est - prev_est) <= rtol * est:
            break
        prev_est = est

    state[state_key] = u

    sigma = (Xf.mT @ u).norm()
    if not torch.isfinite(sigma):
        state.pop(state_key, None)
        return fro.to(dtype=X.dtype)

    sigma = sigma.clamp(min=sigma_lo, max=fro)
    return sigma.clamp_min(eps).to(dtype=X.dtype)


#margin to prevent sv max going above 1
SPECTRAL_MARGIN = 1.05


def _ns_iterations(X, ns_steps, a, b, c):
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def zeropower_via_newtonschulz(G, ns_steps, state=None, state_key="power_v"):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.float()
    #if tall matrix, transpose
    #muon more stable with wider matrices
    if G.size(-2) > G.size(-1):
        X = X.mT

    sigma = power_iteration_spectral_norm(X, state, state_key)
    Xn = (X / (sigma.float() * SPECTRAL_MARGIN)).bfloat16()
    out = _ns_iterations(Xn, ns_steps, a, b, c)


    if not torch.isfinite(out).all():
        if state is not None:
            state.pop(state_key, None)
        print(f"[power_muon] NS diverged for shape {list(G.shape)}; redoing with Frobenius norm this step")
        Xn = (X / X.norm().clamp_min(1e-7)).bfloat16()
        out = _ns_iterations(Xn, ns_steps, a, b, c)
    X = out

    #transpose back the tall matrices
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


#update the weights
#takes in the gradient
#smoothens gradient using momentum
#orthogonalize it
#rescale
def muon_update(grad, momentum, state, beta=0.95, ns_steps=5, nesterov=True):
    #ema
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    #split grouped QKV parameters
    if update.size(0) == 3 * update.size(1):
        update = torch.cat([
            zeropower_via_newtonschulz(g, ns_steps, state, state_key=f"power_v_{i}")
            for i, g in enumerate(update.split(update.size(1)))
        ])
        scale = update.size(1)**0.5
    else:
        update = zeropower_via_newtonschulz(update, ns_steps, state)
        scale = max(update.size(0), update.size(1))**0.5
    return update, scale


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)



# https://github.com/KellerJordan/Muon/blob/master/muon.py#L228
class Muon(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['use_muon']:
                #muon update for 2d layers
                for p in group['params']:
                    if p.grad is None:
                        continue
                    #state to remember prev momentum
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    #ema of past gradients
                    mbuf = state['momentum_buffer']
                    update, scale = muon_update(p.grad, mbuf, state, beta=group['momentum'])
                    #weight decay w ← (1 − lr * weight_decay) w
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update.reshape(p.shape), alpha=-group['lr'] * scale)

            else:
                #adam for linear
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['step'] = 0
                    state['step'] += 1
                    update = adam_update(p.grad, state['exp_avg'], state['exp_avg_sq'], state['step'], group['betas'], group['eps'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])



def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class RoPE(nn.Module):
    def __init__(self, emb_dim, base=10000):
        super().__init__()
        #self attributes
        self.freqs = 1 / (base ** (torch.arange(0, emb_dim, 2) / emb_dim))


    #return cos and sin
    def forward(self, x):
        # x: (B, T, nh, head_dim)
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device).float()
        theta = torch.outer(pos, self.freqs.to(x.device)) #(T, head_dim//2)
        cos_theta = torch.cos(theta).unsqueeze(0).unsqueeze(2) #(1, T, 1, head_dim//2)
        sin_theta = torch.sin(theta).unsqueeze(0).unsqueeze(2) #(1, T, 1, head_dim//2)
        return cos_theta, sin_theta  #broadcast over B and nh in rope_rotate



def rope_rotate(tok, cos, sin):
    #forming pairs
    x_even = tok[...,0::2] #even
    y_odd = tok[..., 1::2] #odd
    x_rot = x_even * cos - y_odd * sin
    y_rot = x_even * sin + y_odd * cos
    #reconstruct the embedding again
    out = torch.stack([x_rot, y_rot], dim=-1)
    out = out.flatten(-2)
    return out



class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #qkv proj for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape to (B, T, nh, hs) first so rope sees T at dim 1
        q = q.view(B, T, self.n_head, self.head_dim)  # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rope(q)  
        q = rope_rotate(q, cos, sin)
        k = rope_rotate(k, cos, sin)
        #QKNorm
        q, k = rmsnorm(q), rmsnorm(k)
        # now transpose to (B, nh, T, hs) for flash attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        # y = y / math.sqrt(24)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        # https://arxiv.org/abs/2109.08668v2
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


#GPT2

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # initialize the position embedding at std=0.02 to match the scale of the token embedding.
        if isinstance(module, nn.Embedding) and not hasattr(module, 'LLMC_SKIP_INIT'):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        assert t <= self.config.block_size

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        x = tok_emb  # no absolute position embedding — RoPE handles position inside attention

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    # def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(), lr=learning_rate,
    #         weight_decay=weight_decay, betas=betas,
    #     )
    #     return optimizer




#Distributed Dataloader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        exit(1)
    assert header[1] == 1, "unsupported version"
    return header[2]

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


#logging helpers

def print0(s, console=False):
    if master_process:
        with open(logfile, 'a') as f:
            if console:
                print(s)
            print(s, file=f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin",     type=str, default="dev/data/fineweb10B/fineweb_train_*.bin",
                        help="glob pattern for training .bin shards")
    parser.add_argument("--input_val_bin", type=str, default="dev/data/fineweb10B/fineweb_val_*.bin",
                        help="glob pattern for validation .bin shards")
    parser.add_argument("--output_dir",    type=str, default="", help="directory for logs and checkpoints")
    parser.add_argument("--model",         type=str, default="d12", help="model size: d12")
    # token layout
    parser.add_argument("--batch_size",       type=int, default=64,
                        help="micro-batch size per device (sequences per forward pass)")
    parser.add_argument("--sequence_length",  type=int, default=1024, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=262144,
                        help="total token budget per gradient step; must equal "
                             "batch_size * sequence_length * world_size * grad_accum_steps")
    # training length
    parser.add_argument("--num_iterations", type=int, default=24576,
                        help="number of gradient steps (~6.44B tokens at default batch size)")
    # optimisation
    parser.add_argument("--learning_rate", type=float, default=1.5e-3)
    parser.add_argument("--warmup_iters",  type=int,   default=0,
                        help="linear LR warmup steps (0 = no warmup, for Muon)")
    parser.add_argument("--warmdown_iters", type=int, default=1800,
                        help="linear LR warmdown steps at end of training (trapezoidal schedule)")
    parser.add_argument("--weight_decay",  type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="gradient accumulation steps (set automatically if 0)")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=250,
                        help="evaluate validation loss every N steps")
    parser.add_argument("--val_max_steps",  type=int, default=20,
                        help="number of val batches to average")
    parser.add_argument("--target_val_loss", type=float, default=None,
                        help="stop early once val loss reaches this value (e.g. 3.28)")
    parser.add_argument("--disable_wandb",  action="store_true")
    args = parser.parse_args()



    # -------------------------------------------------------------------------
    # Distributed setup — works both with `python train.py` (single GPU)
    # and `torchrun --nproc_per_node=N train.py` (multi-GPU).
    assert torch.cuda.is_available(), "CUDA is required"
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank       = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device         = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
    else:
        ddp_rank       = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device         = 'cuda'

    # begin logging
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs('logs', exist_ok=True)
        logfile = f'logs/{run_id}.txt'
        print(f"Logfile: {logfile}")
        if not args.disable_wandb:
            wandb.init(project="nanogpt-speedrun", name=str(run_id), config=args)
    else:
        # non-master processes need logfile defined even though they never write to it
        logfile = '/dev/null'



    # Validate batch-size accounting
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024, "sequence_length must be in [1, 1024]"
    assert args.model == "d12", f"Only d12 is supported in this baseline; got {args.model!r}"

    grad_accum_steps = args.grad_accum_steps
    tokens_per_step  = B * T * ddp_world_size * grad_accum_steps
    assert tokens_per_step == args.total_batch_size, (
        f"batch_size({B}) * sequence_length({T}) * world_size({ddp_world_size}) "
        f"* grad_accum_steps({grad_accum_steps}) = {tokens_per_step} "
        f"!= total_batch_size({args.total_batch_size})"
    )


    # Log environment info
    print0(code)
    print0('=' * 100)
    print0(f'Running Python {sys.version}')
    print0(f'Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}')
    print0(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    print0('=' * 100)
    print0(f"ddp={ddp} world_size={ddp_world_size} device={device}")
    print0(f"B={B} T={T} grad_accum_steps={grad_accum_steps} tokens_per_step={tokens_per_step}", console=True)

    # Build model
    model_configs = {
        # vocab padded to nearest multiple of 128 for better tensor-core utilisation
        #changing n_heads to 6, so that n_heads(6) x head_dim(128) = n_embd(768)
        #since 128 head dim better for qknorm
        #see https://x.com/kellerjordan0/status/1845865698532450646
        "d12": GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=6, n_embd=768),
    }
    model = GPT(model_configs[args.model]).train().cuda()

    if hasattr(inductor_config, "coordinate_descent_tuning"):
        inductor_config.coordinate_descent_tuning = True
    print0("Compiling model...", console=True)
    model = torch.compile(model)

    if ddp:
        raw_model = model
        model = DDP(model, device_ids=[ddp_local_rank])
    else:
        raw_model = model



    # Data loaders
    # train loader: load a full grad-accum batch at once (B * grad_accum_steps rows)
    train_loader = DistributedDataLoader(
        args.input_bin, B * grad_accum_steps, T, ddp_rank, ddp_world_size
    )
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(
            args.input_val_bin, B, T, ddp_rank, ddp_world_size
        )


    #Optimizer
    # optimizer = raw_model.configure_optimizers(
    #     weight_decay=args.weight_decay,
    #     learning_rate=args.learning_rate,
    #     betas=(0.9, 0.95),
    #     device_type=device,
    # )
    param_groups = [
    dict(params=list(raw_model.transformer.h.parameters()), use_muon=True),
    dict(params=list(raw_model.lm_head.parameters()), use_muon=False, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay),
    ]
    optimizer = Muon(param_groups)

    # LR schedule: trapezoidal (warmup → constant → linear warmdown to zero)
    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            return (it + 1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return decay_ratio

    
    # Training loop
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    tokens_seen  = 0
    training_time_ms = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_loader.reset()
    for step in range(args.num_iterations + 1):
        last_step   = (step == args.num_iterations)
        # Ignore timing for first 10 steps (kernel warm-up / compile effects)
        if step == 10:
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1

        
        # Validation
        if (args.val_loss_every > 0
                and (step % args.val_loss_every == 0 or last_step)
                and val_loader is not None):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)

            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    xv, yv = val_loader.next_batch()
                    with ctx:
                        _, loss = model(xv, yv, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps

            print0(
                f"step:{step}/{args.num_iterations} "
                f"val_loss:{val_loss:.4f} "
                f"train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / max(timed_steps - 1, 1):.2f}ms",
                console=True,
            )
            if master_process and not args.disable_wandb:
                wandb.log({
                    "val_loss": val_loss,
                    "train_time": training_time_ms,
                    "step": step,
                    "step_avg": training_time_ms / max(timed_steps - 1, 1),
                })

            # Early stopping: report tokens needed and exit
            if args.target_val_loss is not None and val_loss <= args.target_val_loss:
                tokens_to_target = tokens_seen + (
                    # add tokens from steps that happened after the last logged tokens_seen
                    # (tokens_seen is updated after training, before val; so it's current)
                    0
                )
                print0(
                    f"\n*** TARGET REACHED ***\n"
                    f"val_loss {val_loss:.4f} <= target {args.target_val_loss}\n"
                    f"tokens_seen: {tokens_seen:,}\n"
                    f"steps: {step}\n"
                    f"train_time: {training_time_ms / 1000:.1f}s ({training_time_ms / 60000:.2f} min)\n",
                    console=True,
                )
                if master_process and not args.disable_wandb:
                    wandb.summary["tokens_to_target"] = tokens_seen
                    wandb.summary["steps_to_target"]  = step
                    wandb.summary["time_to_target_s"]  = training_time_ms / 1000
                break

            torch.cuda.synchronize()
            t0 = time.perf_counter()


        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # ------------------------------------------------------------------
        # Training step
        model.train()
        optimizer.zero_grad(set_to_none=True)

        x, y = train_loader.next_batch()   # shape: (B * grad_accum_steps, T)

        with ctx:
            for micro_x, micro_y in zip(
                x.chunk(grad_accum_steps, dim=0),
                y.chunk(grad_accum_steps, dim=0),
            ):
                # Disable gradient sync on all but the last micro-batch when using DDP
                sync_ctx = contextlib.nullcontext() if not ddp else (
                    model.no_sync() if micro_x is not x.chunk(grad_accum_steps, dim=0)[-1]
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    _, loss = model(micro_x, micro_y, return_logits=False)
                    # divide by grad_accum_steps so accumulated gradient equals
                    # the gradient of the mean loss over the full batch
                    (loss / grad_accum_steps).backward()
            train_loss = loss.detach()  # loss of last micro-batch (for logging)

        lr_schedule = get_lr(step)
        for pg in optimizer.param_groups:
            if pg['use_muon']:
                pg['lr'] = 0.1 * args.learning_rate * lr_schedule
            else:
                pg['lr'] = args.learning_rate * lr_schedule
        optimizer.step()


        
        # Diagnostics
        if master_process:
            tokens_seen += tokens_per_step
            approx_time  = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(
                f"step:{step + 1}/{args.num_iterations} "
                f"train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_time:.0f}ms "
                f"step_avg:{approx_time / timed_steps:.2f}ms "
                f"tokens_seen:{tokens_seen:.2e} "
                f"lr_scale:{lr_schedule:.3e}",
                console=True,
            )
            if not args.disable_wandb:
                wandb.log({
                    "train_loss": train_loss.item(),
                    "train_time": approx_time,
                    "step": step + 1,
                    "step_avg": approx_time / timed_steps,
                    "tokens_seen": tokens_seen,
                    "lr": lr_schedule,
                })

    # -------------------------------------------------------------------------
    print0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", console=True)

    if ddp:
        destroy_process_group()

    if master_process and not args.disable_wandb:
        wandb.finish()
