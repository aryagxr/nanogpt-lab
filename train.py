"""Baseline from modded NanoGPT track 3"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW

from pathlib import Path



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
    assert batch_size % 8 == 0
    local_batch_size = batch_size // 8
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        #check if reading the next batch of tokens will exceed the current 100M tok file
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        #distribute the data across ranks
        buf = tokens[pos + 1 * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cpu", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cpu", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)





########################################
#             Architecture             #
########################################

#rms norm
#input hidden state B T D
class RMSnorm(nn.Module):
    def __init__(self, dim):
        super().__init()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x / F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


#Linear
class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features



#RoPE
class RoPE(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x):
        pass

#Causal self attention
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, head_dim=128):
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        #Wq, Wk, Wv
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
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

        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), V.transpose(1, 2), scale=0.12, is_causal=True)
        out = out.contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out



#mlp
class MLP(nn.Module):
    def __init__(self, dim):
        pass

#block
class Block(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

#gpt
class GPT(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


########################################
#              Optimizer               #
########################################

class MuonH(torch.optim.Optimizer):
    def __init__(self):
        pass

    def step(self):
        pass


########################################
#                Setup                 #
########################################

train_steps = 3325
batch_size = 8 * 64 * 1024 #8 ranks, shape 64,1024
mbs = 64

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()



#initialize model params 



#3 param groups for adamW
#each dict is a param group
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                    betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)

optimizer2 = MuonH([p for p in model.blocks.params if p.ndim == 2], lr=0.018)
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
            print(grp)
            #if starting phase
            if progress < 1 - grp["cooldown_frac"]:
                eta = 1.0
            else:
                #cooldown phase, linear decrease
                eta = (1 - progress) / grp["cooldown_frac"]
            grp["lr"] = grp["initial_lr"] * eta

            


#train loop

train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)

for step in range(train_steps + 1):
    #validation

    #training
    inputs, targets = next(train_loader)
    #gradient accumulation in microbatches
    assert len(inputs) % mbs == 0
    num_mbs = len(inputs) // mbs
    for i in range(num_mbs):
        #run fwd and bwd in microbatches
        model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        #sum up the gradients for every param across the gpu ranks
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

    #set optimization hyperparams and take step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)


dist.destroy_process_group()




print("Inputs shape:", inputs.shape)
print("Targets shape:", targets.shape)
print("\nFirst sequence inputs:\n", inputs[0])
print("\nFirst sequence targets:\n", targets[0])