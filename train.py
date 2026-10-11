"""HC/mHC depth experiment on the existing modded NanoGPT backbone."""


import os
import sys
import argparse
import json
import random
import hashlib
from datetime import timedelta

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW

from pathlib import Path
from torch.utils.checkpoint import checkpoint

SHAPES = {"shallow-wide": (8, 768), "middle": (18, 512), "deep-narrow": (32, 384)}
CONFIGURATIONS = tuple(f"{arch}-{shape}" for arch in ("hc", "mhc") for shape in SHAPES)
SEED = 1337


def configuration(name):
    if name not in CONFIGURATIONS:
        raise ValueError(f"Choose one of {CONFIGURATIONS}, got {name!r}")
    architecture, regime = name.split("-", 1)
    layers, width = SHAPES[regime]
    return dict(configuration=name, architecture=architecture, regime=regime,
                num_layers=layers, model_dim=width, num_heads=width // 64, head_dim=64,
                aspect_ratio=width / layers, residual_streams=4, sinkhorn_iterations=20,
                seed=SEED, train_steps=3325, batch_tokens=524288, seq_len=1024,
                microbatch_sequences=4, val_tokens=20 * 524288, val_interval=125,
                routing_lr=0.001, routing_dynamic_weight_decay=0.1,
                vocab_size=50304, backbone_lr=0.018, embedding_lr=0.3,
                head_lr=1 / 320, scalar_lr=0.01,
                sinkhorn_implementation="log-softmax-v2" if architecture == "mhc" else None,
                compile_scope="block-and-head" if architecture == "mhc" else "model")


def plot_rank(suite_dir):
    """One six-panel training-by-depth figure from saved rank summaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suite_dir = Path(suite_dir)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True, constrained_layout=True)
    batch_hashes = set()
    last_image = None
    for row, architecture in enumerate(("hc", "mhc")):
        for col, (regime, (layers, width)) in enumerate(SHAPES.items()):
            ax = axes[row, col]
            directory = suite_dir / f"{architecture}-{regime}"
            saved = directory / "rank_progression.jsonl"
            if saved.exists():
                snapshots = [json.loads(line) for line in saved.read_text().splitlines() if line]
                records = [(item["step"], item["ranks_by_depth"]) for item in snapshots]
                batch_hashes.update(item["batch_sha256"] for item in snapshots)
            else:
                records = []
                for path in sorted((directory / "diagnostics").glob("step_*.json")):
                    item = json.loads(path.read_text())
                    ranks = sorted((entry for entry in item["statistics"]
                                    if entry["scope"] == "composite"
                                    and entry["metric"] == "effective_rank"
                                    and entry["component"] is None),
                                   key=lambda entry: entry["residual_depth"])
                    records.append((item["step"], [entry["mean"] for entry in ranks]))
                    batch_hashes.add(item["batch_sha256"])
            ax.set_title(f"{architecture.upper()} · {regime} ({layers}×{width})")
            ax.set_xlabel("Residual mapping (attention, MLP)")
            if col == 0:
                ax.set_ylabel("Training completed (%)")
            ax.set_ylim(0, 100)
            if not records:
                ax.text(0.5, 0.5, "Awaiting run", ha="center", va="center", transform=ax.transAxes)
                continue
            records.sort(key=lambda item: item[0])
            if any(len(values) != 2 * layers for _, values in records):
                raise ValueError(f"Incomplete rank-by-depth data for {directory.name}")
            steps = np.asarray([step for step, _ in records], dtype=float) / 3325 * 100
            values = np.asarray([values for _, values in records], dtype=float)
            if len(steps) == 1:
                edges = np.array([max(0, steps[0] - 2.5), min(100, steps[0] + 2.5)])
            else:
                edges = np.r_[max(0, steps[0] - (steps[1] - steps[0]) / 2),
                              (steps[:-1] + steps[1:]) / 2,
                              min(100, steps[-1] + (steps[-1] - steps[-2]) / 2)]
            last_image = ax.pcolormesh(np.arange(0.5, 2 * layers + 1.5), edges, values,
                                       cmap="viridis", vmin=1, vmax=4, shading="flat")
            ax.set_xlim(0.5, 2 * layers + 0.5)
    if len(batch_hashes) > 1:
        plt.close(fig)
        raise ValueError("Diagnostic validation tokens differ between runs")
    if last_image is None:
        plt.close(fig)
        return None
    fig.colorbar(last_image, ax=axes, label="Mean effective rank (1–4)", shrink=0.86)
    fig.suptitle(f"Residual transport rank across training and depth · seed {SEED}")
    path = suite_dir / "effective_rank_progression.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path



########################################
#              Dataloader              #
########################################


#return all tokens as a 1D tensor in a single .bin input file
def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
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
    def __init__(self, dim, head_dim=64):
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


def sinkhorn_log(logits, iterations=20):
    """Row then column Sinkhorn in log space; A[source, destination].

    Keep tiny entries in log space until all iterations finish. Exponentiating
    first can overflow the backward pass even when the forward output is finite.
    log_softmax needs no denominator epsilon and preserves the paper's projection.
    Its native backward avoids the large graph from repeated x - logsumexp(x).
    """
    log_matrix = logits - logits.amax(dim=(-2, -1), keepdim=True)
    for _ in range(iterations):
        log_matrix = F.log_softmax(log_matrix, dim=-1)
        log_matrix = F.log_softmax(log_matrix, dim=-2)
    return log_matrix.exp()


def nonfinite_names(named_tensors):
    """One host transfer; infinity norms avoid overflow from squaring finite values."""
    names, tensors = zip(*named_tensors)
    maxima = torch.stack(torch._foreach_norm(list(tensors), float("inf")))
    bad = (~maxima.isfinite()).nonzero().flatten().cpu().tolist()
    return [names[i] for i in bad]


class HyperConnection(nn.Module):
    """Four source streams; A[source, destination] is the applied residual map."""

    def __init__(self, dim, architecture, sublayer_index):
        super().__init__()
        self.architecture = architecture
        if architecture == "hc":
            self.dynamic_alpha = nn.Parameter(torch.zeros(dim, 5))
            self.dynamic_beta = nn.Parameter(torch.zeros(dim))
            self.alpha_scale = nn.Parameter(torch.tensor(0.01))
            self.beta_scale = nn.Parameter(torch.tensor(0.01))
            read = torch.zeros(4)
            read[sublayer_index % 4] = 1
            self.static_alpha = nn.Parameter(torch.cat((read[:, None], torch.eye(4)), dim=1))
            self.static_beta = nn.Parameter(torch.ones(4))
        elif architecture == "mhc":
            self.dynamic = nn.Parameter(torch.zeros(4 * dim, 24))
            self.scales = nn.Parameter(torch.full((3,), 0.01))
            read = torch.full((4,), -8.0)
            read[sublayer_index % 4] = 8.0
            residual = torch.full((4, 4), -8.0)
            residual.fill_diagonal_(0)
            self.bias = nn.Parameter(torch.cat((read, torch.zeros(4), residual.flatten())))
        else:
            raise ValueError(architecture)

    def forward(self, x, branch, norm, rank_product=None):
        if self.architecture == "hc":
            normalized = F.rms_norm(x.float(), (x.size(-1),), eps=1e-6)
            alpha = self.static_alpha + self.alpha_scale * (normalized @ self.dynamic_alpha).tanh()
            post = self.static_beta + self.beta_scale * (normalized @ self.dynamic_beta).tanh()
            pre, hres = alpha[..., 0], alpha[..., 1:]
        else:
            flat = x.float().flatten(-2)
            normalized = F.rms_norm(flat, (flat.size(-1),), eps=1e-6)
            dynamic = normalized @ self.dynamic
            pre = (self.scales[0] * dynamic[..., :4] + self.bias[:4]).sigmoid()
            post = 2 * (self.scales[1] * dynamic[..., 4:8] + self.bias[4:8]).sigmoid()
            logits = (self.scales[2] * dynamic[..., 8:] + self.bias[8:]).unflatten(-1, (4, 4))
            hres = sinkhorn_log(logits, iterations=20)
        if rank_product is not None:
            matrix = hres.double()
            product = matrix if rank_product[0] is None else rank_product[0] @ matrix
            # Scalar rescaling preserves effective rank and prevents HC product overflow.
            rank_product[0] = product / product.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-300)
            rank_product[1].append(rank_product[0])
        branch_input = (pre.unsqueeze(-1) * x.float()).sum(dim=-2).to(x.dtype)
        update = branch(norm(branch_input))
        return (hres.transpose(-1, -2) @ x.float()
                + post.unsqueeze(-1) * update.float().unsqueeze(-2)).to(x.dtype)


#block
class Block(nn.Module):
    def __init__(self, dim, architecture, index):
        super().__init__()
        self.index = index
        self.norm1 = RMSnorm(dim)
        self.norm2 = RMSnorm(dim)
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.attn_connection = HyperConnection(dim, architecture, 2 * index)
        self.mlp_connection = HyperConnection(dim, architecture, 2 * index + 1)

    def forward(self, x, rank_product=None):
        x = self.attn_connection(x, self.attn, self.norm1, rank_product)
        x = self.mlp_connection(x, self.mlp, self.norm2, rank_product)
        return x

#gpt
class GPT(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers, architecture):
        super().__init__()
        self.architecture = architecture
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, architecture, i) for i in range(num_layers)])
        self.norm1 = RMSnorm(model_dim)
        self.norm2 = RMSnorm(model_dim)
        self.proj = Linear(model_dim, vocab_size)


    def forward(self, inputs, targets=None, rank_product=None):
        x = self.norm1(self.embed(inputs))
        x = x.unsqueeze(-2).expand(-1, -1, 4, -1)
        for block in self.blocks:
            if rank_product is not None:
                x = block.forward(x, rank_product)  # Bypass compiled calls for eager diagnostics.
            elif self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        if rank_product is not None:
            return rank_product[0]
        # Match DepthBench's HC sum / mHC mean readout; final RMSNorm follows.
        x = x.sum(dim=-2) if self.architecture == "hc" else x.mean(dim=-2)
        return self.compute_loss(x, targets)

    def compute_loss(self, x, targets):
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
parser = argparse.ArgumentParser()
parser.add_argument("--configuration", choices=CONFIGURATIONS)
parser.add_argument("--output-dir")
parser.add_argument("--events-file")
parser.add_argument("--plot-rank", action="store_true")
parser.add_argument("--suite-dir")
args = parser.parse_args()
if args.plot_rank:
    if not args.suite_dir:
        parser.error("--plot-rank needs --suite-dir")
    plot_rank(args.suite_dir)
    sys.exit(0)
if not all((args.configuration, args.output_dir, args.events_file)):
    parser.error("training needs --configuration, --output-dir, and --events-file")
code = Path(__file__).read_text()
config = configuration(args.configuration)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(hours=2))
dist.barrier()
#the code can run equivalently with 1, 2, 4, or 8 GPUs
assert 8 % dist.get_world_size() == 0

#logging setup
if dist.get_rank() == 0:
    logfile = output_dir / "train.log"

def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)


def emit_metrics(step, payload):
    if dist.get_rank() == 0:
        with open(args.events_file, "a") as stream:
            stream.write(json.dumps({"train/step": step,
                                     "train/tokens": step * config["batch_tokens"], **payload}) + "\n")
        print("EXPERIMENT_EVENT_READY", flush=True)

#log this exact source file and the environment used for the run
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

train_steps = config["train_steps"]
batch_size = config["batch_tokens"]
mbs = config["microbatch_sequences"]
val_tokens = config["val_tokens"]
rank_step_index = {round(i * train_steps / 20): i for i in range(21)}

val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens,
                                                         seq_len=config["seq_len"]))
model = GPT(vocab_size=config["vocab_size"], num_layers=config["num_layers"], model_dim=config["model_dim"],
            architecture=config["architecture"]).cuda()
# Avoid one enormous backward graph containing 20 Sinkhorn iterations per sublayer.
# Equivalent blocks can reuse compiled code; diagnostics call block.forward directly.
if config["architecture"] == "mhc":
    for block in model.blocks:
        block.compile(dynamic=False)
    model.compute_loss = torch.compile(model.compute_loss, dynamic=False)
    training_model = model
else:
    training_model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
print0(f"num_params:{num_params}", console=True)
routing_params = [p for module in model.modules() if isinstance(module, HyperConnection)
                  for p in module.parameters()]
routing_ids = {id(p) for p in routing_params}
config.update(parameter_count=num_params, world_size=dist.get_world_size())



#initialize model params 
for name, p in model.named_parameters():
    if id(p) in routing_ids:
        continue  # connector-specific identity / GAP-8 initialization must survive
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
        w.mul_(1.25 / 2)
    elif name.endswith(".mlp.proj.weight"):
        w.mul_(3.0 / 2)
    elif name.endswith(".mlp.fc.weight"):
        w.mul_(1.5)


#3 param groups for adamW
#each dict is a param group
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=config["embedding_lr"]),
                    dict(params=[model.proj.weight], lr=config["head_lr"]),
                    dict(params=[p for p in model.parameters() if p.ndim < 2 and id(p) not in routing_ids], lr=config["scalar_lr"])],
                    betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)

optimizer2 = MuonH([p for p in model.blocks.parameters() if p.ndim == 2 and id(p) not in routing_ids], lr=config["backbone_lr"])
# Zero-initialized routing projections cannot use MuonH's norm-preserving update.
# They would be frozen forever at norm zero. All routing uses AdamW instead.
routing_dynamic = [p for module in model.modules() if isinstance(module, HyperConnection)
                   for name, p in module.named_parameters() if name.startswith("dynamic")]
dynamic_ids = {id(p) for p in routing_dynamic}
optimizer3 = AdamW([
    dict(params=routing_dynamic, weight_decay=config["routing_dynamic_weight_decay"]),
    dict(params=[p for p in routing_params if id(p) not in dynamic_ids], weight_decay=0.0),
], lr=config["routing_lr"], betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizers = [optimizer1, optimizer2, optimizer3]
assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for grp in opt.param_groups:
        grp["initial_lr"] = grp["lr"]

for group in optimizer1.param_groups + optimizer3.param_groups:
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

train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size,
                                          seq_len=config["seq_len"])
#make every rank start with rank 0's parameters
for p in model.parameters():
    dist.broadcast(p.detach(), 0)

if dist.get_rank() == 0:
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def fixed_rank_batch():
    shard = Path("data/fineweb10B/fineweb_val_000000.bin")
    with shard.open("rb") as stream:
        header = np.frombuffer(stream.read(1024), dtype=np.int32)
        if header[0] != 20240520 or header[1] != 1 or header[2] < 2049:
            raise ValueError("Invalid fixed validation shard")
        token_bytes = stream.read(2049 * 2)
    if len(token_bytes) != 2049 * 2:
        raise ValueError("Fixed validation batch is incomplete")
    tokens = np.frombuffer(token_bytes, dtype=np.uint16)
    inputs = torch.from_numpy(tokens[:-1].astype(np.int32).reshape(2, 1024)).to(device)
    return inputs, hashlib.sha256(token_bytes).hexdigest()


if dist.get_rank() == 0:
    rank_inputs, rank_batch_hash = fixed_rank_batch()
    rank_values = np.lib.format.open_memmap(
        output_dir / "rank_per_token.npy", mode="w+", dtype=np.float32,
        shape=(len(rank_step_index), 2 * config["num_layers"], 2, 1024))


def save_rank_snapshot(step):
    """Compose actual routing per token, then average rank at every residual depth."""
    model.eval()
    product = [None, []]
    with torch.no_grad():
        model(rank_inputs, rank_product=product)
        matrices = torch.stack(product[1])  # [2L, batch, time, 4, 4]
        singular = torch.linalg.svdvals(matrices)
        probabilities = singular / singular.sum(dim=-1, keepdim=True).clamp_min(1e-300)
        ranks = (-(probabilities * probabilities.clamp_min(1e-300).log()).sum(dim=-1)).exp()
    means = ranks.mean(dim=(1, 2)).cpu().tolist()
    rank_values[rank_step_index[step]] = ranks.cpu().numpy().astype(np.float32)
    rank_values.flush()
    snapshot = dict(step=step, tokens_seen=step * batch_size,
                    batch_sha256=rank_batch_hash, batch_shape=[2, 1024],
                    ranks_by_depth=means)
    with (output_dir / "rank_progression.jsonl").open("a") as stream:
        stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
    print0(f"step:{step}/{train_steps} final-depth effective rank:{means[-1]:.4f}", console=True)
    model.train()


def stop_nonfinite(stage, step, inputs, targets, bad_gradients=()):
    # Written only on failure, while weights are still available for diagnosis.
    # These files are for replaying the failing batch, not optimizer resumption.
    rank = dist.get_rank()
    evidence = dict(stage=stage, step=step, config=config,
                    bad_gradients=list(bad_gradients),
                    bad_parameters=nonfinite_names(list(model.named_parameters())),
                    inputs=inputs.detach().cpu(), targets=targets.detach().cpu(),
                    torch_version=torch.__version__,
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    if rank == 0:
        evidence["model"] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        evidence["bad_gradient_values"] = {
            name: p.grad.detach().cpu() for name, p in model.named_parameters()
            if name in bad_gradients}
    path = output_dir / f"failure_rank{rank}.pt"
    torch.save(evidence, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)
    dist.barrier()  # Finish every rank's evidence before torchrun shuts workers down.
    raise FloatingPointError(
        f"Nonfinite {stage} at step {step}; bad gradients: {list(bad_gradients)}; "
        f"bad parameters: {evidence['bad_parameters']}; evidence saved to {path}")


best_val_loss = float("inf")
dist.barrier()

for step in range(train_steps + 1):
    #validation
    if step == train_steps or step % config["val_interval"] == 0:
        dist.barrier()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += training_model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        if not torch.isfinite(val_loss):
            stop_nonfinite("validation loss", step, val_inputs, val_targets)
        best_val_loss = min(best_val_loss, val_loss.item())
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f}", console=True)
        emit_metrics(step, {"validation/loss": val_loss.item(), "validation/best_loss": best_val_loss})
        model.train()
        dist.barrier()

    if step in rank_step_index:
        dist.barrier()
        if dist.get_rank() == 0:
            save_rank_snapshot(step)
        dist.barrier()

    #the extra loop iteration is only for final validation
    if step == train_steps:
        break

    #training
    if step == 0:
        print0("Starting first training step (includes forward/backward and optimizer compilation).",
               console=True)
    inputs, targets = next(train_loader)
    #gradient accumulation in microbatches
    assert len(inputs) % mbs == 0
    num_mbs = len(inputs) // mbs
    train_loss = 0
    for i in range(num_mbs):
        #run fwd and bwd in microbatches
        loss = training_model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        train_loss += loss.detach()
        loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        #sum up the gradients for every param across the gpu ranks
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    train_loss /= batch_size
    if not torch.isfinite(train_loss):
        stop_nonfinite("training loss", step + 1, inputs, targets)
    bad_gradients = nonfinite_names([(name, p.grad) for name, p in model.named_parameters()])
    if bad_gradients:
        stop_nonfinite("training gradients", step + 1, inputs, targets, bad_gradients)
    #set optimization hyperparams and take step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.5f} tokens:{(step+1)*batch_size}",
           console=True)
    emit_metrics(step + 1, {"training/loss": train_loss.item()})


dist.destroy_process_group()
