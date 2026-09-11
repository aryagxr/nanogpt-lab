"""Baseline from modded NanoGPT track 3"""


import os
import sys
import time
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import tomllib

from nanogpt.model import GPT
from nanogpt.optim import build_optimizers
from nanogpt.schedule import build_schedule

from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/baseline.toml")
args = parser.parse_args()
config_text = Path(args.config).read_text()
config = tomllib.loads(config_text)
training = config["training"]

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
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{uuid.uuid4()}.txt"
    print(logfile)

def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

#log this exact source file and the environment used for the run
print0(config_text)
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}", console=True)
print0("="*100)

train_steps = training["steps"]
batch_size = training["batch_tokens"]
mbs = training["microbatch_size"]
seq_len = training["sequence_length"]
val_tokens = training["val_tokens"]
assert train_steps > 0 and training["val_every"] > 0
assert batch_size % (dist.get_world_size() * mbs * seq_len) == 0
assert val_tokens % (dist.get_world_size() * mbs * seq_len) == 0
if "seed" in training:
    torch.manual_seed(training["seed"])

val_inputs, val_targets = next(distributed_data_generator(config["data"]["val"], val_tokens, seq_len))
model = GPT(**config["model"], attention=config["attention"], position=config["position"],
            norm=config["norm"], residual=config["residual"]).cuda()
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


optimizers = build_optimizers(model, config["optimizer"])
lr_schedule = build_schedule(optimizers, train_steps, config["schedule"])


#train loop

train_loader = distributed_data_generator(config["data"]["train"], batch_size, seq_len)
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
    if step == train_steps or step % training["val_every"] == 0:
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
    lr_schedule.step(step)
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
           + "".join(f" lr_{group['name']}:{group['lr']:.6e}"
                     for opt in optimizers for group in opt.param_groups)
           + f" train_time:{approx_training_time:.3f}s", console=True, log=False)


dist.destroy_process_group()
