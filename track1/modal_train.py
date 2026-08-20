import os
import modal


app = modal.App("nanogpt-baseline")


DATA_VOL  = modal.Volume.from_name("fineweb-data", create_if_missing=True)
LOGS_VOL  = modal.Volume.from_name("nanogpt-logs", create_if_missing=True)

DATA_DIR  = "/mnt/fineweb" 
LOGS_DIR  = "/mnt/logs"      


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy",
        "wandb",
        "huggingface_hub",
        "tqdm",
    )
    .add_local_file("train.py", "/root/train.py")
)

# data

MAX_SHARDS = 103  # kjj0/fineweb10B-gpt2 has shards 000001–000103

@app.function(
    image=image,
    volumes={DATA_DIR: DATA_VOL},
    cpu=4,
    memory=8192,
    timeout=7200,
)
def download_data(num_shards: int = 9):
    """
    Download pre-tokenised FineWeb shards from HuggingFace (kjj0/fineweb10B-gpt2).
    Each shard is ~100 M tokens. 
    """
    num_shards = min(num_shards, MAX_SHARDS)
    from huggingface_hub import hf_hub_download

    os.makedirs(DATA_DIR, exist_ok=True)

    def fetch(fname):
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest):
            print(f"  already present: {fname}")
            return
        print(f"  downloading: {fname} ...", flush=True)
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        print(f"  done: {fname}", flush=True)

    #val shard
    fetch("fineweb_val_000000.bin")

    #training shards
    for i in range(1, num_shards + 1):
        fetch(f"fineweb_train_{i:06d}.bin")

    DATA_VOL.commit()
    print(f"\nFinished. {num_shards} training shard(s) in {DATA_DIR}.")


#training

@app.function(
    image=image,
    gpu="H100",
    volumes={
        DATA_DIR: DATA_VOL,
        LOGS_DIR: LOGS_VOL,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=86400, #24 hrs
)
def train(
    num_iterations: int = 40000, # ceiling: ~10.3B tokens, safely above what baseline needs
    target_val_loss: float = 3.28, #stop as soon as val loss hits this; set to 0 to disable
    disable_wandb: bool = False,
):
    import subprocess

    cmd = [
        "python", "/root/train.py",
        "--input_bin",       f"{DATA_DIR}/fineweb_train_*.bin",
        "--input_val_bin",   f"{DATA_DIR}/fineweb_val_*.bin",
        "--model",           "d12",
        "--batch_size",      "64",
        "--sequence_length", "1024",
        "--total_batch_size","262144",
        "--grad_accum_steps","4",
        "--num_iterations",  str(num_iterations),
        "--learning_rate",   "1.5e-3",
        "--warmup_iters",    "0",
        "--warmdown_iters",  "1800",
        "--weight_decay",    "0.1",
        "--val_loss_every",  "250",
        "--val_max_steps",   "20",
    ]
    if target_val_loss > 0:
        cmd += ["--target_val_loss", str(target_val_loss)]

    if disable_wandb or not os.environ.get("WANDB_API_KEY"):
        cmd.append("--disable_wandb")

    # Write logs directly into the persistent volume
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, cwd=LOGS_DIR, env=env)

    LOGS_VOL.commit()
    return result.returncode


#local entrypoint

@app.local_entrypoint()
def main(
    num_iterations: int = 40000,
    target_val_loss: float = 3.28,
    disable_wandb: bool = False,
):
    """
    modal run modal_train.py                          # full baseline run, stops at 3.28
    modal run modal_train.py --num-iterations 3814    # 1B token ablation run
    modal run modal_train.py --target-val-loss 0      # disable early stop, run all steps
    """
    rc = train.remote(
        num_iterations=num_iterations,
        target_val_loss=target_val_loss,
        disable_wandb=disable_wandb,
    )
    if rc != 0:
        raise SystemExit(f"Training exited with code {rc}")
