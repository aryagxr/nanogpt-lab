import os
import re
import subprocess

import modal


app = modal.App("nanogpt-ablations")

data_volume = modal.Volume.from_name("fineweb-data", create_if_missing=True)
logs_volume = modal.Volume.from_name("nanogpt-logs", create_if_missing=True)

data_dir = "/root/data/fineweb10B"
logs_dir = "/root/logs"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "huggingface_hub", "wandb")
    .add_local_file("train.py", "/root/train.py")
)


@app.function(
    image=image,
    volumes={data_dir: data_volume},
    timeout=7200,
)
def download_data(num_shards: int = 18):
    from huggingface_hub import hf_hub_download

    os.makedirs(data_dir, exist_ok=True)
    filenames = ["fineweb_val_000000.bin"] + [
        f"fineweb_train_{i:06d}.bin" for i in range(1, num_shards + 1)
    ]
    for filename in filenames:
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir,
        )
    data_volume.commit()


def run_training(gpus: int, project: str, name: str):
    import wandb

    run = wandb.init(
        project=project,
        name=name or None,
        config={
            "gpus": gpus,
            "train_steps": 3325,
            "tokens_per_step": 524288,
            "total_train_tokens": 3325 * 524288,
        },
    )
    run.define_metric("train/tokens")
    run.define_metric("training/*", step_metric="train/tokens")
    run.define_metric("validation/*", step_metric="train/tokens")
    run.define_metric("learning_rate/*", step_metric="train/tokens")
    run.define_metric("optimization/*", step_metric="train/tokens")
    run.define_metric("model/*", step_metric="train/tokens")
    run.define_metric("system/*", step_metric="train/tokens")
    training_line = re.compile(
        r"step:(\d+)/(\d+) train_loss:([0-9.eE+-]+) tokens:(\d+) "
        r"step_time:([0-9.eE+-]+)s grad_norm:([0-9.eE+-]+) "
        r"param_norm:([0-9.eE+-]+) peak_gpu_memory_bytes:(\d+) "
        r"lr_embed:([0-9.eE+-]+) lr_head:([0-9.eE+-]+) "
        r"lr_scalar:([0-9.eE+-]+) lr_muonh:([0-9.eE+-]+)"
    )
    validation_line = re.compile(
        r"step:(\d+)/(\d+) val_loss:([0-9.]+) best_val_loss:([0-9.]+)"
    )
    parameter_line = re.compile(r"num_params:(\d+)")

    try:
        process = subprocess.Popen(
            [
                "torchrun",
                "--standalone",
                f"--nproc-per-node={gpus}",
                "/root/train.py",
            ],
            cwd="/root",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            parameter_match = parameter_line.search(line)
            if parameter_match:
                parameter_count = int(parameter_match.group(1))
                run.config.update({"parameter_count": parameter_count})
                run.summary["model/parameter_count"] = parameter_count

            training_match = training_line.search(line)
            if training_match:
                run.log(
                    {
                        "train/tokens": int(training_match.group(4)),
                        "training/loss": float(training_match.group(3)),
                        "system/step_time_seconds": float(training_match.group(5)),
                        "optimization/gradient_norm": float(training_match.group(6)),
                        "model/parameter_norm": float(training_match.group(7)),
                        "system/peak_gpu_memory_gb": int(training_match.group(8)) / 1e9,
                        "learning_rate/embed": float(training_match.group(9)),
                        "learning_rate/head": float(training_match.group(10)),
                        "learning_rate/scalar": float(training_match.group(11)),
                        "learning_rate/muonh": float(training_match.group(12)),
                    }
                )
            validation_match = validation_line.search(line)
            if validation_match:
                step = int(validation_match.group(1))
                run.log(
                    {
                        "train/tokens": step * 524288,
                        "validation/loss": float(validation_match.group(3)),
                        "validation/best_loss": float(validation_match.group(4)),
                    }
                )
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, process.args)
    finally:
        run.finish()
        logs_volume.commit()


train_options = {
    "image": image,
    "volumes": {data_dir: data_volume, logs_dir: logs_volume},
    "secrets": [modal.Secret.from_name("wandb-secret")],
    "timeout": 86400,
}


@app.function(gpu="H100:8", **train_options)
def train_8(project: str, name: str):
    run_training(8, project, name)


@app.function(gpu="H100:4", **train_options)
def train_4(project: str, name: str):
    run_training(4, project, name)


@app.local_entrypoint()
def main(gpus: int = 8, project: str = "nanogpt-ablations", name: str = ""):
    if gpus == 8:
        train_8.remote(project, name)
    elif gpus == 4:
        train_4.remote(project, name)
    else:
        raise ValueError("--gpus must be 4 or 8")
