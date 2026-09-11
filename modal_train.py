import json
import os
import time
import tomllib
import re
import subprocess
from datetime import datetime
from pathlib import Path

import modal


app = modal.App("nanogpt-lab")

data_volume = modal.Volume.from_name("fineweb-data", create_if_missing=True)
logs_volume = modal.Volume.from_name("nanogpt-logs", create_if_missing=True)

data_dir = "/root/data/fineweb10B"
logs_dir = "/root/logs"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "huggingface_hub", "wandb")
    .add_local_file("train.py", "/root/train.py")
    .add_local_dir("nanogpt", "/root/nanogpt", ignore=["__pycache__"])
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


def run_training(gpus: int, project: str, name: str, track: str, config_text: str, revision: dict):
    import wandb

    config = tomllib.loads(config_text)
    batch_tokens = config["training"]["batch_tokens"]
    run = wandb.init(
        project=project,
        name=name or None,
        config={**config, **revision, "gpus": gpus},
    )
    experiment = re.sub(r"[^a-z0-9]+", "_", (name or run.id).lower()).strip("_") or run.id
    record_dir = Path(logs_dir) / track / f"{datetime.now():%Y%m%d}_{experiment}_{run.id}"
    record_dir.mkdir(parents=True, exist_ok=False)
    config_path = record_dir / "config.toml"
    config_path.write_text(config_text)
    metrics = {**revision, "gpus": gpus, "wandb_url": run.url, "status": "running"}
    started = time.perf_counter()

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
        r"param_norm:([0-9.eE+-]+) peak_gpu_memory_bytes:(\d+)"
    )
    validation_line = re.compile(
        r"step:(\d+)/(\d+) val_loss:([0-9.]+) best_val_loss:([0-9.]+)"
    )
    parameter_line = re.compile(r"num_params:(\d+)")

    output = (record_dir / "output.log").open("w", buffering=1)
    try:
        process = subprocess.Popen(
            [
                "torchrun",
                "--standalone",
                f"--nproc-per-node={gpus}",
                "/root/train.py",
                "--config", str(config_path),
            ],
            cwd="/root",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "LOG_DIR": f"/tmp/nanogpt-logs/{run.id}"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
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
                        **{f"learning_rate/{key}": float(value)
                           for key, value in re.findall(r"lr_(\w+):([0-9.eE+-]+)", line)},
                    }
                )
            validation_match = validation_line.search(line)
            if validation_match:
                step = int(validation_match.group(1))
                run.log(
                    {
                        "train/tokens": step * batch_tokens,
                        "validation/loss": float(validation_match.group(3)),
                        "validation/best_loss": float(validation_match.group(4)),
                    }
                )
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, process.args)
        metrics["status"] = "completed"
    finally:
        output.close()
        metrics.update(dict(run.summary))
        metrics["wall_clock_seconds"] = time.perf_counter() - started
        if metrics["status"] != "completed":
            metrics["status"] = "failed"
        (record_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        logs_volume.commit()
        run.finish()
        print(f"Saved run: {record_dir.relative_to(logs_dir)}")



train_options = {
    "image": image,
    "volumes": {data_dir: data_volume, logs_dir: logs_volume},
    "secrets": [modal.Secret.from_name("wandb-secret")],
    "timeout": 86400,
}


@app.function(gpu="H100:8", **train_options)
def train_8(project: str, name: str, track: str, config_text: str, revision: dict):
    run_training(8, project, name, track, config_text, revision)


@app.function(gpu="H100:4", **train_options)
def train_4(project: str, name: str, track: str, config_text: str, revision: dict):
    run_training(4, project, name, track, config_text, revision)


@app.local_entrypoint()
def main(gpus: int = 8, project: str = "nanogpt-lab", name: str = "", track: str = "dense",
         config: str = "configs/baseline.toml"):
    config_text = Path(config).read_text()
    tomllib.loads(config_text)
    revision = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }
    if revision["git_dirty"]:
        print("Uncommitted changes: the recorded commit alone will not reproduce this run.")
    if track not in {"dense", "sparse"}:
        raise ValueError("--track must be dense or sparse")
    if gpus == 8:
        train_8.remote(project, name, track, config_text, revision)
    elif gpus == 4:
        train_4.remote(project, name, track, config_text, revision)
    else:
        raise ValueError("--gpus must be 4 or 8")
