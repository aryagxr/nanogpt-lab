"""Launch one HC/mHC run at a time on Modal; save losses and rank progression."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal


CONFIGURATIONS = tuple(f"{arch}-{shape}" for arch in ("hc", "mhc")
                       for shape in ("shallow-wide", "middle", "deep-narrow"))
app = modal.App("nanogpt-lab")
data_volume = modal.Volume.from_name("fineweb-data", create_if_missing=True)
logs_volume = modal.Volume.from_name("nanogpt-logs", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy==1.26.4", "huggingface_hub==0.26.2",
                      "wandb==0.18.7", "matplotlib==3.9.2")
         .add_local_file("train.py", "/root/train.py")
         .add_local_file("modal_train.py", "/root/modal_train.py"))


@app.function(image=image, volumes={"/root/data/fineweb10B": data_volume}, timeout=7200)
def download_data(num_shards: int = 18):
    from huggingface_hub import hf_hub_download
    for filename in ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin"
                                                    for i in range(1, num_shards + 1)]:
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=filename,
                        repo_type="dataset", local_dir="/root/data/fineweb10B")
    data_volume.commit()


def prepare_run_directory(suite, config, restart_failed=False):
    directory = suite / config
    if directory.exists():
        metadata_path = directory / "metrics.json"
        status = json.loads(metadata_path.read_text()).get("status") if metadata_path.exists() else None
        if not restart_failed or status != "failed":
            raise ValueError(f"Run already exists: {directory}. Only failed runs can use --restart-failed.")
        archive = suite / "failed" / f"{config}-{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
        print(f"Preserved failed run at {archive}; restarting from step 0.", flush=True)
    elif restart_failed:
        raise ValueError(f"No failed run exists at {directory}; omit --restart-failed.")
    directory.mkdir(parents=True)
    return directory


def run_training(gpus, config, project, name, restart_failed=False):
    import wandb

    logs_volume.reload()
    data_volume.reload()
    suite = Path("/root/logs/dense/hc-mhc-depth-seed1337")
    directory = prepare_run_directory(suite, config, restart_failed)
    for source in ("train.py", "modal_train.py"):
        shutil.copyfile(Path("/root") / source, directory / source)
    logs_volume.commit()

    sinkhorn_version = "log-softmax-v2" if config.startswith("mhc-") else None
    run = wandb.init(project=project, name=name or f"{config}-seed1337",
                     group="hc-mhc-depth-seed1337", tags=[config.split("-", 1)[0], "hc-mhc-depth"],
                     config={"configuration": config, "seed": 1337, "gpus": gpus,
                             "sinkhorn_implementation": sinkhorn_version},
                     dir=str(directory))
    run.define_metric("train/tokens")
    run.define_metric("training/*", step_metric="train/tokens")
    run.define_metric("validation/*", step_metric="train/tokens")
    run.define_metric("routing/*", step_metric="train/tokens")
    metadata = {"configuration": config, "seed": 1337, "gpus": gpus,
                "wandb_url": run.url, "status": "running"}
    (directory / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    last_commit = started
    event_position = 0
    event_path = directory / "events.jsonl"

    def drain_events(history):
        nonlocal event_position
        if not event_path.exists():
            return
        with event_path.open() as events:
            events.seek(event_position)
            while line := events.readline():
                payload = json.loads(line)
                history.write(line)
                try:
                    run.log(payload)
                except Exception as error:
                    print(f"W&B logging error; data saved locally: {error}", flush=True)
            event_position = events.tell()

    try:
        process = subprocess.Popen(
            ["torchrun", "--standalone", f"--nproc-per-node={gpus}", "/root/train.py",
             "--configuration", config, "--output-dir", str(directory),
             "--events-file", str(event_path)],
            cwd="/root", env={**os.environ, "PYTHONUNBUFFERED": "1",
                              "TORCHINDUCTOR_COMPILE_THREADS": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with (directory / "output.log").open("w", buffering=1) as output, \
             (directory / "history.jsonl").open("w", buffering=1) as history:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                drain_events(history)
                if time.monotonic() - last_commit >= 60:
                    logs_volume.commit()
                    last_commit = time.monotonic()
            if process.wait() != 0:
                raise RuntimeError(f"Training exited with code {process.returncode}")
            drain_events(history)
        progression = directory / "rank_progression.jsonl"
        if not (directory / "rank_per_token.npy").exists() \
                or not progression.exists() or len(progression.read_text().splitlines()) != 21:
            raise RuntimeError("Training finished without complete rank data")
        metadata["status"] = "completed"
        try:
            plot = subprocess.run(["python", "/root/train.py", "--plot-rank",
                                   "--suite-dir", str(suite)], cwd="/root", check=True)
            if plot.returncode == 0 and (suite / "effective_rank_progression.png").exists():
                run.log({"plots/effective_rank_progression": wandb.Image(
                    str(suite / "effective_rank_progression.png"))})
        except Exception as error:
            print(f"Rank plot failed; raw rank data is saved: {error}", flush=True)
    except Exception:
        metadata["status"] = "failed"
        raise
    finally:
        metadata["wall_clock_seconds"] = time.monotonic() - started
        (directory / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logs_volume.commit()
        try:
            run.finish(exit_code=0 if metadata["status"] == "completed" else 1)
        finally:
            logs_volume.commit()


train_options = dict(image=image,
                     volumes={"/root/data/fineweb10B": data_volume, "/root/logs": logs_volume},
                     secrets=[modal.Secret.from_name("wandb-secret")], timeout=86400,
                     max_containers=1, cpu=16, memory=65536)


@app.function(gpu="H100:8", **train_options)
def train_8(config: str, project: str, name: str, restart_failed: bool = False):
    run_training(8, config, project, name, restart_failed)


@app.function(gpu="H100:4", **train_options)
def train_4(config: str, project: str, name: str, restart_failed: bool = False):
    run_training(4, config, project, name, restart_failed)


@app.local_entrypoint()
def main(config: str, gpus: int = 8, project: str = "nanogpt-lab", name: str = "",
         restart_failed: bool = False):
    if config not in CONFIGURATIONS:
        raise ValueError(f"Choose one of {CONFIGURATIONS}")
    if gpus == 8:
        train_8.remote(config, project, name, restart_failed)
    elif gpus == 4:
        train_4.remote(config, project, name, restart_failed)
    else:
        raise ValueError("--gpus must be 4 or 8")
