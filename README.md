# nanoGPT lab
Minimal & hackable implementation of nanogpt. This is a toy codebase to run model architecture experiments.
The goal is to lower val loss. 

## Run on Modal

```bash
source .venv/bin/activate
modal run modal_train.py::download_data  # first run only
modal run modal_train.py --gpus <4,8> --name <name>
```

Use `--gpus 8` to run on eight H100s.

## Records

| # | Val loss | Experiment | Wall clock time | Date | Script |
|---:|---:|---|---:|---|---|
| 1 | 3.28023 | MuonH baseline | 18m 31.7s | 2026-09-10 | [train.py](train.py) |

## Failed experiments

| # | Val loss | Experiment | Wall clock time | Date | Script |
|---:|---:|---|---:|---|---|
