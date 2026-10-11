# mHC vs HC for depth utilization

Read full blogpost [here](https://aryagxr.com/blogs/hc-mhc-depth/)

On `research/hc-mhc-depth`, `train.py` contains the four-stream HC/mHC experiment. The six configurations use one seed (1337), 20 Sinkhorn iterations for mHC, and three depth/width pairs: 8/768, 18/512, and 32/384. `modal_train.py` launches one run at a time and logs train/validation loss to the `nanogpt-lab` W&B project.

## Results

All runs use seed 1337 and four residual streams; mHC uses 20 Sinkhorn iterations. Effective rank is the token-mean effective rank of the cumulative residual transport across all attention and MLP sublayers, measured on the fixed diagnostic validation batch.

| Configuration | Layers | Width | Heads | Aspect ratio (d/L) | Validation loss | Effective rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HC shallow-wide | 8 | 768 | 12 | 96.0 | 3.3142 | 2.3160 |
| HC middle | 18 | 512 | 8 | 28.4 | 3.3066 | 2.3919 |
| HC deep-narrow | 32 | 384 | 6 | 12.0 | 3.3327 | 2.2627 |
| mHC shallow-wide | 8 | 768 | 12 | 96.0 | 3.3357 | 1.7058 |
| mHC middle | 18 | 512 | 8 | 28.4 | 3.3325 | 1.5665 |
| mHC deep-narrow (partial) | 32 | 384 | 6 | 12.0 | 3.6712 | 2.2780 |

Completed runs report step 3325 (1,743,257,600 training tokens). The deep-narrow mHC run stopped early: its latest validation loss is from step 1375 and its latest rank measurement is from step 1330, so its values are not a matched-budget comparison with the completed runs.

## Run on Modal

Use the existing `.venv`, or create one with `python3.11 -m venv .venv`:

```bash
source .venv/bin/activate
pip install -r requirements.txt
modal setup  # first-time Modal authentication
modal run modal_train.py::download_data  # first run only
modal run --detach modal_train.py --config hc-shallow-wide --gpus 8
modal run --detach modal_train.py --config hc-middle --gpus 8
modal run --detach modal_train.py --config hc-deep-narrow --gpus 8
modal run --detach modal_train.py --config mhc-shallow-wide --gpus 8
modal run --detach modal_train.py --config mhc-middle --gpus 8
modal run --detach modal_train.py --config mhc-deep-narrow --gpus 8
```


```bash
modal volume get nanogpt-logs dense/hc-mhc-depth-seed1337 /tmp/hc-mhc-depth-seed1337
```

The plot is at `/tmp/hc-mhc-depth-seed1337/effective_rank_progression.png` after download. Rebuild it from saved ranks with `python train.py --plot-rank --suite-dir /tmp/hc-mhc-depth-seed1337`.


