# Trained weights (checkpoints)

The app shows **reported numbers** from `results/` without any weights. To **attack or evaluate** a model live, its `best.pt` must be on disk. Weights are not in git (about 43 MB each for ResNet-18).

## 1. Check what you have

```bash
python -m attack2defense doctor
```

The **Start here** page in the app shows the same check under "Setup status". The output lists every run and whether it has weights and results, then prints the exact paths for any missing reported weights:

```
Missing reported weights (copy each best.pt here):
  checkpoints/resnet18_clean/best.pt
  checkpoints/resnet18_fgsm/best.pt
  checkpoints/resnet18_pgd/best.pt
```

## 2. Download from Google Drive

The Colab notebook saves to the Drive folder `Attack2Defense_results`, with one subfolder per run:

```
Attack2Defense_results/
  resnet18_clean/best.pt
  resnet18_fgsm/best.pt
  resnet18_pgd/best.pt
```

Download the whole folder (Drive gives you a `.zip`) or the three subfolders.

## 3. Install

```bash
# from a zip
python scripts/install_checkpoints.py ~/Downloads/Attack2Defense_results.zip
# or from an unzipped folder
python scripts/install_checkpoints.py ~/Downloads/Attack2Defense_results
```

The script:

- finds every `best.pt` (ignores `last.pt`) and works out the run from its folder name, or from a file name such as `resnet18_pgd_best.pt`;
- loads each file once to confirm it is a real training checkpoint;
- copies it to `checkpoints/<run>/best.pt`;
- adds `config.json` from `results/` when the download did not include one;
- skips runs that are already installed unless you pass `--force`.

Placing files by hand also works. The folder name must match the run name in `results/` so that weights and results are merged into one model card.

## 4. Confirm

Run `python -m attack2defense doctor` again, or refresh the app. The API rescans every 10 seconds. The models appear in the Attack lab and Evaluation pickers with a "Live" badge.

## Where the API looks

| Setting (`A2D_` env var) | Default | Purpose |
| --- | --- | --- |
| `A2D_CHECKPOINT_ROOTS` | `["checkpoints", "outputs/cpu_preview"]` | Folders scanned for `best.pt` |
| `A2D_RESULTS_ROOTS` | `["results"]` | Folders scanned for `config.json` and `test_eval/summary.csv` |
| `A2D_RUNS_DIR` | `outputs/api_runs` | Per-user runs trained from the app |

Runs in other `outputs/` folders (for example `outputs/smoke`, a 3-batch test at chance accuracy) are not scanned and never appear in the app. They can be deleted safely.
