# RTX 3090: Git code updates and one-time dataset import

## Method-v2 protocol (new, explicit profile)

See `PAPER_EXPERIMENT_PLAN.md` for configuration, the full proposed experiment
matrix, limitations and which additional paper studies are NOT implemented.
Use `-Config configs/method_v2.json -OutputRoot .\runs_method_v2` for all v2
stages. This opts into 120 epochs, five warmup epochs and cosine LR decay.
Legacy commands below without `-Config` retain their 80-epoch confirmation
budget and constant LR. Never mix old and v2 comparisons in one output root.

Development `confirm` still uses source chronological validation. AFTER method
freeze, new stage `refit` trains on BOTH source splits with NO validation and
NO early stopping, selecting `last.pt`. New stage `refit-test` is separately
locked, requires `-ConfirmFinalTest`, and refuses mismatched plans/incomplete
refits. No target-lake labels enter training. Do not pull while training.

This repository contains code and all eight small F-Z tables, not satellite
images or masks. Put the 214 image/mask pairs in a sibling `dataset` directory.
Original preprocessing stays outside the package. All models receive the same
already preprocessed THREE-channel RGB images. No raw six-band files are needed.

The project directory contains `code` (this Git checkout) and `dataset`
(`images`, `masks`). Local outputs live in `code/runs`, ignored by Git. Do not
copy old ZIP code over the Git checkout or resume old-layout/split checkpoints.
See `CONFIG_REFERENCES.md` for checked paper/config sources and the distinction
between their settings and the settings actually implemented in this package.

## 1. Clone code once on the RTX 3090

If neither `git` nor `winget` exists, run this in PowerShell first. It reuses an
existing Git installation or installs official MinGit without administrator
rights, verifies the pinned release SHA256, and adds Git to your user PATH:

```powershell
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$installer = Join-Path $env:TEMP "reservoir8-install-git.ps1"
Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/toan3107/m-c-n-c/main/install_git_user.ps1" -OutFile $installer
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& $installer
```

MinGit release source: [Git for Windows 2.56.1](https://github.com/git-for-windows/git/releases/tag/v2.56.1.windows.1).
The helper does not touch Python environments or dataset files.

Choose the project directory in remote PowerShell. It may include spaces and
Unicode characters; all scripts resolve paths from their own location:

```powershell
$ProjectRoot = Join-Path $HOME "reservoir_water"
New-Item -ItemType Directory -Path $ProjectRoot -Force | Out-Null
git clone https://github.com/toan3107/m-c-n-c.git (Join-Path $ProjectRoot "code")
Set-Location (Join-Path $ProjectRoot "code")
```

No credentials, keys or passwords are included in this repository. Set
`ProjectRoot` to your preferred folder name before cloning. Do not clone over
an existing non-Git folder or remove an existing dataset to make the command work.

## 2. Import the dataset once and set up Python

The bootstrap reuses an existing verified dataset, or imports just the dataset
from the previous full ZIP. It never extracts old ZIP code over this repository:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap_3090.ps1
```

Auto-discovery checks `$HOME/reservoir8_temporal/dataset`,
`$HOME/reservoir8_method/dataset`, then the previous ZIPs named
`reservoir8_3090_temporal_20261117.zip` or `reservoir8_3090_method_20261116.zip`
in `$HOME`. You can specify the existing source explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap_3090.ps1 -DataZip "$HOME\reservoir8_3090_temporal_20261117.zip"
# Or: -DataSource "C:\path\to\existing\dataset"
```

SHA256 is verified for every image, mask and F-Z CSV. Existing modified files
are never overwritten. If data is missing entirely, transfer the old ZIP once,
or copy `dataset/images` and `dataset/masks` once. No repeated data SCP is needed.
`-SkipEnvironmentSetup` performs only the data import/check.

The setup creates `$HOME\reservoir8-env`, separate from `eclare-env`; it does
not upgrade or delete packages in `eclare-env`. It installs PyTorch 2.9.1 with
the official CUDA 12.8 wheel and pinned scientific dependencies, audits the
dataset, runs unit tests, and probes a real model forward/backward/optimizer
step at 512x512, batch size 2. Python 3.10-3.13 is required for these pins.
If another base interpreter is needed, pass `-BasePython C:\path\python.exe`.

`nvidia-smi` confirms the driver sees the GPU, not that Python has CUDA PyTorch.
The wheel runtime need not equal the CUDA version printed by `nvidia-smi`.
Wheel source: [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/).
GPU and package details, peak allocated VRAM and installed versions are saved
under `runs/preflight`. The RTX 3090 check itself runs on YOUR machine; the
local software checks do not claim to have verified that remote GPU.

## Later code updates

When no training/evaluation process is using this checkout:

```powershell
git status --short
git pull --ff-only
```

Do not pull while training, and do not mix comparisons from different code
commits into one run directory. The runner records source/manifest hashes.
Changing code requires a new output root for new experiments; do not silently
resume an old partial run under changed source. Run setup again only when
dependencies change; dataset does not need to be transferred again.

## 3. Run one stage at a time

### Smoke: does every comparator run on the actual GPU?

```powershell
.\run_3090.ps1 -Stage smoke
```

One train and one validation batch per experiment, not full training and not
meaningful accuracy. Stops on the first failed process; no silent retry or
automatic batch-size change. Resolve errors before continuing.

### Overfit: can the method learn two training windows?

```powershell
.\run_3090.ps1 -Stage overfit
```

Only CfC+SDF, two training windows, no augmentation, up to 120 epochs.
Validation is capped at two windows for this diagnostic only. Inspect
`runs/overfit/cfc_sdf/seed_42/history.csv` and `loss_curves.png`. Training loss
should fall and training IoU should rise. This checks learning implementation,
not unseen-reservoir generalization; do not report its validation as a result.
There is no guaranteed IoU threshold or automatic scientific pass claim.

### Pilot: optional short exploratory comparison

```powershell
.\run_3090.ps1 -Stage pilot
```

Seven experiments, seed 42, maximum 20 epochs each, patience 8, run sequentially
on one GPU. Same 512x512 input, base width 16, batch size 2, context length 3,
AdamW learning rate 0.0003 and weight decay 0.0001. Only validation is scored.
Review the results BEFORE starting confirmation. This pilot is exploratory,
not a final comparison and not grounds to select the best random seed.

### Confirm: the main one-seed method check

```powershell
.\run_3090.ps1 -Stage confirm
```

Seed 42 ONLY; maximum 80 epochs each; patience 15. All seven comparisons,
same settings within the stage. This still does NOT evaluate final-test lakes.
The 20-epoch pilot is optional, not the main training budget. No scheduler or
warmup is implemented: learning rate is constant at 0.0003. Do not describe
this package as using cosine decay. For a larger predeclared budget, use
`-Epochs 200` with a new `-OutputRoot`; patience remains 15.
Decide whether settings need changes based on development data, document the
changes, and use a NEW output root for a new protocol rather than overwriting.

## 4. Main comparisons

| Experiment ID | What is trained/predicted? | SDF weight |
| --- | --- | ---: |
| `rgb_persistence` | Train last-RGB segmenter on its CURRENT mask; at forecast time carry that predicted mask forward unchanged | 0 |
| `last_frame` | Last RGB -> FUTURE mask, without timing or earlier images | 0 |
| `convgru` | Spatial ConvGRU sequence baseline, without actual timestamp/calendar cues | 0 |
| `dt_convgru` | Same spatial framework with elapsed-time channel and calendar cues | 0 |
| `cfc_notime` | CfC architecture with constant time steps and zeroed calendar cues | 0 |
| `cfc_no_sdf` | CfC with actual timing and calendar, mask loss only | 0 |
| `cfc_sdf` | Same CfC with auxiliary SDF loss | 0.1 |

Persistence is OPERATIONAL: it never inputs a manual context mask from a held-out
reservoir. It is not oracle ground-truth-mask persistence. It uses the same
training-window last-context labels, with current-mask rather than future-mask
supervision; all other models use future-mask labels. Its checkpoint is selected
on forecast validation performance, just like the other comparators.

The ConvGRU models use the same CNN, spatial resolution, future-query decoder
and skip-conditioning framework as CfC. The delta-time-aware ConvGRU concatenates
`log1p(delta_days/30)` as a spatial input channel; it is NOT GRU-D. Parameter
counts are recorded: cells differ in size, so this is architecture-controlled,
not an exactly parameter-matched comparison. The single-frame CNNs also have
different parameter counts. Do not claim equal counts.

The no-time comparisons remove BOTH actual gaps and calendar features. They
test total timing information, not the isolated causal effect of delta-time.
A separate delta-only/calendar-only ablation would be needed for that claim.

## 5. Inspect outputs

Each run saves `best.pt`, resumable `last.pt`, `config.json`, actual forecast
window CSVs, console log, loss components, IoU history and loss curves.
The runner also stores the complete stage plan, manifest SHA256 and code hashes.
Changed code/settings cannot silently share the same stage output directory.

Each pilot/confirmation stage produces:

- `comparison.csv`: score across completed seeds; SD is blank for one seed.
- `all_seed_metrics.csv`: individual seed values, parameter counts and Z coverage.
- `per_reservoir_comparison.csv`: separate lake results.
- `paired_seed_differences.csv`: CfC+SDF minus each baseline on matched seeds.
- `comparison.png`: native IoU, Boundary F1 at 20 m, and area MAE.
- `aggregation_status.json`: missing runs; incomplete results are not hidden.
- Per-run `val_evaluation/per_window.csv` and `metrics.json`: native-grid scores
  and 1-30 / 31-60 / 61-90 day horizon breakdowns.

Checkpoint selection uses letterboxed validation macro-reservoir IoU from
training. Post-training comparisons use native-grid metrics. The distinction
is recorded here; these two IoU definitions are not interchangeable.
Fixed maximum target gap is 90 days for the staged experiments. Previous
context gaps can still be longer; no monthly image interpolation is performed.
Changing the horizon cap is a new protocol, not a post-hoc test-set filter.

## 6. Resume and memory problems

Rerunning the SAME completed stage safely skips completed runs. Partial runs
require an explicit resume request:

```powershell
.\run_3090.ps1 -Stage pilot -Resume
```

If 512x512 batch 2 exceeds available VRAM, keep input size and run a new protocol
with batch 1 for ALL comparisons, using a separate output root:

```powershell
.\run_3090.ps1 -Stage smoke -BatchSize 1 -OutputRoot ".\runs_b1"
.\run_3090.ps1 -Stage confirm -BatchSize 1 -OutputRoot ".\runs_b1"
```

Do not compare selected successful batch-2 runs with retried batch-1 runs as
though they used an identical protocol. The runner does not hide OOM retries.
Do not run multiple experiment instances against the same output root.

## 7. Final test stays locked

Development: ALL six lakes, including Thirlmere. The last ceil(20% of dates)
within each lake is validation; all earlier dates are training. Cutoffs are
computed before the horizon cap, without inspecting accuracy or mask values.
`temporal_split.json` records the dates and counts, and
`prepare_temporal_split.py` reproduces the split from the manifest inventory.

There are 131 train images, 36 validation images and 47 test images. With three
context observations and target gaps <=90 days, there are 79 train windows,
30 validation windows and 34 test windows. Training and validation contain all
six development lakes. A validation target uses the previous three available
RGB images, which can lie in the earlier training period or earlier in the
validation period. Context masks are never model inputs. Validation labels
never update weights. This is causal rolling-origin evaluation, not a frozen
forecast of all later dates, and not one global calendar cutoff across lakes.

Final test: El Vado and Riano, never used by pilot/confirmation optimization.
This is rolling-origin RGB-context forecasting on unseen reservoirs, not
forecasting without any observations of a new lake.

Only after settings are fixed and confirmation results reviewed:

```powershell
.\run_3090.ps1 -Stage final-test -ConfirmFinalTest
```

This loads the corresponding confirmation checkpoints and evaluates test;
it does not train. Both the runner and evaluator require explicit unlocking.
Do not rerun method selection based on those test scores.

Z is derived from predicted and labelled mask areas using fixed F-Z curves;
it is NOT an independently gauge-validated water-level error. Report Z support
coverage alongside Z MAE/RMSE; no clamp or extrapolation is used. Thirlmere has
seven masks outside the empirical table's support. Area and mask scores remain
reported even when Z is missing. See `bang_quy_doi/README.md` for evidence status.

The purpose here is to establish whether the method learns and helps compared
with controls. No software smoke test or single-seed pilot proves novelty,
global generalization, statistical significance or suitability for a journal.
