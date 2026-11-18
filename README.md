# Reservoir future-mask forecasting

Git deployment update (2026-11-17): clone this repository into the project's
`code` directory and transfer the sibling `dataset` only once. Start with
`README_3090.md`, `bootstrap_3090.ps1` and `EXPERIMENT_PROTOCOL.md`.
Default: one seed (42); six training lakes with temporal validation and two
locked unseen test lakes. Existing archive deployments are not overwritten.

For the three-arm forecast-head study, use
[`FORECAST_SEARCH.md`](FORECAST_SEARCH.md) and `configs/method_forecast_v4.json`.
This retrains the reference and two residual variants, all at SDF weight 0.1,
with support-aware Z-MAE checkpoint selection instead of IoU. Test lakes stay
locked; old profiles/checkpoint-selection defaults are unchanged.

For the fixed-SDF-weight architecture search, use
[`ARCHITECTURE_SEARCH.md`](ARCHITECTURE_SEARCH.md) and
`configs/method_arch_v3.json`: 13 one-seed development runs, including five
new architecture/control variants. This is not a SDF-weight sweep.

## Input and directory contract

The project has two main directories; only `code` is a Git checkout:

```text
<project>/code/                  This repository
<project>/code/bang_quy_doi/     Eight F-Z CSVs, units and provenance, tracked
<project>/code/runs/             Local checkpoints/results, ignored by Git
<project>/dataset/images/        214 preprocessed water-enhanced RGB PNG images
<project>/dataset/masks/         214 matching binary masks
```

The filename contract is `<reservoir>__<YYYY-MM-DD>.png` in both dataset
directories. Images keep their native dimensions; each reservoir has one
fixed projected 10 m grid. Native pixels represent 100 m2, NOT model pixels.
`splits.csv` and `data_checksums.csv` resolve asset paths relative to this code
directory. Images and masks stay outside Git. Water=255, background=0.
CSV bytes are intentionally not newline-normalized by Git: manifest SHA256
must remain identical after checkout on another machine.

Preprocessing is deliberately outside this review export. Training consumes
the already processed THREE RGB channels in `dataset/images`, normalized by
255. It does not require separate B03/B08/B11, SCL, footprints or six-channel
arrays at runtime. The recovered preprocessing and original baseline code are
preserved separately; they have not been deleted or replaced by this export.

## Implemented model

```text
Three past RGB images + actual acquisition dates
  -> shared multiscale CNN encoder for each image
  -> convolutional spatial CfC recurrent state, using actual time gaps
  -> future-date query (horizon and seasonal sin/cos; no target image)
  -> temporal conditioning of last-observation spatial skip features
  -> shared spatial decoder
       -> future mask logits                [primary output]
       -> future normalized SDF prediction  [auxiliary training head]
  -> native-grid mask -> physical area F -> fixed F-Z interpolation -> Z
```

`CfCMaskForecaster` in `model.py` is the class actually used by `train.py`
and `predict.py`. The recurrent state is a learned spatial feature map, not
an SDF. There is no normal-velocity decoder or level-set PDE. SDF predictions
are not used to generate masks at inference. The standalone single-frame
`UNetWithSDFAux` class is retained as a reference, not the forecasting model.

The cell is a convolutional adaptation of the default gated CfC update:

```text
q = backbone(concat(observation, h))
gate = sigmoid(time_a(q) * (delta_days / 30) + time_b(q))
h_new = (1 - gate) * tanh(branch1(q)) + gate * tanh(branch2(q))
```

Dense transformations are replaced by spatial convolutions. This is an
explicit adaptation, not an unchanged official `ncps` module or an exact ODE
solver. It is not claimed to outperform GRU, ODE or other temporal baselines.
Reference: [Hasani et al., Closed-form continuous-time neural networks](https://doi.org/10.1038/s42256-022-00556-7);
gate reference: [official ncps CfC cell](https://github.com/mlech26l/ncps/blob/master/ncps/torch/cfc_cell.py).

## Loss and supervision

```text
L = L_BCE(future_mask) + L_Dice(future_mask) + lambda_sdf * L_SDF
lambda_sdf = 0.1 by default
```

SDF is derived solely from the target mask. It is negative inside water and
positive outside. Distances are computed BEFORE resizing, clipped at 5% of
the native image diagonal, then normalized to [-1, 1]. SDF supervision uses
boundary-weighted Smooth L1. Padding and a narrow strip at water-touched crop
edges are excluded from this auxiliary loss. Mask loss excludes padding.
Use `--lambda-sdf 0` for the no-SDF-loss ablation; compare the same data,
seeds, training budget and evaluation protocol. No F-Z loss or gauge loss is
used in this implementation.

Only past images are loaded by the dataset. The future MASK supplies the
training label; the future RGB image is never an input. At inference no
target image or target mask is required. Calendar features are known at the
query date and are not future measurements.

## Splits and forecasting protocol

All six development reservoirs participate in training. Within EACH lake,
the last ceil(20% of acquisition dates) is reserved for validation. The cutoff
is selected before filtering target gaps. It is a per-lake time split, not
one global calendar cutoff. El Vado and Riano remain entirely held out.

| Split | Reservoirs | Images | All windows | Windows with target gap <=90 days |
| --- | --- | ---: | ---: | ---: |
| Train | An Khe, Ka Nak, Barrios de Luna, Lake Tschida, Lovewell, Thirlmere | 131 | 113 | 79 |
| Validation | Later dates of the SAME six lakes | 36 | 36 | 30 |
| Final test | El Vado, Riano | 47 | 41 | 34 |

`prepare_temporal_split.py` rebuilds the manifest deterministically.
`temporal_split.json` records every cutoff and retained-window count.
Validation targets can use three earlier RGB observations across the train/val
boundary; later validation forecasts can use earlier observed validation RGB.
No context masks are model inputs, and validation labels never train weights.
This is rolling-origin evaluation, not forecasting the entire validation block
from one frozen origin. Training targets precede validation targets within
each lake. Train and validation may share past RGB observations, which is
intentional; they never share target labels.

Each window uses the previous three available dates and predicts the NEXT
observed date of the same reservoir. There is no monthly image interpolation.
Held-out reservoirs provide past RGB context at inference, but their masks
are not used for learning. Evaluation is rolling-origin next-observation
forecasting, not a free-running multi-step simulation without new images.

Available forecast horizons vary substantially; see `dataset_audit.json`.
These are not all monthly forecasts. `--max-horizon-days 90` excludes longer target gaps
consistently in all splits; choose any cap BEFORE comparing models, report
retained sample counts, and never label all remaining gaps as 30 days.
The flag caps the target gap, not the earlier context gaps. Each run exports
its exact windows and horizons as CSV for audit. No future forcing is used.

## Run

Use Python 3.10 or newer. Install a CUDA-compatible PyTorch build separately
when using a GPU; the requirements do not select a CUDA wheel for you.
Run these commands from the extracted `code` directory:

```sh
python -m pip install -r requirements.txt
python audit_dataset.py
python -m unittest discover -p "test_*.py" -v
python train.py --smoke-test --device cpu --image-size 64 --base-channels 4 --batch-size 1 --output-dir runs/smoke
```

Train the default forecasting model (this is the real experiment, not smoke):

```sh
python train.py --device cuda --epochs 80 --batch-size 2 --image-size 512 --base-channels 16 --context-length 3 --max-horizon-days 90 --lambda-sdf 0.1 --seed 42 --output-dir runs/cfc_sdf_l01
```

The staged runner and command above cap target gaps at 90 days. Direct
`train.py` without `--max-horizon-days` retains all windows, so keep the flag
for comparable runs.
Reduce batch size or image size if GPU memory is insufficient; keep input
size divisible by 16. Image size changes require a new, separate run.
The current method check uses one seed. Multiple seeds are needed to estimate
optimization variability; smoke does not establish
accuracy, superiority, cross-reservoir generalization or publishability.

Resume using the SAME output directory and the same model/data/loss settings:

```sh
python train.py --device cuda --epochs 80 --max-horizon-days 90 --seed 42 --output-dir runs/cfc_sdf_l01 --resume runs/cfc_sdf_l01/last.pt
```

Training writes `best.pt`, `last.pt` (optimizer, scaler and RNG states),
`history.csv`, `loss_curves.png`, `best_val_metrics.json`, `summary.json`,
`train_windows.csv`, `val_windows.csv` and `config.json`. Best checkpoint and
early stopping use validation macro-reservoir IoU. The curve plot contains
total train/validation losses, weighted SDF loss and validation IoU.
Test is opt-in: append `--evaluate-test` to the final pre-specified run only
after model selection is complete. Do not tune settings on final-test scores.

Standalone forecast example, using three An Khe images from the bundle:

```sh
python predict.py --checkpoint runs/cfc_sdf_l01/best.pt --context example_context.json --target-date 2020-03-18 --conversion-table bang_quy_doi/ankhe_fz.csv --pixel-area-m2 100 --output-dir runs/ankhe_forecast
```

Context JSON is a chronologically ordered list of `image_path` and `date`.
Its paths are relative to that JSON file. `predict.py` restores output masks
to native resolution before counting area. It writes a binary mask,
probability PNG, auxiliary normalized SDF and `prediction.json` with F and Z.
Only load checkpoints from sources you trust.

A bounded integration check exercises one tiny training window, one validation
window, checkpoint/resume to epoch two, loss plotting and past-only inference:

```sh
python integration_check.py --output-dir runs/integration_check_fresh
```

This is a software check only, not a complete training experiment. Use a new
output directory for every integration check. Smoke checkpoints are marked
`smoke_only` and are not a trained scientific baseline.

## Physical interpretation and limitations

Letterboxing is inverted before native-grid mask area is measured. Never
multiply the count on a 512x512 tensor directly by 100 m2. F-Z curves are fixed
postprocessing, not learned weights or model inputs. Areas outside support
return `Z_m: null`, never silently clamp or extrapolate. Seven of 29 Thirlmere
masks exceed its empirical F-Z support; image/mask forecasting remains usable,
but those samples cannot provide an in-support F-Z level score.

Reported Z errors compare F-Z-derived predicted-mask and target-mask levels,
NOT an independent gauge validation. Z coverage and out-of-support counts must
accompany Z errors because invalid predictions are excluded from Z MAE/RMSE.
Curve evidence varies across reservoirs; see `bang_quy_doi/README.md`.
No new gauge CSV is asserted to be ground truth by this export.
