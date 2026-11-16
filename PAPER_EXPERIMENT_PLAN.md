# Reservoir forecasting: method-v2 experiment plan

Status: proposed paper protocol, not a claim of positive results or publication
readiness. One seed (42) remains the default requested for method checking.
The locked test reservoirs are El Vado and Riano. Do not change this choice after
seeing their errors. Older runs are exploratory development evidence and remain
preserved. No raw data, labels, or F-Z curves were changed for this protocol.

## 1. Fixed input/output contract and configuration

Three past enhanced RGB acquisitions plus actual gaps and calendar query time
predict ONE future water mask. SDF is an auxiliary head/loss, never the dynamical
state or a replacement output. Mask area is computed on the native projected
grid, then transformed with the reservoir-specific fixed F-Z curve. Neither
future imagery, future gauge data nor manual context masks enter the model.
Preprocessing must be available at forecast origin; any ROI or enhancement
parameters derived using future water labels must be audited before a paper.

`configs/method_v2.json` is an explicit starting profile, NOT an optimum inferred
from test results: 512 input, base 16, GroupNorm, microbatch 2, context 3,
AdamW 3e-4, weight decay 1e-4, gradient clip 5, CUDA AMP, five warmup epochs,
cosine decay to 1% of base LR, 120 epoch budget, seed 42. SDF lambda stays 0.1.
It deliberately does not copy Colubrid-Net's batch 32 or scalar MSE loss.
Its base-64 backbone and hydrological inputs describe a different experiment.
The pasted paper does not specify epochs, L2 coefficient, scheduler, or
checkpoint selection. Its stated 70/20/10 split disagrees with its date ranges.

Warmup/cosine has precedent in the official Earthformer SEVIR configuration,
but OUR LR, five-epoch warmup and other settings are our own declared choices,
not that paper's reproduced config:
https://github.com/amazon-science/earth-forecasting-transformer/blob/main/scripts/cuboid_transformer/sevir/cfg_sevir.yaml

## 2. Development versus final refit

Development (`confirm`): six source reservoirs retain their chronological
validation dates. Maximum 120 epochs, patience 20, best model-grid reservoir-
macro IoU. At context 3 and horizon <=90 days there are 79 train and 30 val
windows. Use this data ONLY to settle method/hyperparameters and epoch policy.
Declare comparable search budgets for baselines, e.g. LR {1e-4, 3e-4, 1e-3},
without using held-out test results to select the winning setting. A shared
default is a controlled comparison, not proof of baseline optimality.

Final refit (`refit`): after freeze, merge the two SOURCE splits `train + val`.
All 167 images from the six source lakes are eligible; this yields 109 forecast
windows at the same horizon cap. Do NOT duplicate acquisitions or include test
labels. Run a declared fixed 120 epochs, no validation and no early stopping;
select `last.pt`, not the checkpoint with the smallest training loss or largest
test IoU. If development suggests a different budget, declare it BEFORE refit
and apply the policy consistently. The fixed policy and use of previous
development validation must both be disclosed; this is not "never used val".

Locked scoring (`refit-test`): requires explicit unlock, a matching frozen run
plan and completed fixed-budget checkpoints. No training occurs. Test forecasts
condition on previous observed RGB of the unseen lake, never its labelled masks.
This is zero-shot parameter transfer with an observed context, not prediction
without any observations of the test lake. The 34 test windows are observation-
aligned targets, not a dense independently observed daily trajectory.

## 3. Main comparison table (implemented internal comparators)

| ID | Question |
| --- | --- |
| rgb_persistence | Can future water extent beat carrying the last predicted mask forward? |
| last_frame | Is one RGB enough, without any temporal module? |
| convgru | Does an ordinary spatial recurrent model suffice? |
| dt_convgru | Does CfC help beyond a time-aware recurrent control? |
| cfc_notime | Effect of removing BOTH elapsed gaps and calendar, no SDF |
| cfc_no_sdf | Full timing, mask loss only |
| cfc_sdf | Full timing plus small auxiliary SDF loss |

The persistence arm trains a CURRENT-mask segmenter; others train FUTURE-mask
forecasters. It never takes a manual mask as test-time input. This label-task
difference must be stated. All arms receive the same eligible dates and RGB
contract. Report parameters/compute, because matched backbone does not imply
exact parameter equality. Keep each baseline's label contract fixed.

For a submission, add at least one independently implemented strong spatial-
temporal baseline, not just our own module substitutions. Candidates still
requiring implementation and fair RGB-to-mask adaptation:

- SimVP (CVPR 2022), using its official code, not a renamed custom CNN:
  https://github.com/A4Bio/SimVP
- ConvLSTM (NeurIPS 2015), same task/observations and a documented implementation:
  https://arxiv.org/abs/1506.04214
- ODE-RNN/Latent ODE for an irregular-time comparison, adapting the official
  implementation to the same image encoder and mask decoder:
  https://github.com/YuliaRubanova/latent_ode

These are candidates, not already integrated or trained baselines. Numerical-
solver costs and any timestamp adapters must be disclosed. A forcing-dependent
Neural CDE comparison is optional only if all compared methods receive the same
forcing available at forecast origin; do not invent missing meteorology.

## 4. Controlled ablations and sensitivity

Implemented time-isolation arms (same SDF lambda=0.1 as full method):
`cfc_no_gap` keeps calendar but fixes elapsed gaps AND forecast query interval;
`cfc_no_calendar` keeps real elapsed times but zeroes calendar embeddings.
Compare each with `cfc_sdf`. The older no-time/no-SDF arm alone cannot isolate
gap effects from seasonality or the auxiliary objective.

SDF sensitivity (implemented IDs): lambda {0, .01, .03, .1}. Set the primary
lambda using DEVELOPMENT evidence, not by choosing the lowest test error.
Report boundary accuracy and area/level errors, not just mask IoU.

Additional paper studies, NOT automatically implemented by this profile:

- Source-data scarcity: use 25/50/100% of source acquisition dates, selected
  before window creation with a recorded rule; keep held-out targets identical.
  Report actual dates/windows/lakes retained, not nominal percentages alone.
  At minimum compare persistence, dt-ConvGRU and full CfC at each budget.
- Missing-observation robustness: remove past observations only, recompute real
  gaps, preserve target/date and equal context rules, and compare the same
  target set across arms. Never simulate clouds by changing target labels.
- Context length {1,3,5} and spatial width {16,32}: CLI supports these settings,
  but restrict evaluation to common targets. New context lengths change the
  window inventory and cannot be compared naively as if only architecture varied.
- Preprocessing ablation requires actual raw RGB and matched labels/dates for
  ALL lakes. Cannot construct a valid raw-RGB baseline by recoloring enhanced RGB.
- Optional SDF-vs-mask-output comparison changes the prediction task and should
  be separate from the main auxiliary-loss ablation, with its own adapter details.

Do the first three on source development data BEFORE locked testing. If testing
these conditions on the final two lakes, prerecord all corruptions/arms and
report them all without retuning against their outcomes.

## 5. Evidence required beyond one mean IoU

Already emitted by native-grid evaluation: per-window and per-reservoir IoU,
Dice, Boundary F1 at 20 m, area MAE/RMSE/relative MAE, supported F-Z level
MAE/RMSE, horizon strata 1-30/31-60/61-90 d and Z coverage counts. Empty strata
are unsupported evidence, not zero error. Every stratum needs sample/lake counts.
Use the same paired targets for model contrasts. Report macro across reservoirs
plus individual-lake errors rather than letting large lakes dominate.

Planned figures/tables to build from those exports: model comparison, horizon
curves, source-data learning curves, missing-observation curves, contour overlays
for prespecified normal/shrinkage/expansion examples, predicted-vs-labelled area
and Z through actual dates, failure examples, and loss/LR curves for all arms.
Fix example-selection criteria before test to avoid showing only easy cases.
F-Z support failures stay in mask/area metrics and are counted in Z coverage.

Z derived from a human-labelled mask is NOT independent gauge ground truth.
For strong physical water-level claims, separately validate against trustworthy
date-aligned gauge records where available, with datum/F-Z uncertainty and
native pixel scale documented. Do not label mask-derived Z errors gauge errors.

## 6. Generalization, repeats and efficiency

The current 6/2 split supports conclusions about TWO held-out reservoirs, not
global generalization. Before test unlock, optionally use leave-one-source-
reservoir-out development folds across the six source lakes, with all methods
using identical folds. This is a PLANNED manifest-generation/evaluation study;
the current runner does not automatically construct these folds. Do not relabel
two test lakes as development folds after inspecting their results.

One seed is for the requested method check. For a later paper, preferably repeat
the main methods/critical ablations with at least three predeclared seeds and
report mean and sample SD, not the best seed. Three seeds do not estimate
geographical uncertainty. Do not treat overlapping windows as independent
replicates or run an ordinary pixel-wise t-test. No significance/global claim
can be established from two held-out lakes alone.

Efficiency: benchmark parameters, peak GPU memory, and synchronized warmup+
repeated inference latency on the SAME RTX 3090, batch/context/size/precision.
Separate model inference from native-grid conversion and F-Z postprocessing;
report training time as observed, not an estimate from laptop smoke tests.
The current runner records epochs/times and parameter counts; a complete
latency/FLOP benchmark is PLANNED, not claimed implemented.

## 7. Run order on RTX 3090

Do not pull code while an older training process uses that checkout. Preserve
old runs; all v2 outputs must be in new roots. No dataset SCP or reinstall is
needed because this change uses existing dependencies and unchanged data.

```powershell
git pull --ff-only
powershell -ExecutionPolicy Bypass -File .\run_3090.ps1 -Stage smoke -Config configs/method_v2.json -OutputRoot .\runs_method_v2
powershell -ExecutionPolicy Bypass -File .\run_3090.ps1 -Stage confirm -Config configs/method_v2.json -OutputRoot .\runs_method_v2
```

After reviewing development evidence, freeze the configuration, tested arms,
seed list, preprocessing and epoch policy. Run the complete source refit:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_3090.ps1 -Stage refit -Config configs/method_v2.json -OutputRoot .\runs_method_v2
```

DO NOT execute this next command until freeze and completion checks are done:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_3090.ps1 -Stage refit-test -Config configs/method_v2.json -OutputRoot .\runs_method_v2 -ConfirmFinalTest
```

For time-only ablations use a separate output root and PowerShell array syntax:

```powershell
& .\run_3090.ps1 -Stage confirm -Config configs/method_v2.json -OutputRoot .\runs_time_v2 -Experiments cfc_sdf,cfc_no_gap,cfc_no_calendar
```

For all SDF weights (again a separate root):

```powershell
& .\run_3090.ps1 -Stage confirm -Config configs/method_v2.json -OutputRoot .\runs_sdf_v2 -Experiments cfc_no_sdf,cfc_sdf_001,cfc_sdf_003,cfc_sdf
```

Only when expanding beyond the requested one-seed check, explicitly pass
`-Seeds 42,43,44` with direct `& .\run_3090.ps1` invocation. Do not silently
change the seed list or run directory after partial completion. Changed profile,
code or settings requires a NEW root, not reuse of incompatible checkpoints.
