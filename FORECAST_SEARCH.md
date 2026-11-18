# Forecast-head development, v4

Exploratory method development, not a claim of scientific improvement. The two
main benchmark tables are deferred. Train on six source reservoirs, validate
chronologically within those reservoirs, and keep the two test lakes locked.

## Three arms

| Experiment | Forecast after CfC observation assimilation | Parameters (base 16) | SDF weight |
| --- | --- | ---: | ---: |
| cfc_sdf | Existing shared-cell future-date query | 4,008,114 | 0.1 |
| cfc_residual_sdf | h_future = h_last + R(h_last, query, log1p(tau)) | 4,320,818 | 0.1 |
| cfc_residual_anchor_sdf | h_future = h_last + (1-exp(-tau)) R(...) | 4,320,818 | 0.1 |

tau is the real future gap in units of 30 days, the existing time scale.
The residual arms share the same convolutional head and parameter count; the
anchor adds no weights. The reference has fewer parameters, so its comparison
does not isolate capacity. The residual-head comparison isolates the fixed
time envelope, not all possible benefits of temporal modeling.

CfC still assimilates every past image at the original 1/16 spatial scale.
Only the two new arms replace its final synthetic-query update with a separate
forecast head. Encoder, skips, decoder, mask and SDF heads are unchanged. SDF
outputs never feed forecasting or mask inference. No new loss, F-Z loss, gauge
input, future image, extra recurrent scale or SDF-weight sweep is introduced.

The anchor guarantees h_future(0) = h_last in latent space. It does NOT prove
physical conservation, correct current segmentation, ODE semigroup consistency,
or bounded shoreline displacement. R may be signed and arbitrarily large.
This is a hypothesis to test, not a new exact ODE solution or a novelty claim.

## Forecast-aligned checkpoint selection

The new profile selects `z_mae_support_first`, not IoU. The previous profiles
retain their IoU default for reproducibility. All v4 arms must be retrained
under this new rule; do not mix their results with v3 checkpoint selections.

At every epoch, FP32 validation restores probabilities to the native grid,
thresholds at 0.5, measures area, then applies the unchanged F-Z curve. The
fixed cohort contains primary validation TARGETS within their own curve's
support. Out-of-support targets are counted and excluded independently of
predictions. Unsupported predictions are never clamped or extrapolated.
The current manifest has 30/30 supported validation targets across all six
source lakes. This does not guarantee predictions will stay in support.

Checkpoint ranking is lexicographic:

1. Minimize unsupported predictions on that fixed target cohort.
2. Once all those predictions are supported, minimize equal-reservoir macro
   Z MAE. Z RMSE remains a secondary endpoint.
3. While coverage is incomplete, break equal-failure ties using macro area
   MAE divided by each lake's F-Z area span, evaluated on ALL eligible targets.
   This fallback is not a water-level MAE and is labeled separately in logs.

Primary Z MAE/RMSE are null unless the ENTIRE cohort has supported predictions.
Conditional valid-pair errors remain diagnostic only. History, best-checkpoint
metadata and native-grid evaluation record cohort size, failures and coverage.
Selection never sees either test reservoir. Z remains mask-derived, not an
independently measured gauge reference. Partial coverage must not be presented
as a lower-error winning model.

## Run on RTX 3090

From the existing code directory after `git pull --ff-only`:

```powershell
& .\run_forecast_v4.ps1
```

The wrapper runs full-size smoke first, then confirm only if smoke succeeds.
It enables UTF-8 for Python/output so Vietnamese directory names are supported.
Use `-DryRun` to print both stages without training, or `-Resume` to resume an
identical interrupted v4 confirmation. No test-unlock argument is exposed.

Three arms, seed 42, 512px, base width 16, batch 2, context 3, target gap at
most 90 days, AdamW 3e-4, weight decay 1e-4, 5 warmup epochs, cosine floor 3e-6,
maximum 120 epochs and patience 20. The same stopping rule does not imply the
same realized training duration. No interpolation or forcing data is added.

Use a NEW output root. Resume only identical v4 code/configuration with
`-Resume`; never pull code while a stage is running. The changed code cannot
resume a frozen v3 experiment plan. Smoke is one train/validation batch per
arm, not an accuracy result. Checkpoints and loss/forecast curves are retained.

Inspect `confirm/comparison.csv`, `per_reservoir_comparison.csv`, per-window
evaluation and `best_selection` in each run's summary. `comparison.png` shows
Z MAE/RMSE only at complete coverage, plus area MAE. Consider all source lakes
and actual horizon bins; one seed cannot establish statistical significance.
No automatic winner or final-test execution is selected by this profile.
