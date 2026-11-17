# Fixed-loss CfC architecture search

This is exploratory method development, not a claim of new state-of-the-art
performance or a completed novelty review. All architecture candidates predict
future MASKS. SDF remains auxiliary, with lambda=0.1 fixed across candidates.
There is no SDF-weight sweep in `configs/method_arch_v3.json`.

## Two hypotheses

1. Irregular feature change: the existing spatial CfC consumes absolute image
   features. Add an explicit residual feature difference so it can use observed
   changes without first discovering subtraction within its recurrent state.
   Compare raw difference against difference divided by the REAL acquisition
   interval. This tests interval normalization separately from the added layer.
2. Finer spatial dynamics: the original model recurs only at 1/16 resolution;
   its multiscale skips are conditioned by that same coarse state. Add a second
   CfC stream at 1/8 resolution, then project its future state into the decoder
   bottleneck and all four skips. This is a NEW recurrent stream, not merely
   renaming the existing skip-conditioning mechanism.

Feature differencing has prior art, including Memory in Memory (CVPR 2019):
https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Memory_in_Memory_A_Predictive_Neural_Network_for_Learning_Higher-Order_CVPR_2019_paper.html
Our simple first-difference residual is NOT a reproduction of MIM, its memory
blocks or its higher-order differencing. CfC uses the existing convolutional
adaptation of the default gated update, not an exact ODE solver:
https://www.nature.com/articles/s42256-022-00556-7

## Candidates and controls

| ID | Change input | Recurrent scales | Parameters (base 16) | Role |
| --- | --- | --- | ---: | --- |
| cfc_sdf | None | 1/16 | 4,008,114 | Existing reference |
| cfc_change_sdf | Raw feature difference | 1/16 | 4,024,498 | Extra change signal |
| cfc_rate_sdf | Interval-normalized difference | 1/16 | 4,024,498 | Separate time-normalization test |
| cfc_multiscale_sdf | None | 1/16 + 1/8 | 4,180,578 | Finer spatial dynamics |
| cfc_multiscale_rate_sdf | Interval-normalized at both scales | 1/16 + 1/8 | 4,201,058 | Combined hypothesis |
| cfc_multiscale_lastfine_sdf | None | Full coarse history; fine stream only sees last image | 4,180,578 | Exactly parameter-matched fine-history control |

For a feature sequence D, the change module is
`D_i + Conv1x1(D_i - D_(i-1))`. The rate version divides the difference by
`max(delta_days_i, 1) / 30` BEFORE projection. The 30-day unit matches the
existing cell's time scale; it is not a monthly interpolation or a forecast
horizon restriction. The first observation has zero difference. No derivative
is carried across reservoir/window boundaries. The projection has no bias, so
zero differences give exactly the original input. Raw and normalized change
have IDENTICAL parameter counts. This feature-space rate is NOT a measured
shoreline velocity, water-level derivative or physical conservation law.

The fine stream has half the coarse hidden width (64 vs 128 at base width 16).
The matched `lastfine` control has exactly the same weights/layers as the full
two-stream model, but resets the fine history and assimilates only the last
context image before the future-date query. Coarse history remains unchanged.
This separates additional fine-history processing from additional capacity;
it does NOT make two-stream and one-stream models parameter-matched.

All candidates use only past RGB and known dates. No target RGB, target mask,
target SDF, gauge or F-Z table enters either recurrent stream. Feature
differences may include illumination, vegetation and registration effects;
their usefulness is an empirical hypothesis, not a guaranteed hydrological
signal. SDF-head outputs NEVER feed the mask head, gates or recurrent state.

Four comparators remain in the full run: `rgb_persistence`, `last_frame`,
`convgru`, `dt_convgru`. Three component ablations are also included:
`cfc_no_sdf` (lambda=0 ONLY to test the auxiliary loss), `cfc_no_gap`,
`cfc_no_calendar` (both lambda=0.1). The legacy confounded `cfc_notime` arm and
the lambda=0.01/0.03 sweep are deliberately absent from this profile.
These three component ablations target the ORIGINAL reference. If an enhanced
architecture is retained, its own SDF/time component ablations must subsequently
hold that enhanced architecture fixed; the reference's ablations cannot be
presented as ablations of a different winning model.

## Full execution on RTX 3090

```powershell
cd "$HOME\mực nước\code"
git pull --ff-only
& .\run_3090.ps1 -Stage smoke -Config configs/method_arch_v3.json -OutputRoot .\runs_arch_v3
& .\run_3090.ps1 -Stage confirm -Config configs/method_arch_v3.json -OutputRoot .\runs_arch_v3
```

These commands run 13 configurations sequentially on ONE GPU, seed 42 only.
Use a NEW root; old smoke/confirmation settings and checkpoints are untouched.
All runs use 512 pixels, base width 16, batch 2, context 3, target gap <=90 days,
AdamW 3e-4, weight decay 1e-4, five warmup epochs and cosine decay to 3e-6.
Development budget is MAXIMUM 120 epochs, early-stopping patience 20, not an
unconditional 120 epochs. Same budget/stopping RULE does not mean identical
actual optimizer-step counts. No reduction of image size/width for full runs.

The full-resolution smoke checks a single train and validation batch per arm;
it does not measure forecasting accuracy. Z may be null when an almost
untrained prediction lies outside F-Z support. Do not clamp or extrapolate it.
If a full-size candidate OOMs, stop and declare a new COMMON memory-compatible
configuration/root for comparisons. Do not silently shrink only that model.
The entire search should use the same code snapshot. Do not `git pull` while
it is running. An interrupted run can resume with the identical command plus
`-Resume`; completed runs are skipped. `best.pt`, `last.pt`, history and loss
curves are retained per arm.

## Interpretation and freeze

The six source lakes retain the SAME chronological validation dates/windows.
The two unseen lakes stay LOCKED. Checkpoint selection remains model-grid
reservoir-macro IoU; final validation tables also report native-grid IoU,
Boundary F1 and area errors. Do not relabel whichever metric wins as the
predeclared primary endpoint. Z is mask-derived and must include support
coverage; compare Z only on a common supported target set.

Read these controlled contrasts before selecting a candidate:

- Raw change vs reference: does an explicit difference help at all?
- Rate vs raw change: does elapsed-time normalization help beyond the layer?
- Two-stream vs reference: does finer dynamic modeling help?
- Two-stream vs matched lastfine: is fine HISTORY useful beyond capacity?
- Two-stream+rate vs rate and vs two-stream: does combining the ideas help?

`architecture_contrasts.csv` records these paired-SEED descriptive differences
in IoU, Boundary F1 and area MAE. It does not establish statistical significance
with one seed. All arms use the same target windows; overlapping windows are
not independent replicates. The generic older `paired_seed_differences.csv`
continues to use `cfc_sdf` as a REFERENCE, not a predetermined winning method.

All candidates may fail to beat persistence. Keep/report those results; do not
keep adding mechanisms until the held-out score increases. This small search
does not prove originality, Q1 readiness or broad geographical generalization.
Once the architecture and training policy are settled on DEVELOPMENT data,
freeze them before source-only refit and any explicit final-test unlock, as
described in `PAPER_EXPERIMENT_PLAN.md`.
