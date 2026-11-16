# Prespecified one-seed method-checking protocol, 2026-11-17

## Research questions

1. Does a learned temporal model improve future-mask prediction over operational
   RGB-based persistence and a last-image-only forecaster?
2. Does spatial CfC help beyond a delta-time-aware ConvGRU using the same
   encoder/decoder framework and calendar information?
3. Does the small auxiliary SDF loss improve native shoreline accuracy without
   harming future-mask overlap or area estimation?

## Design

The unit being forecast is one future observation after three irregularly
spaced past RGB acquisitions of the same reservoir. Known query time and
calendar are allowed. No future RGB, manual context masks, future gauge value,
meteorological forcing or target F-Z-derived level enters the model.

Use the chronological manifest, identical across comparisons: six development
lakes each contribute early training dates and later validation dates. Hold
out the last ceil(20% of acquisitions) in each development lake BEFORE filtering
horizons. The boundary differs by lake; it is not a global calendar cutoff.
El Vado and Riano are the two locked unseen test lakes. The
maximum target gap is fixed to 90 days BEFORE comparing performance. Context
gaps are retained and audited; this is not a regular monthly series.
Validation is rolling-origin: use the previous three already-observed RGB
images, including earlier training RGB or earlier validation RGB. Do not
discard the first three validation targets merely because their context is
before the split boundary. Training targets and all their context stay in the
early train period. Validation labels never train weights; no context mask is
an input. Audit cutoffs and exact counts in `temporal_split.json`.

Image counts: train 131, validation 36, test 47. Uncapped windows: 113, 36, 41.
At the fixed <=90-day horizon cap: train 79, validation 30, test 34. All six
development lakes have nonempty train and validation window sets.

The seed-42 pilot is exploratory (20 epochs maximum). Confirmation repeats all
seven main comparisons at seed 42 ONLY (80 epochs maximum). Early stopping
uses validation macro IoU only, with patience 8 and 15 respectively. Model
selection never uses final-test scores. Data preprocessing, 512 input size,
base width 16, batch 2, context 3, AdamW, learning rate 0.0003, weight decay
0.0001 and augmentation are fixed within each comparison stage. There is no
learning-rate scheduler or warmup in this version. Persistence
uses current-mask training labels; other comparators use future-mask labels.

The proposed mask loss is BCE + soft Dice. Auxiliary SDF uses normalized
native-grid distances and boundary-weighted Smooth L1 at lambda=0.1. The
matched no-SDF arm sets lambda to zero without changing model architecture.
Optional, separately declared sensitivity runs have IDs `cfc_sdf_001` and
`cfc_sdf_003` for lambda=0.01 and 0.03. Do not pick lambda on final-test scores.

The no-time arms jointly remove actual gap and calendar information. They do
not isolate delta-time from seasonality. The CfC cell is an explicit spatial
adaptation, not an unmodified official ncps implementation or an ODE solver.

## Outcomes

Primary checkpoint criterion: validation macro-reservoir IoU on the model grid.
Comparison outcomes: native-grid future-mask IoU, Dice, shoreline Boundary F1
within 20 metres, area MAE/RMSE/relative error and mask-derived F-Z level
MAE/RMSE with valid-support counts. Native binary masks are obtained by
unletterboxing, bilinear probability upsampling and thresholding at 0.5.
Area is counted on native 10 m grids, not on resized tensors.

Per-window scores are averaged within each reservoir and then equally across
reservoirs; report per-lake results and horizon strata as well. Present mean
over all prespecified seeds, not only the best seed. In the current one-seed
check, sample SD is unavailable, not zero. More seeds would be needed to
estimate optimization variability. Seed SD
describes optimization variability, not geographic sampling uncertainty.
Do not use overlapping windows as independent replicates for significance
tests. With two held-out lakes, global generalization is not established.

Z is a downstream transformation of labelled/predicted masks. It is not a
new independent gauge ground truth and may be unavailable outside curve
support. Z coverage failures must not remove those cases from mask/area
outcomes or be hidden when reporting Z accuracy.

## Gates

1. CUDA probe and all-comparator smoke must succeed on the RTX 3090.
2. Two-window overfit must show learning, not merely successful execution.
3. Review the pilot validation outputs before launching three-seed confirmation.
4. Freeze settings and review the seed-42 confirmation before explicitly unlocking test.

This package has a different manifest from the older five-train/one-validation
lake package. Old checkpoints cannot be resumed into this experiment. Use the
new extracted directory and new run outputs; earlier packages are preserved.

If the proposed method does not consistently help, report that result and
inspect the method on development data. Do not repeatedly tune against locked
test lakes. A larger benchmark, stronger external baselines, independent level
validation and more held-out reservoirs may be needed for later paper claims;
this first experiment set does not certify a publication tier.
