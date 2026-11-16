# Configuration rationale for the first method check

This page documents the original constant-LR profile. For the opt-in 120-epoch
warmup/cosine profile, fixed-budget no-validation refit, and paper experiment
plan see `configs/method_v2.json` and `PAPER_EXPERIMENT_PLAN.md`. Baseline paper
settings below are references, not claims that v2 reproduces their full recipes.

Protocol date: 2026-11-17. These are starting settings for a one-seed experiment,
not a claim that any configuration guarantees publication or optimal accuracy.

## Primary-source comparisons

The following settings refer to particular datasets/experiments. They are not
universal defaults for a venue and should not be copied across tasks blindly.

| Work | Verified setting | Primary source |
| --- | --- | --- |
| SimVP, CVPR 2022 | Adam, learning rate 0.01; 80 epochs on TrafficBJ, 100 on Human3.6, 2000 on Moving MNIST. Official code uses OneCycleLR. | [Paper, Section 4.3](https://arxiv.org/pdf/2206.05099), [optimizer code](https://github.com/A4Bio/SimVP/blob/master/exp.py) |
| Earthformer, NeurIPS 2022, SEVIR config | AdamW, LR 0.001, weight decay 0, max 100 epochs, total batch 32, microbatch 2, cosine decay, 20% warmup, gradient clip 1, early-stop patience 20. | [Official SEVIR YAML](https://github.com/amazon-science/earth-forecasting-transformer/blob/main/scripts/cuboid_transformer/sevir/cfg_sevir.yaml) |
| Contextformer / GreenEarthNet, CVPR 2024 | 100 epochs, AdamW, LR 0.00004, total batch 32 across two GPUs; pretrained spatial encoder and meteorological inputs. | [Supplement, A.2](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Benson_Multi-modal_Learning_for_CVPR_2024_supplemental.pdf), [official config](https://github.com/vitusbenson/greenearthnet/blob/main/model_configs/contextformer/contextformer6M/seed%3D42.yaml) |
| CfC, Nature Machine Intelligence 2022, Walker2d default experiment | 50 epochs, Adam, initial LR 0.02, batch 256, exponential LR decay factor 0.95 per epoch, gradient clip 1. This is a vector-sequence task, not satellite mask forecasting. | [Official training script](https://github.com/raminmh/CfC/blob/main/train_walker.py) |
| PredRNN-V2, TPAMI, Moving MNIST launch script | 80000 iterations, LR 0.0001, batch 8. Iterations are not epochs. | [Official launch script](https://github.com/thuml/predrnn-pytorch/blob/master/mnist_script/predrnn_v2_mnist_train.sh), [Adam implementation](https://github.com/thuml/predrnn-pytorch/blob/master/core/models/model_factory.py) |

These examples show why an epoch threshold alone cannot certify an experiment.
The original SimVP repository also warns that its short example run is not the
full reproduction budget. Effective batch size must not be confused with a
single GPU microbatch. The GreenEarthNet input/target problem differs from ours.

## Settings actually implemented here

| Item | Current value |
| --- | --- |
| Random seed | 42 only |
| Main stage | `confirm`, all seven comparisons |
| Budget | Maximum 80 epochs, early-stop patience 15 |
| Optimizer | AdamW, LR 0.0003, weight decay 0.0001 |
| LR schedule | Constant; no scheduler or warmup |
| Gradient clipping | Global norm 5 |
| Input | Three past enhanced RGB images, letterboxed to 512 x 512 |
| Batch / gradient accumulation | Batch 2, no accumulation |
| Spatial width / normalization | Base channels 16, GroupNorm |
| Precision | CUDA autocast + GradScaler; CPU float32 |
| Target | Next observed future mask, actual target gap <=90 days |
| Loss | BCE + soft Dice + 0.1 * boundary-weighted Smooth L1 SDF |
| Auxiliary controls | Matched lambda=0; optional 0.01 / 0.03 runs |
| Split | Six lakes with chronological validation; two entirely unseen test lakes |
| Checkpoint / stopping | Validation macro-reservoir IoU on the model grid |
| Comparison | Native-grid per-window metrics, averaged per lake then across lakes |

At 79 retained training windows and batch 2, one epoch draws 79 windows with
the reservoir-balanced sampler and takes 40 optimizer steps. The 80-epoch cap
is therefore at most 3200 optimizer steps, not 80 updates. Windows overlap and
must not be counted as independent geographical replicates.

The 20-epoch pilot is optional exploratory screening. It is not the main
budget. Inspect learning curves and development errors before specifying a
different budget or an equal validation-only LR search for every comparator.
Shared default settings are a controlled comparison, not proof that every
baseline has been individually tuned optimally.

A single seed does not estimate run-to-run variability. SD fields remain
blank. These experiments do not establish global generalization from only
two test lakes. F-Z-derived level errors are not independent gauge validation.
Do not reuse the locked test scores to choose hyperparameters.
