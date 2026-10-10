# Frozen confidence transfer test

Learned-gate transfer value: **PASS**.
Learned-gate transfer quality: **PASS**.
Online fallback qualification: **FAIL**.
Recommended confidence default in this lab: **frozen_gate**.

The learned checkpoint/normalization is unchanged; only the conventional baseline was selected on fresh original-distribution training/validation streams.
Shorter mode intervals and higher noise were never available for baseline selection. No held-out retuning.

Baseline selected prior=0.700, share=0.005, eta=1.0; fixed alpha=0.70.

| Case | Policy | Accuracy | Log loss | Brier | ECE | Confidently wrong | High-conf coverage | Mean alpha |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| reference_mixed | uniform | 49.995% | 0.69315 | 0.25000 | 0.00808 | 0.000% | 0.000% | 0.000 |
| reference_mixed | recency | 69.765% | 0.67529 | 0.22298 | 0.13609 | 9.630% | 45.562% | 1.000 |
| reference_mixed | fixed_shrinkage | 69.765% | 0.59788 | 0.20468 | 0.04820 | 0.000% | 0.000% | 0.700 |
| reference_mixed | frozen_gate | 69.765% | 0.54115 | 0.18326 | 0.01570 | 1.782% | 21.753% | 0.541 |
| reference_mixed | online_mixture | 69.767% | 0.53498 | 0.18078 | 0.02182 | 2.279% | 27.235% | 0.518 |
| short_mixed | uniform | 50.290% | 0.69315 | 0.25000 | 0.00979 | 0.000% | 0.000% | 0.000 |
| short_mixed | recency | 68.582% | 0.69328 | 0.22953 | 0.14381 | 9.931% | 44.577% | 1.000 |
| short_mixed | fixed_shrinkage | 68.582% | 0.60884 | 0.20954 | 0.04806 | 0.000% | 0.000% | 0.700 |
| short_mixed | frozen_gate | 68.582% | 0.60065 | 0.20569 | 0.03570 | 1.345% | 8.976% | 0.554 |
| short_mixed | online_mixture | 68.585% | 0.61711 | 0.21066 | 0.06985 | 3.060% | 18.951% | 0.527 |
| noisy_mixed | uniform | 50.027% | 0.69315 | 0.25000 | 0.00971 | 0.000% | 0.000% | 0.000 |
| noisy_mixed | recency | 59.701% | 0.85834 | 0.29041 | 0.21352 | 12.587% | 37.050% | 1.000 |
| noisy_mixed | fixed_shrinkage | 59.701% | 0.71026 | 0.25449 | 0.12133 | 0.000% | 0.000% | 0.700 |
| noisy_mixed | frozen_gate | 59.701% | 0.66646 | 0.23617 | 0.03221 | 0.334% | 1.503% | 0.297 |
| noisy_mixed | online_mixture | 59.703% | 0.67707 | 0.23885 | 0.05295 | 1.530% | 6.700% | 0.275 |
| short_noisy_mixed | uniform | 50.407% | 0.69315 | 0.25000 | 0.01172 | 0.000% | 0.000% | 0.000 |
| short_noisy_mixed | recency | 58.808% | 0.86857 | 0.29444 | 0.22027 | 12.725% | 36.458% | 1.000 |
| short_noisy_mixed | fixed_shrinkage | 58.808% | 0.71683 | 0.25748 | 0.12782 | 0.000% | 0.000% | 0.700 |
| short_noisy_mixed | frozen_gate | 58.808% | 0.67924 | 0.24239 | 0.04081 | 0.109% | 0.410% | 0.261 |
| short_noisy_mixed | online_mixture | 58.811% | 0.68981 | 0.24540 | 0.06290 | 0.936% | 3.152% | 0.204 |
| standard | uniform | 49.981% | 0.69315 | 0.25000 | 0.01044 | 0.000% | 0.000% | 0.000 |
| standard | recency | 88.987% | 0.33226 | 0.09510 | 0.02874 | 4.386% | 61.445% | 1.000 |
| standard | fixed_shrinkage | 88.987% | 0.38548 | 0.11024 | 0.13025 | 0.000% | 0.000% | 0.700 |
| standard | frozen_gate | 88.987% | 0.33438 | 0.09577 | 0.03407 | 3.738% | 53.464% | 0.967 |
| standard | online_mixture | 88.989% | 0.33384 | 0.09566 | 0.03308 | 4.142% | 58.013% | 0.977 |
| noisy_standard | uniform | 50.008% | 0.69315 | 0.25000 | 0.01017 | 0.000% | 0.000% | 0.000 |
| noisy_standard | recency | 69.043% | 0.69516 | 0.22850 | 0.13961 | 10.392% | 44.699% | 1.000 |
| noisy_standard | fixed_shrinkage | 69.043% | 0.60790 | 0.20879 | 0.04331 | 0.000% | 0.000% | 0.700 |
| noisy_standard | frozen_gate | 69.043% | 0.62467 | 0.21503 | 0.06131 | 0.925% | 4.308% | 0.545 |
| noisy_standard | online_mixture | 69.046% | 0.64948 | 0.22209 | 0.09860 | 3.513% | 15.451% | 0.539 |
| random_labels | uniform | 50.133% | 0.69315 | 0.25000 | 0.01147 | 0.000% | 0.000% | 0.000 |
| random_labels | recency | 50.079% | 1.02112 | 0.35254 | 0.28927 | 14.697% | 29.340% | 1.000 |
| random_labels | fixed_shrinkage | 50.079% | 0.81276 | 0.30035 | 0.20252 | 0.000% | 0.000% | 0.700 |
| random_labels | frozen_gate | 50.079% | 0.69698 | 0.25179 | 0.02103 | 0.003% | 0.005% | 0.082 |
| random_labels | online_mixture | 50.079% | 0.69723 | 0.25174 | 0.01713 | 0.027% | 0.071% | 0.045 |
| no_feedback | uniform | 50.407% | 0.69315 | 0.25000 | 0.01172 | 0.000% | 0.000% | 0.000 |
| no_feedback | recency | 50.407% | 0.69315 | 0.25000 | 0.01172 | 0.000% | 0.000% | 1.000 |
| no_feedback | fixed_shrinkage | 50.407% | 0.69315 | 0.25000 | 0.01172 | 0.000% | 0.000% | 0.700 |
| no_feedback | frozen_gate | 50.407% | 0.69315 | 0.25000 | 0.01172 | 0.000% | 0.000% | 0.901 |
| no_feedback | online_mixture | 50.407% | 0.69315 | 0.25000 | 0.01172 | 0.000% | 0.000% | 0.700 |

## Primary comparisons

| Cases | Difference | Mean log loss difference | Paired stream 95% interval |
|---|---|---:|---|
| aggregate_ood | frozen_gate_minus_online_mixture | -0.012546 | [-0.013292, -0.011789] |
| aggregate_ood | online_mixture_minus_fixed_shrinkage | -0.017314 | [-0.020319, -0.014413] |
| short_mixed | frozen_gate_minus_online_mixture | -0.016460 | [-0.017757, -0.015072] |
| short_mixed | online_mixture_minus_fixed_shrinkage | +0.008269 | [+0.005020, +0.011264] |
| noisy_mixed | frozen_gate_minus_online_mixture | -0.010607 | [-0.011952, -0.009271] |
| noisy_mixed | online_mixture_minus_fixed_shrinkage | -0.033190 | [-0.038234, -0.028481] |
| short_noisy_mixed | frozen_gate_minus_online_mixture | -0.010570 | [-0.011701, -0.009412] |
| short_noisy_mixed | online_mixture_minus_fixed_shrinkage | -0.027022 | [-0.031246, -0.022717] |
| reference_mixed | frozen_gate_minus_online_mixture | +0.006164 | [+0.005454, +0.006891] |
| reference_mixed | online_mixture_minus_fixed_shrinkage | -0.062900 | [-0.066127, -0.059521] |

Required primary advantage: <=-.01 nats with upper interval<0, and no unseen case worse by >.01.
The aggregate first averages three related cases within each seed, then resamples the24 independent seeds. Steps and cases are not independent replicates.
Per-case intervals are descriptive. Intervals condition on the frozen checkpoint and one baseline selection; this is transfer within one toy world family.

## First16 predictions after mode changes

| Case | Direction | Policy | Scored steps | Log loss | Confidently wrong | Mean alpha |
|---|---|---|---:|---:|---:|---:|
| reference_mixed | learnable_entry_first16 | recency | 1875 | 0.39794 | 4.418% | 1.000 |
| reference_mixed | learnable_entry_first16 | fixed_shrinkage | 1875 | 0.43076 | 0.000% | 0.700 |
| reference_mixed | learnable_entry_first16 | frozen_gate | 1875 | 0.56889 | 0.000% | 0.273 |
| reference_mixed | learnable_entry_first16 | online_mixture | 1875 | 0.53857 | 0.534% | 0.330 |
| reference_mixed | random_entry_first16 | recency | 1913 | 1.06293 | 17.379% | 1.000 |
| reference_mixed | random_entry_first16 | fixed_shrinkage | 1913 | 0.82927 | 0.000% | 0.700 |
| reference_mixed | random_entry_first16 | frozen_gate | 1913 | 0.92160 | 7.046% | 0.800 |
| reference_mixed | random_entry_first16 | online_mixture | 1913 | 0.92270 | 8.614% | 0.695 |
| short_mixed | learnable_entry_first16 | recency | 9030 | 0.40176 | 4.782% | 1.000 |
| short_mixed | learnable_entry_first16 | fixed_shrinkage | 9030 | 0.43319 | 0.000% | 0.700 |
| short_mixed | learnable_entry_first16 | frozen_gate | 9030 | 0.50889 | 0.199% | 0.462 |
| short_mixed | learnable_entry_first16 | online_mixture | 9030 | 0.52888 | 0.610% | 0.389 |
| short_mixed | random_entry_first16 | recency | 9071 | 1.03608 | 16.565% | 1.000 |
| short_mixed | random_entry_first16 | fixed_shrinkage | 9071 | 0.81460 | 0.000% | 0.700 |
| short_mixed | random_entry_first16 | frozen_gate | 9071 | 0.82864 | 3.480% | 0.629 |
| short_mixed | random_entry_first16 | online_mixture | 9071 | 0.88603 | 7.875% | 0.645 |
| noisy_mixed | learnable_entry_first16 | recency | 1875 | 0.69976 | 9.453% | 1.000 |
| noisy_mixed | learnable_entry_first16 | fixed_shrinkage | 1875 | 0.61559 | 0.000% | 0.700 |
| noisy_mixed | learnable_entry_first16 | frozen_gate | 1875 | 0.65746 | 0.000% | 0.152 |
| noisy_mixed | learnable_entry_first16 | online_mixture | 1875 | 0.65013 | 0.156% | 0.152 |
| noisy_mixed | random_entry_first16 | recency | 1913 | 1.04996 | 16.711% | 1.000 |
| noisy_mixed | random_entry_first16 | fixed_shrinkage | 1913 | 0.82453 | 0.000% | 0.700 |
| noisy_mixed | random_entry_first16 | frozen_gate | 1913 | 0.75185 | 0.773% | 0.369 |
| noisy_mixed | random_entry_first16 | online_mixture | 1913 | 0.77944 | 2.596% | 0.306 |
| short_noisy_mixed | learnable_entry_first16 | recency | 9030 | 0.73409 | 10.623% | 1.000 |
| short_noisy_mixed | learnable_entry_first16 | fixed_shrinkage | 9030 | 0.63463 | 0.000% | 0.700 |
| short_noisy_mixed | learnable_entry_first16 | frozen_gate | 9030 | 0.65954 | 0.000% | 0.214 |
| short_noisy_mixed | learnable_entry_first16 | online_mixture | 9030 | 0.66639 | 0.236% | 0.142 |
| short_noisy_mixed | random_entry_first16 | recency | 9071 | 1.02582 | 15.656% | 1.000 |
| short_noisy_mixed | random_entry_first16 | fixed_shrinkage | 9071 | 0.81238 | 0.000% | 0.700 |
| short_noisy_mixed | random_entry_first16 | frozen_gate | 9071 | 0.72581 | 0.220% | 0.284 |
| short_noisy_mixed | random_entry_first16 | online_mixture | 9071 | 0.74670 | 1.617% | 0.231 |

Startup excluded; windows stop at the next mode change/episode end. Private mode/age fields are scoring-only.

## Frozen gate components

### learned_value

- aggregate_improvement: PASS
- paired_upper_below_zero: PASS
- short_mixed_loss_preserved: PASS
- noisy_mixed_loss_preserved: PASS
- short_noisy_mixed_loss_preserved: PASS

### learned_quality

- short_mixed/learnable_entry_first16: PASS
- short_mixed/random_entry_first16: PASS
- noisy_mixed/learnable_entry_first16: PASS
- noisy_mixed/random_entry_first16: PASS
- short_noisy_mixed/learnable_entry_first16: PASS
- short_noisy_mixed/random_entry_first16: PASS
- standard_loss_preserved: PASS
- random_loss_near_uniform: PASS
- random_confident_errors_halved: PASS
- short_mixed_confident_errors_preserved: PASS
- noisy_mixed_confident_errors_preserved: PASS
- short_noisy_mixed_confident_errors_preserved: PASS

### online_value

- aggregate_improvement: PASS
- paired_upper_below_zero: PASS
- short_mixed_loss_preserved: PASS
- noisy_mixed_loss_preserved: PASS
- short_noisy_mixed_loss_preserved: PASS

### online_quality

- short_mixed/learnable_entry_first16: FAIL
- short_mixed/random_entry_first16: FAIL
- noisy_mixed/learnable_entry_first16: FAIL
- noisy_mixed/random_entry_first16: PASS
- short_noisy_mixed/learnable_entry_first16: FAIL
- short_noisy_mixed/random_entry_first16: PASS
- standard_loss_preserved: PASS
- random_loss_near_uniform: PASS
- random_confident_errors_halved: PASS

Equal positive shrinkage retains recency's predicted class by construction; equal accuracy is not learned recognition. Alpha is a mixture weight, not a hidden-mode probability.
The online baseline updates causal state after labels, while all hyperparameters and learned weights remain fixed. No private mode or current answer enters inference.
Metrics are equal-weight stream means; slice counts are totals. Confidently wrong is a fraction of ALL predictions, with conditional high-confidence error also retained in results.json.
Checkpoints, candidate grids, every trace/per-seed metric, source hashes and failed gates are retained. Fresh-process proof is in verification.json.
No real assistant configuration, semantic-memory claim, quantum result or broad model superiority follows from this experiment.
