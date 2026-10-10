# Continuous sensor confidence transfer

Cross-task predictive value: **FAIL**.
Cross-task quality: **PASS**.
Random-label controls: **PASS**.
Sensor-family qualification: **FAIL**.

Primary gate minus online loss: -0.005716 nats, paired-seed95% interval [-0.006882, -0.004492].
Required: <=-.01 nats, upper interval<0, no stress case worse by>.02. All failed components are retained.

Synthetic fault propensity from eight continuous normal readings;36 learned base parameters. This is not measured sensor data or a hardware diagnostic.
The gate checkpoint and its normalization are unchanged. Online Platt calibration directly updates slope/intercept from past released feedback.

Validation-selected online rate=0.08, decay=0.01; fixed slope/intercept=[0.7755946216147885, 0.026910376655559634].

| Case | Policy | Log loss | Brier | Accuracy | ECE | Confidently wrong | High-conf coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| stable | uniform | 0.69315 | 0.25000 | 50.288% | 0.01090 | 0.000% | 0.000% |
| stable | base | 0.40501 | 0.12821 | 81.676% | 0.03945 | 0.556% | 26.948% |
| stable | fixed_calibration | 0.42705 | 0.13356 | 81.665% | 0.07880 | 0.008% | 4.650% |
| stable | frozen_gate | 0.41823 | 0.13105 | 81.676% | 0.05733 | 0.201% | 12.519% |
| stable | online_calibration | 0.40188 | 0.12845 | 81.535% | 0.01836 | 1.847% | 41.040% |
| mixed | uniform | 0.69315 | 0.25000 | 50.404% | 0.01082 | 0.000% | 0.000% |
| mixed | base | 0.71116 | 0.24073 | 65.346% | 0.12931 | 7.123% | 26.950% |
| mixed | fixed_calibration | 0.66447 | 0.23020 | 65.318% | 0.08848 | 1.104% | 4.650% |
| mixed | frozen_gate | 0.58400 | 0.20152 | 65.346% | 0.02637 | 0.157% | 3.673% |
| mixed | online_calibration | 0.59557 | 0.20643 | 65.028% | 0.03544 | 0.987% | 12.299% |
| short_mixed | uniform | 0.69315 | 0.25000 | 50.255% | 0.01270 | 0.000% | 0.000% |
| short_mixed | base | 0.70410 | 0.23800 | 65.782% | 0.12513 | 6.993% | 26.950% |
| short_mixed | fixed_calibration | 0.65903 | 0.22790 | 65.769% | 0.08439 | 1.147% | 4.650% |
| short_mixed | frozen_gate | 0.61755 | 0.21454 | 65.782% | 0.02138 | 0.109% | 0.906% |
| short_mixed | online_calibration | 0.63278 | 0.22013 | 64.345% | 0.03056 | 0.686% | 3.958% |
| weak_signal | uniform | 0.69315 | 0.25000 | 50.136% | 0.01031 | 0.000% | 0.000% |
| weak_signal | base | 0.74474 | 0.25593 | 62.364% | 0.15902 | 6.887% | 26.950% |
| weak_signal | fixed_calibration | 0.69059 | 0.24253 | 62.343% | 0.11667 | 0.795% | 4.650% |
| weak_signal | frozen_gate | 0.66850 | 0.23723 | 62.364% | 0.04877 | 0.011% | 0.043% |
| weak_signal | online_calibration | 0.66461 | 0.23504 | 61.155% | 0.03738 | 0.144% | 0.621% |
| rotating | uniform | 0.69315 | 0.25000 | 50.529% | 0.00974 | 0.000% | 0.000% |
| rotating | base | 0.93771 | 0.32394 | 53.258% | 0.24988 | 11.933% | 26.945% |
| rotating | fixed_calibration | 0.84015 | 0.30166 | 53.315% | 0.20668 | 2.094% | 4.652% |
| rotating | frozen_gate | 0.66623 | 0.23777 | 53.258% | 0.04029 | 0.041% | 0.930% |
| rotating | online_calibration | 0.66618 | 0.23718 | 58.244% | 0.04662 | 0.298% | 3.410% |
| random_labels | uniform | 0.69315 | 0.25000 | 50.420% | 0.00936 | 0.000% | 0.000% |
| random_labels | base | 1.00347 | 0.34792 | 49.878% | 0.28376 | 13.517% | 26.950% |
| random_labels | fixed_calibration | 0.89118 | 0.32232 | 49.851% | 0.24137 | 2.238% | 4.650% |
| random_labels | frozen_gate | 0.69788 | 0.25223 | 49.878% | 0.02997 | 0.000% | 0.000% |
| random_labels | online_calibration | 0.71523 | 0.26007 | 50.027% | 0.07739 | 0.033% | 0.052% |
| no_feedback | uniform | 0.69315 | 0.25000 | 50.404% | 0.01082 | 0.000% | 0.000% |
| no_feedback | base | 0.71114 | 0.24072 | 65.354% | 0.12933 | 7.113% | 26.942% |
| no_feedback | fixed_calibration | 0.66446 | 0.23020 | 65.313% | 0.08864 | 1.107% | 4.650% |
| no_feedback | frozen_gate | 0.63451 | 0.22152 | 65.354% | 0.02723 | 0.000% | 0.000% |
| no_feedback | online_calibration | 0.66446 | 0.23020 | 65.313% | 0.08864 | 1.107% | 4.650% |

## Paired loss comparisons

| Case | Difference | Mean | Paired-seed95% interval |
|---|---|---:|---|
| aggregate | frozen_gate_minus_online_calibration | -0.005716 | [-0.006882, -0.004492] |
| aggregate | frozen_gate_minus_fixed_calibration | -0.079491 | [-0.084287, -0.074825] |
| aggregate | online_calibration_minus_fixed_calibration | -0.073775 | [-0.079108, -0.068579] |
| stable | frozen_gate_minus_online_calibration | +0.016351 | [+0.015175, +0.017544] |
| stable | frozen_gate_minus_fixed_calibration | -0.008820 | [-0.010072, -0.007535] |
| stable | online_calibration_minus_fixed_calibration | -0.025171 | [-0.027539, -0.022746] |
| mixed | frozen_gate_minus_online_calibration | -0.011575 | [-0.012766, -0.010345] |
| mixed | frozen_gate_minus_fixed_calibration | -0.080473 | [-0.084809, -0.076138] |
| mixed | online_calibration_minus_fixed_calibration | -0.068898 | [-0.072927, -0.064885] |
| short_mixed | frozen_gate_minus_online_calibration | -0.015234 | [-0.016642, -0.013829] |
| short_mixed | frozen_gate_minus_fixed_calibration | -0.041485 | [-0.046748, -0.036722] |
| short_mixed | online_calibration_minus_fixed_calibration | -0.026251 | [-0.030794, -0.022152] |
| weak_signal | frozen_gate_minus_online_calibration | +0.003885 | [+0.002676, +0.005145] |
| weak_signal | frozen_gate_minus_fixed_calibration | -0.022090 | [-0.025825, -0.018306] |
| weak_signal | online_calibration_minus_fixed_calibration | -0.025975 | [-0.029103, -0.022857] |
| rotating | frozen_gate_minus_online_calibration | +0.000059 | [-0.002693, +0.003044] |
| rotating | frozen_gate_minus_fixed_calibration | -0.173918 | [-0.193492, -0.154396] |
| rotating | online_calibration_minus_fixed_calibration | -0.173977 | [-0.195579, -0.152384] |
| random_labels | frozen_gate_minus_online_calibration | -0.017357 | [-0.018437, -0.016255] |
| random_labels | frozen_gate_minus_fixed_calibration | -0.193304 | [-0.199960, -0.186683] |
| random_labels | online_calibration_minus_fixed_calibration | -0.175947 | [-0.182669, -0.169087] |
| no_feedback | frozen_gate_minus_online_calibration | -0.029948 | [-0.033516, -0.026347] |
| no_feedback | frozen_gate_minus_fixed_calibration | -0.029948 | [-0.033516, -0.026347] |
| no_feedback | online_calibration_minus_fixed_calibration | +0.000000 | [+0.000000, +0.000000] |

The primary aggregate averages four stress cases within each seed, then resamples24 seeds. Cases and individual steps are not independent replicates.
Intervals condition on one learned ensemble and one selection; they exclude training/selection uncertainty. Per-case intervals are descriptive.

## Transitions and response delay

| Case | Policy | Entry | First16 loss | Descriptive response steps | Recovered fraction |
|---|---|---|---:|---:|---:|
| mixed | base | signal | 0.40355 | 15.89206 | 1.00000 |
| mixed | base | noise | 0.96342 | 46.98115 | 0.82381 |
| mixed | fixed_calibration | signal | 0.42507 | 15.70972 | 1.00000 |
| mixed | fixed_calibration | noise | 0.85994 | 38.99722 | 0.90020 |
| mixed | frozen_gate | signal | 0.57733 | 21.03472 | 1.00000 |
| mixed | frozen_gate | noise | 0.79781 | 21.21806 | 0.99306 |
| mixed | online_calibration | signal | 0.57092 | 19.03988 | 1.00000 |
| mixed | online_calibration | noise | 0.93647 | 27.60218 | 0.98710 |
| short_mixed | base | signal | 0.40333 | 16.90528 | 0.98044 |
| short_mixed | base | noise | 1.00717 | 66.80902 | 0.39560 |
| short_mixed | fixed_calibration | signal | 0.42569 | 16.61260 | 0.98385 |
| short_mixed | fixed_calibration | noise | 0.89366 | 55.60241 | 0.54276 |
| short_mixed | frozen_gate | signal | 0.53832 | 26.70236 | 0.88167 |
| short_mixed | frozen_gate | noise | 0.77617 | 27.19476 | 0.88313 |
| short_mixed | online_calibration | signal | 0.52717 | 23.81388 | 0.91184 |
| short_mixed | online_calibration | noise | 0.83354 | 37.19668 | 0.78008 |
| rotating | base | signal | 0.95760 | 72.71543 | 0.40394 |
| rotating | base | noise | n/a | n/a | n/a |
| rotating | fixed_calibration | signal | 0.85566 | 71.66413 | 0.41151 |
| rotating | fixed_calibration | noise | n/a | n/a | n/a |
| rotating | frozen_gate | signal | 0.70536 | 87.07050 | 0.17874 |
| rotating | frozen_gate | noise | n/a | n/a | n/a |
| rotating | online_calibration | signal | 0.74022 | 75.19115 | 0.41273 |
| rotating | online_calibration | noise | n/a | n/a | n/a |

Startup excluded. Search ends at the next switch/end. Unrecovered eligible events score96; report is an equal-stream mean. Loss-window response is stochastic and is not a change detector.

## Every frozen gate component

### value

- minimum_improvement: FAIL
- paired_upper_below_zero: PASS
- mixed_loss_preserved: PASS
- short_mixed_loss_preserved: PASS
- weak_signal_loss_preserved: PASS
- rotating_loss_preserved: PASS

### quality

- mixed_brier_preserved: PASS
- mixed_confident_errors_preserved: PASS
- short_mixed_brier_preserved: PASS
- short_mixed_confident_errors_preserved: PASS
- weak_signal_brier_preserved: PASS
- weak_signal_confident_errors_preserved: PASS
- rotating_brier_preserved: PASS
- rotating_confident_errors_preserved: PASS
- mixed/signal_entry_first16: PASS
- mixed/noise_entry_first16: PASS
- short_mixed/signal_entry_first16: PASS
- short_mixed/noise_entry_first16: PASS
- stable_loss_preserved: PASS
- random_loss_near_uniform: PASS
- random_confident_errors_halved: PASS

### controls

- uniform_random_accuracy_near_chance: PASS
- base_random_accuracy_near_chance: PASS
- fixed_calibration_random_accuracy_near_chance: PASS
- frozen_gate_random_accuracy_near_chance: PASS
- online_calibration_random_accuracy_near_chance: PASS

All predictions precede feedback; hidden coefficients/modes/ages/switches are scoring-only. The identical public observation streams prevent an artificial mode marker.
No-feedback retains the frozen learned predictor and therefore can still recognize signal. It withholds all online feedback, including adapter summaries.
Gate shrinkage preserves base class sign apart from rounding ties; accuracy equality is structural. Platt calibration can adjust bias and invert sign, so any recognition difference belongs to that baseline.
Confidently wrong is a fraction of all predictions; coverage and conditional high-confidence error are also retained in results.json.
All source/checkpoints/candidate grids/traces and failed gates are saved. Fresh-process verification and causal SQLite resume are in verification.json.
Prior research defaults and live assistant configuration remain unchanged. No downloads, API keys, pretrained peptide weights, network services or quantum jobs were used. Python socket guard is not an OS sandbox.
