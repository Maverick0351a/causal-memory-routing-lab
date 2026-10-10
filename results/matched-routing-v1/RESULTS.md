# Matched routing comparison

Primary verdict: **FAIL** for autoregressive predictive value.

Fresh training, initialization and held-out seeds; no tuning after evaluation.

| Condition | Model | Accuracy | Log loss | Brier | ECE | Confidently wrong | Recovery score* |
|---|---|---:|---:|---:|---:|---:|---:|
| standard | fixed_uniform | 49.427% | 0.69315 | 0.25000 | 0.01797 | 0.000% | 27.72 |
| standard | recency | 89.219% | 0.32557 | 0.09257 | 0.03160 | 4.310% | 9.00 |
| standard | autoregressive | 88.862% | 0.32703 | 0.09313 | 0.02617 | 4.393% | 9.38 |
| standard | joint | 88.938% | 0.32635 | 0.09295 | 0.02627 | 4.453% | 9.43 |
| fast_switch | fixed_uniform | 50.247% | 0.69315 | 0.25000 | 0.01081 | 0.000% | 29.15 |
| fast_switch | recency | 82.174% | 0.45471 | 0.13970 | 0.04612 | 6.406% | 11.96 |
| fast_switch | autoregressive | 81.214% | 0.46514 | 0.14451 | 0.04985 | 6.490% | 12.52 |
| fast_switch | joint | 81.081% | 0.46589 | 0.14455 | 0.05065 | 6.513% | 12.58 |
| random_labels | fixed_uniform | 50.260% | 0.69315 | 0.25000 | 0.01302 | 0.000% | 27.72 |
| random_labels | recency | 49.870% | 1.00934 | 0.34944 | 0.29228 | 13.867% | 27.81 |
| random_labels | autoregressive | 50.258% | 1.01108 | 0.35117 | 0.29374 | 14.117% | 27.77 |
| random_labels | joint | 50.143% | 1.00557 | 0.34926 | 0.29154 | 13.544% | 27.55 |
| no_feedback | fixed_uniform | 49.427% | 0.69315 | 0.25000 | 0.01797 | 0.000% | 27.72 |
| no_feedback | recency | 49.427% | 0.69315 | 0.25000 | 0.01797 | 0.000% | 27.72 |
| no_feedback | autoregressive | 49.237% | 0.71393 | 0.26003 | 0.08682 | 0.000% | 25.65 |
| no_feedback | joint | 49.307% | 0.71080 | 0.25858 | 0.07983 | 0.000% | 26.55 |

AR minus joint standard log loss: **0.000685 nats**; paired two-way bootstrap 95% interval: [-0.000030, 0.001397]. Negative favors AR.
Required: <=-0.01 nats and upper interval endpoint <0, with matched size/affine compute.

Five model seeds share ten test streams. The 50 paired cells are not 50 independent environments.
The bootstrap resamples both seed axes; this task-level experiment cannot establish broad superiority.

## Compute and latency

AR: 750 parameters, 720 affine MACs. Joint: 758 parameters, 728 affine MACs.
MAC counts omit activations/softmax; measured forward times include them.
Median marginal forward: AR 36.476 us; joint 18.345 us; ratio 1.988x.
Sampling reuses the already computed conditionals; both use one forward per evaluated step.
Timing is specific to this machine/load; the efficiency gate is separate from the primary outcome.

## Frozen scientific gates

- comparable_size_and_affine_compute: PASS
- ar_predictive_value: FAIL
- ar_latency_within_budget: FAIL
- autoregressive_learning: PASS
- autoregressive_negative_control: PASS
- autoregressive_memory_dependence: PASS
- joint_learning: PASS
- joint_negative_control: PASS
- joint_memory_dependence: PASS

*Recovery score: mean endpoint offset for first 7/8-correct rolling window after an actual switch; 32 is assigned if censored before the next switch/horizon. Lower is better. Startup is excluded. On random labels, chance runs have no adaptation meaning.
Recovery fraction, recovered-only median, all paired measurements and per-seed metrics are in results.json.

Training data order and feature vectors were hash-checked identical across both model families.
Checkpoints and complete traces are saved. Fresh-process replay evidence is written separately to verification.json.
No calibration was fit on test labels. Random-label accuracy passing does not imply calibrated uncertainty.
This uses known toy experts and hand-designed feedback features, with no hardware or semantic-retrieval claim.
