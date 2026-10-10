# Confidence-gating experiment

Adaptive value over fixed shrinkage: **PASS**.
Deployment-quality gate: **PASS**.
Replacement supported: **YES**.

Fresh disjoint train/validation/test seeds; test labels never select parameters. CPU NumPy only.
Validation selected fixed alpha 0.65 and gate L2 0.0.
Eight fitted gate coefficients; feature normalization fit on training data only.

| Condition | Policy | Accuracy | Log loss | Brier | ECE | Confidently wrong | High-conf coverage | Mean alpha |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| standard | uniform | 49.807% | 0.69315 | 0.25000 | 0.00827 | 0.000% | 0.000% | 0.000 |
| standard | recency | 88.631% | 0.33668 | 0.09692 | 0.02640 | 4.351% | 61.312% | 1.000 |
| standard | fixed_shrinkage | 88.631% | 0.40290 | 0.11658 | 0.14547 | 0.000% | 0.000% | 0.650 |
| standard | adaptive_gate | 88.631% | 0.33878 | 0.09757 | 0.03230 | 3.700% | 53.163% | 0.966 |
| fast_switch | uniform | 49.786% | 0.69315 | 0.25000 | 0.00817 | 0.000% | 0.000% | 0.000 |
| fast_switch | recency | 81.717% | 0.46455 | 0.14344 | 0.04810 | 6.624% | 55.263% | 1.000 |
| fast_switch | fixed_shrinkage | 81.717% | 0.47446 | 0.14881 | 0.08629 | 0.000% | 0.000% | 0.650 |
| fast_switch | adaptive_gate | 81.717% | 0.46182 | 0.14375 | 0.03534 | 4.574% | 37.834% | 0.924 |
| random_labels | uniform | 50.529% | 0.69315 | 0.25000 | 0.01299 | 0.000% | 0.000% | 0.000 |
| random_labels | recency | 49.948% | 1.02251 | 0.35290 | 0.29063 | 14.827% | 29.804% | 1.000 |
| random_labels | fixed_shrinkage | 49.948% | 0.79399 | 0.29364 | 0.18887 | 0.000% | 0.000% | 0.650 |
| random_labels | adaptive_gate | 49.948% | 0.69703 | 0.25180 | 0.02266 | 0.000% | 0.000% | 0.083 |
| mixed | uniform | 50.217% | 0.69315 | 0.25000 | 0.01020 | 0.000% | 0.000% | 0.000 |
| mixed | recency | 68.875% | 0.68896 | 0.22830 | 0.14192 | 9.790% | 45.288% | 1.000 |
| mixed | fixed_shrinkage | 68.875% | 0.60361 | 0.20744 | 0.03003 | 0.000% | 0.000% | 0.650 |
| mixed | adaptive_gate | 68.875% | 0.55048 | 0.18714 | 0.01556 | 1.679% | 20.190% | 0.526 |
| no_feedback | uniform | 50.217% | 0.69315 | 0.25000 | 0.01020 | 0.000% | 0.000% | 0.000 |
| no_feedback | recency | 50.217% | 0.69315 | 0.25000 | 0.01020 | 0.000% | 0.000% | 1.000 |
| no_feedback | fixed_shrinkage | 50.217% | 0.69315 | 0.25000 | 0.01020 | 0.000% | 0.000% | 0.650 |
| no_feedback | adaptive_gate | 50.217% | 0.69315 | 0.25000 | 0.01020 | 0.000% | 0.000% | 0.901 |

Primary mixed log-loss difference (adaptive minus fixed): **-0.053134 nats**, 95% paired stream-bootstrap interval [-0.055597, -0.050618].
Required: <=-0.01 nats and upper endpoint <0. Twenty-four independent test seeds; individual steps are not replicates.
Intervals condition on one deterministic fitting/validation selection; they do not measure training-set or optimization uncertainty.

## Quality gates

- standard_log_loss_preserved: PASS
- fast_switch_log_loss_preserved: PASS
- standard_accuracy_preserved: PASS
- standard_switch_log_loss_preserved: PASS
- fast_switch_accuracy_preserved: PASS
- fast_switch_switch_log_loss_preserved: PASS
- mixed_accuracy_preserved: PASS
- mixed_switch_log_loss_preserved: PASS
- random_log_loss_near_uniform: PASS
- random_confident_errors_halved: PASS
- mixed_random_confident_errors_halved: PASS

## Adaptation and mixed-mode diagnostics

| Condition | Policy | Loss in rule-switch windows | Rolling recovery score |
|---|---|---:|---:|
| standard | recency | 0.43681 | 9.176 |
| standard | fixed_shrinkage | 0.45873 | 9.176 |
| standard | adaptive_gate | 0.43412 | 9.176 |
| fast_switch | recency | 0.47596 | 12.147 |
| fast_switch | fixed_shrinkage | 0.48068 | 12.147 |
| fast_switch | adaptive_gate | 0.47225 | 12.147 |
| mixed | recency | 0.44728 | 9.437 |
| mixed | fixed_shrinkage | 0.46443 | 9.437 |
| mixed | adaptive_gate | 0.44870 | 9.437 |

Adaptive shrinkage preserves recency's classification sign by construction. Equal accuracy/rolling recovery is not evidence of a learned benefit.
Switch log loss is the meaningful confidence-adaptation check: first16 steps, truncated at the next rule/mode change. Rolling recovery is first7/8 correct; unrecovered/censored=32, startup excluded.

| Mixed slice | Policy | Steps | Log loss | Confidently wrong | Mean alpha |
|---|---|---:|---:|---:|---:|
| learnable | recency | 18389 | 0.34919 | 4.376% | 1.000 |
| learnable | fixed_shrinkage | 18389 | 0.41072 | 0.000% | 0.650 |
| learnable | adaptive_gate | 18389 | 0.37444 | 2.565% | 0.842 |
| random | recency | 18475 | 1.02795 | 15.182% | 1.000 |
| random | fixed_shrinkage | 18475 | 0.79606 | 0.000% | 0.650 |
| random | adaptive_gate | 18475 | 0.72628 | 0.801% | 0.211 |
| random_entry_first16 | recency | 1949 | 1.05303 | 17.556% | 1.000 |
| random_entry_first16 | fixed_shrinkage | 1949 | 0.80257 | 0.000% | 0.650 |
| random_entry_first16 | adaptive_gate | 1949 | 0.90524 | 7.360% | 0.779 |
| random_late32 | recency | 14238 | 1.02761 | 15.148% | 1.000 |
| random_late32 | fixed_shrinkage | 14238 | 0.79646 | 0.000% | 0.650 |
| random_late32 | adaptive_gate | 14238 | 0.70022 | 0.006% | 0.106 |
| learnable_entry_first16 | recency | 1967 | 0.45245 | 5.435% | 1.000 |
| learnable_entry_first16 | fixed_shrinkage | 1967 | 0.47347 | 0.000% | 0.650 |
| learnable_entry_first16 | adaptive_gate | 1967 | 0.59029 | 0.000% | 0.247 |
| learnable_late32 | recency | 14061 | 0.33634 | 4.251% | 1.000 |
| learnable_late32 | fixed_shrinkage | 14061 | 0.40277 | 0.000% | 0.650 |
| learnable_late32 | adaptive_gate | 14061 | 0.34056 | 3.232% | 0.947 |

Metrics are equal-weight means over independent streams. Mixed-slice metrics average available stream slice means; counts are totals.
Confidently wrong is the fraction of ALL predictions with >=90% confidence that are wrong; conditional error rates are also saved in results.json.
Private mode labels and transition ages are used only for scoring. The gate sees past released feedback and a current public base margin.
All candidate fit records, validation scores, checkpoints, complete traces, per-stream metrics, paired differences and frozen snapshots are retained.
Fresh-process replay and SQLite resume evidence are in verification.json. Prior AR failures remain unchanged.
No claim about semantic embeddings, real-world agency, quantum hardware or general superiority follows from this toy task.
