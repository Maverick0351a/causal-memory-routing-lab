# Recurring-context memory experiment

Functional recall: **PASS**.
Faster return adaptation: **PASS**.
Novel/stale/random quality: **PASS**.
Overall memory qualification: **PASS**.
Extra value over conventional context cache: **PASS**.
Descriptive gated random-loss control: **PASS**.

All qualification endpoints use raw probabilities; confidence shrinkage cannot rescue failed recall, speed or relative quality.
New random cue-to-task mappings are sampled in every stream. Current answers and private context IDs never enter inference or memory.
Memory-disabled and recency are intentionally the same policy; they are one independent baseline, with a duplicate equality check.

| Case | Policy | Raw loss | Gated loss | Raw Brier | Accuracy | Raw confident errors | Gated confident errors |
|---|---|---:|---:|---:|---:|---:|---:|
| recurring | memory | 0.41592 | 0.42380 | 0.12800 | 83.134% | 0.100% | 0.037% |
| recurring | memory_disabled | 0.51153 | 0.52346 | 0.16955 | 74.943% | 0.431% | 0.035% |
| recurring | shuffled_memory | 0.62972 | 0.63629 | 0.22036 | 63.871% | 0.079% | 0.000% |
| recurring | recency | 0.51153 | 0.52346 | 0.16955 | 74.943% | 0.431% | 0.035% |
| recurring | recurrent | 0.51051 | 0.51897 | 0.16858 | 75.216% | 0.201% | 0.016% |
| recurring | context_cache | 0.43375 | 0.44236 | 0.13567 | 82.107% | 0.126% | 0.043% |
| recurring | fixed_uniform | 0.62263 | 0.63426 | 0.21709 | 64.679% | 0.031% | 0.000% |
| noisy_cues | memory | 0.41676 | 0.42472 | 0.12837 | 83.057% | 0.102% | 0.039% |
| noisy_cues | memory_disabled | 0.51153 | 0.52346 | 0.16955 | 74.943% | 0.431% | 0.035% |
| noisy_cues | shuffled_memory | 0.62978 | 0.63621 | 0.22036 | 63.926% | 0.079% | 0.000% |
| noisy_cues | recency | 0.51153 | 0.52346 | 0.16955 | 74.943% | 0.431% | 0.035% |
| noisy_cues | recurrent | 0.51070 | 0.51921 | 0.16866 | 75.203% | 0.203% | 0.016% |
| noisy_cues | context_cache | 0.44512 | 0.45422 | 0.14036 | 81.211% | 0.140% | 0.039% |
| noisy_cues | fixed_uniform | 0.62263 | 0.63426 | 0.21709 | 64.679% | 0.031% | 0.000% |
| novel_contexts | memory | 0.43823 | 0.44730 | 0.13749 | 81.541% | 0.149% | 0.037% |
| novel_contexts | memory_disabled | 0.50952 | 0.52213 | 0.16869 | 75.171% | 0.372% | 0.024% |
| novel_contexts | shuffled_memory | 0.62536 | 0.62924 | 0.21791 | 64.819% | 0.222% | 0.000% |
| novel_contexts | recency | 0.50952 | 0.52213 | 0.16869 | 75.171% | 0.372% | 0.024% |
| novel_contexts | recurrent | 0.49644 | 0.50673 | 0.16265 | 76.217% | 0.149% | 0.026% |
| novel_contexts | context_cache | 0.45599 | 0.46593 | 0.14484 | 80.461% | 0.128% | 0.016% |
| novel_contexts | fixed_uniform | 0.62089 | 0.63339 | 0.21625 | 65.125% | 0.024% | 0.000% |
| mapping_change | memory | 0.50399 | 0.50287 | 0.16564 | 75.812% | 0.122% | 0.018% |
| mapping_change | memory_disabled | 0.51373 | 0.52559 | 0.17045 | 74.758% | 0.437% | 0.031% |
| mapping_change | shuffled_memory | 0.62738 | 0.63445 | 0.21935 | 63.997% | 0.077% | 0.000% |
| mapping_change | recency | 0.51373 | 0.52559 | 0.17045 | 74.758% | 0.437% | 0.031% |
| mapping_change | recurrent | 0.50822 | 0.51700 | 0.16764 | 75.403% | 0.199% | 0.028% |
| mapping_change | context_cache | 0.44070 | 0.44952 | 0.13861 | 81.510% | 0.122% | 0.047% |
| mapping_change | fixed_uniform | 0.62052 | 0.63244 | 0.21613 | 65.023% | 0.028% | 0.000% |
| random_labels | memory | 0.77352 | 0.71050 | 0.28281 | 49.894% | 0.482% | 0.000% |
| random_labels | memory_disabled | 0.86013 | 0.70630 | 0.31105 | 49.896% | 3.320% | 0.000% |
| random_labels | shuffled_memory | 0.77483 | 0.71109 | 0.28337 | 49.890% | 0.456% | 0.000% |
| random_labels | recency | 0.86013 | 0.70630 | 0.31105 | 49.896% | 3.320% | 0.000% |
| random_labels | recurrent | 0.83836 | 0.70856 | 0.30523 | 49.786% | 1.742% | 0.000% |
| random_labels | context_cache | 0.79583 | 0.71057 | 0.29080 | 49.974% | 0.812% | 0.000% |
| random_labels | fixed_uniform | 0.75845 | 0.71019 | 0.27717 | 49.807% | 0.356% | 0.000% |
| no_feedback | memory | 0.62263 | 0.63382 | 0.21709 | 64.679% | 0.031% | 0.000% |
| no_feedback | memory_disabled | 0.62263 | 0.63382 | 0.21709 | 64.679% | 0.031% | 0.000% |
| no_feedback | shuffled_memory | 0.62263 | 0.63382 | 0.21709 | 64.679% | 0.031% | 0.000% |
| no_feedback | recency | 0.62263 | 0.63382 | 0.21709 | 64.679% | 0.031% | 0.000% |
| no_feedback | recurrent | 0.67367 | 0.65665 | 0.23839 | 61.684% | 0.252% | 0.000% |
| no_feedback | context_cache | 0.62263 | 0.63382 | 0.21709 | 64.679% | 0.031% | 0.000% |
| no_feedback | fixed_uniform | 0.62263 | 0.63382 | 0.21709 | 64.679% | 0.031% | 0.000% |

## Primary return comparisons

| Endpoint | Memory minus baseline | Mean difference | Paired-seed 95% interval |
|---|---|---:|---|
| return_first16_log_loss | recency | -0.142088 | [-0.147869, -0.136350] |
| return_first16_log_loss | shuffled_memory | -0.224854 | [-0.231939, -0.217721] |
| return_first16_log_loss | recurrent | -0.201261 | [-0.215841, -0.187671] |
| return_first16_log_loss | context_cache | -0.033344 | [-0.036954, -0.029750] |
| return_delay_penalized_steps | recency | -3.131953 | [-3.629314, -2.647128] |
| return_delay_penalized_steps | shuffled_memory | -10.406256 | [-11.380229, -9.442639] |
| return_delay_penalized_steps | recurrent | -5.714694 | [-6.630135, -4.866562] |
| return_delay_penalized_steps | context_cache | -0.735746 | [-0.892704, -0.582648] |
| raw_log_loss | recency | -0.095185 | [-0.098134, -0.092315] |
| raw_log_loss | shuffled_memory | -0.213410 | [-0.218992, -0.207786] |
| raw_log_loss | recurrent | -0.094264 | [-0.105650, -0.083340] |
| raw_log_loss | context_cache | -0.023093 | [-0.024991, -0.021273] |

Primary cases are recurring/noisy cues, averaged within each seed before bootstrapping 24 seed blocks. Steps and cases are not independent replicates.
Require first16 loss improvement >=.02 nats versus recency/disabled, shuffle and recurrent; return delay >=2 steps faster versus recency and recurrent. Every paired upper bound must be below zero.
Extra value gate versus the simple cache requires >=.01 nats overall raw loss improvement. It is separate from whether memory helps versus context-blind learning.
Intervals condition on one expert fit and one selection and exclude fit/selection uncertainty. They are not simultaneous confidence bands.

## Return behavior

| Case | Policy | First16 return loss | Penalized response steps | Recovered fraction | Total return steps |
|---|---|---:|---:|---:|---:|
| recurring | memory | 0.41857 | 7.509 | 100.000% | 9610 |
| recurring | memory_disabled | 0.56070 | 10.638 | 99.333% | 9610 |
| recurring | shuffled_memory | 0.64322 | 17.906 | 97.505% | 9610 |
| recurring | recency | 0.56070 | 10.638 | 99.333% | 9610 |
| recurring | recurrent | 0.62007 | 13.231 | 99.506% | 9610 |
| recurring | context_cache | 0.43698 | 7.972 | 100.000% | 9610 |
| recurring | fixed_uniform | 0.62278 | 16.284 | 98.152% | 9610 |
| noisy_cues | memory | 0.41865 | 7.504 | 100.000% | 9610 |
| noisy_cues | memory_disabled | 0.56070 | 10.638 | 99.333% | 9610 |
| noisy_cues | shuffled_memory | 0.64371 | 17.919 | 97.505% | 9610 |
| noisy_cues | recency | 0.56070 | 10.638 | 99.333% | 9610 |
| noisy_cues | recurrent | 0.61967 | 13.210 | 99.506% | 9610 |
| noisy_cues | context_cache | 0.46692 | 8.512 | 100.000% | 9610 |
| noisy_cues | fixed_uniform | 0.62278 | 16.284 | 98.152% | 9610 |

Response uses the first8-label window below .55 log loss. Startup/first visits excluded; search truncates at next switch/end; unrecovered events score64. It is a stochastic response metric, not a change detector.

## Frozen components

### recall

- memory_disabled_improvement: PASS
- memory_disabled_upper_below_zero: PASS
- shuffled_memory_improvement: PASS
- shuffled_memory_upper_below_zero: PASS
- recency_improvement: PASS
- recency_upper_below_zero: PASS
- recurrent_improvement: PASS
- recurrent_upper_below_zero: PASS

### speed

- recency_two_steps_faster: PASS
- recency_upper_below_zero: PASS
- recurrent_two_steps_faster: PASS
- recurrent_upper_below_zero: PASS

### quality

- novel_contexts_loss_preserved: PASS
- novel_contexts_brier_preserved: PASS
- novel_contexts_confident_errors_preserved: PASS
- mapping_change_loss_preserved: PASS
- mapping_change_brier_preserved: PASS
- mapping_change_confident_errors_preserved: PASS
- random_labels_loss_preserved: PASS
- random_labels_brier_preserved: PASS
- random_labels_confident_errors_preserved: PASS
- memory_random_accuracy_near_chance: PASS
- memory_disabled_random_accuracy_near_chance: PASS
- shuffled_memory_random_accuracy_near_chance: PASS
- recency_random_accuracy_near_chance: PASS
- recurrent_random_accuracy_near_chance: PASS
- context_cache_random_accuracy_near_chance: PASS
- fixed_uniform_random_accuracy_near_chance: PASS

## Budgets and scope

{
  "shared_expert_parameters": 36,
  "recurrent_parameters": 1036,
  "recurrent_hidden_state": 24,
  "searchable_feedback_records": 1024,
  "cache_slots": 8,
  "equal_compute_or_capacity": false
}
These methods have different capacity and compute. Memory searches at most1024 released feedback records; transcript and RAM audit index retain the complete2048-step episode. Cache holds8 summaries. The recurrent network has1036 weights and24 hidden values, and extra offline training.
Three recurrent candidates each see98304 examples and3072 updates; selection scores8 memory,4 cache and9 recurrent candidates on independent validation streams. Full curves and every checkpoint are saved.
Confidence checkpoint/normalization are frozen. Companion gating changes confidence, not recognition; primary recall and speed use raw probabilities.
Memory search keys are an identity encoding of six measured cue values. This does not establish semantic retrieval, real-world sensor competence or algorithmic novelty.
No-feedback retains learned experts and the neural prior; only feedback-driven state/retrieval is removed. Chance is required only on independent random labels.
Every raw/gated score, ECE, confidence coverage and conditional high-confidence error is retained in results.json. Source/selection hashes precede held-out generation.
Fresh-process refit/reselection/replay and strict prior-feedback SQLite resumes are in verification.json. Existing artifacts/defaults stay preserved. No downloads, keys, services or QPU jobs; Python guard is not an OS sandbox.
