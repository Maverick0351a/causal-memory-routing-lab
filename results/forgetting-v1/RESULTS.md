# Explicit forgetting on changing associations

- entry_value: **FAIL**
- speed: **PASS**
- stable_retention: **FAIL**
- quality: **FAIL**
- fading_ablation: **PASS**
- forgetting_qualified: **FAIL**
- gated_random_loss_control: **PASS**

All qualification gates use raw predictions. The confidence gate is a descriptive companion.
Changed-entry windows contain only the FIRST visit to each cue after each mapping revision, with age <16. Later recovered visits cannot dilute this endpoint.
Revision IDs, context IDs, changed-entry flags, answers and switches are evaluator-only and read after every policy predicts.

| Case | Policy | Raw loss | Gated loss | Raw accuracy | Brier | ECE | Confidently wrong fraction | Changed-entry16 loss | Response steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recurring | forgetting | 0.49140 | 0.50359 | 76.595% | 0.16098 | 0.04542 | 0.181% | n/a | n/a |
| recurring | recent_uniform | 0.42201 | 0.42990 | 83.085% | 0.13014 | 0.09378 | 0.061% | n/a | n/a |
| recurring | shuffled_forgetting | 0.54897 | 0.55935 | 72.290% | 0.18479 | 0.01715 | 0.441% | n/a | n/a |
| recurring | memory | 0.44721 | 0.45655 | 80.943% | 0.14120 | 0.08412 | 0.092% | n/a | n/a |
| recurring | memory_disabled | 0.51541 | 0.52753 | 74.605% | 0.17103 | 0.02775 | 0.350% | n/a | n/a |
| recurring | shuffled_memory | 0.53682 | 0.54981 | 73.863% | 0.17892 | 0.05932 | 0.055% | n/a | n/a |
| recurring | recency | 0.51541 | 0.52753 | 74.605% | 0.17103 | 0.02775 | 0.350% | n/a | n/a |
| recurring | recurrent | 0.49782 | 0.50821 | 76.613% | 0.16289 | 0.04872 | 0.144% | n/a | n/a |
| recurring | context_cache | 0.43965 | 0.44844 | 81.763% | 0.13786 | 0.09039 | 0.071% | n/a | n/a |
| recurring | fixed_uniform | 0.61928 | 0.63141 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| noisy_cues | forgetting | 0.49114 | 0.50322 | 76.613% | 0.16088 | 0.04473 | 0.183% | n/a | n/a |
| noisy_cues | recent_uniform | 0.42269 | 0.43067 | 83.042% | 0.13046 | 0.09310 | 0.061% | n/a | n/a |
| noisy_cues | shuffled_forgetting | 0.56338 | 0.57305 | 71.185% | 0.19085 | 0.01826 | 0.480% | n/a | n/a |
| noisy_cues | memory | 0.44750 | 0.45686 | 80.959% | 0.14133 | 0.08424 | 0.090% | n/a | n/a |
| noisy_cues | memory_disabled | 0.51541 | 0.52753 | 74.605% | 0.17103 | 0.02775 | 0.350% | n/a | n/a |
| noisy_cues | shuffled_memory | 0.53692 | 0.54988 | 73.863% | 0.17896 | 0.05847 | 0.055% | n/a | n/a |
| noisy_cues | recency | 0.51541 | 0.52753 | 74.605% | 0.17103 | 0.02775 | 0.350% | n/a | n/a |
| noisy_cues | recurrent | 0.49817 | 0.50855 | 76.573% | 0.16303 | 0.04839 | 0.142% | n/a | n/a |
| noisy_cues | context_cache | 0.45289 | 0.46213 | 80.794% | 0.14332 | 0.08595 | 0.106% | n/a | n/a |
| noisy_cues | fixed_uniform | 0.61928 | 0.63141 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| mapping_change | forgetting | 0.49434 | 0.50682 | 76.375% | 0.16223 | 0.04303 | 0.175% | 0.55587 | 10.52083 |
| mapping_change | recent_uniform | 0.44380 | 0.45192 | 81.230% | 0.13949 | 0.08074 | 0.081% | 0.77617 | 22.21875 |
| mapping_change | shuffled_forgetting | 0.54708 | 0.55762 | 72.323% | 0.18402 | 0.01708 | 0.439% | 0.66297 | 14.21875 |
| mapping_change | memory | 0.48233 | 0.49326 | 78.068% | 0.15609 | 0.07082 | 0.077% | 0.62608 | 13.55208 |
| mapping_change | memory_disabled | 0.51683 | 0.52928 | 74.542% | 0.17163 | 0.02671 | 0.348% | 0.56534 | 10.56250 |
| mapping_change | shuffled_memory | 0.53776 | 0.55094 | 73.712% | 0.17934 | 0.05708 | 0.055% | 0.56844 | 9.86458 |
| mapping_change | recency | 0.51683 | 0.52928 | 74.542% | 0.17163 | 0.02671 | 0.348% | 0.56534 | 10.56250 |
| mapping_change | recurrent | 0.49392 | 0.50447 | 76.744% | 0.16125 | 0.05078 | 0.140% | 0.59098 | 11.60417 |
| mapping_change | context_cache | 0.44539 | 0.45444 | 81.268% | 0.14029 | 0.08715 | 0.073% | 0.60130 | 12.23958 |
| mapping_change | fixed_uniform | 0.62064 | 0.63263 | 65.163% | 0.21611 | 0.02464 | 0.020% | 0.61610 | 15.72917 |
| repeated_changes | forgetting | 0.50319 | 0.51549 | 75.596% | 0.16584 | 0.03651 | 0.279% | 0.54769 | 10.13927 |
| repeated_changes | recent_uniform | 0.55793 | 0.56611 | 71.399% | 0.18851 | 0.02011 | 0.122% | 0.78268 | 21.66401 |
| repeated_changes | shuffled_forgetting | 0.54624 | 0.55694 | 72.251% | 0.18381 | 0.01563 | 0.415% | 0.68152 | 14.47639 |
| repeated_changes | memory | 0.53832 | 0.55130 | 73.556% | 0.17975 | 0.05182 | 0.081% | 0.59444 | 11.97422 |
| repeated_changes | memory_disabled | 0.51453 | 0.52688 | 74.748% | 0.17062 | 0.02822 | 0.366% | 0.54686 | 10.32516 |
| repeated_changes | shuffled_memory | 0.53719 | 0.55045 | 73.615% | 0.17908 | 0.05628 | 0.071% | 0.56339 | 10.93668 |
| repeated_changes | recency | 0.51453 | 0.52688 | 74.748% | 0.17062 | 0.02822 | 0.366% | 0.54686 | 10.32516 |
| repeated_changes | recurrent | 0.48949 | 0.49972 | 77.193% | 0.15939 | 0.05258 | 0.157% | 0.58859 | 11.61308 |
| repeated_changes | context_cache | 0.47665 | 0.48659 | 78.721% | 0.15348 | 0.07009 | 0.146% | 0.60442 | 12.13840 |
| repeated_changes | fixed_uniform | 0.62198 | 0.63365 | 64.935% | 0.21669 | 0.02283 | 0.022% | 0.62445 | 17.36101 |
| noisy_changes | forgetting | 0.50727 | 0.51951 | 75.234% | 0.16759 | 0.03509 | 0.271% | 0.56619 | 10.78541 |
| noisy_changes | recent_uniform | 0.58334 | 0.58997 | 68.868% | 0.19967 | 0.01806 | 0.149% | 0.77299 | 22.09896 |
| noisy_changes | shuffled_forgetting | 0.55790 | 0.56795 | 71.472% | 0.18862 | 0.01655 | 0.462% | 0.69667 | 15.09657 |
| noisy_changes | memory | 0.53868 | 0.55167 | 73.525% | 0.17990 | 0.05164 | 0.079% | 0.59464 | 11.96120 |
| noisy_changes | memory_disabled | 0.51453 | 0.52688 | 74.748% | 0.17062 | 0.02822 | 0.366% | 0.54686 | 10.32516 |
| noisy_changes | shuffled_memory | 0.53726 | 0.55051 | 73.641% | 0.17912 | 0.05571 | 0.071% | 0.56336 | 10.93483 |
| noisy_changes | recency | 0.51453 | 0.52688 | 74.748% | 0.17062 | 0.02822 | 0.366% | 0.54686 | 10.32516 |
| noisy_changes | recurrent | 0.48980 | 0.50004 | 77.150% | 0.15951 | 0.05277 | 0.155% | 0.58887 | 11.60973 |
| noisy_changes | context_cache | 0.48502 | 0.49557 | 78.105% | 0.15699 | 0.06784 | 0.151% | 0.59765 | 11.92425 |
| noisy_changes | fixed_uniform | 0.62198 | 0.63365 | 64.935% | 0.21669 | 0.02283 | 0.022% | 0.62445 | 17.36101 |
| novel_contexts | forgetting | 0.49781 | 0.51096 | 76.152% | 0.16364 | 0.04053 | 0.222% | n/a | n/a |
| novel_contexts | recent_uniform | 0.44188 | 0.45111 | 81.555% | 0.13867 | 0.08265 | 0.136% | n/a | n/a |
| novel_contexts | shuffled_forgetting | 0.59680 | 0.60272 | 68.313% | 0.20496 | 0.02917 | 0.566% | n/a | n/a |
| novel_contexts | memory | 0.46134 | 0.47200 | 79.803% | 0.14738 | 0.07295 | 0.142% | n/a | n/a |
| novel_contexts | memory_disabled | 0.51116 | 0.52460 | 75.008% | 0.16934 | 0.02919 | 0.275% | n/a | n/a |
| novel_contexts | shuffled_memory | 0.53300 | 0.54671 | 73.983% | 0.17741 | 0.05355 | 0.110% | n/a | n/a |
| novel_contexts | recency | 0.51116 | 0.52460 | 75.008% | 0.16934 | 0.02919 | 0.275% | n/a | n/a |
| novel_contexts | recurrent | 0.48211 | 0.49323 | 77.879% | 0.15628 | 0.05557 | 0.124% | n/a | n/a |
| novel_contexts | context_cache | 0.45896 | 0.46914 | 80.387% | 0.14582 | 0.08389 | 0.116% | n/a | n/a |
| novel_contexts | fixed_uniform | 0.62078 | 0.63332 | 65.267% | 0.21609 | 0.02641 | 0.022% | n/a | n/a |
| random_labels | forgetting | 0.84332 | 0.70796 | 49.797% | 0.30624 | 0.20331 | 2.279% | n/a | n/a |
| random_labels | recent_uniform | 0.77145 | 0.71124 | 49.841% | 0.28226 | 0.14753 | 0.387% | n/a | n/a |
| random_labels | shuffled_forgetting | 0.84434 | 0.70762 | 49.697% | 0.30664 | 0.20331 | 2.309% | n/a | n/a |
| random_labels | memory | 0.79923 | 0.71126 | 49.668% | 0.29228 | 0.17369 | 0.749% | n/a | n/a |
| random_labels | memory_disabled | 0.85524 | 0.70702 | 49.744% | 0.30992 | 0.21068 | 2.850% | n/a | n/a |
| random_labels | shuffled_memory | 0.79678 | 0.71118 | 49.784% | 0.29138 | 0.17112 | 0.722% | n/a | n/a |
| random_labels | recency | 0.85524 | 0.70702 | 49.744% | 0.30992 | 0.21068 | 2.850% | n/a | n/a |
| random_labels | recurrent | 0.83285 | 0.70929 | 49.441% | 0.30388 | 0.20159 | 1.392% | n/a | n/a |
| random_labels | context_cache | 0.79325 | 0.71129 | 49.868% | 0.29013 | 0.16722 | 0.614% | n/a | n/a |
| random_labels | fixed_uniform | 0.75611 | 0.71091 | 49.772% | 0.27649 | 0.13208 | 0.258% | n/a | n/a |
| no_feedback | forgetting | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| no_feedback | recent_uniform | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| no_feedback | shuffled_forgetting | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| no_feedback | memory | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| no_feedback | memory_disabled | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| no_feedback | shuffled_memory | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| no_feedback | recency | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| no_feedback | recurrent | 0.68782 | 0.66560 | 60.356% | 0.24455 | 0.09406 | 0.222% | n/a | n/a |
| no_feedback | context_cache | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |
| no_feedback | fixed_uniform | 0.61928 | 0.63164 | 65.251% | 0.21548 | 0.02580 | 0.020% | n/a | n/a |

## Paired primary comparisons

| Endpoint | Forgetting minus comparator | Difference | Paired-seed 95% interval |
|---|---|---:|---|
| changed_entry_first16_log_loss | memory | -0.048470 | [-0.055360, -0.041867] |
| changed_entry_first16_log_loss | recent_uniform | -0.220698 | [-0.234827, -0.206931] |
| changed_entry_first16_log_loss | shuffled_forgetting | -0.123804 | [-0.136309, -0.111767] |
| changed_entry_first16_log_loss | recency | +0.003564 | [-0.001633, +0.008990] |
| changed_entry_first16_log_loss | context_cache | -0.044538 | [-0.049831, -0.039359] |
| changed_entry_first16_log_loss | recurrent | -0.032899 | [-0.042328, -0.024466] |
| changed_entry_first8_log_loss | memory | -0.018296 | [-0.027756, -0.009716] |
| changed_entry_first8_log_loss | recent_uniform | -0.179579 | [-0.194851, -0.163720] |
| changed_entry_first8_log_loss | shuffled_forgetting | -0.103909 | [-0.125290, -0.083520] |
| changed_entry_first8_log_loss | recency | +0.013955 | [+0.003999, +0.024090] |
| changed_entry_first8_log_loss | context_cache | -0.061907 | [-0.070867, -0.053128] |
| changed_entry_first8_log_loss | recurrent | -0.051632 | [-0.065779, -0.037780] |
| change_delay_penalized_steps | memory | -2.014000 | [-2.570637, -1.448647] |
| change_delay_penalized_steps | recent_uniform | -11.512070 | [-12.803478, -10.330006] |
| change_delay_penalized_steps | shuffled_forgetting | -4.115401 | [-4.766727, -3.506682] |
| change_delay_penalized_steps | recency | +0.077564 | [-0.184550, +0.330160] |
| change_delay_penalized_steps | context_cache | -1.618906 | [-1.983377, -1.289662] |
| change_delay_penalized_steps | recurrent | -1.127156 | [-1.478011, -0.760527] |

Primary cases are once-changing, repeatedly changing and repeatedly changing with noisy cues. Average cases WITHIN each of24 seeds before bootstrapping10000 seed-block resamples.
Response is first endpoint of an8-label raw-loss window <=.55, capped64 and truncated at next switch/end; unrecovered=64. This measures feedback steps, not wall time or a change detector.
All candidate families use the SAME validation objective: .5 stable return-first16 loss + .5 changed-entry-first16 loss. Recurrent training includes stable and changing streams.
Intervals condition on one coefficient world, expert fit and validation selection, exclude fit/selection uncertainty and are not simultaneous confidence bands.

## Every frozen component

### entry_value

- memory_minimum_gain: PASS
- memory_upper_below_zero: PASS
- recency_minimum_gain: FAIL
- recency_upper_below_zero: FAIL
- context_cache_minimum_gain: PASS
- context_cache_upper_below_zero: PASS
- mapping_change_recency_no_hidden_case_harm: PASS
- mapping_change_context_cache_no_hidden_case_harm: PASS
- repeated_changes_recency_no_hidden_case_harm: PASS
- repeated_changes_context_cache_no_hidden_case_harm: PASS
- noisy_changes_recency_no_hidden_case_harm: FAIL
- noisy_changes_context_cache_no_hidden_case_harm: PASS

### speed

- memory_minimum_two_steps: PASS
- memory_upper_below_zero: PASS
- recency_not_slower: PASS
- context_cache_not_slower: PASS

### stable_retention

- recurring_raw_log_loss: FAIL
- recurring_return_first16_log_loss: FAIL
- recurring_confident_errors: PASS
- noisy_cues_raw_log_loss: FAIL
- noisy_cues_return_first16_log_loss: FAIL
- noisy_cues_confident_errors: PASS

### quality

- mapping_change_recency_brier: PASS
- mapping_change_recency_confidently_wrong_rate: PASS
- mapping_change_context_cache_brier: PASS
- mapping_change_context_cache_confidently_wrong_rate: PASS
- repeated_changes_recency_brier: PASS
- repeated_changes_recency_confidently_wrong_rate: PASS
- repeated_changes_context_cache_brier: PASS
- repeated_changes_context_cache_confidently_wrong_rate: PASS
- noisy_changes_recency_brier: FAIL
- noisy_changes_recency_confidently_wrong_rate: PASS
- noisy_changes_context_cache_brier: PASS
- noisy_changes_context_cache_confidently_wrong_rate: PASS
- forgetting_random_accuracy: PASS
- recent_uniform_random_accuracy: PASS
- shuffled_forgetting_random_accuracy: PASS
- memory_random_accuracy: PASS
- memory_disabled_random_accuracy: PASS
- shuffled_memory_random_accuracy: PASS
- recency_random_accuracy: PASS
- recurrent_random_accuracy: PASS
- context_cache_random_accuracy: PASS
- fixed_uniform_random_accuracy: PASS
- random_loss_vs_recency: PASS

### fading_ablation

- recent_uniform_minimum_gain: PASS
- recent_uniform_upper_below_zero: PASS
- shuffled_forgetting_minimum_gain: PASS
- shuffled_forgetting_upper_below_zero: PASS

## Method and scope

Fading retrieval searches the latest1024 eligible public-feedback records, keeps the newest64 cue matches within radius1.25, and weights their ranks by2^(-rank/half_life). Validation selects half-life2,8 or32 matching-feedback observations. No hidden boundary or reset signal is used.
recent_uniform uses those same records, radius, blend, count warmup and ordering with uniform weights; it isolates fading from simply preferring recent matches. Shuffled forgetting causally permutes key-feedback associations with the same key multiset, counts and effective sample sizes.
Original memory retains its distance-nearest retrieval algorithm, cache retains its centroid/error-EWMA algorithm; both are reselected on fresh validation. Disabled memory equals recency as an intentional duplicate.
Audit history is retained; forgetting changes prediction weights rather than physically deleting stored evidence. Effective neighbor counts are recorded. Search, cache and recurrent budgets differ.
{
  "shared_expert_parameters": 36,
  "recurrent_parameters": 1036,
  "recurrent_hidden_state": 24,
  "searchable_feedback_records": 1024,
  "fading_neighbors": 64,
  "cache_slots": 8,
  "equal_compute_or_capacity": false
}
The embedding remains identity encoding of six measured cue values over four known fitted tasks. This is a synthetic concept-drift experiment, not semantic memory, new-task learning, novelty proof, SAM-kNN/ADWIN/Titans reproduction or general architecture superiority.
Sources and exact distinctions are in [the related-work note](../../experiments/forgetting-20261010/RELATED_RESEARCH.md). Fresh-process refit/reselection/replay and all-policy strict prior-feedback resumes are in verification.json.
No keys, downloaded models, API calls, services or QPU jobs are required; the Python process guard is not an OS sandbox. Earlier results, failed gates and live defaults remain unchanged.
