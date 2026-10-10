# Two-speed memory: six independent synthetic worlds

**Qualification: FAIL**.

All thresholds were frozen before registered generation. Raw forecasts decide every scientific gate; the unchanged confidence companion is descriptive only.

| Component | Result |
|---|---|
| adaptation | FAIL |
| retention | FAIL |
| quality | FAIL |
| incremental_value | FAIL |
| causal_controls | PASS |
| two_speed_qualified | FAIL |

Differences below are two-speed minus the comparator; lower is better. Intervals resample six independent world means, after averaging all eight stream seeds and relevant cases within each world. These are exploratory pointwise intervals with only six world clusters.

| Endpoint | Comparator | Difference | 95% interval |
|---|---|---:|---:|
| stable_loss | stable_only | 0.003741 | [0.003002, 0.004346] |
| stable_return_loss | stable_only | 0.010870 | [0.009724, 0.012175] |
| changed_loss | stable_only | -0.211985 | [-0.217972, -0.206488] |
| change_delay | stable_only | -17.644806 | [-18.292330, -17.067676] |
| balanced_loss | stable_only | -0.100557 | [-0.103526, -0.097770] |
| stable_loss | fast_only | -0.050849 | [-0.053545, -0.048593] |
| stable_return_loss | fast_only | -0.088338 | [-0.093307, -0.083908] |
| changed_loss | fast_only | 0.038491 | [0.036864, 0.039909] |
| change_delay | fast_only | 1.331534 | [1.067822, 1.631098] |
| balanced_loss | fast_only | -0.024923 | [-0.027459, -0.022082] |
| stable_loss | recent_uniform | 0.025508 | [0.024463, 0.026673] |
| stable_return_loss | recent_uniform | 0.016654 | [0.015749, 0.017604] |
| changed_loss | recent_uniform | -0.147243 | [-0.151603, -0.142936] |
| change_delay | recent_uniform | -9.034848 | [-9.818773, -8.272589] |
| balanced_loss | recent_uniform | -0.065295 | [-0.067216, -0.063399] |
| stable_loss | fading | -0.020754 | [-0.037953, -0.003263] |
| stable_return_loss | fading | -0.029794 | [-0.047847, -0.010925] |
| changed_loss | fading | 0.004880 | [-0.015291, 0.024645] |
| change_delay | fading | 0.202941 | [-0.654776, 0.987641] |
| balanced_loss | fading | -0.012457 | [-0.015714, -0.008997] |
| stable_loss | adaptive_window | 0.016513 | [0.015509, 0.017703] |
| stable_return_loss | adaptive_window | 0.007511 | [0.006364, 0.008865] |
| changed_loss | adaptive_window | -0.021484 | [-0.024932, -0.017366] |
| change_delay | adaptive_window | -1.018523 | [-1.269853, -0.777701] |
| balanced_loss | adaptive_window | -0.006987 | [-0.008869, -0.004695] |
| stable_loss | recency | -0.066639 | [-0.069238, -0.064655] |
| stable_return_loss | recency | -0.109568 | [-0.115767, -0.104609] |
| changed_loss | recency | 0.031844 | [0.029533, 0.034121] |
| change_delay | recency | 1.231024 | [1.028637, 1.493803] |
| balanced_loss | recency | -0.038862 | [-0.041739, -0.035985] |
| stable_loss | context_cache | -0.002708 | [-0.022605, 0.008901] |
| stable_return_loss | context_cache | -0.009589 | [-0.027821, 0.001566] |
| changed_loss | context_cache | -0.016420 | [-0.027615, 0.000940] |
| change_delay | context_cache | -0.493449 | [-1.149188, 0.342345] |
| balanced_loss | context_cache | -0.013004 | [-0.014157, -0.011744] |

## Calibration and controls

| Policy | Random accuracy | Raw loss | Gated loss | Raw ECE | Raw confidently wrong |
|---|---:|---:|---:|---:|---:|
| two_speed | 49.7823% | 0.804383 | 0.710355 | 0.173508 | 1.2268% |
| stable_only | 49.8454% | 0.801230 | 0.710187 | 0.168227 | 1.1922% |
| fast_only | 49.6175% | 0.847608 | 0.707817 | 0.208295 | 2.5289% |
| recent_uniform | 49.9919% | 0.778338 | 0.711528 | 0.151041 | 0.5107% |
| fading | 49.6623% | 0.826919 | 0.709459 | 0.193434 | 1.8290% |
| adaptive_window | 49.9695% | 0.787656 | 0.711472 | 0.161160 | 0.6185% |
| shuffled_two_speed | 49.9186% | 0.788431 | 0.710910 | 0.160777 | 0.7202% |
| recency | 49.6216% | 0.856159 | 0.707093 | 0.213152 | 2.9297% |
| context_cache | 49.7864% | 0.804185 | 0.710685 | 0.176052 | 1.0539% |
| recurrent | 49.6989% | 0.791398 | 0.711897 | 0.169148 | 0.5107% |
| fixed_uniform | 49.9308% | 0.756883 | 0.711101 | 0.129764 | 0.3011% |

## Boundaries and budgets

The stable and fast stores are disjoint logical views of the same prior-feedback index: older 192 and latest 64 records. Prediction searches at most 256 rows; the persistent audit log and its RAM replay index retain full history. This is not a hard total-RAM cap. Stable routing uses up to 64 matching records; fast uses up to 16 with half-life two. Stable-only and fast-only intentionally have different branch capacities. Adaptive-window uses three candidate sizes (8/32/64 matching records) under the same 256-row search cap. The selectors store at most eight measured-cue prototypes with two/three EMA forecast losses. Context cache and recurrent budgets differ; equal compute is not established.

The adaptive-window method is a prequential mixture over window sizes, not a reproduction of ADWIN. The dual-store direction is established by [SAM-kNN](https://www.ijcai.org/Proceedings/2017/0690.pdf); this study does not reproduce SAM-kNN or establish novelty.

Policies receive only six measured cue values and eight readings, their own prior released feedback and frozen settings. No hidden context IDs, switch ages, generator seeds, current labels or future feedback enter prediction. SQLite reload is restricted to step < cutoff. Identity cue embedding is not semantic memory; the four logistic tasks are fitted and known. New coefficient worlds vary task geometry, not task family.

No API keys, downloads, peptide weights, model inference services, QPU jobs or paid compute. The Python socket guard is not an OS sandbox. Earlier results and v0.1.0 remain frozen.

See results.json for every metric, world, stream, candidate and gate; protocol and source hashes are in preregistration.json. Replay evidence is reported separately in verification.json once completed.
