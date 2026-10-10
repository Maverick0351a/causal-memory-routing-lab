# What this memory experiment establishes

The registered v2 run passed functional recall, response speed, aggregate
novel/stale/random quality, and the separate value gate against a conventional
context cache. It supports a specific local result: retrieving earlier released
feedback by a measured cue helps choose among four already learned tasks when
the same cue-to-task association returns. It does not establish semantic memory,
new task discovery, an autoregressive architecture advantage or general AI.

## Returning contexts

The primary comparison averages recurring and noisy-cue cases within each of
24 test seed blocks. All results below use raw probabilities.

| Comparator | First16 return loss difference, memory minus comparator | Paired-seed 95% interval | Penalized response-step difference |
|---|---:|---|---:|
| Recency / disabled memory | -0.142088 | [-0.147869, -0.136350] | -3.131953 |
| Shuffled memory | -0.224854 | [-0.231939, -0.217721] | -10.406256 |
| Recurrent router | -0.201261 | [-0.215841, -0.187671] | -5.714694 |
| Conventional context cache | -0.033344 | [-0.036954, -0.029750] | -0.735746 |

Overall primary-case loss improved by 0.023093 nats versus the cache, with
interval [-0.024991, -0.021273], passing its separate 0.01-nat value threshold.
The response metric measures feedback steps to an eight-label loss window,
with minimum endpoint seven and an unrecovered penalty of 64. It is not runtime
latency or a change detector. Memory's average response was about 7.51 steps,
close to that measurement floor. Disabled memory and recency are one baseline.

Recurring accuracy was 83.134% for memory, 74.943% for recency, 75.216% for the
recurrent model and 82.107% for the cache. Raw confidently wrong fractions of
all recurring predictions were 0.100%, 0.431%, 0.201% and 0.126%, respectively.
Memory's raw recurring ECE was 0.0862 versus recency's 0.0251: better loss and
accuracy do not imply every calibration metric improved.

## Limits that the aggregate pass does not remove

After the cue-to-task mapping changed, whole-case raw loss was 0.50399 for
memory, 0.51373 for recency and 0.44070 for the cache. The first16 steps of
post-change visits expose a larger stale-memory cost: memory loss 0.63188,
recency 0.55733 and cache 0.48265. These slices were saved in the frozen
evaluation; they were not extra qualification gates. The quality pass compares
whole-case averages to recency, so it does not establish harmless transitions
or superiority to the cache under changed mappings.

Random labels give memory 49.894% accuracy, as expected. Its raw loss 0.77352
still exceeds uniform guessing's 0.69315; raw confidently wrong fraction is
0.482%, compared with recency's 3.320%. The unchanged companion confidence gate
reduces memory loss to 0.71050 and confidently wrong fraction to zero. This
companion check is descriptive and cannot rescue raw gates. On recurring data,
the same companion worsens memory loss from 0.41592 to 0.42380 while reducing
confident mistakes. There is a confidence-versus-sharpness tradeoff.

No-feedback memory is empty and equals recency/disabled/shuffled routing. Its
64.679% accuracy is permitted: the fitted expert ensemble already contains
task information. Independent random labels are the chance-level control.
Novel contexts are new cues over the same four known tasks, with feedback
available within each block. They are not unseen target functions.

The methods have different budgets: memory searches the latest 1,024 eligible
feedback rows and retains the 2,048-row audit transcript; cache stores eight
summaries; the recurrent model has 1,036 weights and 24 hidden values plus extra
offline training. This is not a matched-capacity or compute-efficiency result.
Intervals condition on one expert fit, coefficient world and validation selection;
they exclude fit/selection uncertainty and are not simultaneous confidence bands.

## Reproducibility and retained failure

The initial v1 completed its traces but failed JSON summary export on a NumPy
boolean. It remains incomplete and preserved. V2 uses a separate package,
fresh seeds in every stochastic role, unchanged scientific settings and a
native-bool export repair. The copy also introduced a display-only separator
encoding artifact in the frozen HTML generator; no scientific values changed.
The derived comparison-review.html corrects only those typographic separators;
the originally generated comparison.html is retained with its recorded hash.
The source comparison is saved as repair-audit.json in the experiment directory.

The development suite passed 150 tests. Full registered fresh-process verification
passed: expert refit, all three recurrent refits, all 21 validation candidates,
and 144 streams / 1,008 policy traces / 2,064,384 raw predictions plus the same
number of gated companions reproduced exactly. All four stateful reloads at
step 1,024 excluded 1,024 future database rows. The proof is verification.json.
All 3,025 prior lab files plus three original project files remain unchanged.
Both original and replacement source locks still match. No earlier failed routing or
sensor-transfer gate is superseded. Existing defaults and live assistants remain
unchanged. No access token, API call, download, service or quantum job was needed.

The next useful experiment is a fresh protocol targeting stale associations:
compare explicit forgetting against this unchanged memory and the simple cache,
with gates on post-change entry windows as well as whole-case averages. This is
a proposed next experiment, not an additional run or a deployed change.
