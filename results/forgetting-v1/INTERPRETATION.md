# Fading helps stale associations, but fails qualification

The registered result is **FAIL overall**. Speed and fading-ablation gates
pass; entry value, stable retention and quality gates fail. A successful
technical replay does not change this scientific verdict. Full numerical
results and every component are in `RESULTS.md` and `results.json`.

## What improved

These are raw predictions in the first 16 steps of the FIRST visit to each
cue after its mapping changes, averaged over the three changing cases within
each of 24 held-out seeds. Lower loss and response steps are better.

| Policy | Changed-entry16 log loss | Penalized response steps |
|---|---:|---:|
| Fading memory | 0.556583 | 10.481836 |
| Original memory | 0.605053 | 12.495836 |
| Recent uniform, same records | 0.777281 | 21.993906 |
| Recency | 0.553019 | 10.404272 |
| Context cache | 0.601121 | 12.100743 |
| Recurrent | 0.589482 | 11.608992 |
| Fixed uniform routing | 0.621665 | 16.817063 |

Paired differences are fading minus comparator; negative favors fading.

| Comparator | Entry loss difference | Paired-seed 95% interval |
|---|---:|---|
| Original memory | -0.048470 | [-0.055360, -0.041867] |
| Context cache | -0.044538 | [-0.049831, -0.039359] |
| Recurrent | -0.032899 | [-0.042328, -0.024466] |
| Recent uniform | -0.220698 | [-0.234827, -0.206931] |
| Shuffled fading | -0.123804 | [-0.136309, -0.111767] |
| Recency | +0.003564 | [-0.001633, +0.008990] |

The same-record uniform ablation establishes that changing weights, beyond
merely choosing recent cue matches, contributed to the observed transition
gain. The shuffled control supports dependence on the past cue-feedback
association. Neither establishes a general architectural advantage.

Fading reaches the registered rolling-loss response threshold 2.014 feedback
steps sooner than original memory, interval [-2.571, -1.449]. The speed gate
requires a mean gain of at least two steps and an interval below zero; it does
not require the entire interval to exceed two steps. Compared with recency,
response is +0.078 steps, interval [-0.185, +0.330], within the frozen +0.5
tolerance. The endpoint is a capped, feedback-step loss measure, not wall time
or change-detection latency.

## Why it failed

1. **Entry value:** fading does not beat recency by the required 0.01 nats.
   Its aggregate difference is +0.003564 with an interval spanning zero.
   On noisy changes the difference is +0.019329, interval
   [+0.012734, +0.026057]; the upper bound exceeds the +0.02 case-harm limit.
2. **Stable retention:** recurring whole-stream loss rises from 0.447206
   for original memory to 0.491399 for fading. The increase is +0.044193,
   interval [+0.042924, +0.045408], above the +0.02 tolerance. Recurring
   return-first16 loss also rises +0.029048, interval
   [+0.024239, +0.033690]. Noisy-cue whole-stream and return checks fail too.
3. **Quality:** noisy-change entry Brier degradation versus recency is
   +0.008082, interval [+0.005306, +0.010892]. Its mean is below +0.01,
   but the upper bound exceeds that frozen tolerance. The other quality
   components, including confidently-wrong tolerances, pass.

The selected half-life is two matching-feedback examples. The average
effective neighbor count is 5.777 despite an average 59.938 retrieved rows.
Recent uniform achieves stable recurring loss 0.422009 and 83.085% accuracy;
fading gives 0.491399 and 76.595%. This same-record comparison shows the
observed stability-versus-adaptation tradeoff. It does not justify tuning a
new half-life on these exposed test streams.

The cache also beats fading on whole-stream loss in all five stable/changing
selection families. For example, on repeated changes it scores 0.476652
versus fading 0.503188, despite worse initial changed-entry loss. Entry-window
benefits must not be presented as whole-stream superiority.

## Confidence and negative controls

All ten random-label accuracies remain between 49.441% and 49.868%, satisfying
the frozen chance control. Fading's random-label raw loss is still 0.843320,
worse than constant chance loss log(2) = 0.693147. Its raw confidently-wrong
fraction is 2.279% of all predictions, with 4.551% high-confidence coverage
and about 50.042% error among those confident predictions. Better loss than
recency on this case does not make it well calibrated.

The unchanged confidence companion reduces random loss to 0.707955 and
high-confidence coverage to zero, passing its descriptive random-loss gate.
It cannot rescue the raw qualification failure. It also worsens stable fading
loss from 0.491399 to 0.503590. Stable recurring raw ECE is 0.045419;
confidently-wrong fraction is 0.181% versus original memory 0.092%.
All probabilities, calibration metrics, coverage and conditional errors are
retained in the machine-readable results.

## Evidence, related work and scope

The complete suite passed 194 tests before registration. The initial
development failure logs remain: a missing cue-separation setting and proposed
seed overlap were repaired before freeze or registered data generation.
Preregistration precedes all registered fitting; all 24 validation candidates
and selection were locked before any held-out generation. Fresh-process
verification passed: all expert and recurrent parameters, all 24 candidate
scores and selection, every raw/companion prediction and route, metrics,
slices, response endpoints, bootstrap intervals and gates reproduced exactly.
All ten policy reloads passed with future rows excluded. The complete proof
is in `verification.json`.

The test covers 192 streams and 1,920 policy traces, totaling 3,932,160 raw
predictions plus the same number of frozen confidence companions. Every
policy predicts before current feedback and evaluator-private fields are
read. Reload uses only step < 1,024 while 1,024 later feedback rows already
exist. Audit history is retained; fading never deletes the stored evidence.

This is measured-cue recall over four known fitted tasks. Training and
validation include the changing families. Intervals condition on one
coefficient world, expert fit and selection, omit their uncertainty, and are
not simultaneous bands. Search, cache and recurrent budgets differ. Results
are not directly comparable to earlier runs with different fits and seeds.
No semantic embeddings, new tasks, peptide weights, external APIs, services,
QPU jobs, live assistant changes or general AI claims are involved.

The primary-source reference check is in
`source/experiments/forgetting-20261010/RELATED_RESEARCH.md`.
[SAM-kNN](http[redacted-local-path] and
[ADWIN](http[redacted-local-path] are the closest
memory/drift references;
[Dynamic Weighted Majority](http[redacted-local-path] and
[Titans](http[redacted-local-path] provide broader ensemble and neural
memory context. This experiment does not implement those full methods or
establish novelty.

A possible next experiment is a separate stable store and fast store with a
selector driven only by released feedback, compared against an adaptive-window
baseline. This is an untested proposal, requiring a new protocol and fresh
seeds. No replacement default or follow-up experiment has been activated.
