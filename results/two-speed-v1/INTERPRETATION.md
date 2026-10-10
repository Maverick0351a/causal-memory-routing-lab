# Two-speed memory: a useful tradeoff, failed qualification

The registered experiment **fails overall qualification**. It demonstrates
that combining the stable and fast views can improve their tradeoff, but it
does not meet the joint adaptation, retention, quality and incremental-value
requirements. Causal controls pass. No gate, seed or algorithm was changed
after evaluation started.

Six independently generated coefficient worlds each have fresh expert fits,
two recurrent initializations, validation selection and eight held-out stream
seeds across eight cases: **384 streams, 4,224 policy traces and 4,325,376 raw
predictions**, plus the same number of frozen confidence companions. Raw
probabilities decide qualification. Independent-process verification is
recorded separately in [verification.json](verification.json); passing that
check establishes reproduction, not scientific superiority.

## What improved

Two-speed memory improves changed-entry loss over stable-only by 0.211985 nats
and recovers 17.644806 feedback steps sooner (difference interval
[-18.292330, -17.067676] steps). Relative to fast-only, its stable whole-stream
loss improves by 0.050849 nats, while its changed-entry loss worsens by
0.038491. Combining the branches therefore offers a measured tradeoff rather
than dominating both endpoints.

| Policy | Stable whole-stream loss | Changed-entry first16 loss |
|---|---:|---:|
| Two-speed | 0.459898 | 0.582015 |
| Stable-only | 0.456157 | 0.794000 |
| Fast-only | 0.510747 | 0.543524 |
| Uniform recent matching records | 0.434390 | 0.729259 |
| Adaptive-window mixture | 0.443385 | 0.603500 |
| Global recency | 0.526537 | 0.550171 |
| Context cache | 0.462606 | 0.598436 |

Loss uses natural-log nats; lower is better. Stable loss averages the two
stable cases; changed-entry loss averages the three changing cases. The
balanced **selection/value** endpoint instead uses stable **return-first16**
loss plus changed-entry-first16 loss, with equal weight. Do not compute that
endpoint from the whole-stream stable column above.

Two-speed's balanced entry advantage over adaptive-window is 0.006987 nats,
with interval [0.004695, 0.008869] for the improvement. Every world has a
positive observed improvement. This is evidence of a small gain in this setup,
but its lower improvement bound misses the frozen minimum 0.005 requirement.
The **incremental-value gate fails**; a positive mean or interval excluding
zero does not replace that preregistered margin.

## Why qualification fails

- **Adaptation:** compared with recency, two-speed changed-entry loss is worse
  by 0.031844 nats (interval [0.029533, 0.034121]) and recovery is 1.231024
  steps slower ([1.028637, 1.493803]). Both exceed the frozen tolerances.
  Fading comparisons also fail their upper-bound noninferiority checks.
- **Retention:** stable whole-stream loss is worse than uniform by 0.025508
  nats ([0.024463, 0.026673]), exceeding the 0.02 tolerance. Stable return
  loss passes that particular tolerance; the whole-stream failure remains.
- **Quality:** mean whole-stream ECE over the five selection cases exceeds
  recency by 0.025280 ([0.023945, 0.026666]), above the 0.02 tolerance. Brier
  excess over adaptive-window has mean 0.009971 but upper bound 0.010157,
  narrowly above its 0.01 tolerance. Confident-error tolerances pass.
- **Incremental value:** the adaptive-window minimum-gain interval gate fails
  as described above. No threshold was relaxed after seeing this result.

Whole-stream and first-entry results answer different questions. The complete
case-level metrics, paired world differences and every gate component remain
in [results.json](results.json); all 114 validation candidate scores and the
six locked selections remain in [selection.json](selection.json).

## Controls and confidence

Two-speed beats the shuffled-cue control on balanced entry loss by 0.119952
nats; all eleven random-label accuracies fall within the frozen chance range.
The development suite tests that every forecast is committed before private
row fields are read, private metadata/current/future label changes cannot
alter a current prediction, and no-feedback does not update memory/selection.

Passing random accuracy is **not calibration**. Two-speed random accuracy is
49.7823%, while raw log loss is 0.804383 (chance is log(2) = 0.693147), ECE is
0.173508, and confidently-wrong predictions are 1.2268% of all predictions.
The unchanged confidence companion reduces random loss to 0.710355, but this
descriptive adjustment cannot rescue the raw qualification failure. Coverage
and conditional confident error are retained in the machine-readable record.

## Scope and practical conclusion

The stable and fast stores are disjoint FIFO views (older 192/latest 64) of
the same prior-feedback log, not persistent compressed concept banks. The
selector uses only already-committed forecast losses and measured cue
similarity. Prediction searches at most 256 released rows; full audit history
and its replay index are unbounded. Branch neighbor counts and auxiliary
selector/cache/recurrent budgets differ. Equal total RAM/compute is not proved.

Intervals bootstrap six independent world means after averaging eight streams
and relevant cases within each world. Six clusters limit precision; intervals
are pointwise, not simultaneous. This is identity embedding of measured cues
and four fitted logistic tasks. It is not semantic memory, novel-task learning,
a full SAM-kNN/ADWIN implementation, new architecture proof or an AGI result.
See the [protocol](../../experiments/two-speed-20261010/PREREGISTRATION.md) and
[related work](../../experiments/two-speed-20261010/RELATED_RESEARCH.md).

Keep two-speed as an experimental option, not a replacement default. A future
study could test a cue-keyed stable concept bank or a different past-feedback
selector, but would need a new protocol and fresh worlds/seeds. Neither has
been started. No downloads, API inference, peptide weights, QPU jobs or paid
compute were used; the original lab and previous studies are preserved.
