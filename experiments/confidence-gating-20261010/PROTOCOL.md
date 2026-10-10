# Confidence gating: frozen experiment 2026-10-10

Written before generating new training/validation/test streams. This is a small
classical, local synthetic experiment. Earlier AR comparative and latency failures
remain failures. No new packages, models, peptide weights, APIs or QPU work.

## Question and models

Can an eight-coefficient logistic confidence gate improve a recency router's
probabilities on an unsignaled mixture of learnable changing rules and independent
random labels, beyond a validation-selected constant shrinkage coefficient?

All policies share the existing recency distribution q=softmax(-8*fast_loss), with
fast_loss updated at rate .35 from previously released expert errors. Base output
probability p=.05+.9*q@expert_outputs. Compare recency (alpha=1), uniform (alpha=0),
fixed shrinkage, and adaptive sigmoid gating: p_gate=.5+alpha*(p-.5). This also
corresponds to mixing q with a uniform expert distribution. Adaptive alpha>0 keeps
recency's class decision unchanged by construction; equal accuracy or rolling
accuracy recovery therefore does not demonstrate a learned improvement.

The gate's features are intercept, current public base margin, two EWMAs of past
base correctness, two EWMAs of past base log-likelihood gain over uniform, and
best fast/slow expert historical mismatch. All feedback-derived features are
computed strictly before the current answer is released. Training-only feature
means/scales are frozen (scale floor .05). The gate never sees regime IDs, seeds,
switch flags, ages, current answers, future examples or evaluator objects.

## Splits and fitting

All seeds/intervals/thresholds are in protocol.json; splits are disjoint and fresh
relative to both earlier benchmarks. Train on 12 streams per condition, 1,024 steps
each; validation on eight per condition, 1,536 steps; test on 24 per condition,
1,536 steps. Four fit conditions have equal length/weight. Logistic candidates
start at zero weights, use Adam for 60 shuffled epochs, batches 512, rate .01,
with L2 candidates [0,.001,.01] (intercept unpenalized). No early stopping. Choose
lowest validation log loss; exact ties use declared candidate order. Fixed alpha
uses the same validation objective and grid 0,.05,...,1. Validation labels can
select parameters; test labels can only update causal histories and score results.

Before any test generation, write selection.json and all candidate checkpoints;
freeze their hashes. No held-out retuning. One deterministic fitted initialization
per L2 candidate; confidence intervals condition on this training/selection run.

## Environment

Independent PRNG substreams for public observations, changing rules, rule noise,
random labels and mode changes. Observation distributions are identical in every
condition. Learnable labels use four known experts with 5% independent noise.
Rules change to a different expert every 24-64 steps (fast: 8-20). The mixed stream
alternates learnable/random segments lasting 96-192 steps; starting mode is random.
Changes are never signaled to policies. Underlying rule changes continue through
random segments. Only real switches within a learnable segment enter rule recovery
scoring. A separate no-feedback control reuses mixed rows, without state updates.

## Frozen decisions

Primary adaptive-value gate: mixed-stream mean log-loss difference adaptive minus
validation-selected fixed <=-.01 nats, and upper paired episode-bootstrap 95%
endpoint <0 (10,000 resamples of the 24 independent stream seeds). Steps are not
independent replicates. Failure cannot be rescued by auxiliary metrics.

Separate deployment-quality gate requires ALL: standard and fast log loss no more
than .02 above recency; accuracy no more than .005 below recency; mean log loss over
the first 16 steps following actual rule changes no more than .03 above recency;
random-label log loss no more than .03 above uniform; and confidently-wrong fraction
on random labels and on mixed random segments <= half of recency's fraction.
These are mean-based engineering margins, not statistical equivalence claims.
Do not recommend replacement unless both primary and quality gates pass.

Report log loss, Brier, ten-bin ECE, accuracy, coverage at >=.9 confidence,
confidently-wrong fraction of ALL predictions, error rate conditional on high
confidence, switch log loss, and rolling 7/8 recovery (horizon32; censored32,
startup excluded, no crossing another rule/mode boundary). Partition mixed results
by private mode and time since transition ONLY during scoring; those slices never
enter inputs or parameter selection. Preserve all failures.

## Audit and reproducibility

Lock protocol/source hashes before training; save causal feature/label dataset
digests, candidate fit records, validation grid, checkpoints, all test traces,
per-stream metrics, paired bootstrap and source snapshots. A separate process
reconstructs training/validation feature hashes, recomputes selection, replays every
test probability, recomputes metrics/gates, checks frozen parameters, and resumes
from SQLite feedback strictly before step768. It also checks saved artifacts,
the unchanged pinned vendor, old lab artifacts, and three original Quantum files.
Leakage tests perturb the current label, future observations/labels, private
metadata and namespace/future memory rows. Gradient, checkpoint, deterministic-fit,
null/control, switch-window boundaries and gate-failure tests are required.

Python's audit hook blocks socket operations for these runs; it is not an OS
sandbox. Only new experiment files plus backed-up launcher/README/AGENTS updates
are permitted. No past experiment source or result files are changed.
