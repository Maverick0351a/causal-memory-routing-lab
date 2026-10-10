# Frozen confidence transfer test: 2026-10-10

Written before any new training, validation or held-out stream generation. Keep
the learned gate and previous results unchanged. No packages, models, keys, QPU
jobs or network services are required. New code is isolated in transfer_lab and
transfer_tests; use Run-Transfer.ps1 to avoid modifying prior frozen entry points.

## Question and models

Does the existing confidence gate transfer to faster mode changes and noisier
learnable rules, and add value beyond a simple adaptive online probability mixture?
Compare uniform, recency, validation-tuned fixed shrinkage, the original frozen
learned gate, and an online mixture of recency and uniform. All use the same
causal recency probabilities and public observations. The learned gate's fitted
coefficients and normalization MUST remain byte-identical to the selected v1
checkpoint (hash in protocol.json); no fitting, feature changes or retuning.

Online mixture: initialize alpha at a training-selected prior. Predict with
p=.5+alpha*(base-.5). AFTER label release, compute
gain=log(P_recency(label)/.5), posterior=sigmoid(logit(alpha)+eta*gain), then
alpha_next=(1-share)*posterior+share*prior. Log odds are clipped to [-20,20].
No hidden mode, boundary, age, seed, current label or future example enters any
policy's inference. The online mixture updates its state from released feedback;
its selected eta/share/prior remain fixed throughout testing. Thus evaluation
state changes, but there is no held-out hyperparameter fitting. Alpha is a mixing
weight, not the probability that a hidden regime is learnable.

## Fitting and selection

Use the original environment distribution ONLY for baseline development: four
equal-weight fit conditions, 12 fresh training streams of1,024 steps each and
eight fresh validation streams of1,536 steps each. Streams reset state. Choose
the online prior from the training fixed-alpha grid; clip to [1e-6,1-1e-6]. Select
share x eta (5x3 candidates) by lowest validation mean log loss. Independently
choose the constant-shrinkage baseline on the same validation grid. Exact ties
use declared candidate order. Retain the learned gate's validation score as a
diagnostic only. No shorter/noisier environment is available to selection.

All split and bootstrap seeds are fresh relative to the earlier three experiments.
Before generating test data, write and hash selection.json and the gate copy.
Do not retune after test results. Test streams are 24 independent seeds,1,536 steps
each, shared across cases and policies for paired comparisons. Public observations
are identical across cases for a seed. Controls: standard5% noise, noisy-standard
20%, pure random labels, and no feedback on the short/noisy mixed stream.

## Transfer cases and transitions

Reference mixed mode durations are96-192 steps,5% learnable-label noise. Unseen
cases are16-48-step durations,20% learnable noise, and both. Rule durations remain
24-64 steps. Initial mode is randomly selected. Random labels are independent
coin flips and have the same observation distribution. No transition signals
are provided. Generator and independent PRNG substreams are reused unchanged.

Score the first16 observations after EVERY mode transition, separated by entry
into learnable versus random segments. Exclude startup; never cross the next
mode boundary. Mode durations are >=16, so complete windows are available except
at episode end; retain available truncated final windows and report their counts.
Also report original rule-switch scoring and longer-term slices. Private regime
metadata is used only after predictions, for scoring.

## Frozen gates and default decision

Primary learned-gate transfer value: mean log-loss difference gate minus online
over the THREE unseen cases <=-.01 nats, with upper paired bootstrap95% endpoint
<0. Average the three case differences WITHIN each stream seed first, then
resample the24 seeds (10,000 draws). Cases/steps are not independent replicates.
Additionally, no unseen case may be worse than online by >.01 mean log loss.
All three per-case bootstrap intervals are descriptive, not separate significance
claims. These are transfer cases within one toy world family, not general transfer.

Separate learned-gate quality: both directions' first16 transition loss in EACH
unseen case <=online+.03; standard loss <=recency+.02; pure-random loss
<=uniform+.03; pure-random confidently-wrong fraction <=half recency; and each
unseen case's confidently-wrong fraction <=online+.01. All are mean-based
engineering margins, not equivalence tests. Gate becomes default only if both
primary and quality pass. Accuracy is unchanged by positive shrinkage, so it
cannot rescue a failed gate or demonstrate a learned recognition benefit.

Online-baseline fallback is considered separately: its aggregate unseen-case
loss must beat tuned fixed shrinkage by >=.01 with upper paired interval<0,
no unseen case worse than fixed by >.01, both transition directions in EACH
unseen case <=fixed+.03, standard loss <=recency+.02, random loss <=uniform+.03,
and random confidently-wrong fraction <=half recency. If the learned gate does
not qualify but online does, online becomes the local confidence reference.
If neither qualifies, do not promote either; preserve recency as the underlying
router and treat the v1 confidence result as distribution-specific.

## Evidence and limitations

Save protocol/source/prior-artifact hashes, training/validation data digests, the
full candidate grid, frozen selection, copied checkpoint, all per-stream metrics,
complete test traces, paired intervals and all failed gates. Report log loss,
Brier, ECE, accuracy, >=90% confidence coverage, confidently-wrong fraction of ALL
predictions, conditional high-confidence error and transition loss/alpha by mode.

A separate process recomputes baseline selection from training/validation only,
replays every test trace and score, recomputes bootstrap/default choice, verifies
frozen parameters, and reconstructs learned and online predictions from SQLite
feedback strictly before step768. Tests must cover the online update formula,
predict-before-feedback ordering, current/future/private metadata invariance,
state reset, selection isolation, transition boundaries, controls, checkpoint/
memory reload, seed freshness and frozen gate/default decision logic.

All old lab source/results and the pinned vendor remain unchanged; only the
backed-up AGENTS.md may be updated to record results. Python socket guards are
process-level checks, not an OS sandbox. No real assistant is reconfigured.
