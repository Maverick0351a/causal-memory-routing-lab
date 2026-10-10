# Frozen cross-task confidence experiment

This is a local synthetic sensor-fault-propensity classification benchmark. It
does not represent a measured machine, a validated sensor diagnostic or quantum
calibration. All code, thresholds, seeds and selection rules must be locked before
any held-out stream is generated. No test-directed repairs or retuning.

## Claim and primary endpoint

Does the unchanged eight-weight confidence gate generalize from its original
two-bit task to continuous observations and a supervised learned base predictor,
against a direct conventional online calibrator? The gate checkpoint and its
normalization are copied byte-for-byte. No gate training is permitted.

The primary endpoint is gate minus online-calibrator mean log loss, averaged
equally over four stress cases WITHIN each of 24 independent test seeds. Require
at least .01 nats improvement, paired bootstrap upper 95% bound below zero and
no individual stress case worse by >.02 nats. Bootstrap resamples whole seed
blocks, never individual steps or correlated cases. Other intervals are
descriptive, without simultaneous-coverage claims. One predictor fit and one
validation selection are fixed; intervals do not include fit/selection variance.

Quality gate: every stress-case Brier degradation <=.005 and confident-error
excess <=.01 versus online; first16 signal/noise entry loss degradation <=.03 in
both mixed cases; stable loss <=fixed+.02; random-label loss <=uniform+.03 and
confident errors <=half the base rate. Every random-label accuracy must lie
within .02 of .5. Both scientific gates and controls must pass for qualification.
Keep every failed component. The existing research reference registry is not
changed by this experiment, regardless of outcome.

## Public observations and private labels

Eight iid standard-normal continuous readings are the only observed input.
Independent PRNG substreams generate observations, label uniforms, schedules and
changing coefficients. Public readings are identical for a seed across ALL cases;
no distribution marker reveals signal/noise mode. Binary labels sample a logistic
fault propensity w dot x. Stable coefficients are specified in protocol.json.
Noise mode uses independent balanced labels. Mixed intervals are 64..192 steps;
short mixed intervals are16..48. Weak signal multiplies coefficients by .25.
Rotating coupling starts stable and then draws new isotropic coefficient vectors
with the original norm every64..192 steps. No-feedback repeats the mixed world
but withholds all labels from every policy. It retains the trained base prior:
it is not expected to become uniform or have chance accuracy in signal modes.

Private coefficient vectors, regime IDs, switches, ages, label probabilities,
seeds, current labels and future examples never enter inference or memory.
The evaluator may attach private mode/age/switch metadata AFTER all policies have
committed predictions. Labels are released only then, for subsequent steps.

## Predictor and feature adapter

Fit four nine-parameter logistic models on bootstraps of the same 8192 stable
training samples (fresh streams). Zero initialization,160 full-batch Adam steps,
learning rate .05, L2 .001 (intercept exempt). Model bootstrap seeds are frozen.
These36 learned parameters are frozen for validation and test. There are no
pretrained weights or installed dependencies. Models never receive hidden
coefficients; ordinary supervised training receives public readings and labels.

The base prediction is .05+.9 times an exponentially weighted mixture of four
learned expert probabilities. Expert weights are softmax(-8*fast-error). Fast
and slow binary mismatch EWMAs have rates .35 and .04, initially .5. Base
correctness and log-likelihood gain versus uniform use rates .04 and .15,
initially .5 and0. The unchanged gate input meanings and order are:
intercept1, absolute(2*base-1), two past correctness EWMAs minus .5, two past
log-gain EWMAs, .5-min(fast-error), .5-min(slow-error). All history updates follow
feedback. The new adapter preserves these meanings without inventing a regime
mapping. Different feature distributions are part of the transfer test.

## Calibration baselines and selection

Compare uniform, learned base, fixed Platt calibration, frozen gate and online
Platt calibration. Platt prediction is sigmoid(a*logit(base)+b). Fit fixed slope
and intercept on pooled validation streams (stable/mixed only,24576 examples)
with300 full-batch Adam updates, learning rate .03, initialization(1,0), no L2,
parameter clip[-4,4]. Allow negative slopes; a conventional baseline may invert
the base or correct class bias, while gate shrinkage cannot.

Online calibration starts from the fixed fitted weights for each stream and
directly minimizes log loss by SGD ONLY after each label. Gradient is
(prediction-label)*(logit(base),1) + decay*(weights-initial); clip weights[-4,4].
Select rate[.005,.02,.08,.32] and decay[0,.001,.01] by mean causal validation log
loss; deterministic enumeration resolves ties. Validation can fit offline
parameters; no held-out labels are available for selection. Save selection and
timestamp/hash before generating ANY test data. No short/weak/rotating data enter
selection. Gate weights/normalization and base predictors stay immutable.

## Metrics, uncertainty and replay

Report per-stream and equal-stream mean log loss, Brier, ten-bin confidence ECE,
accuracy, confidently-wrong fraction of ALL predictions (confidence>=.9),
high-confidence coverage and conditional error. Positive shrinkage preserves
the base class except numerical ties; confidence improvement is not recognition.
Online Platt can change classes; report its accuracy without attributing this
capacity to the gate.

Report first16 and late32+ signal/noise slices. Exclude startup from entry slices.
Descriptive response delay is the endpoint (zero-indexed from transition) of the
first16-label window with mean loss <=.55 in signal or <=log(2)+.05 in noise.
Search up to96 steps, truncate at next switch/end, exclude episodes shorter than
16, penalize unrecovered events as96 and report recovery fraction/counts. These
stochastic window scores are not a change-detection or causal-learning claim.

Save all168 test streams with five policies (840 policy traces /1290240
predictions), source snapshots, candidate grids, checkpoints and selection hashes.
In a separate process refit the base ensemble, repeat calibration/selection,
regenerate every stream, replay all probabilities exactly, recompute every metric,
bootstrap and gate, and resume gate/online predictions from SQLite feedback
strictly before step768. Verify preservation of all prior lab/project artifacts,
pinned clean vendor, absent provider keys and active Python socket guard. This
guard is not an OS sandbox. No API keys, network, GPU/QPU jobs or Robin weights.

Implementation tests use reserved development seeds970xxx only, no held-out seeds.
Freeze the final code after tests and before experiment generation. If execution
finds a scientific defect after test exposure, preserve this run as invalid and
use a new protocol with fresh seeds; do not silently repair this experiment.
