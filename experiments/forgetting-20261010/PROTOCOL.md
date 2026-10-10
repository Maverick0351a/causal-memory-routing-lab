# Explicit forgetting: frozen protocol

Freeze this protocol, configuration, implementation and tests after development
tests and BEFORE any registered calibration/training/validation/test generation.
Development uses reserved 990xxx seeds only. Keep every failed gate. No held-out
retuning or source repairs; a defect after exposure needs a separately preserved
run, new protocol and fresh seeds. Earlier experiments/defaults stay unchanged.

## Question and information boundary

Does fading past cue-matched feedback improve the FIRST predictions when that
cue's rule changes, compared with unchanged memory, a simple context cache and
recency, while preserving useful stable recall? A secondary ablation asks whether
fading adds value beyond choosing recent matches with uniform weights.

The task family remains four learned nine-parameter logistic experts over eight
continuous readings, plus six observed cue values. Calibration fits experts from
8192 public examples with four supervised targets,160 Adam steps,.05 learning
rate,.001 L2, bootstrap seeds, zero initialization; private coefficient vectors
never enter fitting or inference. Each stream has fresh separated cue templates
and an independent random permutation from cues to the four known tasks. Random
substreams separately generate cues, mappings, readings, labels, schedules and
jitter. Controllers see only public Observation(cue,readings) and past released
labels. All policies commit before evaluator reads current labels/private fields.
No seed, task/context ID, revision, age, switch or changed-entry flag reaches the
policy, its embedding or stored feedback. SQLite schema remains public-only.

## Cases and exact transition scoring

Blocks last48..96 observations. Each four-block cycle visits all four contexts,
with fresh permutations and no adjacent repeat. Cases:

- recurring: unchanged cue-task associations; normal cue noise .12/coordinate.
- noisy_cues: same world with noise .3.
- mapping_change: cyclically shift all mappings by+1 at the first block boundary
  at/after the midpoint; retain public cues and past feedback.
- repeated_changes: after the first four-block cycle, shift all mappings by+1
  at every new four-block cycle, without a public change signal.
- noisy_changes: identical repeated changes with cue noise .3.
- novel_contexts: new separated cue for every block, random known task.
- random_labels: independent .5 labels, identical public observations to recurring.
- no_feedback: recurring world with every label withheld from every policy.

Evaluator-only changed_entry is true for a block iff revision>0 and this is the
FIRST visit to that cue since that mapping revision. It stays true throughout
that block, but scoring uses only age<16 (also report age<8). Subsequent visits
at the same revision are excluded from changed-entry scoring. This prevents
recovered later visits or the unchanged first half from diluting transition harm.
Stable-return scoring includes returned cues with changed_entry=false, age<16.
Every transition flag is attached after predictions. It never triggers forgetting.

Response uses the first endpoint of an8-label rolling raw log-loss window<=.55
within a changed-entry visit, capped64 and truncated at next switch/end. Visits
shorter than8 are excluded. Unrecovered events score64. This is a stochastic
feedback-step endpoint (minimum7), not wall-time latency or a change detector.

## Policies and controlled fading

Ten policies: forgetting, recent_uniform, shuffled_forgetting, unchanged memory,
memory_disabled, shuffled_memory, recency, recurrent, context_cache, fixed_uniform.
Disabled memory is exactly recency; it is a duplicate integrity control.

Fading searches the latest1024 rows with step<before_step and matching namespace,
finds observed cues within radius1.25, keeps the NEWEST64 matches, then weights
their rank r (newest=0) by2^(-r/h). Choose h in[2,8,32] matched-feedback examples
on validation. Weights normalize to one. Its count warmup is min(1,count/8),
blend1; routing is softmax(-8*blended expert mismatch), raw p=.05+.9*q@experts.
No explicit change detector, hidden reset, task ID or learned semantic embedding.
The complete2048-row audit transcript remains; forgetting changes predictive
weights rather than deleting evidence. Save effective sample size1/sum(w^2).

recent_uniform uses EXACTLY the same selected settings, newest64 matches, radius,
blend and count warmup, but uniform weights. It isolates fading from changed
neighbor ordering. shuffled_forgetting permutes key-to-feedback associations
causally over the same past candidate records before matching; key multiset,
neighbor counts and weight effective sample sizes stay identical per query.

Unchanged memory uses the preserved distance-nearest algorithm:1024 past rows,
neighbors[32,64], radius[.75,1.25], blend[.5,1] (8 candidates). Conventional
cache uses at most8 centroids, centroid update.05, associated four-error EWMA,
radius[.75,1.25], rate[.1,.35], blend1 (4 candidates). Both are reselected on
fresh validation. Recency uses preserved .35/.04 fast/slow summaries.

Recurrent baseline has tanh hidden24,14 public inputs (cue/3,fast/slow errors),
four-route softmax,1036 parameters. Train3 fresh initializations for12 epochs
on8 fresh1024-step streams,98304 examples/3072 BPTT32 updates each, Adam.003.
Four training streams are stable, four have changing mappings; exact assignments
are in training_cases. Use preserved analytic continuous-expert likelihood
gradients; no private scoring metadata enters inputs or loss. Select initialization
and temperature[.5,1,2] (9 candidates). No additional trained memory parameters.

All policies share experts and label-release order. Every policy also gets the
same unchanged confidence-gate checkpoint/normalization as a DESCRIPTIVE companion,
with the preserved public/past-feedback feature meanings. Gate fitting is absent.
Every qualification gate uses RAW probabilities. Companion shrinkage cannot rescue
failure. Storage, compute and offline training differ; no equal-budget or general
model-superiority claim is permitted.

## Splits and fair selection

All stochastic seeds are fresh/disjoint, recorded in protocol.json. Calibration:
1010001..1010008; coefficient1010901; expert1030001..1030004. Training1011001..1011008,
recurrent1031001..1031003, training shuffle1032001. Validation1020001..1020006,1536
steps each on recurring/noisy_cues/mapping_change/repeated_changes/noisy_changes.
Tests1060001..1060024,2048 steps each on all8 cases. Memory shuffle1033001;
paired bootstrap1040001,10000 resamples. Resume at step1024 of repeated_changes.

Every family uses the SAME validation objective: .5 times mean stable-return16
raw loss over2 stable cases and6 seeds PLUS .5 times mean changed-entry16 raw loss
over3 changing cases and6 seeds. Select8 memory+4 cache+9 recurrent+3 fading=24
candidates. Ties use first grid entry. Recent-uniform and shuffled-fading controls
inherit the chosen fading settings. Save all scores and lock selection hash/time
BEFORE generating any held-out stream. Novel/random/no-feedback cases are not
used for selection. Training/validation contain the changing families; this is
fresh-seed testing within that synthetic family, not unseen-family transfer.

## Frozen gates (conjunction, never whichever comparator looks best)

Paired differences always mean forgetting minus comparator. Primary changing
cases are mapping_change/repeated_changes/noisy_changes, averaged WITHIN each
seed before bootstrapping24 seed blocks. Steps and cases are not replicates.

1. Entry value: aggregate changed-entry16 loss improvement>=.03 nats versus
   unchanged memory AND>=.01 versus recency AND>=.01 versus cache; every paired
   95% upper bound<0. Additionally EACH changing case must have mean and upper
   bound<=+.02 versus recency and cache so average wins cannot hide large harm.
2. Speed: aggregate changed-entry response>=2 steps faster versus unchanged memory,
   paired upper<0. Difference versus recency AND cache must have mean and upper
   bound<=+.5 steps (noninferiority to these fast baselines).
3. Stable retention: for EACH stable case whole-stream and return16 loss difference
   versus unchanged memory has mean and upper<=+.02; confidently-wrong fraction
   difference has mean and upper<=+.005.
4. Quality: for EACH changing case changed-entry16 Brier difference versus recency
   AND cache has mean and upper<=+.01, confidently-wrong fraction difference has
   mean and upper<=+.005. EVERY policy's random-label accuracy is within.02 of.5.
   Random raw-loss difference versus recency has mean and upper<=+.02.
5. Fading ablation: aggregate changed-entry16 improvement>=.01 versus recent_uniform
   AND>=.02 versus shuffled_forgetting; each paired upper<0.

Overall forgetting_qualified requires ALL five gates. Separately report companion
random loss<=log(2)+.03; it cannot rescue a failed raw gate. Report every component,
raw/gated log loss, Brier, accuracy,10-bin confidence ECE, confidently-wrong fraction
of ALL predictions at confidence>=.9, coverage and conditional high-confidence error.
Intervals condition on one coefficient world, fit and selection; no fit/selection
uncertainty or simultaneous confidence-band guarantee is claimed.

## Verification and preservation

Save192 streams*10 policies*2048=3,932,160 raw predictions plus the same number of
gated companion probabilities; source snapshots, all fits/curves/selection scores,
SQLite representative history, full per-seed metrics/slices/delays/bootstraps/gates.
Fresh-process verification refits experts and all recurrent candidates, recomputes
selection, regenerates/replays every prediction/route/score and every gate exactly.
Reload ALL10 policy states from1024 earlier feedback rows while1024 future rows
already exist in the full SQLite DB, asserting their next predictions and cutoff.
Check original-baseline predictions against preserved implementation in development.
Check shuffled counts/effective sizes, prediction-before-answer, metadata/current/
future-answer invariance, empty no-feedback memory and public-only stored errors.

Protect every previous lab file via preservation hashes and back up AGENTS.md.
Verify three original Quantum files and clean pinned NVIDIA checkout. No downloads,
models/Robin weights, API keys/calls, services, GPU/QPU jobs or live-default changes.
Research is read from primary public web sources; no project content is uploaded.
Python socket guard and removed listed provider keys cover only the child process,
not the whole operating system. Related-work distinctions are in RELATED_RESEARCH.md.
