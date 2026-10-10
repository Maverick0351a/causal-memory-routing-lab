# Two-speed memory: protocol before registered generation

The question is whether stable plus fast memory can improve adaptation without
losing stable recall, and provide value beyond an adaptive window. This protocol
and all source/test hashes are locked by the runner before generating any
registered calibration, training, validation or test data. Development tests
use a separate 12900000 seed range. No held-out retuning or optional stopping.

Six independent coefficient worlds each have four bootstrapped logistic expert
fits, two recurrent initializations and independently seeded training,
validation and eight held-out streams. Stable/noisy cues, one mapping change,
repeated/noisy changes, novel contexts, random labels and no feedback are the
eight cases. Within a world the same seed across cases pairs the public cues and
random draws; those cases/streams are dependent. Tasks remain four known
logistic functions; independent worlds are not cross-family generalization.

The active retrieval budget is 256 prior released records. The stable view is
the older 192 and the fast view the latest 64, disjoint and FIFO. Stable uses
up to 64 matching records uniformly; fast uses up to 16, half-life two matching
feedback examples. These are logical stores in one SQLite-backed history index.
Full audit history and the RAM replay index are retained outside the prediction
search budget; total RAM is not capped at 256 records. Context selectors retain
at most eight cue prototypes and prequential EMA forecast losses, updated only
after the label is released. No label is used to choose its own prediction.
An initial 0.05-nat fast penalty favors stable memory when forecast evidence ties.
Absent matching records, a branch uses the same global-recency fallback.

Two-speed and adaptive-window each select four candidates (EMA rates .05/.2,
softmin temperatures 4/12). Adaptive-window mixes forecasts from 8/32/64 matching
examples within the same 256-record search cap. It is not ADWIN. Fading selects
half-life 2/8/32 over the latest 64 matching records; uniform uses those records
without decay. Cache selects rate .1/.35 with eight prototypes. Recurrent
selects two trained initializations times temperature .5/1/2. Stable-only,
fast-only, shuffled-cue two-speed, global recency and fixed uniform routing
complete the controls. The shuffled control permutes stored cue/error pairing;
it uses only prior records. Each family sees the same fitted experts and streams.

Within each world, validation minimizes .5 times stable return-first16 loss
(averaged over recurring/noisy cues) plus .5 times changed-entry-first16 loss
(averaged over mapping/repeated/noisy changes). Every grid and deterministic
first-minimum tie is saved. All six selections are locked before any held-out
generation. No weights, gate or selection is updated from held-out results.

The primary endpoint is RAW prediction, with every gate in protocol.json:

- Adaptation: changed-entry loss improves stable-only by at least .01 nats and
  recovery delay by two steps, with upper world-bootstrap bounds at those
  margins. Changed loss must be within +.01 and recovery within +.5 steps of
  recency, cache, adaptive-window and fading.
- Retention: whole-stream and return-first16 stable loss are within +.02 nats
  of stable-only, uniform and cache, including upper interval bounds.
- Quality: whole-stream selection-case Brier/ECE/confidently-wrong excess is
  within +.01/+.02/+.005 of recency, cache and adaptive-window, including bounds.
- Incremental value: balanced entry loss improves adaptive-window by .005 nats,
  including the upper interval bound.
- Causal controls: balanced entry loss improves shuffled two-speed by .01,
  including the upper bound; every policy's pooled random-label accuracy is
  within .02 of chance. No-feedback and leakage invariants must pass separately.

Qualification requires every component. Raw random-label log loss, ECE,
confidently-wrong fraction, confidence coverage and conditional error are always
reported even if the accuracy control passes. The unchanged historical frozen
confidence companion is descriptive, cannot rescue a raw failure, and may not
transfer. No raw-random calibration success is implied by a chance-accuracy gate.

Recovery starts at the first visit to a context in each new mapping revision,
excludes startup and later already-recovered visits, and requires an eight-step
mean log loss <= .55; failure before the next switch or 64-step horizon is
penalized by 64. First16 windows remain distinct from whole-stream results.
Intervals resample six independent world means after averaging the eight stream
seeds and relevant cases within each world (10,000 draws, fixed seed). They are
pointwise, not simultaneous, and only six clusters limit certainty. Worlds are
weighted equally. No individual prediction is treated as an independent trial.

History reload must reconstruct all eleven policies at step 512 while the log
already contains 512 future rows. Full independent-process replay refits experts
and all recurrent candidates, reselects every validation grid, and checks every
trace array, metric, interval and gate. Hashes and failed development checks
are retained. Previous lab, results and v0.1.0 remain untouched.

This study uses only local NumPy and SQLite, six measured cue values and eight
readings. No semantic embeddings, pretrained weights, API keys, downloads,
peptide weights, QPU jobs or paid compute. Python's socket guard is a process
guard, not an OS sandbox. Source isolation is an interface guarantee, not an
adversarial sandbox against hostile policy code.
