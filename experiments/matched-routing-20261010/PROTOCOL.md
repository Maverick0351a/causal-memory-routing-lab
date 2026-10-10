# Matched routing comparison — frozen protocol version 1

This follow-up asks whether the AR factorization improves held-out predictive
log loss relative to a conventional joint categorical router at comparable
parameter count and affine compute. It follows the previous experiment's failed
comparative gate. It changes no prior protocol or results.

The machine-readable settings are in protocol.json. Training, model initialization
and test seeds are all fresh and disjoint from the initial experiment. This is
new random streams of the same four-expert synthetic task, not unseen rule families.
No selection, early stopping, tuning or calibration will use held-out examples.

## Models and equal treatment

- AR: existing 24-unit tanh encoder, two categorical token heads. 750 parameters;
  720 multiply-accumulates in the affine layers per marginal forward pass.
- Joint: 26-unit tanh encoder and one four-class softmax. 758 parameters; 728
  affine multiply-accumulates per forward pass. The 1.07% size and 1.11% MAC
  gaps are disclosed. These are close matches, not exact equality. Softmax and
  activation costs are excluded from MAC counts and included in latency timing.
- Recency and fixed uniform remain untrained reference baselines.

Both neural models get exactly the same 24 public features, past-feedback
summaries, four known experts, label likelihood with 5% noise, data order, 12
epochs, 16-example update batches and Adam settings. Model weights remain frozen
during evaluation. Only released feedback updates the state. Five independently
initialized pairs are trained, using six identical training streams per model.
Only final-epoch checkpoints are evaluated; neither model receives extra tuning.

Controllers receive features only. Hidden rules, current labels, switch markers,
age, RNG state, test seed and evaluator references never enter the model API.
Every probability is fixed before feedback release. Memory resets for each run.
Any evaluator-only adaptation metrics are calculated after predictions. The four
toy experts are enumerated for both models; there is no sparse-execution speedup.

## Primary outcome and success threshold

Primary estimand: standard-condition mean log loss of AR minus joint, averaged
equally over five initialization seeds and ten independent test episode seeds.
AR adds predictive value only if:

1. Parameter counts differ by <=2% and affine MACs differ by <=5% of AR's count.
2. Mean AR log loss is at least 0.01 nats lower than joint log loss.
3. The upper endpoint of a paired two-way bootstrap 95% interval is below zero.

Bootstrap resamples model-seed indices and episode-seed indices independently,
then averages the selected paired matrix; 10,000 draws, fixed seed 173205.
Report all paired values, model-seed and episode-seed averages, plus the interval.
The 50 cells share models and streams; they are not 50 independent environments.
The bootstrap is descriptive uncertainty for this finite design, not a universal
architecture claim. No secondary result can rescue a failed primary gate.

Report a separate efficiency gate: median AR marginal-forward latency must be
<=1.25x joint. Report five interleaved timing blocks per pair after warm-up, 256
fixed synthetic feature vectors per block. Exclude I/O, feature generation and
route sampling from this timing. Also report end-to-end evaluation times. Timing
is machine/load dependent and never decides the primary predictive gate.

## Secondary outcomes and controls

Report accuracy, Brier score, log loss, 10-bin ECE, confidence>=0.9 error rate
and coverage, sampled-route accuracy and first-eight-step post-switch accuracy.
Run standard rules, faster 8–20 step switches, independent random labels, and
no-feedback ablation. Controls are evaluated with unchanged frozen checkpoints.

Adaptation speed: at each real switch, search within that regime and at most 32
steps for the first rolling 8 predictions with >=7 correct. Record the endpoint
offset (7 is earliest), recovered fraction, and median endpoint among recoveries.
If none exists before 32 steps or the next switch, mark censored and assign 32 to
the explicitly penalized recovery score. Include a final regime only with >=8
observations. Exclude startup and windows crossing the next switch. This metric
uses labels in the scorer only; its forward-looking measurement does not feed
back into prediction. On random labels it measures chance runs, not adaptation.

Engineering gates: analytic joint gradients match finite differences; causal
current-label/future perturbations leave prefix predictions unchanged; checkpoints
reload identically; test evaluation cannot update weights; test seeds are disjoint;
both models see byte-identical input vectors; and every saved test trace replays
exactly in a fresh process. The training routine is also tested for determinism.

Negative control: each neural family's aggregate independent-random-label accuracy
must be in [0.45,0.55]. Memory dependence: each neural family's standard accuracy
must exceed its no-feedback accuracy by >=0.10. Learning: each neural family must
reach >=0.70 standard accuracy and exceed uniform by >=0.10. Record each gate.

## Audit and preservation

The runner creates a new output directory and writes a UTC timestamp, protocol
hashes and source snapshot before generating training or test data. It refuses to
overwrite a run and rejects source changes during execution. Previous benchmark
artifacts and original Quantum outputs are hashed before changes and checked after.
Existing embedding, memory and QCA behavior are preserved. No downloads, API keys,
cloud resources or third-party code execution are introduced by this follow-up.
