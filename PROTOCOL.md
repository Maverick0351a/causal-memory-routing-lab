# Frozen protocol, version 1 — 2026-10-10

Write and hash this protocol before training/evaluation. This is a small classical
architecture experiment, not a quantum learner or a reproduction of Robin.

## Information boundary

At step t the public observation is two independently sampled bits x0,x1. The
environment privately chooses a rule (a,b) and emits y = x0 XOR (a AND x1) XOR b,
with 5% independent label noise. Rules change at independently sampled intervals
of 24–64 steps and cannot repeat immediately. No rule ID, seed, switch boundary,
current label, evaluator object, or future example is passed to a controller.
All controllers receive the same public features: a deterministic local hash
embedding of the bits and fast/slow exponentially weighted mismatch rates of
four known expert formulas, calculated exclusively from past released feedback.
The expert formulas are an explicit task-specific inductive bias.

Predict first; record prediction; score privately; only then release y and update
the memory summaries. Training uses the same order. Test weights remain frozen;
feedback updates only recurrent state and memory. Memory is reset per episode,
seed, model and condition. Persistent records are namespace scoped; retrieval
requires a strict earlier-step cutoff. Ground truth rules are never stored there.

## Models

- Fixed expert: one of four experts selected using training data only.
- Fixed uniform: constant equal mixture of four routes.
- Recency: untrained softmax of fast past mismatch rates (strong simple baseline).
- Autoregressive: 24-unit tanh encoder and categorical route tokens a then b,
  q(a,b|h) = q(a|h) q(b|a,h). Exact normalized route likelihoods; sequential sampling
  and temperature/prefix intervention. Train on marginal observed-label likelihood,
  never on hidden route labels. This borrows factorization from ArBG; it does not
  implement molecular energies, SNIS, twisted SMC, Robin, or claim their results.
- Recurrent: 24-unit Elman network, joint four-route softmax, truncated BPTT of 16
  steps. Same public inputs, expert bank, label likelihood and training budget.

The main metric is exact route-marginal label probability, computed by enumerating
four tiny experts. Also report sampled-route accuracy, route likelihood and wall
time. This is not a sparse-expert compute saving demonstration.

## Reproduction and controls

Training episode seeds: 101, 202, 303, 404, 505, 606; 512 steps each; 12 epochs;
Adam learning rate 0.003, gradient norm clip 5; parameter seeds 11, 29, 47.
Test episode seeds: 1001, 2002, 3003, 4004, 5005; 768 steps each.
No test-seed tuning or early stopping. Three independently trained model pairs.

Conditions: standard changing rules; faster switches (8–20 steps); independent
random labels (negative control); standard rules with no feedback/memory (ablation).
Random labels are generated independently of observations, rules and model RNG.
All models get identical streams within a condition. Report accuracy, Brier score,
log loss, 10-bin ECE, confidence >=0.9 error rate/coverage, first-eight-step accuracy
after switches (evaluator only), sampled accuracy, per-seed results and parameter
counts. Paired mean differences and spread across episode seeds are descriptive;
the small sample does not establish general superiority.

## Gates

Engineering PASS: isolation, no-key synthetic execution, saved HDF5/SQLite reload,
deterministic replay, gradient checks, causal prefix/future invariance, namespace
and temporal memory isolation, checkpoint equality and normalized route likelihood.
Learning PASS: standard AR mean accuracy >=0.70 and at least 0.10 above fixed uniform.
Negative control PASS: aggregate random-label accuracy within [0.45,0.55].
Memory dependence PASS: standard AR accuracy at least 0.10 above no-feedback AR.
Comparative PASS only if AR exceeds recurrent mean accuracy by >=0.02 AND improves
mean log loss. Failure of any scientific gate remains in the report. Passing tests
does not prove absence of every possible leakage path or general intelligence.
