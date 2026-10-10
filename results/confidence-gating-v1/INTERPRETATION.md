# What the confidence result establishes

Both frozen gates passed. An eight-coefficient logistic gate can use only causal
released feedback to modulate recency's confidence in this synthetic family.
Against validation-tuned fixed alpha=.65, it improved mixed-stream log loss by
0.053134 nats (8.80%); the 24-seed paired interval was [-.055597,-.050618] for
adaptive minus fixed. All 24 per-stream differences favored adaptive gating.

The test streams and current answers were inaccessible to parameter selection.
Training and validation each used 49,152 examples from disjoint seeds. The
separate verifier refit all three candidates exactly, recomputed validation
selection, replayed 737,280 predictions, checked frozen coefficients and source
hashes, and reconstructed prediction768 from strictly earlier SQLite feedback.

## What remains weak

- Standard log loss rose slightly: .33878 versus recency .33668. Standard Brier
  and ECE also worsened slightly. Passing a .02 mean-loss margin is not statistical
  equivalence or a win on every metric.
- The first16 observations after random labels give way to a learnable rule had
  loss .59029 versus recency .45245; the gate initially remains cautious. Mixed
  learnable segments overall had loss .37444 versus recency .34919. The measured
  rule-switch quality gates do not eliminate this separate mode-transition cost.
- The first16 observations after a random segment begins had loss .90524 versus
  fixed shrinkage .80257; stale confidence decays with feedback. Random-only loss
  .69703 nearly matches uniform .69315, but mixed random-segment loss .72628 is
  still worse because of detection delay.
- Fixed shrinkage also eliminates >=90%-confidence errors by capping confidence;
  its mixed loss is worse. The primary proper-scoring comparison, not counting
  zero confident errors alone, is the evidence that adaptive weighting adds value.
- Adaptive alpha is positive, so the predicted class and accuracy-based recovery
  are unchanged by construction. This improves probability estimates, not class
  recognition. Alpha is a mixture weight, not a calibrated probability that a
  hidden mode is learnable.
- One world family, a known toy expert bank and hand-designed temporal statistics
  were used. The bootstrap measures held-out stream variation conditional on one
  fitted/selected gate. It does not establish transfer, training robustness,
  semantic-memory benefit, agent capability, quantum benefit or general intelligence.

No real assistant was reconfigured. This is a local research checkpoint and
benchmark interface. A useful next test would freeze a stronger online calibration
baseline and unseen, shorter/noisier mode changes with new train/validation/test
seeds. The present checkpoints, seeds, gates and negative results must stay frozen.
