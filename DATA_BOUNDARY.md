# Public observations, feedback and evaluator-only fields

The raw archives preserve the evaluation record, including the correct answer.
Publishing that record does not authorize using answers in policy inference.
Persistent retrieval requires namespace agreement and `step < cutoff`.

| Field category | Examples | Permitted role |
|---|---|---|
| Public observation | bits, measured cue, continuous readings | Current prediction input |
| Prediction | probability, route, alpha, retrieved_count, effective_neighbors | Output committed before current feedback |
| Released feedback | past label and expert mismatches derived from it | Update only after prediction; retrieval cutoff is strict |
| Evaluator-only | current/future label, rule/context IDs, revision, switched, age, mapping_changed, changed_entry, seed | Scoring, reproducibility and post-run audit only |

Training is explicit supervised training on registered training streams, not
an online test-time answer channel. Validation settings are locked before any
held-out generation. Test weights stay frozen; permitted online state uses
released past feedback. Random-label, no-feedback, shuffled-memory and
same-record weighting controls bound the conclusions.

This is an exposed evaluation archive with all seeds and evaluation answers. Treat these
streams as consumed evaluation evidence. New scientific confirmation must
freeze new splits/seeds and controls before generating data. No benchmark
contamination resistance or secret-test protection is claimed for this release.
