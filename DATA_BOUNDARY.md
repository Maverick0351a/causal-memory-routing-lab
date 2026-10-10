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

## Two-speed extension

The new archive has 384 NPZ streams and eleven policy rows per stream. Numeric
arrays include predictions, routes, selector weights, counts, measured cues and
readings, plus evaluator-only labels, context IDs, revisions, ages and flags.
No object arrays or executable model files are needed. `two-speed-index.json`
specifies its format, digest and byte size separately from the historical index.

Controllers receive a whitelist of public settings; generator seeds, case
labels, switch schedules and test seeds are excluded. The registered shuffle
control has a fixed RNG seed used only to permute eligible past cue/error
pairings. Candidate forecast losses update the selector after current feedback,
using the forecasts committed before that label. Full audit history is kept;
strict cutoffs and the 256-row prediction search budget apply on reload too.
