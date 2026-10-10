# Causal Memory & Routing Lab

Reproducible, local synthetic research on autoregressive routing, confidence,
recurring-context memory and forgetting. **Negative results are first-class results.**

[Interactive research explorer](https://huggingface.co/spaces/Maverick03511/causal-memory-routing-lab) · [Dataset and full trace archives](https://huggingface.co/datasets/Maverick03511/causal-memory-routing-lab)

The explorer has switch-aligned adaptation curves, individual stream/model
seed distributions and pooled confidence-versus-accuracy plots. Toggle policies,
choose a case and raw/gated endpoint, change the event window/bin width, inspect
exact values and download SVG figures or selected seed CSV values. These are
descriptive charts of recorded evidence; they do not add new hypothesis tests.

This is a personal research release developed with Codex assistance. It is not
peer reviewed, a general-AI claim, a quantum hardware result or an NVIDIA product.
The known-task generators and resource budgets bound every conclusion.

| Research stage | What the evidence actually supports |
|---|---|
| NVIDIA QCA synthetic T1 | Keyless local experiment, saved HDF5/SQLite results, history reload and seeded replay. No hardware or VLM/API call. |
| Initial autoregressive routing | Learns the toy task and passes controls; comparative superiority fails. Recurrent accuracy 89.149% vs AR 87.778%. |
| Matched conventional routing | AR 750 vs joint 758 parameters; predictive-value and measured latency gates fail. |
| Confidence gating | Original and within-family shifted tests pass frozen gates; this is a restricted synthetic confidence result. |
| Continuous sensor family | Quality passes, but primary improvement is too small; overall sensor qualification fails. |
| Recurring-context memory | Registered recall/speed/quality/cache-value gates pass over four known fitted tasks. Stale associations still cause transition harm. |
| Explicit forgetting | Speed and fading ablations pass; entry value, stable retention and quality fail. Overall qualification fails. |

The earlier benchmark-v1 metric issue, recurring-memory-v1 JSON export failure,
and failed scientific gates remain visible. No test stream was retuned to turn
a failure into a pass. Do not average these distinct studies into a universal score.

## Run the public code

Use an isolated Python 3.11 environment. The research core needs only NumPy;
pytest is used for tests. Installation below downloads those packages from PyPI.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
.\.venv\Scripts\python.exe research.py test
.\.venv\Scripts\python.exe research.py run forgetting --output results/my-forgetting-reproduction
```

On Linux/macOS use `.venv/bin/python` instead. Existing output directories are
preserved. The launcher strips provider credentials from its child, disables
tracing and enables a Python socket audit guard. This is not an OS sandbox.

Supported run families are `matched`, `confidence`, `transfer`, `sensor`,
`memory` and `forgetting`. Their registered configurations are under
`experiments/`. Reusing exposed seeds is reproduction, not fresh confirmation.
Changed methods need new protocols and fresh seeds before held-out evaluation.

## Replay the released forgetting result

Download `artifacts/forgetting-v1.tar.gz` from the dataset explicitly, then run:

```powershell
.\.venv\Scripts\python.exe research.py verify-published-forgetting --archive PATH/forgetting-v1.tar.gz --receipt results/my-public-replay.json
```

This verifies the archive digest, refits experts and all three recurrent
candidates, reselects all 24 settings, replays 192 streams/1,920 policy traces/
3,932,160 raw predictions plus the same confidence companions, recomputes
metrics and gates, and reloads all ten policies while excluding 1,024 future
SQLite rows. The scientific verdict must remain FAIL overall.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for provenance, portability changes,
test scopes and the optional pinned QCA setup. Code and synthetic data are
released under Apache-2.0. No pretrained peptide or language-model weights
are included. Small `.npz` files are fitted toy research parameters.

## Causal boundary and limitations

Controllers receive only public observations and previously released feedback.
Every current prediction is committed before current answers and private
scoring metadata are read. SQLite retrieval uses namespace and `step < cutoff`.
The embedding interface is local hashing or identity encoding of measured cues;
these experiments do not establish semantic memory or new-task learning.

Full evaluation traces include labels, hidden rule/context IDs and switch
metadata for auditing. **Those fields are evaluator-only, never policy inputs.**
The dataset is an exposed evaluation archive, not a secret held-out benchmark
or a clean training corpus. Using it for training/selection invalidates later
held-out claims on these same streams. See [DATA_BOUNDARY.md](DATA_BOUNDARY.md).

Confidence, Brier, loss and confidently-wrong fractions are retained alongside
accuracy. Intervals are conditional on the specified fits and selection; some
studies use different budgets and task families. Passing a local causal test
is not a proof against every possible leakage path or a deployment guarantee.

## Related work

- [NVIDIA Quantum Calibration Agent Blueprint](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint), pinned at `d8d37bc4de28d2347d8e21405903b29a2891f793` for the keyless synthetic foundation.
- [Autoregressive Boltzmann Generators](https://arxiv.org/abs/2606.27361), inspiration for sequential route factorization; no peptide model or full method reproduction.
- [SAM-kNN](https://www.ijcai.org/Proceedings/2017/0690.pdf), [ADWIN](https://epubs.siam.org/doi/10.1137/1.9781611972771.42), [Dynamic Weighted Majority](https://www.jmlr.org/papers/v8/kolter07a.html), [Titans](https://arxiv.org/abs/2501.00663).

The checked distinctions are in [the related-work note](experiments/forgetting-20261010/RELATED_RESEARCH.md).
This focused reference check does not establish novelty or reproduce these full methods.

## Two-speed memory follow-up

The six-world two-speed study is complete and **fails overall qualification**.
It adapts faster than stable-only memory and retains more than fast-only memory,
but recency adapts faster, uniform retains more, and quality/minimum-value gates fail.
See the [interpretation](results/two-speed-v1/INTERPRETATION.md),
[full results](results/two-speed-v1/RESULTS.md), and
[frozen protocol](experiments/two-speed-20261010/PREREGISTRATION.md).
The explorer adds a learning-versus-retention plot with independent-world dots,
world filters, stable-endpoint controls, keyboard tooltips and SVG legends.

This additive study has its own [index](two-speed-index.json) and numeric NPZ
archive in the dataset; the historical nine-study indexes remain unchanged.
Run it locally with the existing NumPy environment:

```text
python two_speed.py test extension_tests
python two_speed.py run --output results/my-two-speed-reproduction
python two_speed.py verify --output results/my-two-speed-reproduction
```

The code requires no keys, downloads or model services. The full trace archive
is exposed evaluation evidence and must not be reused as hidden test data.

All 243 tests and the complete refit/reselection/replay passed, including all
eleven history reloads excluding future feedback. Publication checks and their
limits are in [the receipt](provenance/two-speed-publication.json): provider
files match and the Space reports Running; local interactive QA passed, while
live visual verification was blocked on this workstation and the new dataset
table viewer was still processing. Scientific qualification remains **FAIL**.
