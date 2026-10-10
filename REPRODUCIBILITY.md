# Reproducibility and publication provenance

This is a sanitized derivative of the isolated local research workspace.
Scientific kernels, training, inference, selection, metrics and thresholds
are retained. `provenance/source-map.json` gives original and public SHA-256
digests and explicitly lists every adapted copied source/test file.

Portability adaptations replace only `check_preservation()` and `hashes()` in
experiment orchestration. Public checks protect released source/configuration
and frozen toy checkpoints. They do not pretend to inspect the original
user's private files, workstation or vendor Git checkout. Host-only test
assertions are correspondingly replaced with actual public-manifest checks.
The rest of the scientific test assertions are retained. Empty per-experiment
`preservation.json` files are compatibility placeholders; the real public
integrity inventory is `provenance/portable-source-manifest.json`.

Historical result/verification JSON retains numerical values and candid
failures. Absolute local path values are redacted. Their original file
digests are recorded in `research-index.json`; the public files have separate
release digests. An original source digest is not a digest of adapted public
orchestration. Original artifact manifests and machine audit logs are omitted.

The local original suite passed 194 tests, with 94 selected upstream QCA tests
at initial setup. Public-package tests and published-artifact replay have
separate receipts under `provenance/`; do not conflate their scopes. Earlier
development failures are summarized in `provenance/development-failures.md`.

Historical receipts attest to those recorded local runs. Public replay checks
released bytes and results independently; it does not re-attest private host
preservation. Exact floating-point replay was checked in Python 3.11/NumPy
2.4.6 on Windows. Other BLAS/platforms may differ; report a mismatch honestly
instead of loosening thresholds to make a replay pass. Runtime/latency values
are observations of the original machine and load, not portable guarantees.

## Optional NVIDIA QCA reproduction

The research core does not need QCA packages. For the keyless synthetic demo,
in a separate environment, clone the official NVIDIA repository and check out
`d8d37bc4de28d2347d8e21405903b29a2891f793` at
`vendor/Quantum-Calibration-Agent-Blueprint`. Review its code before running it.
`audit/qca-requirements.lock` records the original official-PyPI package hashes;
QCA also needs the pinned vendor editable install. The `forbiddenfruit`
dependency was the reviewed source-build exception. `scripts/qca_local.py`
provides the external Windows path compatibility adapter without modifying
vendor code. It permits explicit experiments/history/data subcommands only.

After that optional setup:

```text
python scripts/qca_local.py experiments run t1_measurement
python scripts/qca_local.py history list
python scripts/qca_local.py history show EXPERIMENT_ID
```

`results/qca-synthetic.json` publishes the original synthetic measurements,
arrays and sanitized save/reload receipt. No QPU, VLM call or API key is needed.
Local inventory, credentials, old project files, vendor source, installed
environments, service configuration and unrelated security research are absent.

## Additive two-speed study

`two_speed.py` preserves the earlier entry point and source inventory. Its new
protocol and package are hashed in `results/two-speed-v1/preregistration.json`,
in addition to the unchanged historical source manifest.

```text
python two_speed.py test extension_tests
python two_speed.py run --output results/my-two-speed-reproduction
python two_speed.py verify --output results/my-two-speed-reproduction
```

For the published run, explicitly download `artifacts/two-speed-v1.tar.gz`
from the dataset and extract it into a fresh directory; never overwrite a prior
run. Validate its digest against `two-speed-index.json`. The archive uses
numeric NPZ traces (`numpy.load(..., allow_pickle=False)`), checkpoints,
SQLite feedback, frozen source copies, selection and receipts. Then use
`python two_speed.py verify --output PATH/TO/EXTRACTED_RUN`. Verification
refits all six worlds and twelve recurrent candidates, repeats all 114
validation candidates, and reproduces every forecast, metric, interval, gate
and all eleven strict-prior-feedback history reloads. This repeats the
consumed evaluation, not a new scientific confirmation.

The new graphs are in `site/two-speed-charts.json`; they are combined with
the unchanged historical graph data only in the static explorer.
