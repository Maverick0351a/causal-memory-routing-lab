# What this experiment establishes

The unchanged confidence gate transfers a modest amount of probability-quality
benefit to a new synthetic continuous-input task. It does **not** meet the frozen
practical-value threshold, so sensor-family qualification is **FAIL**. Quality
and random-label controls are **PASS**. These are scientific gates; the software
tests and exact replay passed independently.

Gate minus direct online Platt calibration, averaged within each seed over four
stress cases, is **-.005716 nats**, paired-seed95% interval
**[-.006882,-.004492]**. The interval excludes zero, but the predeclared target
was at least .01 nats improvement. The entire interval falls short of that target.
Twenty-two of 24 paired seed aggregates favor the gate. Do not relabel this as a
passed hypothesis by lowering the threshold after seeing the test results.

| Case | Frozen gate loss | Online calibration loss | Difference |
|---|---:|---:|---:|
| Mixed signal/noise | .58400 | .59557 | -.01157 |
| Short mixed intervals | .61755 | .63278 | -.01523 |
| Weak signal | .66850 | .66461 | +.00388 |
| Rotating coupling | .66623 | .66618 | +.00006 |
| Stable control | .41823 | .40188 | +.01635 |
| Random-label control | .69788 | .71523 | -.01736 |

The gate has its clearest benefit at uncertainty changes. Weak signal slightly
favors online calibration; changing couplings are essentially tied on log loss.
Stable operation favors online calibration. Gate beats the pooled fixed
calibrator on the four-stress-case aggregate by .079491 nats; online beats that
fixed calibrator by .073775 nats. Improvement against an offline baseline does
not satisfy the harder comparison with causal online adaptation.

On random labels the gate makes **0 confident mistakes out of 36,864 predictions**
at confidence>=.9, while the uncalibrated base rate is 13.517%. Gate log loss is
.69788 versus uniform .69315. This supports uncertainty suppression in this
generator; it does not identify hidden faults or demonstrate new recognition.
Gate and base accuracy are exactly equal in every case because positive
shrinkage preserves class sign. Online calibration can change bias or invert
sign: rotating-case accuracy is 58.244% online versus 53.258% base/gate. Its extra
recognition capacity comes from the conventional calibrator, not the gate.

Retain the cost of recovering confidence when signal returns. In mixed streams,
first 16 signal-entry loss is .57733 gate versus .42507 fixed/.40355 base and
.57092 online. The descriptive penalized response score is 21.03 steps gate
versus 19.04 online; noise-entry response instead favors gate 21.22 versus 27.60.
Short mixed signal response is 26.70 versus 23.81, noise 27.19 versus 37.20. These
are stochastic loss-window metrics, with unrecovered eligible events penalized
at96 and search truncated at the next change. They are not change-detector
latencies or proof of learning a new fault model.

## Evidence and scope

- 92 local tests passed, including 22 new tests for analytic gradients, public
  input boundaries, predict-before-answer access, future-label invariance,
  feedback ablation, frozen weights and strict memory cutoffs.
- A separate process exactly refit the 36-parameter base ensemble, refit fixed
  calibration, reselected all 12 online candidates and replayed 168 streams /
  840 policy traces / 1,290,240 predictions. Metrics, slices, response scores,
  bootstraps and every scientific gate were recomputed exactly.
- Gate and online policies both resumed their step 768 predictions from 768
  SQLite rows with step<768. The schema contains only public readings and
  released labels. This proves persistence, not a benefit from episodic recall.
- Gate checkpoint SHA256 remains
  `ea13900be7487f6030189f4fc80545d975ee47a58aa46b2c3117412eab37fe4f`;
  weights/normalization and the learned ensemble stayed immutable during test.
- 2,552 prior lab files and three original Quantum files retained their hashes.
  Vendor stays clean at the pinned NVIDIA commit. All earlier experiments,
  defaults, source and test decisions are preserved.
- Fitting/selection took 3.59 seconds; evaluation and writing traces 18.22 seconds,
  excluding fresh-process replay and hashing. No GPU/QPU work, API calls,
  downloads, pretrained weights or new services were used.

The model and observation family changed substantially from the two-bit task,
but this is still a deliberately small synthetic benchmark with balanced binary
labels and Gaussian inputs. It shares the original signal/noise uncertainty
structure and uses a chosen causal feature adapter. It is not independent
evidence over realistic sensor distributions. Intervals condition on one trained
ensemble and one validation selection, and do not include fit/selection variance.
Other base architectures, class imbalance, delayed feedback and real anomalies
remain untested. The Python socket guard is not an OS sandbox.

Keep the gate experimental for this new task. The existing reference registry
remains scoped to the original two-bit family; no live assistant configuration
was changed. All failed gates and held-out traces remain available for review.
