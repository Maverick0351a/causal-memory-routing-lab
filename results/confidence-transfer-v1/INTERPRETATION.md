# What the transfer test establishes

The unchanged learned gate passed both frozen transfer gates. Across three
previously unseen mode/noise settings, gate minus online mean log loss was
-0.012546 nats with paired stream-bootstrap 95% interval [-.013292,-.011789].
All 24 seed-level three-case averages favored the gate. The gate met the per-case
loss bounds and all transition/standard/noise confidence requirements.

| Shifted case | Fixed shrinkage | Online mixture | Frozen gate |
|---|---:|---:|---:|
| 16-48-step modes, 5% learnable noise | .60884 | .61711 | .60065 |
| 96-192-step modes, 20% learnable noise | .71026 | .67707 | .66646 |
| 16-48-step modes, 20% learnable noise | .71683 | .68981 | .67924 |

The online baseline also beat fixed shrinkage on the aggregate unseen cases
(-.017314 nats, 95% interval [-.020319,-.014413]), but failed four of its six
transition requirements against fixed shrinkage. It does not qualify as the
default under its frozen fallback rule. This failed quality gate remains recorded.
The decision favors the frozen learned gate as a local confidence reference.

## What did not improve universally

- On the original mixed distribution, online was better: .53498 versus gate
  .54115. Gate-minus-online difference +.006164, descriptive interval
  [.005454,.006891]. The transfer result does not establish universal superiority.
- Recovery after noise remains costly. On short 5%-noise modes, the first 16
  learnable steps had loss .50889 for the gate versus fixed .43319 and recency
  .40176. Gate still beat online .52888. Passing a transition margin against
  online does not mean the gate is the fastest possible learner.
- On long 20%-noise modes, entry into learnable segments had loss .65746 for the
  gate versus online .65013 and fixed .61559. This passes the frozen +.03 margin
  relative to online, not an equivalence test or a victory on that slice.
- On continuously learnable 20%-noise rules, fixed shrinkage was best: .60790
  versus gate .62467 and online .64948. This control was retained; it was not
  used to select hyperparameters or redefine the three-case primary objective.
- Standard loss is .33438 versus recency .33226. Random-label gate loss .69698
  is close to uniform .69315; confident errors were .003%, not exactly zero.

## Numerical ties and scope

The gate and online mixture change confidence, not classification sign in exact
arithmetic. There are six online-versus-recency threshold differences in the saved
traces: the same seed 920023/step 8 across six cases. Every base probability is
.49999999999999994 and online shrinkage rounds it to .5. The largest distance
from the threshold is 5.55e-17. See rounding-audit.json. The tiny displayed online
accuracy increase is numerical tie handling, not improved recognition. No scores
or frozen rules were changed after observing this issue.

These are shifts within the same two-bit, known-expert world family. The gate saw
similar original-distribution training in the earlier experiment; the baseline
used fresh training/validation seeds from that distribution. Neither saw the new
shorter/higher-noise test settings during selection. Intervals describe variation
over 24 independent held-out streams conditional on these fitted/selected models;
they do not measure training uncertainty, all possible alternatives or transfer
to different tasks. Private modes and transitions enter scoring only.

No pretrained model or package was installed, and no real assistant was
reconfigured. This is evidence for a causal probability-calibration component,
not better class recognition, semantic retrieval, molecular ArBG reproduction,
quantum advantage or general intelligence. A later experiment should change the
task family and compare another conventional online calibration method before
claiming broader architecture value.

Separate-process verification passed: all 960 traces and 1,474,560 predictions
replayed exactly, baseline selection and all score/default decisions recomputed,
and both gate/online predictions resumed from SQLite before step 768. The 1,530
old lab files, three original Quantum files and pinned vendor were unchanged.
Seventy local tests passed. No provider credentials were present in the guarded
experiment processes. These checks support reproducibility and causal data flow;
the Python socket guard remains a process-level check, not an OS sandbox.
