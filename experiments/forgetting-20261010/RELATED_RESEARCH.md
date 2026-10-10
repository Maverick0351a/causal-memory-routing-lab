# Related research checked before the experiment

Primary sources inspected on2026-10-10. No repositories, packages, PDFs or weights
were downloaded into this workspace. These are references and design context;
the experiment is not a reproduction or a comparison against these full methods.

| Reference | Relevant idea | Relationship to this experiment |
|---|---|---|
| Losing, Hammer, Wersing (2017), [Self-Adjusting Memory: How to Deal with Diverse Drift Types](https://www.ijcai.org/Proceedings/2017/0690.pdf), IJCAI, DOI10.24963/ijcai.2017/690 | SAM-kNN maintains short-term and long-term memories, adapts the recent window, cleans conflicting historical examples and selects among memory models by past performance. | Closest classical memory reference. Our fading rule does not implement adaptive windows, consistency cleaning, long-term compression or model selection among stores. |
| Bifet, Gavalda (2007), [Learning from Time-Changing Data with Adaptive Windowing](https://epubs.siam.org/doi/10.1137/1.9781611972771.42), SDM | ADWIN adjusts its observation window from observed changes rather than a fixed time scale; the paper provides statistical guarantees under its assumptions. | Motivates testing stale-data handling. Our validation-selected fixed half-life is not ADWIN, has no adaptive cut detector and inherits none of its guarantees. Publisher abstract inspected; the author-hosted PDF fetch failed. |
| Kolter, Maloof (2007), [Dynamic Weighted Majority: An Ensemble Method for Drifting Concepts](https://www.jmlr.org/papers/v8/kolter07a.html), JMLR8:2755-2790 | DWM updates and weights online experts and can add or remove them as performance changes. | Related routing/ensemble context. Our four fitted experts are frozen; fading affects retrieved mismatch summaries and never creates/deletes experts. Publisher abstract inspected. |
| Behrouz, Zhong, Mirrokni (2025), [Titans: Learning to Memorize at Test Time](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf), NeurIPS38; [arXiv2501.00663](https://arxiv.org/abs/2501.00663) | Neural long-term associative memory is updated at test time using gradient-based surprise, momentum and a data-dependent forgetting gate. Section2.1, especially equation3, describes forgetting as gated weight decay. | Broader architecture reference. Our measured-cue lookup and fixed exponential rank weights have no neural associative memory, learned forgetting gate, attention integration or language-model training. |

The local inference from these sources is that forgetting is an established
stability-versus-adaptation problem, not a new architectural discovery here.
This protocol therefore keeps a recent-uniform ablation to distinguish fading
from recency selection, gives all tuned baselines the same validation objective,
and separately gates stable recall and post-change entry quality. Those are
our experimental choices, not claims of implementing the referenced algorithms.

This is a focused reference check, not an exhaustive review or proof of novelty.
The earlier autoregressive routing failures still stand; the current experiment
changes only feedback-memory use and baseline training/selection for this task.
