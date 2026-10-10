"""Evaluator-private worlds, causal data collection and probability scoring."""
from dataclasses import dataclass
import hashlib
import numpy as np
from archlab.public import Observation, experts
from .policy import ConfidenceState

FAMILIES = ("uniform", "recency", "fixed_shrinkage", "adaptive_gate")


@dataclass(frozen=True)
class PrivateRow:
    observation: Observation
    label: int
    mode: str
    age: int
    mode_age: int
    rule_switched: bool
    mode_switched: bool


def episode(seed, steps, condition, cfg):
    if condition not in cfg["conditions"]:
        raise ValueError(condition)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    xr, rr, nr, yr, mr = streams
    low, high = cfg["fast_rule_interval" if condition=="fast_switch" else "standard_rule_interval"]
    mode_low, mode_high = cfg["mixed_mode_interval"]
    rule = int(rr.integers(4))
    remaining = int(rr.integers(low, high+1))
    mixed_mode = "learnable" if mr.integers(2) else "random"
    mode_remaining = int(mr.integers(mode_low, mode_high+1))
    age, mode_age = 0, 0
    rows = []
    for _ in range(steps):
        rule_changed, mode_changed = False, False
        if remaining == 0:
            rule = (rule + int(rr.integers(1,4))) % 4
            remaining = int(rr.integers(low, high+1))
            age, rule_changed = 0, True
        if mode_remaining == 0:
            mixed_mode = "random" if mixed_mode=="learnable" else "learnable"
            mode_remaining = int(mr.integers(mode_low, mode_high+1))
            mode_age, mode_changed = 0, True
        obs = Observation(tuple(int(x) for x in xr.integers(0,2,2)))
        learned_y = int(experts(obs)[rule]) ^ int(nr.random()<cfg["noise_probability"])
        random_y = int(yr.integers(2))
        mode = (mixed_mode if condition in ("mixed","no_feedback") else
                "random" if condition=="random_labels" else "learnable")
        # The random stream and public observations do not depend on hidden mode.
        y = learned_y if mode=="learnable" else random_y
        visible_mode_change = mode_changed and condition in ("mixed","no_feedback")
        rows.append(PrivateRow(obs,y,mode,age,mode_age if condition in ("mixed","no_feedback") else len(rows),
                               rule_changed and mode=="learnable" and not visible_mode_change,
                               visible_mode_change))
        remaining -= 1
        mode_remaining -= 1
        age += 1
        mode_age += 1
    return rows


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array, dtype="<f8")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def collect(seeds, steps, cfg):
    features, base, labels = [], [], []
    for condition in cfg["fit_conditions"]:
        for seed in seeds:
            state = ConfidenceState(cfg["history_rates"])
            for row in episode(seed,steps,condition,cfg):
                p = state.base(row.observation)
                features.append(state.features(p))
                base.append(p)
                # Current label is appended only after features/prediction are fixed.
                labels.append(row.label)
                state.observe(row.observation,row.label)
    arrays = tuple(np.asarray(x,dtype=float) for x in (features,base,labels))
    return (*arrays, digest_arrays(*arrays))


def losses(probability, label):
    p, y = np.asarray(probability), np.asarray(label)
    return -y*np.log(p)-(1-y)*np.log1p(-p)


def basic_metrics(trace):
    if not trace:
        return None
    p = np.array([r["probability"] for r in trace])
    y = np.array([r["label"] for r in trace])
    correct = (p>=.5)==y
    confidence = np.maximum(p,1-p)
    high = confidence>=.9
    ece = 0.
    for i in range(10):
        mask = (confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1)
        if mask.any():
            ece += mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return {"accuracy":float(correct.mean()),"log_loss":float(losses(p,y).mean()),
            "brier":float(np.mean((p-y)**2)),"ece":float(ece),
            "confidently_wrong_rate":float((high & ~correct).mean()),
            "high_confidence_coverage":float(high.mean()),
            "high_confidence_error_rate":float((~correct[high]).mean()) if high.any() else None,
            "mean_alpha":float(np.mean([r["alpha"] for r in trace]))}


def metrics(trace, cfg):
    score = basic_metrics(trace)
    starts = [i for i,r in enumerate(trace) if r["rule_switched"]]
    switch_rows, penalties, recovered = [], [], 0
    window, required, horizon = (cfg[k] for k in ("recovery_window","recovery_correct","recovery_horizon"))
    for start in starts:
        end = next((i for i in range(start+1,len(trace))
                    if trace[i]["rule_switched"] or trace[i]["mode_switched"]),len(trace))
        switch_rows.extend(trace[start:min(end,start+cfg["switch_score_window"])])
        stop = min(end,start+horizon)
        if stop-start < window:
            continue
        correct = [int((r["probability"]>=.5)==r["label"]) for r in trace[start:stop]]
        endpoint = next((i+window-1 for i in range(len(correct)-window+1)
                         if sum(correct[i:i+window])>=required),None)
        penalties.append(horizon if endpoint is None else endpoint)
        recovered += endpoint is not None
    score.update(rule_switch_log_loss=basic_metrics(switch_rows)["log_loss"] if switch_rows else None,
                 rule_switch_scored_steps=len(switch_rows),switches_scored=len(penalties),
                 recovery_penalized_score=float(np.mean(penalties)) if penalties else None,
                 recovery_fraction=recovered/len(penalties) if penalties else None)
    return score


def slices(trace):
    masks = {
        "learnable":lambda r:r["mode"]=="learnable",
        "random":lambda r:r["mode"]=="random",
        "random_entry_first16":lambda r:r["mode"]=="random" and r["mode_age"]<16 and r["step"]-r["mode_age"]>0,
        "random_late32":lambda r:r["mode"]=="random" and r["mode_age"]>=32,
        "learnable_entry_first16":lambda r:r["mode"]=="learnable" and r["mode_age"]<16 and r["step"]-r["mode_age"]>0,
        "learnable_late32":lambda r:r["mode"]=="learnable" and r["mode_age"]>=32}
    return {name:{"steps":len(selected),"metrics":basic_metrics(selected)}
            for name,mask in masks.items() for selected in [[r for r in trace if mask(r)]]}


def evaluate(family, rows, gate, fixed_alpha, cfg, *, feedback=True, persistent=None, namespace=""):
    if family not in FAMILIES:
        raise ValueError(family)
    state = ConfidenceState(cfg["history_rates"])
    trace, feature_rows = [], []
    for step,row in enumerate(rows):
        base = state.base(row.observation)
        features = state.features(base)
        feature_rows.append(features)
        alpha = (gate.alpha(features) if family=="adaptive_gate" else
                 fixed_alpha if family=="fixed_shrinkage" else 1. if family=="recency" else 0.)
        p = float(.5+alpha*(base-.5))
        # These evaluator-only values are attached after inference has finished.
        trace.append({"step":step,"observation":list(row.observation.bits),"base_probability":base,
                      "probability":p,"alpha":alpha,"label":row.label,"mode":row.mode,
                      "age":row.age,"mode_age":row.mode_age,"rule_switched":row.rule_switched,
                      "mode_switched":row.mode_switched})
        if feedback:
            state.observe(row.observation,row.label)
            if persistent is not None:
                persistent.append(namespace,step,row.observation.bits,row.label)
    return metrics(trace,cfg), slices(trace), trace, digest_arrays(feature_rows)


def bootstrap(differences, cfg):
    values = np.asarray(differences,dtype=float)
    if values.ndim != 1 or len(values)==0 or not np.isfinite(values).all():
        raise ValueError("finite per-stream paired differences required")
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    indices = rng.integers(len(values),size=(cfg["bootstrap_replicates"],len(values)))
    draws = values[indices].mean(axis=1)
    return {"mean":float(values.mean()),"ci95":np.quantile(draws,[.025,.975]).tolist(),
            "per_stream":values.tolist(),"replicates":cfg["bootstrap_replicates"],
            "seed":cfg["bootstrap_seed"]}
