"""Reuse the unchanged generator, score transfer cases and causal predictions."""
import numpy as np
from confidence_lab.benchmark import (episode as old_episode, collect, digest_arrays,
                                     metrics, slices, losses, bootstrap)
from confidence_lab.policy import sigmoid
from .policy import Policy, OnlineMixture

FAMILIES = ("uniform","recency","fixed_shrinkage","frozen_gate","online_mixture")


def case_config(cfg, original, case):
    return {**original,**cfg["cases"][case]}


def episode(seed, steps, case, cfg, original):
    settings = case_config(cfg,original,case)
    return old_episode(seed,steps,settings["condition"],settings)


def fixed_grid(base, labels, cfg):
    return [{"alpha":a,"log_loss":float(losses(.5+a*(base-.5),labels).mean())}
            for a in cfg["alpha_grid"]]


def select(train, valid, gate, cfg):
    training_grid = fixed_grid(train[1],train[2],cfg)
    best_prior = min(training_grid,key=lambda r:r["log_loss"])["alpha"]
    prior = float(np.clip(best_prior,cfg["prior_clip"],1-cfg["prior_clip"]))
    validation_grid = fixed_grid(valid[1],valid[2],cfg)
    fixed = min(validation_grid,key=lambda r:r["log_loss"])["alpha"]
    candidates = []
    steps = cfg["validation_steps"]
    for share in cfg["sharing_candidates"]:
        for rate in cfg["learning_rate_candidates"]:
            total_loss = 0.
            for start in range(0,len(valid[2]),steps):
                online = OnlineMixture(prior,share,rate,cfg["log_odds_clip"])
                for base,label in zip(valid[1][start:start+steps],valid[2][start:start+steps]):
                    base,label = float(base),int(label)
                    p = .5+online.alpha*(base-.5)
                    total_loss -= np.log(p if label else 1-p)
                    online.observe(base,label)
            candidates.append({"prior":prior,"share":share,"learning_rate":rate,
                               "log_odds_clip":cfg["log_odds_clip"],"log_loss":float(total_loss/len(valid[2]))})
    best = min(range(len(candidates)),key=lambda i:candidates[i]["log_loss"])
    alpha = sigmoid(((valid[0]-gate.mean)/gate.scale) @ gate.weights)
    return {"training_data_sha256":train[3],"validation_data_sha256":valid[3],
            "training_examples":len(train[2]),"validation_examples":len(valid[2]),
            "prior_training_grid":training_grid,"prior_before_clip":best_prior,
            "fixed_validation_grid":validation_grid,"fixed_alpha":fixed,
            "online_validation_grid":candidates,"selected_online_index":best,
            "online":{k:v for k,v in candidates[best].items() if k!="log_loss"},
            "frozen_gate_validation_log_loss":float(losses(.5+alpha*(valid[1]-.5),valid[2]).mean())}


def evaluate(family, rows, gate, selection, original, *, feedback=True, persistent=None, namespace=""):
    policy = Policy(family,gate,selection,original["history_rates"])
    trace, features = [], []
    for step,row in enumerate(rows):
        prediction,x = policy.predict(row.observation)
        features.append(x)
        # No hidden row attribute except observation is accessed before prediction.
        trace.append({"step":step,"observation":list(row.observation.bits),**prediction,
                      "label":row.label,"mode":row.mode,"age":row.age,"mode_age":row.mode_age,
                      "rule_switched":row.rule_switched,"mode_switched":row.mode_switched})
        if feedback:
            policy.observe(row.observation,row.label)
            if persistent is not None:
                persistent.append(namespace,step,row.observation.bits,row.label)
    return metrics(trace,original),slices(trace),trace,digest_arrays(features)


def summary(runs):
    result = {}
    for case in sorted({r["case"] for r in runs}):
        result[case] = {}
        for family in FAMILIES:
            records = [r for r in runs if r["case"]==case and r["family"]==family]
            scores = [r["metrics"] for r in records]
            average = {k:float(np.mean([v[k] for v in scores if v[k] is not None]))
                       if any(v[k] is not None for v in scores) else None for k in scores[0]}
            average["slices"] = {}
            for key in records[0]["slices"]:
                parts = [r["slices"][key] for r in records]
                available = [p["metrics"] for p in parts if p["metrics"] is not None]
                average["slices"][key] = {"steps":sum(p["steps"] for p in parts),
                    "metrics":{k:float(np.mean([p[k] for p in available if p[k] is not None]))
                               if any(p[k] is not None for p in available) else None
                               for k in available[0]} if available else None}
            result[case][family] = average
    return result


def comparisons(runs, cfg):
    indexed = {(r["case"],r["seed"],r["family"]):r for r in runs}
    pairs = {}
    for left,right in (("frozen_gate","online_mixture"),("online_mixture","fixed_shrinkage"),
                       ("frozen_gate","fixed_shrinkage")):
        per_case = {}
        for case in cfg["cases"]:
            diffs = [indexed[(case,s,left)]["metrics"]["log_loss"]-
                     indexed[(case,s,right)]["metrics"]["log_loss"] for s in cfg["test_seeds"]]
            per_case[case] = diffs
            pairs[f"{case}/{left}_minus_{right}"] = bootstrap(diffs,cfg)
        # Reuse the same stream indices across the three related transfer cases.
        aggregate = np.array([per_case[c] for c in cfg["ood_cases"]]).mean(axis=0)
        pairs[f"aggregate_ood/{left}_minus_{right}"] = bootstrap(aggregate,cfg)
    return pairs


def scientific_gates(summaries, paired, cfg):
    def predictive(left,right):
        p = paired[f"aggregate_ood/{left}_minus_{right}"]
        components = {"aggregate_improvement":p["mean"] <= -cfg["minimum_log_loss_improvement"],
                      "paired_upper_below_zero":p["ci95"][1]<0}
        for case in cfg["ood_cases"]:
            components[case+"_loss_preserved"] = (summaries[case][left]["log_loss"]-
                summaries[case][right]["log_loss"] <= cfg["maximum_case_degradation"])
        return components
    def quality(family,reference):
        checks = {}
        for case in cfg["ood_cases"]:
            for direction in ("learnable_entry_first16","random_entry_first16"):
                a = summaries[case][family]["slices"][direction]["metrics"]["log_loss"]
                b = summaries[case][reference]["slices"][direction]["metrics"]["log_loss"]
                checks[case+"/"+direction] = a-b <= cfg["maximum_transition_degradation"]
        checks["standard_loss_preserved"] = (summaries["standard"][family]["log_loss"]-
            summaries["standard"]["recency"]["log_loss"] <= cfg["maximum_standard_degradation"])
        random = summaries["random_labels"]
        checks["random_loss_near_uniform"] = (random[family]["log_loss"]-random["uniform"]["log_loss"]
                                               <= cfg["maximum_random_loss_above_uniform"])
        checks["random_confident_errors_halved"] = (random[family]["confidently_wrong_rate"] <=
            cfg["maximum_confident_error_ratio"]*random["recency"]["confidently_wrong_rate"])
        return checks
    learned_value = predictive("frozen_gate","online_mixture")
    learned_quality = quality("frozen_gate","online_mixture")
    for case in cfg["ood_cases"]:
        learned_quality[case+"_confident_errors_preserved"] = (
            summaries[case]["frozen_gate"]["confidently_wrong_rate"]-
            summaries[case]["online_mixture"]["confidently_wrong_rate"] <= cfg["maximum_ood_confident_error_excess"])
    online_value = predictive("online_mixture","fixed_shrinkage")
    online_quality = quality("online_mixture","fixed_shrinkage")
    gate_ok = all(learned_value.values()) and all(learned_quality.values())
    online_ok = all(online_value.values()) and all(online_quality.values())
    return {"learned_transfer_value":all(learned_value.values()),
            "learned_transfer_quality":all(learned_quality.values()),"learned_default_qualified":gate_ok,
            "online_value_over_fixed":all(online_value.values()),"online_quality":all(online_quality.values()),
            "online_default_qualified":online_ok,
            "recommended_confidence_default":"frozen_gate" if gate_ok else "online_mixture" if online_ok else "none",
            "components":{"learned_value":learned_value,"learned_quality":learned_quality,
                          "online_value":online_value,"online_quality":online_quality}}
