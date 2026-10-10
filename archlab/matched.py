"""Preregistered AR versus joint routing comparison with matched training budgets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
from .benchmark import episode, metrics
from .controllers import Adam, AutoregressiveRouter, JointRouter, softmax
from .public import FeedbackState, experts
from .memory import Memory

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/"experiments/matched-routing-20261010"


def config():
    return json.loads((EXPERIMENT/"protocol.json").read_text())


def input_digest(hasher, x):
    hasher.update(np.asarray(x, dtype="<f8").tobytes())


def train(model, cfg):
    opt = Adam(model.params, rate=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["shuffle_seed"])
    examples = {seed:episode(seed, cfg["train_steps"]) for seed in cfg["training_seeds"]}
    digest = hashlib.sha256()
    epoch_losses, seen, updates = [], 0, 0
    started = time.perf_counter()
    for _ in range(cfg["epochs"]):
        losses = []
        for seed in rng.permutation(cfg["training_seeds"]):
            state = FeedbackState()
            model.reset()
            caches, outputs, labels = [], [], []
            for i, (obs, y, _, _) in enumerate(examples[int(seed)]):
                x = state.features(obs)
                input_digest(digest, x)
                _, cache = model.forward(x)
                # Only after forward do the released label and expert outcomes enter loss.
                caches.append(cache)
                outputs.append(experts(obs))
                labels.append(y)
                state.observe(obs, y)
                seen += 1
                if len(caches) == cfg["batch_size"] or i == cfg["train_steps"]-1:
                    grads, loss = model.gradients(caches, outputs, labels)
                    opt.step(model.params, grads)
                    updates += 1
                    losses.append(loss)
                    caches, outputs, labels = [], [], []
        epoch_losses.append(float(np.mean(losses)))
    model.reset()
    return {"loss_by_epoch":epoch_losses, "seconds":time.perf_counter()-started,
            "examples_seen":seen, "optimizer_updates":updates,
            "input_sha256":digest.hexdigest(), "parameters":model.parameter_count,
            "forward_affine_macs":sum(v.size for v in model.params.values() if v.ndim == 2)}


def adaptation(trace, cfg):
    starts = [i for i,r in enumerate(trace) if r["switched"]]
    endpoints, penalties = [], []
    included = 0
    window, required, horizon = (cfg[k] for k in ("recovery_window", "recovery_correct", "recovery_horizon"))
    for index, start in enumerate(starts):
        end = starts[index+1] if index+1 < len(starts) else len(trace)
        limit = min(end-start, horizon)
        if limit < window:
            continue
        included += 1
        correct = [int((r["probability"] >= .5) == r["label"]) for r in trace[start:start+limit]]
        endpoint = next((off+window-1 for off in range(limit-window+1)
                         if sum(correct[off:off+window]) >= required), None)
        if endpoint is not None:
            endpoints.append(endpoint)
        penalties.append(horizon if endpoint is None else endpoint)
    return {"switches_scored":included, "recovered_switches":len(endpoints),
            "recovery_fraction":len(endpoints)/included if included else None,
            "recovery_endpoint_median":float(np.median(endpoints)) if endpoints else None,
            "recovery_penalized_score":float(np.mean(penalties)) if penalties else None}


def evaluate(model, rows, cfg, *, name, rng_seed, feedback=True, persistent=None, namespace=""):
    state = FeedbackState()
    rng = np.random.default_rng(rng_seed)
    digest = hashlib.sha256()
    if model is not None:
        model.reset()
    trace = []
    started = time.perf_counter()
    for step, (obs, y, age, switched) in enumerate(rows):
        x = state.features(obs)
        input_digest(digest, x)
        if model is not None:
            q, cache = model.forward(x)
        elif name == "recency":
            q = softmax(-8*state.fast)
        elif name == "fixed_uniform":
            q = np.full(4, .25)
        else:
            raise ValueError(name)
        probability = float(.05+.9*(q @ experts(obs)))
        if isinstance(model, AutoregressiveRouter):
            # Use the already computed conditionals; avoid repeated forward passes.
            _, _, pa, pb = cache
            a = int(rng.choice(2, p=pa))
            b = int(rng.choice(2, p=pb[a]))
            action = 2*a+b
        else:
            action = int(rng.choice(4, p=q))
            a, b = action//2, action%2
        # No evaluator-only value is consulted until the probability and route are fixed.
        trace.append({"step":step, "observation":list(obs.bits), "probability":probability,
                      "route":[a,b], "log_q":float(np.log(q[action])), "label":y,
                      "age":age, "switched":switched,
                      "sampled_correct":bool(experts(obs)[action] == y)})
        if feedback:
            state.observe(obs, y)
            if persistent is not None:
                persistent.append(namespace, step, obs.bits, y)
    elapsed = time.perf_counter()-started
    score = metrics(trace)
    score.update(adaptation(trace, cfg))
    return score, trace, {"seconds":elapsed, "input_sha256":digest.hexdigest()}


def paired_bootstrap(matrix, *, seed, replicates):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("finite nonempty paired matrix required")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for i in range(replicates):
        model_indices = rng.integers(matrix.shape[0], size=matrix.shape[0])
        episode_indices = rng.integers(matrix.shape[1], size=matrix.shape[1])
        draws[i] = matrix[np.ix_(model_indices, episode_indices)].mean()
    return {"mean":float(matrix.mean()),
            "ci95":[float(x) for x in np.quantile(draws, [.025,.975])],
            "per_model_seed":matrix.mean(axis=1).tolist(),
            "per_episode_seed":matrix.mean(axis=0).tolist(), "paired_matrix":matrix.tolist(),
            "bootstrap_seed":seed,"bootstrap_replicates":replicates}


def summary(runs):
    result = {}
    for condition in sorted({r["condition"] for r in runs}):
        result[condition] = {}
        for family in ("fixed_uniform","recency","autoregressive","joint"):
            values = [r["metrics"] for r in runs if r["condition"]==condition and r["family"]==family]
            # These are equally weighted run means. Counts get totals for clarity.
            result[condition][family] = {
                k:float(np.mean([v[k] for v in values if v[k] is not None]))
                if any(v[k] is not None for v in values) else None for k in values[0]}
            for key in ("switches_scored","recovered_switches"):
                result[condition][family][key] = sum(v[key] for v in values)
    return result


def comparisons(runs, cfg):
    indexed = {(r["condition"], r["family"], r.get("model_seed"),r["seed"]):r for r in runs}
    result = {}
    for metric in ("log_loss","accuracy","confidently_wrong_rate","recovery_penalized_score"):
        matrix = [[indexed[("standard","autoregressive",m,e)]["metrics"][metric] -
                   indexed[("standard","joint",m,e)]["metrics"][metric]
                   for e in cfg["test_seeds"]] for m in cfg["model_seeds"]]
        result[metric] = paired_bootstrap(matrix, seed=cfg["bootstrap_seed"], replicates=cfg["bootstrap_replicates"])
    return result


def scientific_gates(summaries, paired, training, latency, cfg):
    ar = training[f"autoregressive_{cfg['model_seeds'][0]}"]
    joint = training[f"joint_{cfg['model_seeds'][0]}"]
    parameter_gap = abs(joint["parameters"]-ar["parameters"])/ar["parameters"]
    mac_gap = abs(joint["forward_affine_macs"]-ar["forward_affine_macs"])/ar["forward_affine_macs"]
    matched = parameter_gap <= cfg["maximum_parameter_gap_fraction"] and mac_gap <= cfg["maximum_forward_mac_gap_fraction"]
    predictive = (matched and paired["log_loss"]["mean"] <= -cfg["minimum_log_loss_improvement"]
                  and paired["log_loss"]["ci95"][1] < 0)
    gates = {"comparable_size_and_affine_compute":matched, "ar_predictive_value":predictive,
             "ar_latency_within_budget":latency["ar_over_joint_ratio"] <= cfg["maximum_forward_latency_ratio"]}
    for family in ("autoregressive","joint"):
        standard = summaries["standard"][family]
        gates[family+"_learning"] = standard["accuracy"] >= .70 and standard["accuracy"] >= summaries["standard"]["fixed_uniform"]["accuracy"]+.10
        gates[family+"_negative_control"] = .45 <= summaries["random_labels"][family]["accuracy"] <= .55
        gates[family+"_memory_dependence"] = standard["accuracy"] >= summaries["no_feedback"][family]["accuracy"]+.10
    return gates, {"parameter_gap_fraction":parameter_gap,"forward_mac_gap_fraction":mac_gap}


def timing(models, cfg):
    vectors = np.random.default_rng(cfg["timing_seed"]).normal(0,.25,(cfg["timing_vectors"],24))
    blocks = []
    for i, seed in enumerate(cfg["model_seeds"]):
        pair = {family:models[f"{family}_{seed}"] for family in ("autoregressive","joint")}
        for model in pair.values():
            for x in vectors[:128]:
                model.forward(x)
        for rep in range(cfg["timing_repetitions"]):
            order = ["autoregressive","joint"] if (rep+i)%2==0 else ["joint","autoregressive"]
            for family in order:
                started = time.perf_counter_ns()
                for x in vectors:
                    pair[family].forward(x)
                blocks.append({"family":family,"model_seed":seed,"repetition":rep,
                               "nanoseconds_per_forward":(time.perf_counter_ns()-started)/len(vectors)})
    medians = {f:float(np.median([b["nanoseconds_per_forward"] for b in blocks if b["family"]==f]))
               for f in ("autoregressive","joint")}
    return {"median_nanoseconds_per_forward":medians,
            "ar_over_joint_ratio":medians["autoregressive"]/medians["joint"],
            "blocks":blocks,"scope":"marginal forward only; warm-up excluded; interleaved timing"}


def hashes():
    """Public adapter: snapshot portable sources and frozen configuration."""
    from release_tools.integrity import source_hashes
    return source_hashes(ROOT)


def check_preservation():
    """Public adapter: verifies released files, never original private projects."""
    from release_tools.integrity import verify_release
    return verify_release(ROOT)


def write_report(result, output):
    rows = ["# Matched routing comparison", "", "Primary verdict: **"+
            ("PASS" if result["gates"]["ar_predictive_value"] else "FAIL")+"** for autoregressive predictive value.",
            "", "Fresh training, initialization and held-out seeds; no tuning after evaluation.", "",
            "| Condition | Model | Accuracy | Log loss | Brier | ECE | Confidently wrong | Recovery score* |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for condition in result["config"]["conditions"]:
        for family, score in result["summary"][condition].items():
            recovery = score["recovery_penalized_score"]
            rows.append(f"| {condition} | {family} | {score['accuracy']:.3%} | {score['log_loss']:.5f} | {score['brier']:.5f} | {score['ece']:.5f} | {score['confidently_wrong_rate']:.3%} | {recovery:.2f} |")
    primary = result["paired"]["log_loss"]
    rows += ["", "AR minus joint standard log loss: **"+f"{primary['mean']:.6f}"+" nats**; paired two-way bootstrap 95% interval: "+
             f"[{primary['ci95'][0]:.6f}, {primary['ci95'][1]:.6f}]. Negative favors AR.",
             "Required: <=-0.01 nats and upper interval endpoint <0, with matched size/affine compute.",
             "", "Five model seeds share ten test streams. The 50 paired cells are not 50 independent environments.",
             "The bootstrap resamples both seed axes; this task-level experiment cannot establish broad superiority.",
             "", "## Compute and latency", "",
             "AR: 750 parameters, 720 affine MACs. Joint: 758 parameters, 728 affine MACs.",
             "MAC counts omit activations/softmax; measured forward times include them."]
    latency = result["latency"]
    rows += [f"Median marginal forward: AR {latency['median_nanoseconds_per_forward']['autoregressive']/1000:.3f} us; joint {latency['median_nanoseconds_per_forward']['joint']/1000:.3f} us; ratio {latency['ar_over_joint_ratio']:.3f}x.",
             "Sampling reuses the already computed conditionals; both use one forward per evaluated step.",
             "Timing is specific to this machine/load; the efficiency gate is separate from the primary outcome.",
             "", "## Frozen scientific gates", ""]
    rows += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name,passed in result["gates"].items()]
    rows += ["", "*Recovery score: mean endpoint offset for first 7/8-correct rolling window after an actual switch; 32 is assigned if censored before the next switch/horizon. Lower is better. Startup is excluded. On random labels, chance runs have no adaptation meaning.",
             "Recovery fraction, recovered-only median, all paired measurements and per-seed metrics are in results.json.",
             "", "Training data order and feature vectors were hash-checked identical across both model families.",
             "Checkpoints and complete traces are saved. Fresh-process replay evidence is written separately to verification.json.",
             "No calibration was fit on test labels. Random-label accuracy passing does not imply calibrated uncertainty.",
             "This uses known toy experts and hand-designed feedback features, with no hardware or semantic-retrieval claim."]
    (output/"RESULTS.md").write_text("\n".join(rows)+"\n")


def run(output):
    if os.environ.get("ARCHLAB_OFFLINE") != "1":
        raise RuntimeError("Use Run-Lab.ps1 -Action matched for offline execution")
    cfg = config()
    if set(cfg["training_seeds"]) & set(cfg["test_seeds"]):
        raise RuntimeError("Training/test seed overlap")
    before = check_preservation()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    locked = hashes()
    prereg = {"created_utc":datetime.now(timezone.utc).isoformat(), "config":cfg,"hashes":locked,
              "python":sys.version,"executable":sys.executable,"platform":platform.platform(),
              "phase":"written before model initialization and any experiment data generation"}
    (output/"preregistration.json").write_text(json.dumps(prereg, indent=2))
    for path in locked:
        destination = output/"source"/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path,destination)
    print("Protocol and source locked before training",flush=True)
    models, training = {}, {}
    for i, seed in enumerate(cfg["model_seeds"]):
        order = ["autoregressive","joint"] if i%2==0 else ["joint","autoregressive"]
        for family in order:
            model = (AutoregressiveRouter(seed,hidden=cfg["ar_hidden"]) if family=="autoregressive"
                     else JointRouter(seed,hidden=cfg["joint_hidden"]))
            name = f"{family}_{seed}"
            training[name] = train(model,cfg)
            model.save(output/"checkpoints"/(name+".npz"))
            models[name] = model
            print(name, "trained",flush=True)
    if len({r["input_sha256"] for r in training.values()}) != 1:
        raise RuntimeError("Training input/data-order mismatch")
    latency = timing(models,cfg)
    runs = []
    store = Memory(output/"feedback.sqlite")
    try:
        for condition in cfg["conditions"]:
            for seed in cfg["test_seeds"]:
                rows = episode(seed,cfg["test_steps"],condition)
                inputs = []
                for name, model in {"fixed_uniform":None,"recency":None,**models}.items():
                    family = name.rsplit("_",1)[0] if model is not None else name
                    mseed = int(name.rsplit("_",1)[1]) if model is not None else None
                    persist = store if condition=="standard" and seed==cfg["test_seeds"][0] and mseed==cfg["model_seeds"][0] else None
                    score, trace, evidence = evaluate(model,rows,cfg,name=family,rng_seed=seed+123456,
                        feedback=condition!="no_feedback",persistent=persist,namespace=f"{condition}/{seed}/{name}")
                    inputs.append(evidence["input_sha256"])
                    runs.append({"condition":condition,"seed":seed,"model":name,"family":family,
                                 "model_seed":mseed,"metrics":score,**evidence})
                    trace_path = output/"traces"/condition/str(seed)/(name+".jsonl")
                    trace_path.parent.mkdir(parents=True,exist_ok=True)
                    trace_path.write_text("".join(json.dumps(row)+"\n" for row in trace))
                if len(set(inputs)) != 1:
                    raise RuntimeError("Evaluation input mismatch")
            print(condition, "evaluated",flush=True)
    finally:
        store.close()
    summaries = summary(runs)
    paired = comparisons(runs,cfg)
    gates, gaps = scientific_gates(summaries,paired,training,latency,cfg)
    if hashes() != locked:
        raise RuntimeError("Source/protocol changed during run")
    result = {"id":cfg["id"],"config":cfg,"training":training,"runs":runs,"summary":summaries,
              "paired":paired,"gates":gates,"matching":gaps,"latency":latency,
              "preservation":check_preservation(),"source_hashes":locked}
    (output/"results.json").write_text(json.dumps(result,indent=2))
    write_report(result,output)
    print(json.dumps(gates),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
