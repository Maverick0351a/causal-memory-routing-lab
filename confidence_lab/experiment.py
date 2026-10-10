"""Frozen fit/select/test lifecycle, artifact preservation and candid reporting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
from archlab.memory import Memory
from .policy import fit, sigmoid
from .benchmark import FAMILIES, collect, episode, evaluate, losses, bootstrap

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/"experiments/confidence-gating-20261010"


def config():
    return json.loads((EXPERIMENT/"protocol.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def hashes():
    """Public adapter: snapshot portable sources and frozen configuration."""
    from release_tools.integrity import source_hashes
    return source_hashes(ROOT)


def check_preservation():
    """Public adapter: verifies released files, never original private projects."""
    from release_tools.integrity import verify_release
    return verify_release(ROOT)


def validation(gates, arrays, cfg):
    x, base, y, _ = arrays
    fixed_scores = [{"alpha":alpha,"log_loss":float(losses(.5+alpha*(base-.5),y).mean())}
                    for alpha in cfg["fixed_alphas"]]
    adaptive_scores = []
    for penalty,gate in zip(cfg["regularization_candidates"],gates):
        alpha = sigmoid(((x-gate.mean)/gate.scale) @ gate.weights)
        adaptive_scores.append({"penalty":penalty,"log_loss":float(losses(.5+alpha*(base-.5),y).mean())})
    fi = min(range(len(fixed_scores)),key=lambda i:fixed_scores[i]["log_loss"])
    ai = min(range(len(adaptive_scores)),key=lambda i:adaptive_scores[i]["log_loss"])
    return {"fixed_grid":fixed_scores,"adaptive_candidates":adaptive_scores,
            "selected_fixed_alpha":fixed_scores[fi]["alpha"],"selected_gate_index":ai,
            "selected_penalty":adaptive_scores[ai]["penalty"],
            "validation_data_sha256":arrays[3],"validation_examples":len(y)}


def summary(runs):
    result = {}
    for condition in sorted({r["condition"] for r in runs}):
        result[condition] = {}
        for family in FAMILIES:
            records = [r for r in runs if r["condition"]==condition and r["family"]==family]
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
            result[condition][family] = average
    return result


def comparisons(runs, cfg):
    indexed = {(r["condition"],r["seed"],r["family"]):r for r in runs}
    paired = {}
    for condition in cfg["conditions"]:
        for reference in ("fixed_shrinkage","recency","uniform"):
            differences = [indexed[(condition,s,"adaptive_gate")]["metrics"]["log_loss"]-
                           indexed[(condition,s,reference)]["metrics"]["log_loss"] for s in cfg["test_seeds"]]
            paired[condition+"/adaptive_minus_"+reference] = bootstrap(differences,cfg)
    return paired


def scientific_gates(summaries, paired, cfg):
    p = paired["mixed/adaptive_minus_fixed_shrinkage"]
    primary = p["mean"] <= -cfg["minimum_mixed_log_loss_improvement"] and p["ci95"][1]<0
    quality = {}
    for condition,threshold in (("standard","maximum_standard_log_loss_degradation"),
                                ("fast_switch","maximum_fast_log_loss_degradation")):
        a, r = summaries[condition]["adaptive_gate"],summaries[condition]["recency"]
        quality[condition+"_log_loss_preserved"] = a["log_loss"]-r["log_loss"] <= cfg[threshold]
    for condition in ("standard","fast_switch","mixed"):
        a, r = summaries[condition]["adaptive_gate"],summaries[condition]["recency"]
        quality[condition+"_accuracy_preserved"] = a["accuracy"] >= r["accuracy"]-cfg["maximum_accuracy_degradation"]
        quality[condition+"_switch_log_loss_preserved"] = (a["rule_switch_log_loss"]-r["rule_switch_log_loss"]
                                                          <= cfg["maximum_switch_log_loss_degradation"])
    random = summaries["random_labels"]
    quality["random_log_loss_near_uniform"] = (random["adaptive_gate"]["log_loss"]-random["uniform"]["log_loss"]
                                                <= cfg["maximum_random_log_loss_above_uniform"])
    quality["random_confident_errors_halved"] = (random["adaptive_gate"]["confidently_wrong_rate"]
        <= cfg["maximum_confident_error_ratio"]*random["recency"]["confidently_wrong_rate"])
    mixed = summaries["mixed"]
    quality["mixed_random_confident_errors_halved"] = (mixed["adaptive_gate"]["slices"]["random"]["metrics"]["confidently_wrong_rate"]
        <= cfg["maximum_confident_error_ratio"]*mixed["recency"]["slices"]["random"]["metrics"]["confidently_wrong_rate"])
    return {"adaptive_value_over_fixed":primary,"deployment_quality":all(quality.values()),
            "replacement_supported":primary and all(quality.values()),"quality_components":quality}


def write_report(result, output):
    cfg, gates = result["config"],result["gates"]
    rows = ["# Confidence-gating experiment", "",
        "Adaptive value over fixed shrinkage: **"+("PASS" if gates["adaptive_value_over_fixed"] else "FAIL")+"**.",
        "Deployment-quality gate: **"+("PASS" if gates["deployment_quality"] else "FAIL")+"**.",
        "Replacement supported: **"+("YES" if gates["replacement_supported"] else "NO")+"**.","",
        "Fresh disjoint train/validation/test seeds; test labels never select parameters. CPU NumPy only.",
        f"Validation selected fixed alpha {result['selection']['selected_fixed_alpha']:.2f} and gate L2 {result['selection']['selected_penalty']}.",
        "Eight fitted gate coefficients; feature normalization fit on training data only.","",
        "| Condition | Policy | Accuracy | Log loss | Brier | ECE | Confidently wrong | High-conf coverage | Mean alpha |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in cfg["conditions"]:
        for family in FAMILIES:
            s = result["summary"][condition][family]
            rows.append(f"| {condition} | {family} | {s['accuracy']:.3%} | {s['log_loss']:.5f} | {s['brier']:.5f} | {s['ece']:.5f} | {s['confidently_wrong_rate']:.3%} | {s['high_confidence_coverage']:.3%} | {s['mean_alpha']:.3f} |")
    p = result["paired"]["mixed/adaptive_minus_fixed_shrinkage"]
    rows += ["",f"Primary mixed log-loss difference (adaptive minus fixed): **{p['mean']:+.6f} nats**, 95% paired stream-bootstrap interval [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}].",
             "Required: <=-0.01 nats and upper endpoint <0. Twenty-four independent test seeds; individual steps are not replicates.",
             "Intervals condition on one deterministic fitting/validation selection; they do not measure training-set or optimization uncertainty.",
             "", "## Quality gates", ""]
    rows += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name,passed in gates["quality_components"].items()]
    rows += ["", "## Adaptation and mixed-mode diagnostics", "",
             "| Condition | Policy | Loss in rule-switch windows | Rolling recovery score |",
             "|---|---|---:|---:|"]
    for condition in ("standard","fast_switch","mixed"):
        for family in ("recency","fixed_shrinkage","adaptive_gate"):
            s = result["summary"][condition][family]
            rows.append(f"| {condition} | {family} | {s['rule_switch_log_loss']:.5f} | {s['recovery_penalized_score']:.3f} |")
    rows += ["", "Adaptive shrinkage preserves recency's classification sign by construction. Equal accuracy/rolling recovery is not evidence of a learned benefit.",
             "Switch log loss is the meaningful confidence-adaptation check: first16 steps, truncated at the next rule/mode change. Rolling recovery is first7/8 correct; unrecovered/censored=32, startup excluded.","",
             "| Mixed slice | Policy | Steps | Log loss | Confidently wrong | Mean alpha |",
             "|---|---|---:|---:|---:|---:|"]
    for key in ("learnable","random","random_entry_first16","random_late32","learnable_entry_first16","learnable_late32"):
        for family in ("recency","fixed_shrinkage","adaptive_gate"):
            part = result["summary"]["mixed"][family]["slices"][key]
            s = part["metrics"]
            rows.append(f"| {key} | {family} | {part['steps']} | {s['log_loss']:.5f} | {s['confidently_wrong_rate']:.3%} | {s['mean_alpha']:.3f} |")
    rows += ["", "Metrics are equal-weight means over independent streams. Mixed-slice metrics average available stream slice means; counts are totals.",
             "Confidently wrong is the fraction of ALL predictions with >=90% confidence that are wrong; conditional error rates are also saved in results.json.",
             "Private mode labels and transition ages are used only for scoring. The gate sees past released feedback and a current public base margin.",
             "All candidate fit records, validation scores, checkpoints, complete traces, per-stream metrics, paired differences and frozen snapshots are retained.",
             "Fresh-process replay and SQLite resume evidence are in verification.json. Prior AR failures remain unchanged.",
             "No claim about semantic embeddings, real-world agency, quantum hardware or general superiority follows from this toy task."]
    (output/"RESULTS.md").write_text("\n".join(rows)+"\n",encoding="utf-8")


def run(output):
    if os.environ.get("ARCHLAB_OFFLINE") != "1":
        raise RuntimeError("Use Run-Lab.ps1 -Action confidence")
    cfg = config()
    seeds = [*cfg["training_seeds"],*cfg["validation_seeds"],*cfg["test_seeds"]]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Seed overlap")
    check_preservation()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    locked = hashes()
    write_json(output/"preregistration.json",{
        "created_utc":datetime.now(timezone.utc).isoformat(),"config":cfg,"hashes":locked,
        "phase":"before training/validation/test data generation", "python":sys.version,
        "executable":sys.executable,"platform":platform.platform()})
    for relative in locked:
        target = output/"source"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    print("Protocol/source locked before data generation",flush=True)
    started = time.perf_counter()
    train = collect(cfg["training_seeds"],cfg["train_steps"],cfg)
    valid = collect(cfg["validation_seeds"],cfg["validation_steps"],cfg)
    gates, training = [], []
    (output/"checkpoints").mkdir()
    for i,penalty in enumerate(cfg["regularization_candidates"]):
        gate,record = fit(*train[:3],penalty,cfg)
        record["training_data_sha256"] = train[3]
        record["checkpoint"] = f"gate_{i}.npz"
        gate.save(output/"checkpoints"/record["checkpoint"])
        gates.append(gate)
        training.append(record)
        print("Trained candidate",i,"L2",penalty,flush=True)
    selection = validation(gates,valid,cfg)
    selection.update(selected_checkpoint=training[selection["selected_gate_index"]]["checkpoint"],
                     phase="locked before any test generation",created_utc=datetime.now(timezone.utc).isoformat(),
                     checkpoint_hashes={r["checkpoint"]:sha(output/"checkpoints"/r["checkpoint"]) for r in training})
    write_json(output/"training.json",training)
    write_json(output/"selection.json",selection)
    selection_sha = sha(output/"selection.json")
    train_seconds = time.perf_counter()-started
    print("Validation selection locked:",selection["selected_fixed_alpha"],selection["selected_penalty"],flush=True)
    gate = gates[selection["selected_gate_index"]]
    runs = []
    started = time.perf_counter()
    store = Memory(output/"feedback.sqlite")
    try:
        for condition in cfg["conditions"]:
            for seed in cfg["test_seeds"]:
                rows = episode(seed,cfg["test_steps"],condition,cfg)
                for family in FAMILIES:
                    namespace = f"{condition}/{seed}/{family}"
                    persist = store if condition=="mixed" and seed==cfg["test_seeds"][0] and family=="adaptive_gate" else None
                    score,parts,trace,digest = evaluate(family,rows,gate,selection["selected_fixed_alpha"],cfg,
                        feedback=condition!="no_feedback",persistent=persist,namespace=namespace)
                    runs.append({"condition":condition,"seed":seed,"family":family,"metrics":score,
                                 "slices":parts,"features_sha256":digest})
                    target = output/"traces"/condition/str(seed)/(family+".jsonl")
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_text("".join(json.dumps(r)+"\n" for r in trace),encoding="utf-8")
            print(condition,"evaluated",flush=True)
    finally:
        store.close()
    evaluation_seconds = time.perf_counter()-started
    summaries = summary(runs)
    paired = comparisons(runs,cfg)
    scientific = scientific_gates(summaries,paired,cfg)
    if hashes() != locked or sha(output/"selection.json") != selection_sha:
        raise RuntimeError("Source/protocol/selection changed during run")
    if {name:sha(output/"checkpoints"/name) for name in selection["checkpoint_hashes"]} != selection["checkpoint_hashes"]:
        raise RuntimeError("Fitted weights changed during evaluation")
    result = {"config":cfg,"training":training,"selection":selection,"selection_sha256":selection_sha,
              "runs":runs,"summary":summaries,"paired":paired,"gates":scientific,
              "source_hashes":locked,"preservation":check_preservation(),
              "seconds":{"collect_fit_select":train_seconds,"evaluation_and_trace_writes":evaluation_seconds}}
    write_json(output/"results.json",result)
    write_report(result,output)
    write_json(output/"artifact-hashes.json",{str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file()})
    print(json.dumps(scientific),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
