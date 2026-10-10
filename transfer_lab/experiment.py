"""Train/select the conventional baseline, then test the unchanged learned gate."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import numpy as np
from confidence_lab.experiment import sha, write_json
from confidence_lab.policy import Gate
from confidence_lab.benchmark import collect
from archlab.memory import Memory
from .benchmark import (FAMILIES, select, episode, evaluate, summary, comparisons, scientific_gates)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/"experiments/confidence-transfer-20261010"
ORIGINAL = ROOT/"experiments/confidence-gating-20261010/protocol.json"
CREDENTIAL_NAMES = ("NVIDIA_API_KEY","NGC_API_KEY","OPENAI_API_KEY","ANTHROPIC_API_KEY",
                    "OPENROUTER_API_KEY","HF_TOKEN","HUGGINGFACE_HUB_TOKEN","LANGCHAIN_API_KEY","LANGSMITH_API_KEY")


def config():
    return json.loads((EXPERIMENT/"protocol.json").read_text())


def original_config():
    return json.loads(ORIGINAL.read_text())


def hashes():
    """Public adapter: snapshot portable sources and frozen configuration."""
    from release_tools.integrity import source_hashes
    return source_hashes(ROOT)


def offline_proof():
    if os.environ.get("ARCHLAB_OFFLINE")!="1":
        raise RuntimeError("Use Run-Transfer.ps1 for process-scoped offline execution")
    if any(os.environ.get(k) for k in CREDENTIAL_NAMES):
        raise RuntimeError("Provider credentials should be absent from the lab process")
    try:
        socket.getaddrinfo("localhost",9)
    except RuntimeError as exc:
        if "ARCHLAB_OFFLINE blocked" not in str(exc):
            raise
    else:
        raise RuntimeError("Python socket guard is inactive")
    return {"listed_provider_credentials_absent":True,"socket_guard_denied_name_resolution":True,
            "scope":"Python process guard, not an OS sandbox"}


def check_preservation():
    """Public adapter: verifies released files, never original private projects."""
    from release_tools.integrity import verify_release
    return verify_release(ROOT)


def fmt(number, digits=5):
    return "n/a" if number is None else f"{number:.{digits}f}"


def write_report(result, output):
    gates,cfg = result["gates"],result["config"]
    rows = ["# Frozen confidence transfer test", "",
            f"Learned-gate transfer value: **{'PASS' if gates['learned_transfer_value'] else 'FAIL'}**.",
            f"Learned-gate transfer quality: **{'PASS' if gates['learned_transfer_quality'] else 'FAIL'}**.",
            f"Online fallback qualification: **{'PASS' if gates['online_default_qualified'] else 'FAIL'}**.",
            f"Recommended confidence default in this lab: **{gates['recommended_confidence_default']}**.","",
            "The learned checkpoint/normalization is unchanged; only the conventional baseline was selected on fresh original-distribution training/validation streams.",
            "Shorter mode intervals and higher noise were never available for baseline selection. No held-out retuning.","",
            f"Baseline selected prior={result['selection']['online']['prior']:.3f}, share={result['selection']['online']['share']}, eta={result['selection']['online']['learning_rate']}; fixed alpha={result['selection']['fixed_alpha']:.2f}.","",
            "| Case | Policy | Accuracy | Log loss | Brier | ECE | Confidently wrong | High-conf coverage | Mean alpha |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case in cfg["cases"]:
        for family in FAMILIES:
            s = result["summary"][case][family]
            rows.append(f"| {case} | {family} | {s['accuracy']:.3%} | {s['log_loss']:.5f} | {s['brier']:.5f} | {s['ece']:.5f} | {s['confidently_wrong_rate']:.3%} | {s['high_confidence_coverage']:.3%} | {s['mean_alpha']:.3f} |")
    rows += ["", "## Primary comparisons", "",
             "| Cases | Difference | Mean log loss difference | Paired stream 95% interval |",
             "|---|---|---:|---|"]
    for case in ["aggregate_ood",*cfg["ood_cases"],"reference_mixed"]:
        for pair in ("frozen_gate_minus_online_mixture","online_mixture_minus_fixed_shrinkage"):
            p = result["paired"][case+"/"+pair]
            rows.append(f"| {case} | {pair} | {p['mean']:+.6f} | [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}] |")
    rows += ["", "Required primary advantage: <=-.01 nats with upper interval<0, and no unseen case worse by >.01.",
             "The aggregate first averages three related cases within each seed, then resamples the24 independent seeds. Steps and cases are not independent replicates.",
             "Per-case intervals are descriptive. Intervals condition on the frozen checkpoint and one baseline selection; this is transfer within one toy world family.",
             "", "## First16 predictions after mode changes", "",
             "| Case | Direction | Policy | Scored steps | Log loss | Confidently wrong | Mean alpha |",
             "|---|---|---|---:|---:|---:|---:|"]
    for case in ("reference_mixed",*cfg["ood_cases"]):
        for key in ("learnable_entry_first16","random_entry_first16"):
            for family in ("recency","fixed_shrinkage","frozen_gate","online_mixture"):
                part = result["summary"][case][family]["slices"][key]
                s = part["metrics"]
                rows.append(f"| {case} | {key} | {family} | {part['steps']} | {s['log_loss']:.5f} | {s['confidently_wrong_rate']:.3%} | {s['mean_alpha']:.3f} |")
    rows += ["", "Startup excluded; windows stop at the next mode change/episode end. Private mode/age fields are scoring-only.",
             "", "## Frozen gate components", ""]
    for group,components in gates["components"].items():
        rows.append(f"### {group}")
        rows.append("")
        rows.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name,passed in components.items())
        rows.append("")
    rows += ["Equal positive shrinkage retains recency's predicted class by construction; equal accuracy is not learned recognition. Alpha is a mixture weight, not a hidden-mode probability.",
             "The online baseline updates causal state after labels, while all hyperparameters and learned weights remain fixed. No private mode or current answer enters inference.",
             "Metrics are equal-weight stream means; slice counts are totals. Confidently wrong is a fraction of ALL predictions, with conditional high-confidence error also retained in results.json.",
             "Checkpoints, candidate grids, every trace/per-seed metric, source hashes and failed gates are retained. Fresh-process proof is in verification.json.",
             "No real assistant configuration, semantic-memory claim, quantum result or broad model superiority follows from this experiment."]
    (output/"RESULTS.md").write_text("\n".join(rows)+"\n",encoding="utf-8")


def run(output):
    environment = offline_proof()
    cfg,original = config(),original_config()
    seeds = cfg["training_seeds"]+cfg["validation_seeds"]+cfg["test_seeds"]
    if len(seeds)!=len(set(seeds)):
        raise RuntimeError("Seed overlap")
    check_preservation()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    locked = hashes()
    write_json(output/"preregistration.json",{
        "created_utc":datetime.now(timezone.utc).isoformat(),"config":cfg,"source_hashes":locked,
        "phase":"before all new training/validation/test generation","python":sys.version,
        "executable":sys.executable,"platform":platform.platform(),"offline_proof":environment})
    for relative in locked:
        target = output/"source"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    (output/"checkpoints").mkdir()
    shutil.copyfile(ROOT/cfg["frozen_gate"],output/"checkpoints/frozen_gate.npz")
    gate = Gate.load(output/"checkpoints/frozen_gate.npz")
    print("Protocol/source locked; gate copied unchanged",flush=True)
    started = time.perf_counter()
    development = {**original,"fit_conditions":cfg["fit_conditions"]}
    train = collect(cfg["training_seeds"],cfg["train_steps"],development)
    valid = collect(cfg["validation_seeds"],cfg["validation_steps"],development)
    selection = select(train,valid,gate,cfg)
    selection.update(phase="locked before any held-out data generation",
                     created_utc=datetime.now(timezone.utc).isoformat(),frozen_gate_sha256=sha(output/"checkpoints/frozen_gate.npz"))
    write_json(output/"selection.json",selection)
    selection_sha = sha(output/"selection.json")
    select_seconds = time.perf_counter()-started
    print("Baseline selection locked:",selection["online"],"fixed",selection["fixed_alpha"],flush=True)
    frozen = (gate.weights.copy(),gate.mean.copy(),gate.scale.copy())
    started = time.perf_counter()
    runs = []
    store = Memory(output/"feedback.sqlite")
    try:
        for case in cfg["cases"]:
            for seed in cfg["test_seeds"]:
                rows = episode(seed,cfg["test_steps"],case,cfg,original)
                for family in FAMILIES:
                    namespace = f"{case}/{seed}/{family}"
                    persist = store if case=="short_noisy_mixed" and seed==cfg["test_seeds"][0] and family in ("frozen_gate","online_mixture") else None
                    scores,parts,trace,digest = evaluate(family,rows,gate,selection,original,
                        feedback=cfg["cases"][case]["feedback"],persistent=persist,namespace=namespace)
                    runs.append({"case":case,"seed":seed,"family":family,"metrics":scores,"slices":parts,"features_sha256":digest})
                    path = output/"traces"/case/str(seed)/(family+".jsonl")
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text("".join(json.dumps(r)+"\n" for r in trace),encoding="utf-8")
            print(case,"evaluated",flush=True)
    finally:
        store.close()
    evaluate_seconds = time.perf_counter()-started
    summaries = summary(runs)
    paired = comparisons(runs,cfg)
    gates = scientific_gates(summaries,paired,cfg)
    if hashes()!=locked or sha(output/"selection.json")!=selection_sha:
        raise RuntimeError("Frozen source/selection changed")
    if not all(np.array_equal(a,b) for a,b in zip(frozen,(gate.weights,gate.mean,gate.scale))):
        raise RuntimeError("Learned gate changed in memory")
    result = {"config":cfg,"original_config":original,"selection":selection,"selection_sha256":selection_sha,
              "runs":runs,"summary":summaries,"paired":paired,"gates":gates,"source_hashes":locked,
              "offline_proof":environment,"preservation":check_preservation(),
              "seconds":{"development_and_selection":select_seconds,"test_and_trace_writes":evaluate_seconds}}
    write_json(output/"results.json",result)
    write_report(result,output)
    write_json(output/"artifact-hashes.json",{str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file()})
    print(json.dumps(gates),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
