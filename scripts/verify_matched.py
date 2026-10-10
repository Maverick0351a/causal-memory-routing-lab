"""Fresh-process replay of the matched comparison, including memory and weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from archlab.benchmark import episode
from archlab.controllers import load_model
from archlab.matched import (evaluate, summary, comparisons, scientific_gates,
                             check_preservation, hashes)
from archlab.memory import Memory
from archlab.public import FeedbackState, Observation, experts


def verify(run):
    output = Path(run).resolve()
    result = json.loads((output/"results.json").read_text())
    cfg = result["config"]
    locked = json.loads((output/"preregistration.json").read_text())
    assert locked["config"] == cfg
    assert hashes() == locked["hashes"] == result["source_hashes"]
    for relative,digest in locked["hashes"].items():
        assert hashlib.sha256((output/"source"/relative).read_bytes()).hexdigest() == digest
    models = {name:load_model(output/"checkpoints"/(name+".npz")) for name in result["training"]}
    initial = {name:{k:v.copy() for k,v in m.params.items()} for name,m in models.items()}
    checked, checked_steps = 0, 0
    for recorded in result["runs"]:
        model = models.get(recorded["model"])
        rows = episode(recorded["seed"],cfg["test_steps"],recorded["condition"])
        score, trace, evidence = evaluate(model,rows,cfg,name=recorded["family"],
            rng_seed=recorded["seed"]+123456,feedback=recorded["condition"]!="no_feedback")
        assert score == recorded["metrics"],recorded
        assert evidence["input_sha256"] == recorded["input_sha256"]
        saved = [json.loads(s) for s in (output/"traces"/recorded["condition"]/str(recorded["seed"])/(recorded["model"]+".jsonl")).read_text().splitlines()]
        assert trace == saved, recorded
        checked += 1
        checked_steps += len(trace)
    for name,model in models.items():
        assert all(np.array_equal(v,model.params[k]) for k,v in initial[name].items())
    assert summary(result["runs"]) == result["summary"]
    paired = comparisons(result["runs"],cfg)
    assert paired == result["paired"]
    gates,gaps = scientific_gates(result["summary"],paired,result["training"],result["latency"],cfg)
    assert gates == result["gates"] and gaps == result["matching"]
    # Hash equality establishes exact training input order and equal update counts.
    assert len({r["input_sha256"] for r in result["training"].values()}) == 1
    assert len({r["optimizer_updates"] for r in result["training"].values()}) == 1
    assert len({r["examples_seen"] for r in result["training"].values()}) == 1
    for condition in cfg["conditions"]:
        for seed in cfg["test_seeds"]:
            assert len({r["input_sha256"] for r in result["runs"] if r["condition"]==condition and r["seed"]==seed}) == 1
    store = Memory(output/"feedback.sqlite")
    resumes = {}
    for family in ("autoregressive","joint"):
        name = f"{family}_{cfg['model_seeds'][0]}"
        namespace = f"standard/{cfg['test_seeds'][0]}/{name}"
        state = FeedbackState()
        for _,bits,y in store.history(namespace,before_step=512):
            state.observe(Observation(bits),y)
        saved = [json.loads(s) for s in (output/"traces/standard"/str(cfg["test_seeds"][0])/(name+".jsonl")).read_text().splitlines()]
        obs = Observation(tuple(saved[512]["observation"]))
        q,_ = models[name].forward(state.features(obs))
        probability = float(.05+.9*(q @ experts(obs)))
        assert probability == saved[512]["probability"]
        resumes[name] = {"step":512,"probability":probability}
    store.close()
    proof = {"status":"PASS","replayed_runs":checked,"replayed_steps":checked_steps,
             "exact_trace_equality":True,"frozen_evaluation_weights":True,
             "training_inputs_and_update_budgets_equal":True,"evaluation_inputs_equal":True,
             "source_snapshot_hashes_equal":True,"metrics_and_bootstrap_recomputed":True,
             "persistent_memory_resumes":resumes,"preservation":check_preservation(),
             "scientific_gates":gates}
    (output/"verification.json").write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",default=str(ROOT/"results/matched-routing-v1"))
    verify(parser.parse_args().run)
