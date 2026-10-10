"""Separate-process baseline reselection, complete replay and two memory resumes."""
import argparse
import json
from pathlib import Path
import numpy as np
from archlab.memory import Memory
from archlab.public import Observation
from confidence_lab.policy import Gate
from confidence_lab.benchmark import collect
from confidence_lab.experiment import sha,write_json
from .policy import Policy
from .benchmark import (select,episode,evaluate,summary,comparisons,scientific_gates)
from .experiment import (ROOT,hashes,offline_proof,check_preservation)


def verify(run):
    environment = offline_proof()
    output = Path(run).resolve()
    result = json.loads((output/"results.json").read_text())
    cfg,original = result["config"],result["original_config"]
    lock = json.loads((output/"preregistration.json").read_text())
    assert lock["config"]==cfg
    assert hashes()==lock["source_hashes"]==result["source_hashes"]
    for relative,digest in lock["source_hashes"].items():
        assert sha(output/"source"/relative)==digest,relative
    artifacts = json.loads((output/"artifact-hashes.json").read_text())
    for relative,digest in artifacts.items():
        assert sha(output/relative)==digest,relative
    assert sha(output/"selection.json")==result["selection_sha256"]
    assert sha(output/"checkpoints/frozen_gate.npz")==cfg["frozen_gate_sha256"]
    assert sha(ROOT/cfg["frozen_gate"])==cfg["frozen_gate_sha256"]
    gate = Gate.load(output/"checkpoints/frozen_gate.npz")
    frozen = [x.copy() for x in (gate.weights,gate.mean,gate.scale)]
    development = {**original,"fit_conditions":cfg["fit_conditions"]}
    train = collect(cfg["training_seeds"],cfg["train_steps"],development)
    valid = collect(cfg["validation_seeds"],cfg["validation_steps"],development)
    selection = select(train,valid,gate,cfg)
    assert all(result["selection"][k]==v for k,v in selection.items())
    selection = result["selection"]
    assert selection==json.loads((output/"selection.json").read_text())
    print("Frozen checkpoint and baseline reselection verified",flush=True)
    checked,steps = 0,0
    records = []
    for recorded in result["runs"]:
        rows = episode(recorded["seed"],cfg["test_steps"],recorded["case"],cfg,original)
        scores,parts,trace,digest = evaluate(recorded["family"],rows,gate,selection,original,
            feedback=cfg["cases"][recorded["case"]]["feedback"])
        assert scores==recorded["metrics"]
        assert parts==recorded["slices"]
        assert digest==recorded["features_sha256"]
        path = output/"traces"/recorded["case"]/str(recorded["seed"])/(recorded["family"]+".jsonl")
        assert trace==[json.loads(line) for line in path.read_text().splitlines()],path
        records.append({**recorded,"metrics":scores,"slices":parts,"features_sha256":digest})
        checked+=1
        steps+=len(trace)
        if checked%120==0:
            print(checked,"traces replayed",flush=True)
    assert all(np.array_equal(a,b) for a,b in zip(frozen,(gate.weights,gate.mean,gate.scale)))
    for case in cfg["cases"]:
        for seed in cfg["test_seeds"]:
            assert len({r["features_sha256"] for r in records if r["case"]==case and r["seed"]==seed})==1
    summaries = summary(records)
    assert summaries==result["summary"]
    paired = comparisons(records,cfg)
    assert paired==result["paired"]
    assert scientific_gates(summaries,paired,cfg)==result["gates"]
    store = Memory(output/"feedback.sqlite")
    resumes = {}
    for family in ("frozen_gate","online_mixture"):
        namespace = f"short_noisy_mixed/{cfg['test_seeds'][0]}/{family}"
        cutoff = cfg["resume_step"]
        history = store.history(namespace,before_step=cutoff)
        assert len(history)==cutoff
        policy = Policy(family,gate,selection,original["history_rates"])
        for _,bits,label in history:
            obs = Observation(bits)
            policy.predict(obs)
            policy.observe(obs,label)
        saved = [json.loads(line) for line in (output/"traces/short_noisy_mixed"/str(cfg["test_seeds"][0])/(family+".jsonl")).read_text().splitlines()]
        prediction,_ = policy.predict(Observation(tuple(saved[cutoff]["observation"])))
        assert all(prediction[k]==saved[cutoff][k] for k in prediction)
        resumes[family] = {"step":cutoff,"history_strictly_before_step":len(history),**prediction}
    store.close()
    proof = {"status":"PASS","replayed_traces":checked,"replayed_predictions":steps,
             "all_probabilities_and_traces_exact":True,"baseline_selection_recomputed_exactly":True,
             "frozen_gate_byte_identical":True,"frozen_gate_coefficients_and_normalization_unchanged":True,
             "metrics_slices_paired_bootstrap_and_default_recomputed":True,
             "all_policies_saw_equal_causal_base_features":True,"all_recorded_artifact_hashes_equal":True,
             "persistent_memory_resumes":resumes,"offline_proof":environment,"preservation":check_preservation(),
             "scientific_gates":result["gates"]}
    write_json(output/"verification.json",proof)
    print(json.dumps(proof,indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",default=str(ROOT/"results/confidence-transfer-v1"))
    verify(parser.parse_args().run)
