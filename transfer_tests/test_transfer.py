from dataclasses import replace
import inspect
import json
import math
from pathlib import Path
import numpy as np
import pytest
from archlab.public import Observation
from archlab.memory import Memory
from archlab.benchmark import TRAIN_SEEDS,TEST_SEEDS,MODEL_SEEDS
from archlab.matched import config as matched_config
from confidence_lab.policy import Gate
from confidence_lab.benchmark import collect,slices
from transfer_lab.policy import OnlineMixture,Policy
from transfer_lab.benchmark import (FAMILIES,select,episode,evaluate,comparisons,scientific_gates)
from transfer_lab.experiment import config,original_config,ROOT,check_preservation,offline_proof


def gate():
    return Gate.load(ROOT/config()["frozen_gate"])


def selection():
    return {"fixed_alpha":.65,"online":{"prior":.65,"share":.08,"learning_rate":1.,"log_odds_clip":20.}}


def test_exact_online_bayes_and_sharing_update():
    online = OnlineMixture(.6,.1,1.)
    assert online.alpha==.6
    posterior = (.6*.8)/(.6*.8+.4*.5)
    online.observe(.8,1)
    assert online.alpha==pytest.approx(.9*posterior+.1*.6,abs=1e-15)
    unchanged = OnlineMixture(.6,.1,4.)
    unchanged.observe(.5,1)
    assert unchanged.alpha==pytest.approx(.6)
    reset = OnlineMixture(.6,1.,1.)
    reset.observe(.01,1)
    assert reset.alpha==.6


def test_mixture_responds_to_evidence_and_remains_finite():
    online = OnlineMixture(.5,.02,4.)
    for _ in range(100):
        online.observe(.95,1)
    assert online.alpha>.95
    for _ in range(100):
        online.observe(.95,0)
    assert online.alpha<.05
    assert 0<online.alpha<1
    with pytest.raises(ValueError):
        OnlineMixture(0,.1,1.)
    with pytest.raises(ValueError):
        online.observe(.7,2)


def test_policy_requires_prediction_before_feedback_and_resets():
    a = Policy("online_mixture",gate(),selection())
    obs = Observation((0,1))
    with pytest.raises(RuntimeError):
        a.observe(obs,1)
    prediction,_ = a.predict(obs)
    assert prediction["probability"]==.5
    with pytest.raises(RuntimeError):
        a.observe(Observation((1,1)),1)
    a.observe(obs,1)
    with pytest.raises(RuntimeError):
        a.observe(obs,1)
    fresh = Policy("online_mixture",gate(),selection())
    assert fresh.online.alpha==.65 and fresh.history.base(obs)==.5


def test_policy_interfaces_have_no_hidden_state_or_evaluator_imports():
    code = (ROOT/"transfer_lab/policy.py").read_text()
    assert "from .benchmark" not in code and "from .experiment" not in code
    assert list(inspect.signature(Policy.predict).parameters)==["self","observation"]
    assert list(inspect.signature(Policy.observe).parameters)==["self","observation","label"]
    assert list(inspect.signature(OnlineMixture.observe).parameters)==["self","base","label"]


@pytest.mark.parametrize("family",FAMILIES)
def test_current_future_and_private_metadata_invariance(family):
    cfg,original = config(),original_config()
    rows = episode(961001,100,"short_noisy_mixed",cfg,original)
    changed = [replace(r,label=1-r.label if i>=40 else r.label,
                       observation=Observation((1-r.observation.bits[0],r.observation.bits[1])) if i>40 else r.observation,
                       mode="random",age=99999,mode_age=99999,rule_switched=False,mode_switched=False)
               for i,r in enumerate(rows)]
    _,_,a,_ = evaluate(family,rows,gate(),selection(),original)
    _,_,b,_ = evaluate(family,changed,gate(),selection(),original)
    for key in ("probability","alpha","base_probability"):
        assert [r[key] for r in a[:41]]==[r[key] for r in b[:41]]


def test_generator_public_inputs_and_feedback_ablations_match():
    cfg,original = config(),original_config()
    rows = episode(961002,500,"short_noisy_mixed",cfg,original)
    assert rows==episode(961002,500,"no_feedback",cfg,original)
    for case in cfg["cases"]:
        other = episode(961002,500,case,cfg,original)
        assert [r.observation for r in rows]==[r.observation for r in other]
    boundaries = [i for i,r in enumerate(rows) if r.mode_switched]
    assert all(16<=b-a<=48 for a,b in zip([0]+boundaries,boundaries))
    for family in FAMILIES:
        _,_,trace,_ = evaluate(family,rows,gate(),selection(),original,feedback=False)
        assert all(r["probability"]==.5 for r in trace)


def test_equal_features_frozen_checkpoint_and_positive_shrinkage():
    cfg,original = config(),original_config()
    g = gate()
    frozen = (g.weights.copy(),g.mean.copy(),g.scale.copy())
    rows = episode(961003,120,"short_noisy_mixed",cfg,original)
    digests = []
    for family in FAMILIES:
        _,_,trace,digest = evaluate(family,rows,g,selection(),original)
        digests.append(digest)
        if family in ("frozen_gate","online_mixture"):
            assert all((r["probability"]>=.5)==(r["base_probability"]>=.5) for r in trace)
            assert all(0<r["alpha"]<1 for r in trace)
    assert len(set(digests))==1
    assert all(np.array_equal(a,b) for a,b in zip(frozen,(g.weights,g.mean,g.scale)))
    with pytest.raises(ValueError):
        g.weights[0]=0


def test_selection_reproducible_and_ignores_heldout_case_definitions():
    cfg = {**config(),"training_seeds":[961004],"validation_seeds":[961005],
           "train_steps":32,"validation_steps":32,"sharing_candidates":[.02,.08],"learning_rate_candidates":[1.]}
    original = original_config()
    train = collect(cfg["training_seeds"],32,original)
    valid = collect(cfg["validation_seeds"],32,original)
    a = select(train,valid,gate(),cfg)
    b = select(train,valid,gate(),{**cfg,"cases":{"inaccessible":{"noise_probability":999}}})
    assert a==b
    assert len(a["online_validation_grid"])==2
    assert a["training_data_sha256"]!=a["validation_data_sha256"]
    assert a["training_examples"]==128 and a["validation_examples"]==128


@pytest.mark.parametrize("family",["online_mixture","frozen_gate"])
def test_sqlite_resume_namespace_and_future_poison(family,tmp_path):
    cfg,original = config(),original_config()
    rows = episode(961006,100,"short_noisy_mixed",cfg,original)
    store = Memory(tmp_path/"m.sqlite")
    _,_,trace,_ = evaluate(family,rows,gate(),selection(),original,persistent=store,namespace="good")
    store.append("bad",0,(1,1),1)
    store.append("good",9999,(0,0),0)
    store.close()
    store = Memory(tmp_path/"m.sqlite")
    history = store.history("good",before_step=40)
    assert len(history)==40
    policy = Policy(family,gate(),selection(),original["history_rates"])
    for _,bits,y in history:
        obs = Observation(bits)
        policy.predict(obs)
        policy.observe(obs,y)
    p,_ = policy.predict(rows[40].observation)
    assert all(p[k]==trace[40][k] for k in p)
    store.close()


def test_transition_windows_exclude_startup_and_respect_boundaries():
    trace = [{"step":i,"mode":"learnable" if i<16 or i>=32 else "random",
              "mode_age":i%16,"probability":.9,"alpha":.8,"label":1}
             for i in range(40)]
    parts = slices(trace)
    assert parts["random_entry_first16"]["steps"]==16
    assert parts["learnable_entry_first16"]["steps"]==8
    assert parts["random_entry_first16"]["metrics"]["log_loss"]==pytest.approx(-math.log(.9))


def test_paired_bootstrap_aggregates_related_cases_within_seed():
    cfg = {**config(),"test_seeds":[961007,961008],"bootstrap_replicates":100}
    records = []
    for i,case in enumerate(cfg["cases"]):
        for seed in cfg["test_seeds"]:
            for family in FAMILIES:
                value = .4+.01*i if family=="frozen_gate" else .4
                records.append({"case":case,"seed":seed,"family":family,"metrics":{"log_loss":value}})
    paired = comparisons(records,cfg)
    expected = np.mean([.01*list(cfg["cases"]).index(c) for c in cfg["ood_cases"]])
    p = paired["aggregate_ood/frozen_gate_minus_online_mixture"]
    assert p["mean"]==pytest.approx(expected)
    assert len(p["per_stream"])==2
    assert p["ci95"]==pytest.approx([expected,expected])


def gate_fixture():
    cfg = config()
    s = {case:{f:{"log_loss":.4,"confidently_wrong_rate":.01,
                    "slices":{d:{"metrics":{"log_loss":.4}} for d in
                              ("learnable_entry_first16","random_entry_first16")}}
               for f in FAMILIES} for case in cfg["cases"]}
    s["random_labels"]["uniform"]["log_loss"]=math.log(2)
    for f in ("frozen_gate","online_mixture"):
        s["random_labels"][f]["log_loss"]=.7
    s["random_labels"]["recency"]["confidently_wrong_rate"]=.15
    paired = {"aggregate_ood/"+p:{"mean":-.02,"ci95":[-.03,-.01]} for p in
              ("frozen_gate_minus_online_mixture","online_mixture_minus_fixed_shrinkage")}
    return cfg,s,paired


def test_default_requires_primary_and_transition_quality():
    cfg,s,paired = gate_fixture()
    assert scientific_gates(s,paired,cfg)["recommended_confidence_default"]=="frozen_gate"
    s["short_mixed"]["frozen_gate"]["slices"]["learnable_entry_first16"]["metrics"]["log_loss"]+=.1
    gates = scientific_gates(s,paired,cfg)
    assert gates["learned_transfer_value"] and not gates["learned_transfer_quality"]
    assert gates["recommended_confidence_default"]=="online_mixture"
    s["standard"]["online_mixture"]["log_loss"]+=.1
    assert scientific_gates(s,paired,cfg)["recommended_confidence_default"]=="none"


def test_pooled_advantage_does_not_rescue_a_failed_case_or_interval():
    cfg,s,paired = gate_fixture()
    s["short_noisy_mixed"]["frozen_gate"]["log_loss"]+=.02
    gates = scientific_gates(s,paired,cfg)
    assert not gates["learned_transfer_value"]
    assert gates["recommended_confidence_default"]=="online_mixture"
    paired["aggregate_ood/online_mixture_minus_fixed_shrinkage"]["ci95"][1]=.001
    assert scientific_gates(s,paired,cfg)["recommended_confidence_default"]=="none"


def test_seed_freshness_and_original_frozen_artifacts():
    cfg,original,matched = config(),original_config(),matched_config()
    new = cfg["training_seeds"]+cfg["validation_seeds"]+cfg["test_seeds"]
    old = (original["training_seeds"]+original["validation_seeds"]+original["test_seeds"]+
           matched["training_seeds"]+matched["test_seeds"]+matched["model_seeds"]+
           TRAIN_SEEDS+TEST_SEEDS+MODEL_SEEDS)
    assert len(new)==len(set(new)) and not set(new)&set(old)
    check = check_preservation()
    assert check["release_files_match"]
    assert check["scope"]=="public release only; original host preservation is historical"


def test_keyless_socket_guard_is_active():
    proof = offline_proof()
    assert proof["listed_provider_credentials_absent"]
    assert proof["socket_guard_denied_name_resolution"]
