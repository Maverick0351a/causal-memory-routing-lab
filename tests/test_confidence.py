from dataclasses import replace
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from archlab.public import Observation
from archlab.memory import Memory
from archlab.benchmark import TRAIN_SEEDS, TEST_SEEDS, MODEL_SEEDS
from archlab.matched import config as matched_config
from confidence_lab.policy import Gate, ConfidenceState, objective, fit
from confidence_lab.benchmark import (FAMILIES, episode, collect, evaluate, metrics,
                                      basic_metrics, bootstrap, slices)
from confidence_lab.experiment import config, validation, scientific_gates, check_preservation

ROOT = Path(__file__).resolve().parents[1]


def gate():
    return Gate(np.array([.3,.2,1.,2.,1.5,2.,1.,1.]),np.zeros(8),np.ones(8))


def test_seeds_fresh_disjoint_and_fixed_protocol():
    cfg = config()
    now = cfg["training_seeds"]+cfg["validation_seeds"]+cfg["test_seeds"]
    old = matched_config()
    prior = TRAIN_SEEDS+TEST_SEEDS+MODEL_SEEDS+old["training_seeds"]+old["test_seeds"]+old["model_seeds"]
    assert len(now)==len(set(now))
    assert not set(now)&set(prior)
    assert len(cfg["test_seeds"])==24
    assert cfg["minimum_mixed_log_loss_improvement"]==.01


def test_policy_has_no_evaluator_import_or_metadata_arguments():
    code = (ROOT/"confidence_lab/policy.py").read_text()
    assert "from .benchmark" not in code and "from .experiment" not in code
    assert list(inspect.signature(ConfidenceState.base).parameters)==["self","observation"]
    assert list(inspect.signature(ConfidenceState.observe).parameters)==["self","observation","label"]
    assert list(inspect.signature(Gate.alpha).parameters)==["self","features"]


def test_world_reproducible_public_distribution_and_independent_noise():
    cfg = config()
    mixed = episode(951001,800,"mixed",cfg)
    assert mixed==episode(951001,800,"mixed",cfg)
    assert mixed==episode(951001,800,"no_feedback",cfg)
    assert {r.mode for r in mixed}=={"random","learnable"}
    for condition in cfg["conditions"]:
        rows = episode(951001,800,condition,cfg)
        assert [r.observation for r in rows]==[r.observation for r in mixed]
    modes = [i for i,r in enumerate(mixed) if r.mode_switched]
    assert all(96<=b-a<=192 for a,b in zip([0]+modes,modes))
    assert all(not r.rule_switched or (r.mode=="learnable" and not r.mode_switched) for r in mixed)


@pytest.mark.parametrize("family",FAMILIES)
def test_current_label_future_and_hidden_metadata_cannot_change_prediction(family):
    cfg = config()
    rows = episode(951002,90,"mixed",cfg)
    changed = [replace(r,label=1-r.label if i>=40 else r.label,
                       observation=Observation((1-r.observation.bits[0],r.observation.bits[1])) if i>40 else r.observation,
                       mode="random",age=99999,mode_age=9999,rule_switched=False,mode_switched=False)
               for i,r in enumerate(rows)]
    _,_,a,_ = evaluate(family,rows,gate(),.7,cfg)
    _,_,b,_ = evaluate(family,changed,gate(),.7,cfg)
    for key in ("probability","alpha","base_probability"):
        assert [r[key] for r in a[:41]]==[r[key] for r in b[:41]]


def test_feedback_influences_future_and_no_feedback_stays_uniform():
    cfg = config()
    rows = episode(951003,128,"standard",cfg)
    _,_,fed,digest_fed = evaluate("adaptive_gate",rows,gate(),.7,cfg)
    _,_,unfed,digest_unfed = evaluate("adaptive_gate",rows,gate(),.7,cfg,feedback=False)
    assert all(r["probability"]==.5 for r in unfed)
    assert any(r["probability"]!=.5 for r in fed)
    assert digest_fed!=digest_unfed
    digests = [evaluate(f,rows,gate(),.7,cfg)[3] for f in FAMILIES]
    assert len(set(digests))==1


def test_sigmoid_mixture_keeps_class_sign_and_valid_probabilities():
    cfg = config()
    _,_,trace,_ = evaluate("adaptive_gate",episode(951004,200,"mixed",cfg),gate(),.7,cfg)
    assert all(0<r["alpha"]<1 and .05<=r["probability"]<=.95 for r in trace)
    assert all((r["probability"]>=.5)==(r["base_probability"]>=.5) for r in trace)
    for r in trace:
        assert r["probability"]==pytest.approx(.5+r["alpha"]*(r["base_probability"]-.5))


def test_objective_gradient_and_intercept_regularization():
    rng = np.random.default_rng(951005)
    x = rng.normal(size=(31,8))
    x[:,0] = 1
    base = rng.uniform(.05,.95,31)
    y = rng.integers(2,size=31)
    w = rng.normal(0,.2,8)
    _,grad = objective(w,x,base,y,.01)
    for index in range(8):
        a,b = w.copy(),w.copy()
        a[index]+=1e-5
        b[index]-=1e-5
        numeric = (objective(a,x,base,y,.01)[0]-objective(b,x,base,y,.01)[0])/2e-5
        assert grad[index]==pytest.approx(numeric,abs=1e-8)
    zero_grad = objective(w,x,base,y,0)[1]
    assert grad[0]==zero_grad[0]
    assert np.allclose(grad[1:]-zero_grad[1:],.01*w[1:])


def test_collect_predicts_before_feedback_and_fit_reproducible():
    cfg = {**config(),"fit_conditions":["standard"],"epochs":3}
    a = collect([951006],64,cfg)
    b = collect([951006],64,cfg)
    assert a[3]==b[3]
    assert a[0][0].tolist()==[1.,0.,0.,0.,0.,0.,0.,0.]
    assert a[1][0]==.5
    ga,ra = fit(*a[:3],.001,cfg)
    gb,rb = fit(*b[:3],.001,cfg)
    assert ra==rb
    assert np.array_equal(ga.weights,gb.weights)
    assert ga.mean[0]==0 and ga.scale[0]==1
    assert np.all(ga.scale>=cfg["feature_scale_floor"])


def test_checkpoint_roundtrip_and_readonly_parameters(tmp_path):
    g = gate()
    frozen = [a.copy() for a in (g.weights,g.mean,g.scale)]
    evaluate("adaptive_gate",episode(951007,50,"standard",config()),g,.7,config())
    assert all(np.array_equal(a,b) for a,b in zip(frozen,(g.weights,g.mean,g.scale)))
    with pytest.raises(ValueError):
        g.weights[0]=999
    path = tmp_path/"gate.npz"
    g.save(path)
    restored = Gate.load(path)
    assert g.alpha(np.arange(8)*.1)==restored.alpha(np.arange(8)*.1)
    with pytest.raises(ValueError):
        Gate(np.ones(7),np.zeros(8),np.ones(8))
    with pytest.raises(ValueError):
        Gate(np.ones(8),np.zeros(8),np.zeros(8))


def test_validation_selects_constant_uniform_for_unpredictable_balanced_labels():
    cfg = config()
    x = np.zeros((40,8))
    x[:,0] = 1
    base = np.tile([.9,.9,.1,.1],10)
    y = np.tile([0,1,0,1],10)
    gates = [Gate(np.r_[b,np.zeros(7)],np.zeros(8),np.ones(8)) for b in (-2.,0.,2.)]
    selected = validation(gates,(x,base,y,"synthetic"),cfg)
    assert selected["selected_fixed_alpha"]==0
    assert selected["selected_gate_index"]==0
    assert len(selected["fixed_grid"])==21 and len(selected["adaptive_candidates"])==3


def test_memory_resume_cutoff_namespace_and_future_poison(tmp_path):
    cfg = config()
    rows = episode(951008,100,"mixed",cfg)
    path = tmp_path/"feedback.sqlite"
    m = Memory(path)
    _,_,trace,_ = evaluate("adaptive_gate",rows,gate(),.7,cfg,persistent=m,namespace="good")
    m.append("other",0,(1,1),1)
    m.append("good",9999,(0,0),1)
    m.close()
    m = Memory(path)
    state = ConfidenceState(cfg["history_rates"])
    history = m.history("good",before_step=40)
    assert len(history)==40
    for _,bits,y in history:
        state.observe(Observation(bits),y)
    p = state.base(rows[40].observation)
    alpha = gate().alpha(state.features(p))
    assert .5+alpha*(p-.5)==trace[40]["probability"]
    m.close()


def test_scoring_switch_boundaries_and_no_startup_or_random_recovery():
    cfg = config()
    trace = [{"step":i,"probability":.9,"label":int(i<8 or i>=16),"alpha":1.,
              "rule_switched":i in (8,16),"mode_switched":i==24,
              "mode":"learnable" if i<24 else "random","mode_age":i if i<24 else i-24}
             for i in range(40)]
    s = metrics(trace,cfg)
    assert s["switches_scored"]==2
    assert s["rule_switch_scored_steps"]==16
    assert s["recovery_penalized_score"]==19.5
    assert s["recovery_fraction"]==.5
    assert metrics(trace[:8],cfg)["switches_scored"]==0
    parts = slices(trace)
    assert parts["learnable_entry_first16"]["steps"]==0
    assert parts["random_entry_first16"]["steps"]==16


def test_confident_error_denominator_and_null_control():
    trace = [{"probability":p,"label":y,"alpha":1.} for p,y in ((.95,0),(.95,1),(.5,0),(.5,1))]
    m = basic_metrics(trace)
    assert m["confidently_wrong_rate"]==.25
    assert m["high_confidence_coverage"]==.5
    assert m["high_confidence_error_rate"]==.5
    cfg = config()
    _,_,null,_ = evaluate("adaptive_gate",episode(951009,4096,"random_labels",cfg),gate(),.7,cfg)
    assert .45<basic_metrics(null)["accuracy"]<.55


def test_bootstrap_pairs_streams_reproducibly():
    cfg = {**config(),"bootstrap_replicates":1000}
    assert bootstrap([-.02]*24,cfg)["ci95"]==pytest.approx([-.02,-.02])
    a = bootstrap(np.arange(24)*.001,cfg)
    assert a==bootstrap(np.arange(24)*.001,cfg)
    with pytest.raises(ValueError):
        bootstrap([[1,2]],cfg)


def test_failed_primary_or_quality_cannot_be_rescued():
    cfg = config()
    s = {c:{f:{"accuracy":.9,"log_loss":.3,"rule_switch_log_loss":.4,"confidently_wrong_rate":0.,
                  "slices":{"random":{"metrics":{"confidently_wrong_rate":0.}}}}
             for f in FAMILIES} for c in cfg["conditions"]}
    paired = {"mixed/adaptive_minus_fixed_shrinkage":{"mean":-.005,"ci95":[-.009,-.001]}}
    gates = scientific_gates(s,paired,cfg)
    assert gates["deployment_quality"] and not gates["adaptive_value_over_fixed"]
    assert not gates["replacement_supported"]
    paired["mixed/adaptive_minus_fixed_shrinkage"]={"mean":-.02,"ci95":[-.03,-.01]}
    assert scientific_gates(s,paired,cfg)["replacement_supported"]
    s["standard"]["adaptive_gate"]["log_loss"] += .1
    gates = scientific_gates(s,paired,cfg)
    assert gates["adaptive_value_over_fixed"] and not gates["deployment_quality"]
    assert not gates["replacement_supported"]


def test_prior_artifacts_and_vendor_preserved():
    assert check_preservation()["release_files_match"]
