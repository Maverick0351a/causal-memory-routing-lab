import json
from pathlib import Path
import numpy as np
import pytest
from archlab.controllers import AutoregressiveRouter, JointRouter, load_model
from archlab.benchmark import episode, evaluate as old_evaluate, TRAIN_SEEDS, TEST_SEEDS, MODEL_SEEDS
from archlab.public import Observation, experts
from archlab.matched import (config, train, evaluate, adaptation, paired_bootstrap,
                             scientific_gates, check_preservation)


def test_matched_parameters_and_compute():
    ar, joint = AutoregressiveRouter(), JointRouter()
    assert ar.parameter_count == 750
    assert joint.parameter_count == 758
    assert sum(v.size for v in ar.params.values() if v.ndim==2) == 720
    assert sum(v.size for v in joint.params.values() if v.ndim==2) == 728
    assert abs(joint.parameter_count-ar.parameter_count)/ar.parameter_count < .02


def test_joint_gradient_against_finite_difference():
    m = JointRouter(707)
    xs = np.random.default_rng(91).normal(0,.2,(3,24))
    outcomes = [experts(Observation(bits)) for bits in ((0,1),(1,0),(1,1))]
    labels = [0,1,1]
    def loss_and_grad():
        caches = [m.forward(x)[1] for x in xs]
        return m.gradients(caches,outcomes,labels)
    grads, _ = loss_and_grad()
    for key, param in m.params.items():
        for index in (0,param.size//2,param.size-1):
            old = param.flat[index]
            param.flat[index] = old+1e-5
            plus = loss_and_grad()[1]
            param.flat[index] = old-1e-5
            minus = loss_and_grad()[1]
            param.flat[index] = old
            assert grads[key].flat[index] == pytest.approx((plus-minus)/2e-5,abs=2e-7)


@pytest.mark.parametrize("hidden",[7,26])
def test_joint_checkpoint_roundtrip(hidden,tmp_path):
    m = JointRouter(707,hidden=hidden)
    q, _ = m.forward(np.ones(24)*.2)
    assert q.sum() == pytest.approx(1)
    path = tmp_path/"joint.npz"
    m.save(path)
    restored = load_model(path)
    assert restored.parameter_count == m.parameter_count
    assert np.array_equal(q,restored.forward(np.ones(24)*.2)[0])


def test_training_budget_features_and_determinism():
    cfg = {**config(),"epochs":1,"train_steps":32,"training_seeds":[50101,50102]}
    ar, joint, again = AutoregressiveRouter(707),JointRouter(707),JointRouter(707)
    records = [train(m,cfg) for m in (ar,joint,again)]
    assert len({r["input_sha256"] for r in records}) == 1
    assert all(r["examples_seen"]==64 and r["optimizer_updates"]==4 for r in records)
    assert all(np.array_equal(joint.params[k],again.params[k]) for k in joint.params)


@pytest.mark.parametrize("cls",[AutoregressiveRouter,JointRouter])
def test_future_current_label_invariance_and_frozen_weights(cls):
    cfg = config()
    rows = episode(50201,80)
    changed = []
    for i,(obs,y,age,switched) in enumerate(rows):
        new_obs = Observation((1-obs.bits[0],obs.bits[1])) if i>40 else obs
        changed.append((new_obs,1-y if i>=40 else y,9999,not switched))
    model = cls(707)
    weights = {k:v.copy() for k,v in model.params.items()}
    _, a, _ = evaluate(model,rows,cfg,name=model.kind,rng_seed=191)
    _, b, _ = evaluate(cls(707),changed,cfg,name=model.kind,rng_seed=191)
    assert [r["probability"] for r in a[:41]] == [r["probability"] for r in b[:41]]
    assert [r["route"] for r in a[:41]] == [r["route"] for r in b[:41]]
    assert all(np.array_equal(v,model.params[k]) for k,v in weights.items())


def test_cached_ar_sampling_matches_original():
    rows = episode(50301,80)
    _, old, _ = old_evaluate(AutoregressiveRouter(707),rows,name="ar",rng_seed=192)
    _, new, _ = evaluate(AutoregressiveRouter(707),rows,config(),name="autoregressive",rng_seed=192)
    assert json.loads(json.dumps(old)) == new


def test_evaluation_features_match_both_families():
    rows = episode(50302,80)
    cfg = config()
    digests = [evaluate(m,rows,cfg,name=m.kind,rng_seed=193)[2]["input_sha256"]
               for m in (AutoregressiveRouter(707),JointRouter(707))]
    assert len(set(digests)) == 1


def test_adaptation_excludes_startup_and_crossing_boundaries():
    trace = [{"step":i,"switched":i in (16,24),"probability":.9,
              "label":int(i<16 or i>=24)} for i in range(40)]
    score = adaptation(trace,config())
    assert score["switches_scored"] == 2
    assert score["recovered_switches"] == 1
    assert score["recovery_fraction"] == .5
    assert score["recovery_endpoint_median"] == 7
    assert score["recovery_penalized_score"] == 19.5
    assert adaptation(trace[:16],config())["switches_scored"] == 0
    assert adaptation(trace[:29],config())["switches_scored"] == 1


def test_bootstrap_paired_axes_and_reproducible():
    matrix = np.arange(12,dtype=float).reshape(3,4)*.01
    a = paired_bootstrap(matrix,seed=99,replicates=1000)
    b = paired_bootstrap(matrix,seed=99,replicates=1000)
    assert a == b
    assert a["mean"] == pytest.approx(matrix.mean())
    assert a["per_model_seed"] == matrix.mean(axis=1).tolist()
    assert a["per_episode_seed"] == matrix.mean(axis=0).tolist()
    constant = paired_bootstrap(np.full((3,4),-.02),seed=99,replicates=1000)
    assert constant["ci95"] == pytest.approx([-.02,-.02])


def test_primary_gate_cannot_be_rescued_by_accuracy_or_latency():
    cfg = config()
    summaries = {c:{f:{"accuracy":.95 if c=="standard" else .5}
                       for f in ("autoregressive","joint","fixed_uniform")}
                 for c in cfg["conditions"]}
    summaries["standard"]["fixed_uniform"]["accuracy"] = .5
    training = {"autoregressive_71":{"parameters":750,"forward_affine_macs":720},
                "joint_71":{"parameters":758,"forward_affine_macs":728}}
    paired = {"log_loss":{"mean":-.005,"ci95":[-.009,-.001]}}
    assert not scientific_gates(summaries,paired,training,{"ar_over_joint_ratio":.5},cfg)[0]["ar_predictive_value"]
    paired["log_loss"] = {"mean":-.02,"ci95":[-.03,-.01]}
    gates, _ = scientific_gates(summaries,paired,training,{"ar_over_joint_ratio":2},cfg)
    assert gates["ar_predictive_value"]
    assert not gates["ar_latency_within_budget"]
    paired["log_loss"]["ci95"][1] = .001
    assert not scientific_gates(summaries,paired,training,{"ar_over_joint_ratio":1},cfg)[0]["ar_predictive_value"]


def test_all_experiment_seeds_are_fresh():
    cfg = config()
    new = [*cfg["training_seeds"],*cfg["test_seeds"],*cfg["model_seeds"]]
    assert len(new) == len(set(new))
    assert not set(new) & set([*TRAIN_SEEDS,*TEST_SEEDS,*MODEL_SEEDS])


def test_previous_artifacts_preserved():
    assert check_preservation()["release_files_match"]
