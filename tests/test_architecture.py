import dataclasses
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import numpy as np
import pytest
from archlab.embedding import HashEmbedder
from archlab.memory import Memory
from archlab.public import Observation, FeedbackState, experts
from archlab.controllers import AutoregressiveRouter, RecurrentRouter, load_model
from archlab.benchmark import episode, evaluate, train, metrics, TRAIN_SEEDS, TEST_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def test_embedding_stable_and_local():
    e = HashEmbedder()
    assert np.array_equal(e.encode(["bit0_1 bit1_0"]), HashEmbedder().encode(["bit0_1 bit1_0"]))
    assert np.linalg.norm(e.encode([""])) == 0
    assert np.linalg.norm(e.encode(["bit0_1 bit1_0"])[0]) == pytest.approx(1)


def test_public_observation_has_no_hidden_state():
    assert [f.name for f in dataclasses.fields(Observation)] == ["bits"]
    with pytest.raises(TypeError):
        Observation((0, 1), rule=3)
    assert not (set(TRAIN_SEEDS) & set(TEST_SEEDS))
    # Controllers cannot import the private generator or inspect an environment object.
    code = (ROOT/"archlab/controllers.py").read_text()
    assert "from .benchmark" not in code


def test_memory_namespace_time_cutoff_and_reopen(tmp_path):
    path = tmp_path/"m.sqlite"
    m = Memory(path)
    m.append("train", 0, (0, 1), 0)
    m.append("test", 0, (1, 0), 1)
    m.append("test", 1, (1, 1), 0)
    m.append("test", 999, (0, 0), 1)
    with pytest.raises(sqlite3.IntegrityError):
        m.append("test", 0, (0, 0), 0)
    assert m.history("test", before_step=1) == [(0, (1, 0), 1)]
    assert all(x["step"] < 1 for x in m.search("test", "bit0_1", before_step=1))
    m.close()
    m = Memory(path)
    assert len(m.history("test", before_step=2)) == 2
    m.close()


def test_memory_rejects_embedding_version(tmp_path):
    path = tmp_path/"m.sqlite"
    m = Memory(path)
    m.append("one", 0, (0, 0), 1)
    m.close()
    m = Memory(path, HashEmbedder(32))
    with pytest.raises(ValueError, match="version"):
        m.search("one", "bit0_0", before_step=1)
    m.close()


def test_ar_exact_likelihood_and_prefix():
    m = AutoregressiveRouter()
    x = np.random.default_rng(19).normal(size=24)
    q, _ = m.forward(x)
    assert q.sum() == pytest.approx(1)
    assert (q > 0).all()
    for a in (0, 1):
        for b in (0, 1):
            assert q[2*a+b] == pytest.approx(m.next_token(x)[a]*m.next_token(x, (a,))[b])
    first = m.next_token(x).copy()
    m.next_token(x, (1,))
    assert np.array_equal(first, m.next_token(x))
    for _ in range(10):
        route, logq = m.sample(x, np.random.default_rng(_), prefix=(1,))
        assert route[0] == 1
        assert logq == pytest.approx(np.log(q[2+route[1]]))
    with pytest.raises(ValueError):
        m.forward(x, temperature=0)


@pytest.mark.parametrize("cls", [AutoregressiveRouter, RecurrentRouter])
def test_gradients_match_finite_difference(cls):
    m = cls(17)
    xs = np.random.default_rng(99).normal(size=(3, 24))*.2
    outputs = [experts(Observation(x)) for x in ((0, 1), (1, 0), (1, 1))]
    labels = [1, 0, 1]
    def loss_and_grad():
        m.reset()
        caches = [m.forward(x)[1] for x in xs]
        return m.gradients(caches, outputs, labels)
    analytical, _ = loss_and_grad()
    for key, param in m.params.items():
        for index in (0, param.size//2, param.size-1):
            old = param.flat[index]
            param.flat[index] = old+1e-5
            plus = loss_and_grad()[1]
            param.flat[index] = old-1e-5
            minus = loss_and_grad()[1]
            param.flat[index] = old
            assert analytical[key].flat[index] == pytest.approx((plus-minus)/2e-5, abs=2e-7)


@pytest.mark.parametrize("cls", [AutoregressiveRouter, RecurrentRouter])
def test_current_label_and_future_cannot_change_prediction(cls):
    rows = episode(1001, 80)
    altered = list(rows)
    for i in range(40, len(rows)):
        obs, y, age, switched = rows[i]
        altered[i] = (obs, 1-y, 9999, not switched)
    _, a, _ = evaluate(cls(11), rows, name="test")
    _, b, _ = evaluate(cls(11), altered, name="test")
    # Current label is also unavailable at t=40, not just the strict prefix.
    assert [r["probability"] for r in a[:41]] == [r["probability"] for r in b[:41]]
    assert [r["route"] for r in a[:41]] == [r["route"] for r in b[:41]]


@pytest.mark.parametrize("cls", [AutoregressiveRouter, RecurrentRouter])
def test_checkpoint_roundtrip_and_frozen_weights(cls, tmp_path):
    m = cls(29)
    before = {k:v.copy() for k,v in m.params.items()}
    evaluate(m, episode(31, 40), name="test")
    assert all(np.array_equal(v, m.params[k]) for k,v in before.items())
    path = tmp_path/"checkpoint.npz"
    m.save(path)
    loaded = load_model(path)
    x = FeedbackState().features(Observation((1, 0)))
    assert np.array_equal(m.forward(x)[0], loaded.forward(x)[0])


def test_training_deterministic():
    a, b = AutoregressiveRouter(47), AutoregressiveRouter(47)
    train(a, epochs=1, steps=32)
    train(b, epochs=1, steps=32)
    assert all(np.array_equal(a.params[k], b.params[k]) for k in a.params)


def test_independent_random_label_control():
    _, trace, _ = evaluate(AutoregressiveRouter(11), episode(9123, 4096, "random_labels"), name="ar")
    accuracy = np.mean([(r["probability"] >= .5) == r["label"] for r in trace])
    assert .45 < accuracy < .55


def test_offline_child_rejects_socket():
    env = dict(os.environ, ARCHLAB_OFFLINE="1", PYTHONPATH=str(ROOT/"scripts/offline_guard"))
    p = subprocess.run([sys.executable, "-c", "import socket; socket.create_connection(('127.0.0.1',9))"],
                       env=env, capture_output=True, text=True, timeout=10)
    assert p.returncode != 0
    assert "ARCHLAB_OFFLINE blocked" in p.stderr


def test_persistent_memory_reconstructs_feedback_state(tmp_path):
    m = Memory(tmp_path/"memory.sqlite")
    state = FeedbackState()
    for i, (obs, y, _, _) in enumerate(episode(2002, 31)):
        state.observe(obs, y)
        m.append("test", i, obs.bits, y)
    m.close()
    m = Memory(tmp_path/"memory.sqlite")
    restored = FeedbackState()
    for _, bits, y in m.history("test", before_step=31):
        restored.observe(Observation(bits), y)
    assert np.array_equal(state.fast, restored.fast)
    assert np.array_equal(state.slow, restored.slow)
    m.close()


def test_credentials_removed_in_qca_child():
    code = f"import sys,os; sys.path.insert(0,{str(ROOT/'scripts')!r}); import qca_local; qca_local.prepare(); assert all(k not in os.environ for k in ['NVIDIA_API_KEY','OPENAI_API_KEY','ANTHROPIC_API_KEY']); print('KEYLESS_OK')"
    env = dict(os.environ, NVIDIA_API_KEY="synthetic", OPENAI_API_KEY="synthetic", ANTHROPIC_API_KEY="synthetic")
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    assert p.returncode == 0, p.stderr
    assert "KEYLESS_OK" in p.stdout


def test_switch_metric_excludes_episode_start():
    trace = [{"step":i, "age":i%16, "probability":.9, "label":int(i>=16), "sampled_correct":True}
             for i in range(24)]
    assert metrics(trace)["early_accuracy"] == 1.0
