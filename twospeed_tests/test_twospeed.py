"""Causal boundaries, bounded search, fresh worlds and reconstruction."""
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import numpy as np
import pytest
from archlab.controllers import RecurrentRouter
from confidence_lab.policy import Gate
from forget_lab.world import calibration_data, episode
from forget_lab.memory import Memory
from forget_lab.training import clone_router, train_router
from recall_recovery.public import Observation, fit_experts
from twospeed_lab.experiment import ROOT, config, world_config, arrays
from twospeed_lab.policy import Policy, Selector, FAMILIES, public_config
from twospeed_lab.benchmark import evaluate, initial_selection, select, interval


def dev_config(**changes):
    cfg = world_config(config(), config()['worlds'][0])
    cfg.pop('worlds')
    cfg.update(coefficient_seed=12900602, calibration_seeds=[12900001,12900002], calibration_steps=64,
               expert_seeds=[12900101,12900102,12900103,12900104], expert_epochs=15,
               training_seeds=[12900201,12900202], training_cases=['recurring','repeated_changes'],
               recurrent_seeds=[12900301], recurrent_hidden=6, recurrent_epochs=1, training_steps=64,
               training_shuffle_seed=12900603, validation_seeds=[12900401], validation_steps=128,
               test_seeds=[12900501,12900502], test_steps=160, shuffled_memory_seed=12900601,
               block_interval=[16,24], memory_capacity=24, fast_capacity=8, neighbors=8,
               fast_neighbors=4, adaptive_windows=[2,4,8], bootstrap_replicates=100, bootstrap_seed=12900999)
    return {**cfg, **changes}


@pytest.fixture(scope='module')
def experts():
    cfg = dev_config()
    return fit_experts(*calibration_data(cfg), cfg)[0]


@pytest.fixture(scope='module')
def gate():
    return Gate.load(ROOT/config()['frozen_gate'])


@pytest.fixture
def router():
    return RecurrentRouter(seed=12900302, dim=14, hidden=6)


def seed_values(value):
    result = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k.endswith('_seed'):
                result.append(v)
            elif k.endswith('_seeds'):
                result.extend(v)
            elif isinstance(v, (dict, list)):
                result.extend(seed_values(v))
    elif isinstance(value, list):
        for v in value:
            result.extend(seed_values(v))
    return result


def test_six_independent_worlds_and_all_seeds_fresh():
    cfg = config()
    seeds = seed_values(cfg)
    assert len(cfg['worlds']) == 6 and len(seeds) == len(set(seeds))
    assert all(len(w['test_seeds']) == 8 for w in cfg['worlds'])
    for path in (ROOT/'experiments').glob('*/protocol.json'):
        if path.parent.name != cfg['id']:
            assert not set(seeds)&set(seed_values(json.loads(path.read_text(encoding='utf-8'))))
    assert not set(seeds)&set(seed_values(dev_config()))


def test_public_config_excludes_private_world_and_test_fields():
    cfg = public_config(dev_config())
    assert not any(k in cfg for k in ('coefficient_seed','test_seeds','block_interval','cases','return_window'))
    assert list(inspect.signature(Policy.predict).parameters) == ['self','observation']
    assert 'world import' not in inspect.getsource(Policy)


def test_selector_uses_committed_forecasts_and_is_bounded():
    cfg = public_config(dev_config())
    selector = Selector(2, cfg, {'rate':.2, 'temperature':12.})
    cue = (0.,)*6
    w = selector.weights(cue, [True,True])
    assert selector.entries == [] and w[0] > w[1]
    for _ in range(20):
        selector.observe(cue, np.array([.9,.1]), 1)
    assert selector.weights(cue,[True,True])[0] > .99
    for _ in range(30):
        selector.observe(cue, np.array([.9,.1]), 0)
    assert selector.weights(cue,[True,True])[1] > .99
    for i in range(30):
        selector.observe((10.*(i+1),)*6, np.array([.5,.5]), 0)
    assert len(selector.entries) == cfg['selector_capacity']
    assert np.array_equal(selector.weights(cue,[True,False]), [1.,0.])


def test_store_cutoff_namespace_and_disjoint_views(experts,gate):
    cfg = dev_config()
    store = Memory()
    obs = Observation((0.,)*6,(0.,)*8)
    for n in range(100):
        store.append('episode',n,obs,n%2,np.array([n%2]*4,dtype=float))
        store.append('other',n,obs,1,np.ones(4))
    policy = Policy('two_speed',experts,gate,initial_selection(cfg),public_config(cfg),store)
    policy.step = 40
    errors, steps, fast, scanned = policy.candidates(obs)
    assert scanned == cfg['memory_capacity'] and len(steps) == scanned
    assert steps.min() == 16 and steps.max() == 39
    assert fast.sum() == 8 and (~fast).sum() == 16
    assert set(steps[fast]).isdisjoint(steps[~fast])
    policy.namespace = 'empty'
    assert len(policy.candidates(obs)[1]) == 0
    store.close()


@pytest.mark.parametrize('family',FAMILIES)
def test_current_future_labels_and_private_fields_cannot_change_current_prediction(family,experts,gate,router):
    cfg = dev_config()
    rows = episode(12900701,60,'repeated_changes',cfg)
    changed = [replace(r,label=1-r.label if i>=20 else r.label,context_id=999,age=999,
                       revision=999,changed_entry=False,mapping_changed=False,returned=False,switched=False)
               for i,r in enumerate(rows)]
    a = evaluate(rows,experts,gate,initial_selection(cfg),cfg,router,families=(family,))[1][family]
    b = evaluate(changed,experts,gate,initial_selection(cfg),cfg,router,families=(family,))[1][family]
    for key in ('probability','gated_probability','route','selector_weights','retrieved_max_step'):
        assert [r[key] for r in a[:21]] == [r[key] for r in b[:21]]


def test_every_policy_commits_before_any_private_row_access(monkeypatch,experts,gate,router):
    cfg = dev_config()
    commits = []
    original = Policy.predict
    def predict(self,obs):
        result = original(self,obs)
        commits.append(self.family)
        return result
    monkeypatch.setattr(Policy,'predict',predict)
    class Guarded:
        def __init__(self,row,index):
            self.row,self.index = row,index
        def __getattr__(self,key):
            if key != 'observation':
                assert len(commits) >= (self.index+1)*len(FAMILIES)
            return getattr(self.row,key)
    rows = [Guarded(r,i) for i,r in enumerate(episode(12900702,20,'recurring',cfg))]
    evaluate(rows,experts,gate,initial_selection(cfg),cfg,router)
    assert len(commits) == len(rows)*len(FAMILIES)


@pytest.mark.parametrize('family',FAMILIES)
def test_every_policy_reload_excludes_future_rows(family,tmp_path,experts,gate,router):
    cfg = dev_config()
    rows = episode(12900703,80,'repeated_changes',cfg)
    setting = initial_selection(cfg)
    path = tmp_path/'feedback.sqlite'
    store = Memory(path)
    _, traces = evaluate(rows,experts,gate,setting,cfg,router,persistent=store)
    store.close()
    store = Memory(path)
    policy = Policy(family,experts,gate,setting,public_config(cfg),store,clone_router(router) if family=='recurrent' else None)
    for _,obs,label,_ in store.history('episode',before_step=40):
        policy.predict(obs)
        policy.observe(obs,label)
    pred,_ = policy.predict(rows[40].observation)
    for key in pred:
        assert pred[key] == traces[family][40][key], key
    assert pred['retrieved_max_step'] is None or pred['retrieved_max_step'] < 40
    store.close()


def test_no_feedback_has_no_memory_or_selector_learning(experts,gate,router):
    cfg = dev_config()
    store = Memory()
    metrics,traces = evaluate(episode(12900704,80,'no_feedback',cfg),experts,gate,initial_selection(cfg),cfg,router,feedback=False,persistent=store)
    assert store.history('episode',before_step=100) == []
    assert all(metrics[f]['selector_slots_used'] == 0 for f in FAMILIES)
    assert [r['probability'] for r in traces['two_speed']] == [r['probability'] for r in traces['recency']]
    store.close()


def test_validation_does_not_depend_on_test_seeds_and_keeps_grids(experts,gate,router):
    cfg = dev_config(selector_rates=[.2],selector_temperatures=[4.],fading_half_lives=[2.],cache_rates=[.1],recurrent_temperatures=[1.])
    a = select(experts,gate,{cfg['recurrent_seeds'][0]:router},cfg)
    b = select(experts,gate,{cfg['recurrent_seeds'][0]:router},{**cfg,'test_seeds':[1], 'test_steps':1})
    assert a == b
    for grid in a['candidate_grids'].values():
        for row in grid:
            assert row['validation_objective'] == pytest.approx(.5*(row['validation_stable_loss']+row['validation_changed_loss']))


def test_training_refits_and_trace_arrays_reproduce(experts,gate):
    cfg = dev_config()
    a,ha = train_router(experts,cfg['recurrent_seeds'][0],cfg)
    b,hb = train_router(experts,cfg['recurrent_seeds'][0],cfg)
    assert ha == hb and all(np.array_equal(v,b.params[k]) for k,v in a.params.items())
    rows = episode(12900705,80,'repeated_changes',cfg)
    p = arrays(rows,evaluate(rows,experts,gate,initial_selection(cfg),cfg,a)[1])
    q = arrays(rows,evaluate(rows,experts,gate,initial_selection(cfg),cfg,b)[1])
    assert all(np.array_equal(v,q[k]) for k,v in p.items())
    assert all(v.dtype.kind != 'O' for v in p.values())


def test_world_bootstrap_uses_world_means_not_predictions():
    cfg = dev_config()
    p = interval([-.1,-.2,-.3,-.4,-.5,-.6],cfg)
    assert p['independent_worlds'] == 6 and p['mean'] == pytest.approx(-.35)
    assert p == interval([-.1,-.2,-.3,-.4,-.5,-.6],cfg)
    assert p['ci95'][0] < p['mean'] < p['ci95'][1]


def test_development_end_to_end_run_refit_reselection_and_history(monkeypatch,tmp_path):
    import twospeed_lab.experiment as experiment
    cfg = dev_config(selector_rates=[.2],selector_temperatures=[4.],fading_half_lives=[2.],
                     cache_rates=[.1],recurrent_temperatures=[1.],resume_step=80)
    cfg['worlds'] = [{'id':'development', 'test_seeds':cfg['test_seeds']}]
    monkeypatch.setattr(experiment,'config',lambda: cfg)
    output = tmp_path/'new-run'
    experiment.run(output)
    experiment.verify(output)
    receipt = json.loads((output/'verification.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'PASS' and receipt['policy_history_reloads_exact'] == len(FAMILIES)
    assert receipt['raw_predictions_replayed'] == 2*8*160*len(FAMILIES)
    assert receipt['future_rows_excluded'] == 80
