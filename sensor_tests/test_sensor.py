from dataclasses import replace
import inspect
import json
import numpy as np
import pytest
from confidence_lab.policy import Gate
from sensor_lab.public import Observation,Ensemble,Adapter,fit_ensemble,logistic_objective
from sensor_lab.policy import Policy,OnlineCalibration,FAMILIES,logit
from sensor_lab.world import episode,training_data
from sensor_lab.memory import Memory
from sensor_lab.benchmark import evaluate,select,comparisons,scientific_gates,delay,slices
from sensor_lab.experiment import ROOT,config,check_preservation,pack
from transfer_lab.experiment import offline_proof


@pytest.fixture
def model():
    cfg = {**config(),'training_seeds':[970101,970102],'train_steps':128,'model_epochs':35,
           'model_seeds':[970111,970112,970113,970114]}
    x,y = training_data(cfg)
    return fit_ensemble(x,y,cfg)[0]


@pytest.fixture
def gate():
    return Gate.load(ROOT/config()['frozen_gate'])


def selection():
    return {'fixed_weights':[.7,0.],'online':{'rate':.08,'decay':.001}}


def test_seeds_new_disjoint_from_all_previous_experiments():
    cfg = config()
    seeds = sum([cfg[k] for k in ('training_seeds','validation_seeds','test_seeds','model_seeds')],[])
    assert len(seeds)==len(set(seeds))
    for path in (ROOT/'experiments').glob('*/protocol.json'):
        if path.parent.name==cfg['id']:
            continue
        old = json.loads(path.read_text())
        for key,value in old.items():
            if key.endswith('seeds') and isinstance(value,list):
                assert not set(seeds)&set(value)
    assert len(cfg['test_seeds'])==24
    assert cfg['selection_cases']==['stable','mixed']
    assert not set(cfg['selection_cases'])&{'short_mixed','weak_signal','rotating'}


def test_logistic_gradient_finite_difference():
    rng = np.random.default_rng(970201)
    x,y,w = rng.normal(size=(25,9)),rng.integers(2,size=25),rng.normal(size=9)*.2
    _,gradient = logistic_objective(w,x,y,.001)
    numerical = []
    for index in range(9):
        delta = np.zeros(9)
        delta[index] = 1e-6
        numerical.append((logistic_objective(w+delta,x,y,.001)[0]-logistic_objective(w-delta,x,y,.001)[0])/2e-6)
    np.testing.assert_allclose(gradient,numerical,atol=1e-9)


def test_training_reproducible_and_lowers_supervised_loss(tmp_path):
    cfg = {**config(),'training_seeds':[970202],'train_steps':256,'model_epochs':50,
           'model_seeds':[970211,970212,970213,970214]}
    x,y = training_data(cfg)
    a,history = fit_ensemble(x,y,cfg)
    b,history_b = fit_ensemble(x,y,cfg)
    assert np.array_equal(a.weights,b.weights) and history==history_b
    assert all(h[-1]<h[0] for h in history)
    assert a.weights.shape==(4,9) and not a.weights.flags.writeable
    a.save(tmp_path/'model.npz')
    assert np.array_equal(a.weights,Ensemble.load(tmp_path/'model.npz').weights)


def test_online_update_is_direct_log_loss_gradient_and_bounded():
    online = OnlineCalibration([.7,-.1],.08,.01)
    p = online.predict(.8)
    expected = online.weights-.08*(p-1)*np.array([logit(.8),1.])
    online.observe(.8,1)
    np.testing.assert_allclose(online.weights,expected,rtol=0,atol=1e-15)
    for _ in range(2000):
        online.observe(.95,0)
    assert (np.abs(online.weights)<=4).all() and np.isfinite(online.weights).all()
    assert online.weights[0]<0 # Inversion is allowed for the conventional baseline.


def test_feature_adapter_preserves_original_meanings_and_feedback_order(model):
    obs = Observation((1.,-.2,.3,.7,-.8,1.1,.5,-.4))
    adapter = Adapter(model)
    with pytest.raises(RuntimeError):
        adapter.observe(obs,1)
    base,features = adapter.predict(obs)
    assert base==pytest.approx(.05+.9*model.predict(obs).mean(),abs=1e-15)
    np.testing.assert_allclose(features,[1.,abs(2*base-1),0,0,0,0,0,0],atol=1e-15)
    with pytest.raises(RuntimeError):
        adapter.predict(obs)
    with pytest.raises(RuntimeError):
        adapter.observe(Observation((0.,)*8),1)
    adapter.observe(obs,1)
    np.testing.assert_allclose(adapter.correct,.5+np.array([.04,.15])*(float(base>=.5)-.5))
    np.testing.assert_allclose(adapter.gain,np.array([.04,.15])*np.log(base/.5))
    expected = (model.predict(obs)<.5).astype(float)
    np.testing.assert_allclose(adapter.fast,.5+.35*(expected-.5))
    np.testing.assert_allclose(adapter.slow,.5+.04*(expected-.5))
    with pytest.raises(RuntimeError):
        adapter.observe(obs,1)


def test_world_reproducible_observations_identical_and_ablations_match():
    cfg = config()
    a = episode(970203,400,'mixed',cfg)
    assert a==episode(970203,400,'mixed',cfg)==episode(970203,400,'no_feedback',cfg)
    for case in cfg['cases']:
        other = episode(970203,400,case,cfg)
        assert [r.observation for r in a]==[r.observation for r in other]
    changes = [r for r in a if r.switched]
    assert changes and all(r.age==0 for r in changes)
    assert {r.mode for r in a}=={'signal','noise'}
    assert not a[0].switched


@pytest.mark.parametrize('family',FAMILIES)
def test_label_future_and_private_metadata_invariance(family,model,gate):
    cfg = config()
    rows = episode(970204,90,'short_mixed',cfg)
    changed = [replace(r,label=1-r.label if i>=40 else r.label,
               observation=Observation(tuple(-v for v in r.observation.readings)) if i>40 else r.observation,
               mode='noise',age=999999,switched=False) for i,r in enumerate(rows)]
    _,a = evaluate(rows,model,gate,selection(),cfg)
    _,b = evaluate(changed,model,gate,selection(),cfg)
    for key in ('probability','base_probability','alpha','slope','intercept'):
        assert [r[key] for r in a[family][:41]]==[r[key] for r in b[family][:41]]


def test_all_predictions_commit_before_answer_access(model,gate):
    # A label property checks that each policy has finished prediction first.
    from unittest.mock import patch
    count = []
    original = Policy.predict
    def wrapped(self,base,features):
        prediction = original(self,base,features)
        count.append(self.family)
        return prediction
    class GuardRow:
        observation = Observation((.2,)*8)
        mode,age,switched = 'noise',0,False
        @property
        def label(self):
            assert len(count)==len(FAMILIES)
            return 1
    with patch.object(Policy,'predict',wrapped):
        evaluate([GuardRow()],model,gate,selection(),config())


def test_no_feedback_changes_neither_history_nor_online_weights(model,gate):
    cfg = config()
    rows = episode(970205,100,'mixed',cfg)
    fed,_ = evaluate(rows,model,gate,selection(),cfg)
    unfed,traces = evaluate(rows,model,gate,selection(),cfg,feedback=False)
    assert fed['base']['features_sha256']!=unfed['base']['features_sha256']
    assert len({r['features_sha256'] for r in fed.values()})==1
    assert all(r['slope']==.7 and r['intercept']==0 for r in traces['online_calibration'])
    assert any(r['base_probability']!=.5 for r in traces['base'])


def test_model_gate_arrays_immutable_and_unchanged(model,gate):
    arrays = [model.weights,gate.weights,gate.mean,gate.scale]
    copies = [a.copy() for a in arrays]
    evaluate(episode(970206,90,'mixed',config()),model,gate,selection(),config())
    assert all(np.array_equal(a,b) and not a.flags.writeable for a,b in zip(arrays,copies))
    with pytest.raises(ValueError):
        gate.weights[0] = 1


def test_interfaces_exclude_seed_regime_and_evaluator():
    assert list(inspect.signature(Adapter.predict).parameters)==['self','observation']
    assert list(inspect.signature(Policy.predict).parameters)==['self','base','features']
    for name in ('policy.py','public.py','memory.py'):
        text = (ROOT/'sensor_lab'/name).read_text()
        assert 'from .world' not in text and 'from .benchmark' not in text
    with pytest.raises(ValueError):
        Observation((1.,2.))
    with pytest.raises(ValueError):
        Observation((float('nan'),)*8)


@pytest.mark.parametrize('family',['frozen_gate','online_calibration'])
def test_sqlite_restart_resumes_strictly_prior_feedback(tmp_path,family,model,gate):
    cfg = config()
    rows = episode(970207,80,'mixed',cfg)
    store = Memory(tmp_path/'feedback.sqlite')
    _,traces = evaluate(rows,model,gate,selection(),cfg,persistent=store,namespace='a')
    store.append('other',0,rows[0].observation,1-rows[0].label)
    store.close()
    store = Memory(tmp_path/'feedback.sqlite')
    history = store.history('a',before_step=40)
    assert len(history)==40 and max(r[0] for r in history)==39
    adapter,policy = Adapter(model),Policy(family,gate,selection())
    for _,observation,label in history:
        base,x = adapter.predict(observation)
        policy.predict(base,x)
        policy.observe(label)
        adapter.observe(observation,label)
    base,x = adapter.predict(rows[40].observation)
    prediction = policy.predict(base,x)
    assert all(traces[family][40][k]==v for k,v in prediction.items())
    assert len(store.history('other',before_step=40))==1
    assert store.history('a',before_step=0)==[]
    with pytest.raises(TypeError):
        store.history('a')
    store.close()


def test_baseline_selection_is_reproducible_and_ignores_test_cases(model):
    cfg = {**config(),'validation_seeds':[970208],'validation_steps':64,'calibration_epochs':40,
           'online_rates':[.02,.08],'online_decays':[0.,.01]}
    a = select(model,cfg)
    b = select(model,{**cfg,'test_seeds':[123],'test_steps':3,'primary_cases':['rotating']})
    assert a==b
    assert len(a['online_validation_grid'])==4 and a['validation_examples']==128


def test_transition_slices_exclude_startup_and_delay_censors():
    cfg = config()
    def row(step,age,switched,p=.5,mode='noise'):
        return dict(step=step,age=age,switched=switched,probability=p,label=1,mode=mode)
    startup = [row(i,i,False) for i in range(20)]
    transition = [row(i,i-20,i==20) for i in range(20,40)]
    trace = startup+transition
    assert slices(trace)['noise_entry_first16']['steps']==16
    assert delay(trace,cfg)['noise']=={'events':1,'recovered':1,'penalized_steps':15.,'recovery_fraction':1.}
    failed = startup+[row(i,i-20,i==20,p=.01) for i in range(20,40)]
    assert delay(failed,cfg)['noise']['penalized_steps']==96
    assert delay(startup,cfg)['noise']['events']==0


def test_bootstrap_averages_cases_within_seed_and_failed_gate_stays_failed():
    cfg = config()
    records = []
    for ci,case in enumerate(cfg['cases']):
        for si,seed in enumerate(cfg['test_seeds']):
            for family in FAMILIES:
                records.append(dict(case=case,seed=seed,family=family,
                    metrics={'log_loss':.5+(.001*si+.01*ci if family=='frozen_gate' else 0.)}))
    paired = comparisons(records,cfg)
    expected = [.001*si+np.mean([.01*cfg['cases'].index(c) for c in cfg['primary_cases']]) for si in range(24)]
    np.testing.assert_allclose(paired['aggregate/frozen_gate_minus_online_calibration']['per_stream'],expected)
    summary = {}
    for case in cfg['cases']:
        summary[case] = {}
        for family in FAMILIES:
            summary[case][family] = {'log_loss':.7 if family=='frozen_gate' else .69,'brier':.25,
                'confidently_wrong_rate':0.,'accuracy':.5,
                'slices':{k:{'metrics':{'log_loss':.7}} for k in ('signal_entry_first16','noise_entry_first16')}}
    gates = scientific_gates(summary,paired,cfg)
    assert not gates['cross_task_value'] and not gates['qualified_on_sensor_family']
    assert gates['negative_controls']


def test_uniform_and_trace_packing_are_exact(model,gate):
    _,traces = evaluate(episode(970209,35,'random_labels',config()),model,gate,selection(),config())
    assert all(r['probability']==.5 for r in traces['uniform'])
    packed = pack(traces)
    assert len(packed)==35 and set(packed[0]['predictions'])==set(FAMILIES)
    assert packed[0]['predictions']['frozen_gate']['probability']==traces['frozen_gate'][0]['probability']


def test_keyless_offline_and_preservation():
    assert offline_proof()['listed_provider_credentials_absent']
    assert check_preservation()['release_files_match']
