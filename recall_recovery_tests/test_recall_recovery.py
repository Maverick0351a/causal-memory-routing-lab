from dataclasses import replace
import inspect
import json
import numpy as np
import pytest
from archlab.controllers import RecurrentRouter,load_model
from confidence_lab.policy import Gate
from sensor_lab.public import Observation as SensorObservation
from recall_recovery.public import Observation,CueEmbedder,fit_experts
from recall_recovery.world import calibration_data,episode
from recall_recovery.memory import Memory
from recall_recovery.policy import Policy,FAMILIES,ContextCache
from recall_recovery.training import gradients,train_router,clone_router
from recall_recovery.benchmark import evaluate,initial_selection,select,comparisons,gates,response,scores
from recall_recovery.experiment import ROOT,config,check_preservation,pack
from transfer_lab.experiment import offline_proof


def dev_config(**changes):
    return {**config(),'calibration_seeds':[980101,980102],'calibration_steps':128,
        'coefficient_seed':980161,'expert_seeds':[980111,980112,980113,980114],
        'expert_epochs':40,'training_seeds':[980121,980122],'training_steps':96,
        'recurrent_seeds':[980131,980132,980133],'training_shuffle_seed':980162,
        'recurrent_hidden':6,'recurrent_epochs':2,'bptt_steps':16,
        'validation_seeds':[980141],'validation_steps':128,'shuffled_memory_seed':980163,
        'bootstrap_seed':980164,**changes}


@pytest.fixture(scope='module')
def experts():
    cfg = dev_config()
    x,y = calibration_data(cfg)
    return fit_experts(x,y,cfg)[0]


@pytest.fixture(scope='module')
def gate():
    return Gate.load(ROOT/config()['frozen_gate'])


@pytest.fixture
def router():
    return RecurrentRouter(seed=980134,dim=14,hidden=6)


def test_registered_seeds_are_fresh_disjoint_and_controls_fixed():
    cfg = config()
    seeds = sum([cfg[k] for k in ('calibration_seeds','expert_seeds','training_seeds','recurrent_seeds','validation_seeds','test_seeds')],[])
    seeds += [cfg[k] for k in ('coefficient_seed','training_shuffle_seed','shuffled_memory_seed','bootstrap_seed')]
    assert len(seeds)==len(set(seeds)) and len(cfg['test_seeds'])==24
    for path in (ROOT/'experiments').glob('*/protocol.json'):
        if path.parent.name==cfg['id']:
            continue
        old = json.loads(path.read_text())
        for key,value in old.items():
            if key.endswith('seeds') and isinstance(value,list):
                assert not set(seeds)&set(value)
    assert cfg['selection_cases']==['recurring','noisy_cues']
    assert not set(cfg['quality_cases'])&set(cfg['selection_cases'])


def test_identity_embedding_contains_only_observed_cue():
    a = Observation((1.,2.,3.,4.,5.,6.),(0.,)*8)
    b = Observation(a.cue,(999.,)*8)
    encoder = CueEmbedder()
    np.testing.assert_array_equal(encoder.encode([a,b]),[a.cue,a.cue])
    assert encoder.model_id=='continuous-public-cue-v1'
    assert encoder.encode([]).shape==(0,6)
    with pytest.raises(ValueError):
        Observation((1.,),(0.,)*8)
    with pytest.raises(ValueError):
        Observation((float('nan'),)*6,(0.,)*8)


def test_world_reproducibility_independent_random_labels_and_new_mappings():
    cfg = dev_config()
    rows = episode(980201,800,'recurring',cfg)
    assert rows==episode(980201,800,'recurring',cfg)==episode(980201,800,'no_feedback',cfg)
    random = episode(980201,800,'random_labels',cfg)
    assert [r.observation for r in rows]==[r.observation for r in random]
    assert any(a.label!=b.label for a,b in zip(rows,random))
    assert not any(r.returned for r in episode(980201,800,'novel_contexts',cfg))
    starts = [r for r in rows if r.switched]
    assert starts and all(r.age==0 for r in starts)
    assert any(r.returned for r in starts) and not rows[0].returned
    later = episode(980202,800,'recurring',cfg)
    assert rows[0].observation.cue!=later[0].observation.cue
    changed = episode(980201,800,'mapping_change',cfg)
    assert any(r.mapping_changed for r in changed)
    assert all(not r.mapping_changed for r in changed[:400])


def test_calibration_expert_fits_reproduce_and_lower_loss(experts):
    cfg = dev_config()
    x,y = calibration_data(cfg)
    assert y.shape==(256,4)
    a,history = fit_experts(x,y,cfg)
    assert np.array_equal(a.weights,experts.weights)
    assert all(h[-1]<h[0] for h in history)
    assert not experts.weights.flags.writeable


def test_continuous_expert_bptt_gradient_finite_difference():
    rng = np.random.default_rng(980203)
    router = RecurrentRouter(seed=980204,dim=14,hidden=3)
    xs,outputs,labels = rng.normal(size=(4,14)),rng.uniform(.05,.95,size=(4,4)),rng.integers(2,size=4)
    def loss():
        router.reset()
        caches = [router.forward(x)[1] for x in xs]
        return gradients(router,caches,outputs,labels)
    analytic,_ = loss()
    for name,array in router.params.items():
        numeric = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original+1e-6
            upper = loss()[1]
            array[index] = original-1e-6
            lower = loss()[1]
            array[index] = original
            numeric[index] = (upper-lower)/2e-6
        np.testing.assert_allclose(analytic[name],numeric,atol=2e-9,rtol=1e-5)


def test_recurrent_training_reproducible_checkpoint_and_budget(tmp_path,experts):
    cfg = dev_config()
    a,log = train_router(experts,980131,cfg)
    b,other = train_router(experts,980131,cfg)
    assert log==other and all(np.array_equal(v,b.params[k]) for k,v in a.params.items())
    assert log['examples_seen']==2*2*96 and log['updates']==2*2*6
    a.save(tmp_path/'router.npz')
    loaded = load_model(tmp_path/'router.npz')
    assert all(np.array_equal(v,loaded.params[k]) for k,v in a.params.items())
    copy = clone_router(a)
    assert all(not v.flags.writeable for v in copy.params.values())


def test_sqlite_search_cutoff_namespace_capacity_and_version(tmp_path):
    path = tmp_path/'memory.sqlite'
    store = Memory(path)
    obs = Observation((1.,)*6,(0.,)*8)
    for step in range(10):
        store.append('a',step,obs,step%2,np.full(4,float(step%2)))
    store.append('other',0,obs,0,np.zeros(4))
    store.close()
    store = Memory(path)
    # Future rows are already present in the reloaded RAM index.
    errors,count,steps = store.search('a',obs,before_step=5,capacity=3,limit=10,radius=.1)
    assert steps==[2,3,4] and count==3
    np.testing.assert_array_equal(errors,np.full(4,1/3))
    assert store.history('a',before_step=0)==[]
    assert len(store.history('other',before_step=20))==1
    assert store.search('a',obs,before_step=0,capacity=3,limit=10,radius=.1)==(None,0,[])
    with pytest.raises(TypeError):
        store.history('a')
    with pytest.raises(ValueError):
        store.append('a',2,obs,1,np.ones(4))
    store.db.execute("UPDATE feedback SET model_id='incompatible' WHERE namespace='other'")
    store.close()
    store = Memory(path)
    with pytest.raises(ValueError):
        store.history('other',before_step=5)
    store.close()


def test_shuffle_preserves_neighbor_count_but_breaks_feedback_association():
    store = Memory()
    a = Observation((0.,)*6,(0.,)*8)
    b = Observation((3.,)*6,(0.,)*8)
    for step in range(40):
        obs = a if step%2 else b
        store.append('a',step,obs,1,np.full(4,float(step%2)))
    true,count,steps = store.search('a',a,before_step=40,capacity=40,limit=20,radius=.1)
    shuffled,n,shuffled_steps = store.search('a',a,before_step=40,capacity=40,limit=20,radius=.1,
                                           permutation=np.random.default_rng(980205))
    assert count==n==20 and max(steps+shuffled_steps)<40
    np.testing.assert_array_equal(true,np.ones(4))
    assert (shuffled<1).all()
    store.close()


@pytest.mark.parametrize('family',FAMILIES)
def test_current_label_future_and_private_metadata_cannot_change_prediction(family,experts,gate,router):
    cfg = dev_config()
    rows = episode(980206,120,'recurring',cfg)
    changed = [replace(r,label=1-r.label if i>=60 else r.label,
        observation=Observation(tuple(-v for v in r.observation.cue),r.observation.readings) if i>60 else r.observation,
        context_id=999,age=999,returned=True,switched=False,mapping_changed=True) for i,r in enumerate(rows)]
    _,a = evaluate(rows,experts,gate,initial_selection(cfg),cfg,router)
    _,b = evaluate(changed,experts,gate,initial_selection(cfg),cfg,router)
    for key in ('probability','gated_probability','alpha','route','retrieved_count','retrieved_max_step'):
        assert [r[key] for r in a[family][:61]]==[r[key] for r in b[family][:61]]


def test_all_policy_predictions_commit_before_label_read(experts,gate,router):
    from unittest.mock import patch
    count = []
    original = Policy.predict
    def wrapped(self,observation):
        result = original(self,observation)
        count.append(self.family)
        return result
    class GuardRow:
        observation = Observation((.2,)*6,(.3,)*8)
        context_id,age,switched,returned,mapping_changed = 0,0,False,False,False
        @property
        def label(self):
            assert len(count)==len(FAMILIES)
            return 1
    with patch.object(Policy,'predict',wrapped):
        evaluate([GuardRow()],experts,gate,initial_selection(dev_config()),dev_config(),router)


def test_disabled_is_recency_and_no_feedback_keeps_memory_empty(experts,gate,router):
    cfg = dev_config()
    rows = episode(980207,350,'recurring',cfg)
    _,traces = evaluate(rows,experts,gate,initial_selection(cfg),cfg,router)
    assert traces['memory_disabled']==traces['recency']
    assert [r['retrieved_count'] for r in traces['memory']]==[r['retrieved_count'] for r in traces['shuffled_memory']]
    store = Memory()
    _,unfed = evaluate(rows,experts,gate,initial_selection(cfg),cfg,router,feedback=False,persistent=store)
    assert store.history('episode',before_step=999)==[]
    assert unfed['memory']==unfed['memory_disabled']==unfed['recency']==unfed['shuffled_memory']
    store.close()


def test_unseen_cue_falls_back_to_recency_and_released_evidence_changes_future(experts,gate):
    cfg = dev_config()
    store = Memory()
    obs = Observation((0.,)*6,(.2,)*8)
    for step in range(20):
        store.append('episode',step,obs,1,np.array([0.,1.,1.,1.]))
    memory = Policy('memory',experts,gate,initial_selection(cfg),cfg,store)
    recency = Policy('recency',experts,gate,initial_selection(cfg),cfg,store)
    memory.step = recency.step = 20
    novel = Observation((10.,)*6,(.2,)*8)
    a,_ = memory.predict(novel)
    b,_ = recency.predict(novel)
    assert a==b and a['retrieved_count']==0
    memory.observe(novel,1)
    recency.observe(novel,1)
    recalled,_ = memory.predict(obs)
    assert recalled['retrieved_count']==20 and recalled['route'][0]>recalled['route'][1]
    store.close()


@pytest.mark.parametrize('family',['memory','shuffled_memory','context_cache','recurrent'])
def test_persistent_resume_excludes_future_rows(tmp_path,family,experts,gate,router):
    cfg = dev_config()
    rows = episode(980208,180,'recurring',cfg)
    path = tmp_path/'feedback.sqlite'
    store = Memory(path)
    selection = initial_selection(cfg)
    _,traces = evaluate(rows,experts,gate,selection,cfg,router,persistent=store)
    store.close()
    store = Memory(path)
    cutoff = 100
    history = store.history('episode',before_step=cutoff)
    assert len(history)==100 and len(store.history('episode',before_step=999))==180
    policy = Policy(family,experts,gate,selection,cfg,store,clone_router(router) if family=='recurrent' else None)
    for _,observation,label,_ in history:
        policy.predict(observation)
        policy.observe(observation,label)
    prediction,_ = policy.predict(rows[cutoff].observation)
    assert all(traces[family][cutoff][k]==v for k,v in prediction.items())
    assert prediction['retrieved_max_step'] is None or prediction['retrieved_max_step']<cutoff
    store.close()


def test_context_cache_causal_update_and_bounded_slots():
    cache = ContextCache(dict(capacity=2,radius=.1,rate=.35,blend=1.))
    obs = Observation((0.,)*6,(0.,)*8)
    assert cache.query(obs)==(None,0)
    cache.observe(obs,np.zeros(4))
    errors,count = cache.query(obs)
    np.testing.assert_allclose(errors,.325)
    assert count==1
    cache.observe(Observation((3.,)*6,(0.,)*8),np.ones(4))
    cache.observe(Observation((6.,)*6,(0.,)*8),np.ones(4))
    assert len(cache.entries)==2 and cache.query(obs)==(None,0)


def test_policy_feedback_order_public_interfaces_and_frozen_weights(experts,gate,router):
    cfg = dev_config()
    store = Memory()
    policy = Policy('recurrent',experts,gate,initial_selection(cfg),cfg,store,clone_router(router))
    obs = Observation((0.,)*6,(.2,)*8)
    frozen = [a.copy() for a in (experts.weights,gate.weights,gate.mean,gate.scale,*policy.router.params.values())]
    with pytest.raises(RuntimeError):
        policy.observe(obs,1)
    policy.predict(obs)
    with pytest.raises(RuntimeError):
        policy.predict(obs)
    with pytest.raises(RuntimeError):
        policy.observe(Observation((1.,)*6,(.2,)*8),1)
    policy.observe(obs,1)
    assert all(np.array_equal(a,b) for a,b in zip(frozen,(experts.weights,gate.weights,gate.mean,gate.scale,*policy.router.params.values())))
    assert list(inspect.signature(Policy.predict).parameters)==['self','observation']
    for name in ('public.py','policy.py','memory.py'):
        source = (ROOT/'recall_recovery'/name).read_text()
        assert 'from .world' not in source and 'from .benchmark' not in source
    store.close()


def test_selection_reproducible_ignores_heldout_metadata(experts,gate,router):
    cfg = dev_config(memory_neighbors=[8],memory_radii=[.75],memory_blends=[.5],cache_radii=[.75],
        cache_rates=[.1],recurrent_seeds=[980131],recurrent_temperatures=[1.],validation_steps=80)
    a = select(experts,gate,{980131:router},cfg)
    b = select(experts,gate,{980131:router},{**cfg,'test_seeds':[123],'test_steps':3,'primary_cases':['random_labels']})
    assert a==b and len(a['candidate_grids']['memory'])==1
    assert a['validation_examples']==160


def test_return_slices_delay_exclude_first_visits_and_truncate_changes():
    cfg = dev_config()
    def row(step,age,returned,switched,p=.9):
        return dict(step=step,age=age,returned=returned,switched=switched,probability=p,
                    gated_probability=p,label=1,mapping_changed=False,retrieved_count=0)
    startup = [row(i,i,False,False) for i in range(10)]
    returned = [row(i,i-10,True,i==10) for i in range(10,20)]
    trace = startup+returned
    assert scores(trace,cfg)['return_scored_steps']==10
    assert response(trace,cfg)==dict(events=1,recovered=1,penalized_steps=7.,recovery_fraction=1.)
    bad = startup+[row(i,i-10,True,i==10,p=.1) for i in range(10,20)]
    assert response(bad,cfg)['penalized_steps']==64
    assert response(startup,cfg)['events']==0


def test_bootstrap_pairs_cases_within_seed_and_all_gates_are_required():
    cfg = dev_config(test_seeds=list(range(980301,980325)))
    records = []
    for ci,case in enumerate(cfg['cases']):
        for si,seed in enumerate(cfg['test_seeds']):
            for family in FAMILIES:
                value = .5+(.001*si+.01*ci if family=='memory' else 0.)
                records.append(dict(case=case,seed=seed,family=family,raw={'log_loss':value},
                    gated={'log_loss':value},return_first16={'log_loss':value},
                    return_delay={'penalized_steps':10. if family=='memory' else 15.}))
    paired = comparisons(records,cfg)
    expected = [.001*si+.005 for si in range(24)]
    np.testing.assert_allclose(paired['aggregate/raw_log_loss/memory_minus_recency']['per_stream'],expected)
    summary = {case:{family:{'raw':{'log_loss':.7,'brier':.25,'confidently_wrong_rate':0.,'accuracy':.5},
        'gated':{'log_loss':.69}} for family in FAMILIES} for case in cfg['cases']}
    g = gates(summary,paired,cfg)
    assert not g['functional_recall'] and g['faster_return_adaptation'] and g['quality_preserved']
    assert not g['memory_qualified'] # Good gating and speed cannot rescue failed recall.
    for key in paired:
        paired[key]['mean']=-.1 if 'log_loss' in key else -5.
        paired[key]['ci95']=[-.2,-.01] if 'log_loss' in key else [-6.,-4.]
    assert gates(summary,paired,cfg)['memory_qualified']
    summary['mapping_change']['memory']['raw']['confidently_wrong_rate']=.2
    assert not gates(summary,paired,cfg)['memory_qualified']


def test_packed_traces_and_offline_preservation(experts,gate,router):
    cfg = dev_config()
    _,traces = evaluate(episode(980209,30,'recurring',cfg),experts,gate,initial_selection(cfg),cfg,router)
    packed = pack(traces)
    assert len(packed)==30 and set(packed[0]['predictions'])==set(FAMILIES)
    assert packed[0]['predictions']['memory']['probability']==traces['memory'][0]['probability']
    assert offline_proof()['listed_provider_credentials_absent']
    assert check_preservation()['release_files_match']


def test_frozen_gate_result_is_json_serializable():
    cfg = dev_config(test_seeds=[980301,980302])
    paired = {}
    for right in ('memory_disabled','shuffled_memory','recency','recurrent','context_cache'):
        for endpoint in ('return_first16_log_loss','return_delay_penalized_steps','raw_log_loss'):
            paired[f'aggregate/{endpoint}/memory_minus_{right}'] = {'mean':-.1,'ci95':[-.2,-.01]}
    summary = {case:{family:{'raw':{'log_loss':.7,'brier':.25,'confidently_wrong_rate':0.,'accuracy':.5},
        'gated':{'log_loss':.69}} for family in FAMILIES} for case in cfg['cases']}
    result = gates(summary,paired,cfg)
    assert json.loads(json.dumps(result))==result
    assert type(result['gated_random_loss_control']) is bool


def test_development_end_to_end_export_report_and_reload(tmp_path,monkeypatch):
    import recall_recovery.experiment as experiment
    from recall_recovery.verify import verify
    cfg = dev_config(test_seeds=[980311,980312],test_steps=256,block_interval=[16,24],
        calibration_steps=32,expert_epochs=3,training_steps=64,recurrent_epochs=1,
        recurrent_seeds=[980131],memory_neighbors=[8],memory_radii=[1.25],memory_blends=[1.],
        cache_radii=[1.25],cache_rates=[.1],recurrent_temperatures=[1.],validation_steps=128,
        bootstrap_replicates=20,resume_step=128)
    monkeypatch.setattr(experiment,'config',lambda:cfg)
    output = tmp_path/'development-export'
    experiment.run(output)
    saved = json.loads((output/'results.json').read_text())
    assert len(saved['runs'])==12*7
    assert (output/'RESULTS.md').is_file() and (output/'comparison.html').is_file()
    verify(output)
    proof = json.loads((output/'verification.json').read_text())
    assert proof['status']=='PASS' and proof['raw_predictions_replayed']==12*7*256
    assert proof['persistent_memory_resumes']['memory']['future_rows_excluded']==128
