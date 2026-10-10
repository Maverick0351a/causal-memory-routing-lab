from dataclasses import replace
import inspect
import json
import numpy as np
import pytest
from archlab.controllers import RecurrentRouter
from confidence_lab.policy import Gate
from recall_recovery.public import Observation,fit_experts
from recall_recovery.benchmark import evaluate as original_evaluate
from recall_recovery.world import episode as original_episode
from recall_recovery.policy import FAMILIES as ORIGINAL_FAMILIES
from sensor_lab.public import Observation as SensorObservation
from forget_lab.experiment import ROOT,config,check_preservation,pack
from forget_lab.world import calibration_data,episode
from forget_lab.memory import Memory
from forget_lab.policy import Policy,FAMILIES,NEW_FAMILIES
from forget_lab.training import train_router,clone_router
from forget_lab.benchmark import evaluate,initial_selection,select,scores,response,comparisons,gates
from transfer_lab.experiment import offline_proof


def dev_config(**changes):
    return {**config(),'calibration_seeds':[990101,990102],'calibration_steps':64,
        'coefficient_seed':990161,'expert_seeds':[990111,990112,990113,990114],
        'expert_epochs':20,'training_seeds':[990121,990122],'training_cases':['recurring','repeated_changes'],
        'training_steps':128,'recurrent_seeds':[990131,990132,990133],'training_shuffle_seed':990162,
        'recurrent_hidden':6,'recurrent_epochs':2,'bptt_steps':16,'validation_seeds':[990141],
        'validation_steps':192,'shuffled_memory_seed':990163,'bootstrap_seed':990164,
        'test_seeds':[990301,990302],'test_steps':256,'resume_step':128,'block_interval':[16,24],
        'bootstrap_replicates':40,**changes}


@pytest.fixture(scope='module')
def experts():
    cfg = dev_config()
    return fit_experts(*calibration_data(cfg),cfg)[0]


@pytest.fixture(scope='module')
def gate():
    return Gate.load(ROOT/config()['frozen_gate'])


@pytest.fixture
def router():
    return RecurrentRouter(seed=990134,dim=14,hidden=6)


def test_all_registered_seeds_fresh_and_disjoint():
    cfg = config()
    def seeds(data):
        return [v for k,value in data.items() if k.endswith('_seeds') for v in value]+[
            value for k,value in data.items() if k.endswith('_seed')]
    chosen = seeds(cfg)
    assert len(chosen)==len(set(chosen)) and len(cfg['test_seeds'])==24
    for path in (ROOT/'experiments').glob('*/protocol.json'):
        if path.parent.name!=cfg['id']:
            assert not set(chosen)&set(seeds(json.loads(path.read_text(encoding='utf-8'))))
    assert set(cfg['primary_cases'])<=set(cfg['selection_cases'])
    assert len(cfg['training_cases'])==8 and cfg['selection_stable_weight']==.5
    assert all(s not in chosen for s in seeds(dev_config()))


@pytest.mark.parametrize('case',['recurring','noisy_cues','mapping_change','novel_contexts','random_labels','no_feedback'])
def test_preserved_world_cases_have_identical_public_observations_and_labels(case):
    cfg = dev_config()
    a,b = episode(990201,400,case,cfg),original_episode(990201,400,case,cfg)
    assert [(r.observation,r.label,r.context_id,r.age,r.switched) for r in a]==[
        (r.observation,r.label,r.context_id,r.age,r.switched) for r in b]


def test_changed_entry_is_first_visit_per_context_per_revision_only():
    cfg = dev_config()
    rows = episode(990202,800,'mapping_change',cfg)
    visits = [r for r in rows if r.age==0]
    assert max(r.revision for r in visits)==1
    changed = [r for r in visits if r.changed_entry]
    assert len(changed)==4 and {r.context_id for r in changed}==set(range(4))
    assert all(r.returned for r in changed)
    assert any(r.revision==1 and not r.changed_entry for r in visits)
    repeated = episode(990202,800,'repeated_changes',cfg)
    assert max(r.revision for r in repeated)>3
    keys = [(r.context_id,r.revision) for r in repeated if r.age==0 and r.changed_entry]
    assert len(keys)==len(set(keys))
    assert all(not r.changed_entry for r in repeated if r.revision==0)
    for start in [i for i,r in enumerate(rows) if r.age==0]:
        block = [r for r in rows[start:] if r.context_id==rows[start].context_id and r.revision==rows[start].revision]
        assert all(r.changed_entry==rows[start].changed_entry for r in rows[start:start+min(16,len(block))])


def test_noisy_changes_share_labels_and_readings_random_labels_share_public_inputs():
    cfg = dev_config()
    a,b = episode(990203,400,'repeated_changes',cfg),episode(990203,400,'noisy_changes',cfg)
    assert [(r.observation.readings,r.label,r.revision,r.changed_entry) for r in a]==[
        (r.observation.readings,r.label,r.revision,r.changed_entry) for r in b]
    assert a[0].observation.cue!=b[0].observation.cue
    normal,random = episode(990203,400,'recurring',cfg),episode(990203,400,'random_labels',cfg)
    assert [r.observation for r in normal]==[r.observation for r in random]
    assert any(a.label!=b.label for a,b in zip(normal,random))


def test_fading_formula_recent_selection_and_strict_future_cutoff(tmp_path):
    store = Memory(tmp_path/'fading.sqlite')
    obs = Observation((0.,)*6,(0.,)*8)
    for i in range(8):
        store.append('a',i,obs,int(i<4),np.full(4,float(i<4)))
    kwargs = dict(before_step=8,capacity=8,limit=8,radius=.1)
    loss,count,steps,effective = store.recent('a',obs,half_life=2.,**kwargs)
    weights = np.exp2(-np.arange(7,-1,-1)/2.)
    weights /= weights.sum()
    np.testing.assert_allclose(loss,weights[:4].sum())
    assert count==8 and steps==list(range(8)) and effective==pytest.approx(1/(weights@weights))
    uniform = store.recent('a',obs,half_life=None,**kwargs)
    np.testing.assert_array_equal(uniform[0],np.full(4,.5))
    assert uniform[3]==8.
    early = store.recent('a',obs,half_life=2.,**{**kwargs,'before_step':4})
    np.testing.assert_array_equal(early[0],np.ones(4))
    assert early[2]==[0,1,2,3]
    newest = store.recent('a',obs,half_life=None,**{**kwargs,'limit':3})
    assert newest[2]==[5,6,7] and np.array_equal(newest[0],np.zeros(4))
    assert store.recent('unknown',obs,half_life=2.,**kwargs)==(None,0,[],0.)
    assert store.recent('a',replace(obs,cue=(10.,)*6),half_life=2.,**kwargs)==(None,0,[],0.)
    store.close()


@pytest.mark.parametrize('changes',[{'before_step':-1},{'capacity':0},{'limit':0},{'radius':0},{'half_life':0}])
def test_fading_rejects_invalid_budget_or_cutoff(changes):
    store = Memory()
    with pytest.raises(ValueError):
        store.recent('a',Observation((0.,)*6,(0.,)*8),**{**dict(before_step=0,capacity=5,limit=4,radius=1.,half_life=2.),**changes})
    store.close()


def test_shuffle_preserves_counts_effective_samples_and_past_key_multiset():
    store = Memory()
    for i in range(30):
        store.append('a',i,Observation((float(i%2)*5.,)*6,(0.,)*8),i%2,np.full(4,float(i%2)))
    rng = np.random.default_rng(990204)
    for cutoff in (0,1,8,15,29):
        for key in (0.,5.):
            obs = Observation((key,)*6,(0.,)*8)
            kwargs = dict(before_step=cutoff,capacity=12,limit=7,radius=.1,half_life=2.)
            a = store.recent('a',obs,**kwargs)
            b = store.recent('a',obs,permutation=rng,**kwargs)
            assert a[1]==b[1] and a[3]==b[3]
            assert all(s<cutoff for s in a[2]+b[2])
    store.close()


@pytest.mark.parametrize('family',FAMILIES)
def test_current_future_answers_and_private_metadata_do_not_change_current_prediction(family,experts,gate,router):
    cfg = dev_config()
    rows = episode(990205,60,'repeated_changes',cfg)
    changed = [replace(r,label=1-r.label if i>=20 else r.label,context_id=999,age=999,
        revision=999,changed_entry=False,mapping_changed=False,returned=False,switched=False) for i,r in enumerate(rows)]
    kwargs = dict(families=(family,))
    a = evaluate(rows,experts,gate,initial_selection(cfg),cfg,router,**kwargs)[1][family]
    b = evaluate(changed,experts,gate,initial_selection(cfg),cfg,router,**kwargs)[1][family]
    for key in ('probability','gated_probability','alpha','route','retrieved_count','retrieved_max_step','effective_neighbors'):
        assert [r[key] for r in a[:21]]==[r[key] for r in b[:21]]
    assert list(inspect.signature(Policy.predict).parameters)==['self','observation']


def test_all_policies_commit_before_any_current_label_access(monkeypatch,experts,gate,router):
    cfg = dev_config()
    commits = []
    original = Policy.predict
    def predict(self,obs):
        result = original(self,obs)
        commits.append(self.family)
        return result
    monkeypatch.setattr(Policy,'predict',predict)
    class GuardedRow:
        def __init__(self,row,index):
            self.row,self.index = row,index
        def __getattr__(self,name):
            if name!='observation':
                assert len(commits)>=(self.index+1)*len(FAMILIES)
            return getattr(self.row,name)
    rows = [GuardedRow(r,i) for i,r in enumerate(episode(990206,10,'recurring',cfg))]
    evaluate(rows,experts,gate,initial_selection(cfg),cfg,router)
    assert len(commits)==10*len(FAMILIES)


def test_all_original_baselines_match_preserved_implementation(experts,gate,router):
    cfg = dev_config()
    rows = episode(990207,120,'repeated_changes',cfg)
    setting = initial_selection(cfg)
    old = original_evaluate(rows,experts,gate,setting,cfg,router)[1]
    new = evaluate(rows,experts,gate,setting,cfg,router)[1]
    for family in ORIGINAL_FAMILIES:
        assert [{k:v for k,v in r.items() if k in old[family][i]} for i,r in enumerate(new[family])]==old[family]
    assert new['memory_disabled']==new['recency']


def test_no_feedback_keeps_memory_empty(experts,gate,router):
    cfg = dev_config()
    store = Memory()
    _,traces = evaluate(episode(990208,120,'no_feedback',cfg),experts,gate,initial_selection(cfg),cfg,router,feedback=False,persistent=store)
    assert store.history('episode',before_step=999)==[]
    assert traces['forgetting']==traces['recent_uniform']==traces['shuffled_forgetting']==traces['recency']
    store.close()


@pytest.mark.parametrize('family',FAMILIES)
def test_reload_from_past_feedback_excludes_existing_future_rows(family,tmp_path,experts,gate,router):
    cfg = dev_config()
    selection = initial_selection(cfg)
    rows = episode(990209,100,'repeated_changes',cfg)
    path = tmp_path/(family+'.sqlite')
    store = Memory(path)
    _,traces = evaluate(rows,experts,gate,selection,cfg,router,persistent=store)
    store.close()
    store = Memory(path)
    policy = Policy(family,experts,gate,selection,cfg,store,clone_router(router) if family=='recurrent' else None)
    cutoff = 64
    history = store.history('episode',before_step=cutoff)
    assert len(history)==64 and len(store.history('episode',before_step=100))==100
    for _,obs,label,_ in history:
        policy.predict(obs)
        policy.observe(obs,label)
    prediction,_ = policy.predict(rows[cutoff].observation)
    assert all(prediction[k]==traces[family][cutoff][k] for k in prediction)
    assert prediction['retrieved_max_step'] is None or prediction['retrieved_max_step']<cutoff
    store.close()


def test_training_reproducible_budget_and_changing_exposure(experts):
    cfg = dev_config(recurrent_epochs=1)
    a,ha = train_router(experts,990131,cfg)
    b,hb = train_router(experts,990131,cfg)
    assert ha==hb and all(np.array_equal(v,b.params[k]) for k,v in a.params.items())
    assert ha['examples_seen']==256 and ha['updates']==16
    _,stable = train_router(experts,990131,{**cfg,'training_cases':['recurring','recurring']})
    assert ha['input_sha256']!=stable['input_sha256']


def test_selection_same_objective_and_independent_of_test_metadata(experts,gate,router):
    cfg = dev_config(memory_neighbors=[8],memory_radii=[1.25],memory_blends=[1.],cache_radii=[1.25],
        cache_rates=[.1],recurrent_seeds=[990131],recurrent_temperatures=[1.],fading_half_lives=[8.])
    a = select(experts,gate,{990131:router},cfg)
    b = select(experts,gate,{990131:router},{**cfg,'test_seeds':[1],'test_steps':1})
    assert a==b and a['validation_examples']==5*192
    for grid in a['candidate_grids'].values():
        for row in grid:
            assert row['validation_objective']==pytest.approx(.5*(row['validation_stable_entry_loss']+row['validation_changed_entry_loss']))


def test_changed_slices_cannot_be_diluted_by_later_recovered_visits():
    cfg = dev_config()
    rows = []
    for i in range(64):
        rows.append(dict(probability=.1 if i<16 else .9,gated_probability=.5,label=1,
            returned=True,changed_entry=i<16,age=i%16,switched=i%16==0,retrieved_count=1,effective_neighbors=1.))
    s = scores(rows,cfg)
    assert s['changed_scored_steps']==16 and s['return_scored_steps']==48
    assert s['changed_entry_first16']['log_loss']==pytest.approx(-np.log(.1))
    assert s['return_first16']['log_loss']==pytest.approx(-np.log(.9))
    assert s['change_delay']['penalized_steps']==64
    rows[0]['probability']=.9
    assert response(rows[:6],cfg)['events']==0


def fake_records(cfg):
    records = []
    for case in cfg['cases']:
        for seed in cfg['test_seeds']:
            for family in FAMILIES:
                changed = case in cfg['primary_cases']
                loss = .4 if family=='forgetting' and changed else .5
                metric = {'log_loss':loss,'brier':.2,'confidently_wrong_rate':0.,'accuracy':.5}
                delay = 7. if family in ('forgetting','recency','context_cache') else 12.
                records.append(dict(case=case,seed=seed,family=family,raw=metric,gated=metric,
                    return_first16=metric if case in cfg['stable_cases'] else None,
                    changed_entry_first16=metric if changed else None,changed_entry_first8=metric if changed else None,
                    change_delay={'penalized_steps':delay if changed else None}))
    return records


def test_paired_seed_averaging_and_gate_conjunction_json_serialization():
    cfg = dev_config()
    records = fake_records(cfg)
    paired = comparisons(records,cfg)
    summary = {c:{f:{'raw':{'accuracy':.5},'gated':{'log_loss':.7}} for f in FAMILIES} for c in cfg['cases']}
    g = gates(summary,paired,cfg)
    assert g['forgetting_qualified'] and json.loads(json.dumps(g))==g
    key = 'aggregate_change/changed_entry_first16_log_loss/forgetting_minus_context_cache'
    paired[key]['mean']=-.005
    assert not gates(summary,paired,cfg)['forgetting_qualified']
    paired[key]['mean']=-.1
    key = 'mapping_change/changed_entry_first16_log_loss/forgetting_minus_context_cache'
    paired[key]['ci95'][1]=.03
    assert not gates(summary,paired,cfg)['entry_value']
    pairs = comparisons(fake_records(cfg),cfg)
    assert pairs['aggregate_change/changed_entry_first16_log_loss/forgetting_minus_memory']['per_stream']==pytest.approx([-.1,-.1])


def test_full_development_export_report_and_verification(tmp_path,monkeypatch):
    import forget_lab.experiment as experiment
    from forget_lab.verify import verify
    cfg = dev_config(calibration_steps=32,expert_epochs=3,training_seeds=[990121],training_cases=['repeated_changes'],
        training_steps=64,recurrent_epochs=1,recurrent_seeds=[990131],memory_neighbors=[8],memory_radii=[1.25],
        memory_blends=[1.],cache_radii=[1.25],cache_rates=[.1],recurrent_temperatures=[1.],fading_half_lives=[8.],
        bootstrap_replicates=20)
    monkeypatch.setattr(experiment,'config',lambda:cfg)
    output = tmp_path/'development-pipeline'
    experiment.run(output)
    saved = json.loads((output/'results.json').read_text(encoding='utf-8'))
    assert len(saved['runs'])==8*2*10 and (output/'comparison.html').is_file()
    verify(output)
    proof = json.loads((output/'verification.json').read_text(encoding='utf-8'))
    assert proof['status']=='PASS' and proof['raw_predictions_replayed']==8*2*10*256
    assert len(proof['persistent_memory_resumes'])==10
    assert all(p['future_rows_excluded']==128 for p in proof['persistent_memory_resumes'].values())
    assert offline_proof()['listed_provider_credentials_absent'] and check_preservation()['release_files_match']
