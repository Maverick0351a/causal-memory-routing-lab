"""Fresh-process refit, reselection, full trace replay and future-row exclusion."""
import argparse
import json
from pathlib import Path
import numpy as np
from confidence_lab.policy import Gate
from confidence_lab.experiment import sha,write_json
from confidence_lab.benchmark import digest_arrays
from archlab.controllers import load_model
from sensor_lab.public import Ensemble,Observation as SensorObservation
from transfer_lab.experiment import offline_proof
from recall_recovery.public import fit_experts,Observation
from .world import calibration_data,episode
from .training import train_router,clone_router
from .memory import Memory
from .policy import Policy,FAMILIES
from .benchmark import select,evaluate,summarize,comparisons,gates
from .experiment import ROOT,hashes,check_preservation,pack


def verify(run):
    output = Path(run).resolve()
    environment = offline_proof()
    result = json.loads((output/'results.json').read_text(encoding='utf-8'))
    cfg = result['config']
    locked = json.loads((output/'preregistration.json').read_text(encoding='utf-8'))
    assert cfg==locked['config'] and hashes()==locked['source_hashes']==result['source_hashes']
    for relative,digest in locked['source_hashes'].items():
        assert sha(output/'source'/relative)==digest,relative
    for relative,digest in json.loads((output/'artifact-hashes.json').read_text(encoding='utf-8')).items():
        assert sha(output/relative)==digest,relative
    selection = json.loads((output/'selection.json').read_text(encoding='utf-8'))
    assert selection==result['selection'] and sha(output/'selection.json')==result['selection_sha256']
    start = json.loads((output/'test-generation.json').read_text(encoding='utf-8'))
    assert locked['created_utc']<selection['created_utc']<start['created_utc']
    assert start['selection_sha256']==result['selection_sha256']
    checkpoint = output/'checkpoints'
    assert sha(checkpoint/'frozen_gate.npz')==cfg['frozen_gate_sha256']
    assert sha(checkpoint/'experts.npz')==selection['experts_sha256']
    assert sha(checkpoint/'recurrent_selected.npz')==selection['selected_router_sha256']
    experts = Ensemble.load(checkpoint/'experts.npz')
    x,y = calibration_data(cfg)
    refit,history = fit_experts(x,y,cfg)
    assert np.array_equal(experts.weights,refit.weights)
    calibration = json.loads((output/'calibration.json').read_text(encoding='utf-8'))
    assert calibration['data_sha256']==digest_arrays(x,y) and calibration['loss_by_epoch']==history
    training = json.loads((output/'recurrent-training.json').read_text(encoding='utf-8'))
    routers = {}
    for seed in cfg['recurrent_seeds']:
        routers[seed],trained = train_router(refit,seed,cfg)
        assert trained==training[str(seed)]
        saved = load_model(checkpoint/f'recurrent_{seed}.npz')
        assert all(np.array_equal(v,saved.params[k]) for k,v in routers[seed].params.items())
        print('Recurrent refit exact:',seed,flush=True)
    gate = Gate.load(checkpoint/'frozen_gate.npz')
    reselected = select(refit,gate,routers,cfg)
    assert all(selection[k]==v for k,v in reselected.items())
    router = load_model(checkpoint/'recurrent_selected.npz')
    assert all(np.array_equal(v,router.params[k]) for k,v in routers[selection['recurrent']['seed']].params.items())
    frozen = [a.copy() for a in (experts.weights,gate.weights,gate.mean,gate.scale,*router.params.values())]
    print('All fits and all validation candidates exact',flush=True)
    recorded = {(r['case'],r['seed'],r['family']):r for r in result['runs']}
    records,streams,predictions = [],0,0
    for case in cfg['cases']:
        for seed in cfg['test_seeds']:
            evaluated,traces = evaluate(episode(seed,cfg['test_steps'],case,cfg),experts,gate,selection,cfg,router,feedback=case!='no_feedback')
            saved = [json.loads(line) for line in (output/'traces'/case/(str(seed)+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            assert saved==pack(traces),(case,seed)
            assert traces['memory_disabled']==traces['recency']
            if case=='no_feedback':
                assert traces['forgetting']==traces['recent_uniform']==traces['shuffled_forgetting']==traces['recency']
                assert all(r['retrieved_count']==0 for r in traces['forgetting'])
            for key in ('retrieved_count','effective_neighbors'):
                assert [r[key] for r in traces['forgetting']]==[r[key] for r in traces['shuffled_forgetting']]
            for family in FAMILIES:
                row = {'case':case,'seed':seed,'family':family,**evaluated[family]}
                assert row==recorded[case,seed,family]
                records.append(row)
                predictions += len(traces[family])
            streams += 1
        print(case,'raw/gated replay verified',flush=True)
    assert all(np.array_equal(a,b) for a,b in zip(frozen,(experts.weights,gate.weights,gate.mean,gate.scale,*router.params.values())))
    summary = summarize(records)
    assert summary==result['summary']
    paired = comparisons(records,cfg)
    assert paired==result['paired'] and gates(summary,paired,cfg)==result['gates']
    store = Memory(output/'feedback.sqlite')
    cutoff = cfg['resume_step']
    history = store.history('episode',before_step=cutoff)
    full = store.history('episode',before_step=cfg['test_steps'])
    assert len(history)==cutoff and len(full)==cfg['test_steps']
    for step,observation,label,errors in full:
        p = experts.predict(SensorObservation(observation.readings))
        assert np.array_equal(errors,((p>=.5)!=label).astype(float))
    seed = cfg['test_seeds'][0]
    saved = [json.loads(line) for line in (output/'traces/repeated_changes'/(str(seed)+'.jsonl')).read_text(encoding='utf-8').splitlines()]
    query = Observation(tuple(saved[cutoff]['cue']),tuple(saved[cutoff]['readings']))
    resumes = {}
    for family in FAMILIES:
        policy = Policy(family,experts,gate,selection,cfg,store,clone_router(router) if family=='recurrent' else None)
        for _,observation,label,_ in history:
            policy.predict(observation)
            policy.observe(observation,label)
        prediction,_ = policy.predict(query)
        assert prediction==saved[cutoff]['predictions'][family],family
        assert prediction['retrieved_max_step'] is None or prediction['retrieved_max_step']<cutoff
        resumes[family] = {'step':cutoff,'prior_feedback_rows':len(history),'future_rows_excluded':len(full)-len(history),**prediction}
    store.close()
    proof = {'status':'PASS','test_streams':streams,'policy_traces':streams*len(FAMILIES),
        'raw_predictions_replayed':predictions,'gated_companion_probabilities_replayed':predictions,
        'expert_refit_exact':True,'all_recurrent_refits_exact':True,'all_validation_scores_and_selection_exact':True,
        'all_predictions_routes_metrics_slices_delays_bootstraps_gates_exact':True,
        'source_and_artifact_hashes_match':True,'frozen_weights_and_gate_unchanged':True,
        'disabled_memory_equals_recency':True,'no_feedback_memory_empty':True,
        'shuffle_preserves_neighbor_counts_and_effective_sample_sizes':True,
        'stored_errors_derived_only_from_public_feedback':True,'persistent_memory_resumes':resumes,
        'offline_proof':environment,'preservation':check_preservation(),'scientific_gates':result['gates']}
    write_json(output/'verification.json',proof)
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default=str(ROOT/'results/forgetting-v1'))
    verify(parser.parse_args().run)
