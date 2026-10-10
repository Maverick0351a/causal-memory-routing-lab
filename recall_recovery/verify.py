"""Fresh-process complete refit, selection, trace replay and causal DB resume."""
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
from .public import fit_experts
from .world import calibration_data,episode
from .training import train_router,clone_router
from .memory import Memory
from .policy import Policy,FAMILIES
from .benchmark import select,evaluate,summarize,comparisons,gates
from .experiment import ROOT,hashes,check_preservation,pack


def verify(run):
    output = Path(run).resolve()
    environment = offline_proof()
    result = json.loads((output/'results.json').read_text())
    cfg = result['config']
    locked = json.loads((output/'preregistration.json').read_text())
    assert cfg==locked['config'] and hashes()==locked['source_hashes']==result['source_hashes']
    for relative,digest in locked['source_hashes'].items():
        assert sha(output/'source'/relative)==digest,relative
    for relative,digest in json.loads((output/'artifact-hashes.json').read_text()).items():
        assert sha(output/relative)==digest,relative
    selection = json.loads((output/'selection.json').read_text())
    assert selection==result['selection'] and sha(output/'selection.json')==result['selection_sha256']
    test_start = json.loads((output/'test-generation.json').read_text())
    assert locked['created_utc']<selection['created_utc']<test_start['created_utc']
    assert test_start['selection_sha256']==result['selection_sha256']
    checkpoint = output/'checkpoints'
    assert sha(checkpoint/'frozen_gate.npz')==cfg['frozen_gate_sha256']
    assert sha(checkpoint/'experts.npz')==selection['experts_sha256']
    assert sha(checkpoint/'recurrent_selected.npz')==selection['selected_router_sha256']
    experts = Ensemble.load(checkpoint/'experts.npz')
    x,y = calibration_data(cfg)
    refit,history = fit_experts(x,y,cfg)
    assert np.array_equal(experts.weights,refit.weights)
    calibration = json.loads((output/'calibration.json').read_text())
    assert calibration['data_sha256']==digest_arrays(x,y) and calibration['loss_by_epoch']==history
    saved_training = json.loads((output/'recurrent-training.json').read_text())
    routers = {}
    for seed in cfg['recurrent_seeds']:
        routers[seed],training = train_router(refit,seed,cfg)
        assert training==saved_training[str(seed)]
        saved = load_model(checkpoint/f'recurrent_{seed}.npz')
        assert all(np.array_equal(v,saved.params[k]) for k,v in routers[seed].params.items())
        print('Recurrent refit exact:',seed,flush=True)
    gate = Gate.load(checkpoint/'frozen_gate.npz')
    reselected = select(refit,gate,routers,cfg)
    assert all(selection[k]==v for k,v in reselected.items())
    router = load_model(checkpoint/'recurrent_selected.npz')
    assert all(np.array_equal(v,router.params[k]) for k,v in routers[selection['recurrent']['seed']].params.items())
    frozen = [a.copy() for a in (experts.weights,gate.weights,gate.mean,gate.scale,*router.params.values())]
    print('All fits and all21 validation candidates reproduced exactly',flush=True)
    recorded = {(r['case'],r['seed'],r['family']):r for r in result['runs']}
    records,streams,predictions = [],0,0
    for case in cfg['cases']:
        for seed in cfg['test_seeds']:
            rows = episode(seed,cfg['test_steps'],case,cfg)
            evaluated,traces = evaluate(rows,experts,gate,selection,cfg,router,feedback=case!='no_feedback')
            saved = [json.loads(line) for line in (output/'traces'/case/(str(seed)+'.jsonl')).read_text().splitlines()]
            assert saved==pack(traces),(case,seed)
            assert traces['memory_disabled']==traces['recency']
            if case=='no_feedback':
                assert traces['memory']==traces['recency']==traces['shuffled_memory']
                assert all(r['retrieved_count']==0 for r in traces['memory'])
            # Key shuffling preserves the eligible key multiset and retrieval count.
            assert [r['retrieved_count'] for r in traces['memory']]==[r['retrieved_count'] for r in traces['shuffled_memory']]
            for family in FAMILIES:
                row = {'case':case,'seed':seed,'family':family,**evaluated[family]}
                assert row==recorded[case,seed,family]
                records.append(row)
                predictions += len(traces[family])
            streams += 1
        print(case,'complete raw/gated replay verified',flush=True)
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
        outputs = experts.predict(SensorObservation(observation.readings))
        assert np.array_equal(errors,((outputs>=.5)!=label).astype(float))
    seed = cfg['test_seeds'][0]
    saved = [json.loads(line) for line in (output/'traces/recurring'/(str(seed)+'.jsonl')).read_text().splitlines()]
    from .public import Observation
    query = Observation(tuple(saved[cutoff]['cue']),tuple(saved[cutoff]['readings']))
    resumes = {}
    for family in ('memory','shuffled_memory','context_cache','recurrent'):
        policy = Policy(family,experts,gate,selection,cfg,store,clone_router(router) if family=='recurrent' else None)
        for _,observation,label,_ in history:
            policy.predict(observation)
            policy.observe(observation,label)
        prediction,_ = policy.predict(query)
        assert prediction==saved[cutoff]['predictions'][family]
        assert prediction['retrieved_max_step'] is None or prediction['retrieved_max_step']<cutoff
        resumes[family] = {'step':cutoff,'prior_feedback_rows':len(history),'future_rows_excluded':len(full)-len(history),**prediction}
    store.close()
    proof = {'status':'PASS','test_streams':streams,'policy_traces':streams*len(FAMILIES),
        'raw_predictions_replayed':predictions,'gated_companion_probabilities_replayed':predictions,
        'all_predictions_routes_and_traces_exact':True,'expert_refit_exact':True,
        'all_three_recurrent_refits_exact':True,'all_validation_selection_recomputed':True,
        'metrics_slices_delays_bootstraps_gates_exact':True,'gate_byte_identical_and_immutable':True,
        'all_frozen_weights_unchanged':True,'disabled_memory_equals_recency':True,
        'shuffle_preserves_history_capacity_and_neighbor_counts':True,'no_feedback_memory_empty':True,
        'stored_expert_errors_derived_only_from_public_feedback':True,
        'source_and_recorded_artifact_hashes_match':True,'persistent_memory_resumes':resumes,
        'offline_proof':environment,'preservation':check_preservation(),'scientific_gates':result['gates']}
    write_json(output/'verification.json',proof)
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default=str(ROOT/'results/recurring-memory-v2'))
    verify(parser.parse_args().run)
