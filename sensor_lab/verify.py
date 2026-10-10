"""Fresh-process refit/reselection, complete causal replay and SQLite resume."""
import argparse
import json
from pathlib import Path
import numpy as np
from confidence_lab.policy import Gate
from confidence_lab.benchmark import digest_arrays
from confidence_lab.experiment import sha,write_json
from transfer_lab.experiment import offline_proof
from .public import Ensemble,Adapter,fit_ensemble
from .policy import Policy,FAMILIES
from .world import training_data,episode
from .memory import Memory
from .benchmark import select,evaluate,average,comparisons,scientific_gates
from .experiment import ROOT,hashes,check_preservation,pack


def verify(run):
    output = Path(run).resolve()
    environment = offline_proof()
    result = json.loads((output/'results.json').read_text())
    cfg = result['config']
    locked = json.loads((output/'preregistration.json').read_text())
    assert cfg==locked['config']
    assert hashes()==locked['source_hashes']==result['source_hashes']
    for relative,digest in locked['source_hashes'].items():
        assert sha(output/'source'/relative)==digest,relative
    for relative,digest in json.loads((output/'artifact-hashes.json').read_text()).items():
        assert sha(output/relative)==digest,relative
    selection = json.loads((output/'selection.json').read_text())
    assert selection==result['selection']
    assert sha(output/'selection.json')==result['selection_sha256']
    assert locked['created_utc']<selection['created_utc']
    test_start = json.loads((output/'test-generation.json').read_text())
    assert selection['created_utc']<test_start['created_utc']
    assert test_start['selection_sha256']==result['selection_sha256']
    assert sha(output/'checkpoints/frozen_gate.npz')==cfg['frozen_gate_sha256']
    ensemble = Ensemble.load(output/'checkpoints/ensemble.npz')
    assert sha(output/'checkpoints/ensemble.npz')==selection['ensemble_sha256']
    x,y = training_data(cfg)
    refitted,history = fit_ensemble(x,y,cfg)
    assert np.array_equal(ensemble.weights,refitted.weights)
    training = json.loads((output/'training.json').read_text())
    assert training['data_sha256']==digest_arrays(x,y)
    assert training['losses_by_epoch']==history
    reselected = select(refitted,cfg)
    assert all(selection[k]==v for k,v in reselected.items())
    gate = Gate.load(output/'checkpoints/frozen_gate.npz')
    frozen = [a.copy() for a in (ensemble.weights,gate.weights,gate.mean,gate.scale)]
    print('Base ensemble refit and calibration reselection match exactly',flush=True)
    recorded = {(r['case'],r['seed'],r['family']):r for r in result['runs']}
    records,streams,predictions = [],0,0
    for case in cfg['cases']:
        for seed in cfg['test_seeds']:
            rows = episode(seed,cfg['test_steps'],case,cfg)
            evaluated,traces = evaluate(rows,ensemble,gate,selection,cfg,feedback=case!=cfg['no_feedback_case'])
            saved = [json.loads(line) for line in (output/'traces'/case/(str(seed)+'.jsonl')).read_text().splitlines()]
            assert saved==pack(traces),(case,seed)
            for family in FAMILIES:
                row = {'case':case,'seed':seed,'family':family,**evaluated[family]}
                assert row==recorded[case,seed,family]
                records.append(row)
                predictions += len(traces[family])
            streams += 1
        print(case,'exact replay verified',flush=True)
    assert all(np.array_equal(a,b) for a,b in zip(frozen,(ensemble.weights,gate.weights,gate.mean,gate.scale)))
    summaries = average(records)
    assert summaries==result['summary']
    paired = comparisons(records,cfg)
    assert paired==result['paired']
    assert scientific_gates(summaries,paired,cfg)==result['gates']
    store = Memory(output/'feedback.sqlite')
    seed,cutoff = cfg['test_seeds'][0],cfg['resume_step']
    namespace = f'mixed/{seed}'
    history = store.history(namespace,before_step=cutoff)
    assert len(history)==cutoff and all(step<cutoff for step,_,_ in history)
    saved = [json.loads(line) for line in (output/'traces/mixed'/(str(seed)+'.jsonl')).read_text().splitlines()]
    resumes = {}
    for family in ('frozen_gate','online_calibration'):
        adapter,policy = Adapter(ensemble),Policy(family,gate,selection)
        for _,observation,label in history:
            base,features = adapter.predict(observation)
            policy.predict(base,features)
            policy.observe(label)
            adapter.observe(observation,label)
        from .public import Observation
        obs = Observation(tuple(saved[cutoff]['observation']))
        base,features = adapter.predict(obs)
        prediction = policy.predict(base,features)
        assert prediction==saved[cutoff]['predictions'][family]
        resumes[family] = {'step':cutoff,'history_strictly_before_step':len(history),**prediction}
    store.close()
    proof = {'status':'PASS','test_streams':streams,'policy_traces':streams*len(FAMILIES),
        'replayed_predictions':predictions,'all_traces_and_probabilities_exact':True,
        'base_ensemble_refit_exact':True,'calibration_fit_and_selection_recomputed_exactly':True,
        'gate_checkpoint_byte_identical':True,'gate_normalization_and_weights_unchanged':True,
        'base_predictor_weights_unchanged':True,'metrics_slices_delays_bootstraps_and_gates_recomputed':True,
        'all_recorded_source_and_artifact_hashes_match':True,'persistent_memory_resumes':resumes,
        'offline_proof':environment,'preservation':check_preservation(),'scientific_gates':result['gates']}
    write_json(output/'verification.json',proof)
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default=str(ROOT/'results/sensor-transfer-v1'))
    verify(parser.parse_args().run)
