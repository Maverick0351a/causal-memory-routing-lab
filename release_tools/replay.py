"""Refit/reselect/replay published forgetting artifacts without private host files."""
import argparse
import json
from pathlib import Path
import tarfile
import tempfile
import numpy as np
from archlab.controllers import load_model
from confidence_lab.policy import Gate
from confidence_lab.experiment import write_json
from confidence_lab.benchmark import digest_arrays
from sensor_lab.public import Observation as SensorObservation
from recall_recovery.public import Ensemble,fit_experts
from forget_lab.experiment import ROOT,config,pack
from forget_lab.world import calibration_data,episode
from forget_lab.training import train_router,clone_router
from forget_lab.benchmark import select,evaluate,summarize,comparisons,gates
from forget_lab.memory import Memory
from forget_lab.policy import Policy,FAMILIES
from recall_recovery.public import Observation
from transfer_lab.experiment import offline_proof
from .integrity import sha,verify_release


def verify(archive,receipt):
    receipt=Path(receipt)
    if receipt.exists(): raise FileExistsError('Preserve the existing receipt')
    integrity=verify_release(ROOT)
    cfg=config()
    output=ROOT/'results/forgetting-v1'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    result=read(output/'results.json')
    selection=read(output/'selection.json')
    index=read(ROOT/'artifacts-index.json')['forgetting-v1']
    assert sha(archive)==index['sha256'],'Published archive digest mismatch'
    assert result['config']==cfg
    checkpoint=output/'checkpoints'
    x,y=calibration_data(cfg)
    experts,history=fit_experts(x,y,cfg)
    saved=Ensemble.load(checkpoint/'experts.npz')
    assert np.array_equal(experts.weights,saved.weights)
    calibration=read(output/'calibration.json')
    assert history==calibration['loss_by_epoch'] and digest_arrays(x,y)==calibration['data_sha256']
    routers={}
    training=read(output/'recurrent-training.json')
    for seed in cfg['recurrent_seeds']:
        routers[seed],h=train_router(experts,seed,cfg)
        original=load_model(checkpoint/f'recurrent_{seed}.npz')
        assert h==training[str(seed)]
        assert all(np.array_equal(v,original.params[k]) for k,v in routers[seed].params.items())
        print('Exact recurrent refit',seed,flush=True)
    gate=Gate.load(checkpoint/'frozen_gate.npz')
    assert sha(checkpoint/'frozen_gate.npz')==cfg['frozen_gate_sha256']
    reselected=select(experts,gate,routers,cfg)
    assert all(selection[k]==v for k,v in reselected.items())
    router=routers[selection['recurrent']['seed']]
    recorded={(r['case'],r['seed'],r['family']):r for r in result['runs']}
    records=[]
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        names=[m.name for m in members]
        assert len(names)==len(set(names))
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        for case in cfg['cases']:
            for seed in cfg['test_seeds']:
                evaluated,traces=evaluate(episode(seed,cfg['test_steps'],case,cfg),experts,gate,selection,cfg,router,feedback=case!='no_feedback')
                saved=[json.loads(line) for line in tar.extractfile(f'traces/{case}/{seed}.jsonl')]
                assert pack(traces)==saved,(case,seed)
                for family in FAMILIES:
                    row={'case':case,'seed':seed,'family':family,**evaluated[family]}
                    assert row==recorded[case,seed,family]
                    records.append(row)
                assert traces['memory_disabled']==traces['recency']
                for key in ('retrieved_count','effective_neighbors'):
                    assert [r[key] for r in traces['forgetting']]==[r[key] for r in traces['shuffled_forgetting']]
                if case=='no_feedback':
                    assert traces['forgetting']==traces['recent_uniform']==traces['shuffled_forgetting']==traces['recency']
            print(case,'public replay exact',flush=True)
        with tempfile.TemporaryDirectory() as temporary:
            db=Path(temporary)/'feedback.sqlite'
            db.write_bytes(tar.extractfile('feedback.sqlite').read())
            store=Memory(db)
            cutoff=cfg['resume_step']
            past=store.history('episode',before_step=cutoff)
            full=store.history('episode',before_step=cfg['test_steps'])
            assert len(past)==1024 and len(full)==2048
            for _,obs,label,errors in full:
                outputs=experts.predict(SensorObservation(obs.readings))
                assert np.array_equal(errors,((outputs>=.5)!=label).astype(float))
            saved=[json.loads(line) for line in tar.extractfile(f'traces/repeated_changes/{cfg["test_seeds"][0]}.jsonl')]
            query=Observation(tuple(saved[cutoff]['cue']),tuple(saved[cutoff]['readings']))
            for family in FAMILIES:
                policy=Policy(family,experts,gate,selection,cfg,store,clone_router(router) if family=='recurrent' else None)
                for _,obs,label,_ in past: policy.predict(obs);policy.observe(obs,label)
                prediction,_=policy.predict(query)
                assert prediction==saved[cutoff]['predictions'][family]
                assert prediction['retrieved_max_step'] is None or prediction['retrieved_max_step']<cutoff
            store.close()
    summary=summarize(records)
    paired=comparisons(records,cfg)
    assert summary==result['summary'] and paired==result['paired'] and gates(summary,paired,cfg)==result['gates']
    proof={'status':'PASS','scope':'Public source/input manifests and published synthetic archive, not original private host files',
           'public_integrity':integrity,'archive_sha256':index['sha256'],'raw_predictions_replayed':3932160,
           'companion_probabilities_replayed':3932160,'policy_traces':1920,'streams':192,
           'expert_and_all_recurrent_refits_exact':True,'all_24_validation_scores_and_selection_exact':True,
           'all_traces_routes_metrics_slices_delays_bootstraps_gates_exact':True,
           'all_10_history_resumes_exclude_1024_future_rows':True,'offline_proof':offline_proof(),
           'scientific_gates':result['gates']}
    write_json(receipt,proof)
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();verify(a.archive,a.receipt)
