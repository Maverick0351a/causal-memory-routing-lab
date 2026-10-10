"""Locked fitting/selection then held-out memory evaluation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
from confidence_lab.policy import Gate
from confidence_lab.experiment import sha,write_json
from confidence_lab.benchmark import digest_arrays
from transfer_lab.experiment import offline_proof
from .public import fit_experts
from .world import calibration_data,episode
from .training import train_router
from .memory import Memory
from .policy import FAMILIES
from .benchmark import select,evaluate,summarize,comparisons,gates

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'experiments/recurring-memory-recovery-20261010'


def config():
    return json.loads((EXPERIMENT/'protocol.json').read_text())


def hashes():
    """Public adapter: snapshot portable sources and frozen configuration."""
    from release_tools.integrity import source_hashes
    return source_hashes(ROOT)


def check_preservation():
    """Public adapter: verifies released files, never original private projects."""
    from release_tools.integrity import verify_release
    return verify_release(ROOT)


def pack(traces):
    keys = ('probability','gated_probability','alpha','route','retrieved_count','retrieved_max_step')
    records = []
    for step,row in enumerate(traces['memory']):
        record = {k:v for k,v in row.items() if k not in keys}
        record['predictions'] = {family:{k:traces[family][step][k] for k in keys} for family in FAMILIES}
        records.append(record)
    return records


def run(output):
    cfg = config()
    environment = offline_proof()
    check_preservation()
    seeds = sum([cfg[k] for k in ('calibration_seeds','expert_seeds','training_seeds','recurrent_seeds','validation_seeds','test_seeds')],[])
    seeds += [cfg[k] for k in ('coefficient_seed','training_shuffle_seed','shuffled_memory_seed','bootstrap_seed')]
    assert len(seeds)==len(set(seeds))
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    locked = hashes()
    write_json(output/'preregistration.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'phase':'before all registered calibration/training/validation/test generation',
        'config':cfg,'source_hashes':locked,'python':sys.version,'executable':sys.executable,
        'platform':platform.platform(),'offline_proof':environment})
    for relative in locked:
        target = output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    checkpoint = output/'checkpoints'
    checkpoint.mkdir()
    shutil.copyfile(ROOT/cfg['frozen_gate'],checkpoint/'frozen_gate.npz')
    gate = Gate.load(checkpoint/'frozen_gate.npz')
    frozen_gate = [a.copy() for a in (gate.weights,gate.mean,gate.scale)]
    print('Protocol/source locked before registered generation',flush=True)
    started = time.perf_counter()
    x,y = calibration_data(cfg)
    experts,expert_history = fit_experts(x,y,cfg)
    experts.save(checkpoint/'experts.npz')
    write_json(output/'calibration.json',{'examples':len(x),'learned_parameters':36,
        'data_sha256':digest_arrays(x,y),'loss_by_epoch':expert_history})
    routers,training = {},{}
    for seed in cfg['recurrent_seeds']:
        routers[seed],training[str(seed)] = train_router(experts,seed,cfg)
        routers[seed].save(checkpoint/f'recurrent_{seed}.npz')
        print('Recurrent candidate trained:',seed,'final loss',training[str(seed)]['loss_by_epoch'][-1],flush=True)
    write_json(output/'recurrent-training.json',training)
    selection = select(experts,gate,routers,cfg)
    selected = routers[selection['recurrent']['seed']]
    selected.save(checkpoint/'recurrent_selected.npz')
    selection.update(created_utc=datetime.now(timezone.utc).isoformat(),phase='locked before ANY held-out stream',
        experts_sha256=sha(checkpoint/'experts.npz'),selected_router_sha256=sha(checkpoint/'recurrent_selected.npz'))
    write_json(output/'selection.json',selection)
    selection_sha = sha(output/'selection.json')
    write_json(output/'test-generation.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'phase':'immediately before first held-out generation','selection_sha256':selection_sha})
    seconds_development = time.perf_counter()-started
    frozen_experts = experts.weights.copy()
    frozen_router = {k:v.copy() for k,v in selected.params.items()}
    print('Selection locked:',{k:selection[k] for k in ('memory','cache','recurrent')},flush=True)
    started = time.perf_counter()
    records = []
    for case in cfg['cases']:
        for seed in cfg['test_seeds']:
            rows = episode(seed,cfg['test_steps'],case,cfg)
            store = Memory(output/'feedback.sqlite') if case=='recurring' and seed==cfg['test_seeds'][0] else None
            try:
                evaluated,traces = evaluate(rows,experts,gate,selection,cfg,selected,
                    feedback=case!='no_feedback',persistent=store)
            finally:
                if store is not None:
                    store.close()
            for family in FAMILIES:
                records.append({'case':case,'seed':seed,'family':family,**evaluated[family]})
            # This equality is an intentional retrieval-disabled integrity control.
            assert traces['memory_disabled']==traces['recency']
            path = output/'traces'/case/(str(seed)+'.jsonl')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(row)+'\n' for row in pack(traces)),encoding='utf-8')
        print(case,'evaluated',flush=True)
    seconds_test = time.perf_counter()-started
    summary = summarize(records)
    paired = comparisons(records,cfg)
    scientific = gates(summary,paired,cfg)
    assert hashes()==locked and sha(output/'selection.json')==selection_sha
    assert np.array_equal(experts.weights,frozen_experts)
    assert all(np.array_equal(a,b) for a,b in zip(frozen_gate,(gate.weights,gate.mean,gate.scale)))
    assert all(np.array_equal(v,selected.params[k]) for k,v in frozen_router.items())
    result = {'config':cfg,'selection':selection,'selection_sha256':selection_sha,'runs':records,
        'summary':summary,'paired':paired,'gates':scientific,'source_hashes':locked,
        'offline_proof':environment,'preservation':check_preservation(),
        'seconds':{'development_and_selection':seconds_development,'test_and_trace_writes':seconds_test},
        'budgets':{'shared_expert_parameters':36,'recurrent_parameters':selected.parameter_count,
            'recurrent_hidden_state':cfg['recurrent_hidden'],'searchable_feedback_records':cfg['memory_capacity'],
            'cache_slots':cfg['cache_capacity'],'equal_compute_or_capacity':False}}
    write_json(output/'results.json',result)
    from .report import generate
    generate(output)
    write_json(output/'artifact-hashes.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps(scientific,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
