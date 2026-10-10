"""Freeze sources; fit/select on development splits; then evaluate held-out data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
from confidence_lab.policy import Gate
from confidence_lab.benchmark import digest_arrays
from confidence_lab.experiment import sha,write_json
from transfer_lab.experiment import offline_proof
from .public import fit_ensemble
from .world import training_data,episode
from .memory import Memory
from .policy import FAMILIES
from .benchmark import select,evaluate,average,comparisons,scientific_gates

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'experiments/sensor-transfer-20261010'


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
    keys = ('probability','base_probability','alpha','slope','intercept')
    output = []
    for step,base_row in enumerate(traces['base']):
        row = {k:v for k,v in base_row.items() if k not in keys}
        row['predictions'] = {f:{k:traces[f][step][k] for k in keys} for f in FAMILIES}
        output.append(row)
    return output


def run(output):
    cfg = config()
    environment = offline_proof()
    check_preservation()
    all_seeds = sum([cfg[k] for k in ('training_seeds','validation_seeds','test_seeds','model_seeds')],[])
    if len(all_seeds)!=len(set(all_seeds)):
        raise RuntimeError('Overlapping seeds')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    locked = hashes()
    write_json(output/'preregistration.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'phase':'before training, validation or held-out generation','config':cfg,'source_hashes':locked,
        'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'offline_proof':environment})
    for relative in locked:
        target = output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    (output/'checkpoints').mkdir()
    shutil.copyfile(ROOT/cfg['frozen_gate'],output/'checkpoints/frozen_gate.npz')
    gate = Gate.load(output/'checkpoints/frozen_gate.npz')
    frozen_gate = [a.copy() for a in (gate.weights,gate.mean,gate.scale)]
    print('Protocol and source locked before development generation',flush=True)
    started = time.perf_counter()
    x,y = training_data(cfg)
    ensemble,history = fit_ensemble(x,y,cfg)
    ensemble.save(output/'checkpoints/ensemble.npz')
    write_json(output/'training.json',{'examples':len(y),'learned_parameters':36,
        'data_sha256':digest_arrays(x,y),'bootstrap_seeds':cfg['model_seeds'],'losses_by_epoch':history})
    selection = select(ensemble,cfg)
    selection.update(created_utc=datetime.now(timezone.utc).isoformat(),
        phase='locked before ANY held-out generation',ensemble_sha256=sha(output/'checkpoints/ensemble.npz'))
    write_json(output/'selection.json',selection)
    selection_sha = sha(output/'selection.json')
    seconds_development = time.perf_counter()-started
    print('Selection locked:',selection['online'],'fixed slope/intercept',selection['fixed_weights'],flush=True)
    print('Generating held-out streams only now',flush=True)
    write_json(output/'test-generation.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'phase':'immediately before first held-out generation','selection_sha256':selection_sha})
    frozen_predictor = ensemble.weights.copy()
    started = time.perf_counter()
    records = []
    store = Memory(output/'feedback.sqlite')
    try:
        for case in cfg['cases']:
            for seed in cfg['test_seeds']:
                rows = episode(seed,cfg['test_steps'],case,cfg)
                persist = store if case=='mixed' and seed==cfg['test_seeds'][0] else None
                evaluated,traces = evaluate(rows,ensemble,gate,selection,cfg,
                    feedback=case!=cfg['no_feedback_case'],persistent=persist,namespace=f'{case}/{seed}')
                for family in FAMILIES:
                    records.append({'case':case,'seed':seed,'family':family,**evaluated[family]})
                path = output/'traces'/case/(str(seed)+'.jsonl')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(''.join(json.dumps(r)+'\n' for r in pack(traces)),encoding='utf-8')
            print(case,'evaluated',flush=True)
    finally:
        store.close()
    seconds_test = time.perf_counter()-started
    summaries = average(records)
    paired = comparisons(records,cfg)
    gates = scientific_gates(summaries,paired,cfg)
    if hashes()!=locked or sha(output/'selection.json')!=selection_sha:
        raise RuntimeError('Locked source or selection changed')
    assert np.array_equal(frozen_predictor,ensemble.weights)
    assert all(np.array_equal(a,b) for a,b in zip(frozen_gate,(gate.weights,gate.mean,gate.scale)))
    result = {'config':cfg,'selection':selection,'selection_sha256':selection_sha,'runs':records,
        'summary':summaries,'paired':paired,'gates':gates,'source_hashes':locked,
        'offline_proof':environment,'preservation':check_preservation(),
        'seconds':{'fit_and_selection':seconds_development,'test_and_trace_writes':seconds_test}}
    write_json(output/'results.json',result)
    from .report import generate
    generate(output)
    write_json(output/'artifact-hashes.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps(gates,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
