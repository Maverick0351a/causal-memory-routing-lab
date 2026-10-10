"""Freeze -> all-world fit/selection -> held-out evaluation -> exact replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
from archlab.controllers import load_model
from confidence_lab.policy import Gate
from confidence_lab.experiment import write_json
from confidence_lab.benchmark import digest_arrays
from recall_recovery.public import fit_experts
from sensor_lab.public import Ensemble
from forget_lab.world import calibration_data, episode
from forget_lab.training import train_router, clone_router
from forget_lab.memory import Memory
from release_tools.integrity import source_hashes, sha
from transfer_lab.experiment import offline_proof
from .policy import FAMILIES, Policy, public_config
from .benchmark import evaluate, select, summarize, comparisons, gates

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'experiments/two-speed-20261010/protocol.json'
PREDICTION_FIELDS = ('probability', 'gated_probability', 'alpha', 'route', 'retrieved_count',
                     'retrieved_max_step', 'selector_weights', 'scanned_records', 'stable_count', 'fast_count')


def config():
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def world_config(cfg, world):
    return {**cfg, **world}


def sources():
    before = source_hashes(ROOT)
    paths = [ROOT/'two_speed.py', *sorted((ROOT/'twospeed_lab').glob('*.py')),
             *sorted((ROOT/'twospeed_tests').glob('*.py')),
             *sorted(PROTOCOL.parent.glob('*.md')), PROTOCOL]
    return {**before, **{p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}


def arrays(rows, traces):
    result = {k: np.asarray([[r[k] if r[k] is not None else -1 for r in traces[f]] for f in FAMILIES])
              for k in PREDICTION_FIELDS if k not in ('route', 'selector_weights')}
    result['route'] = np.array([[r['route'] for r in traces[f]] for f in FAMILIES])
    result['selector_weights'] = np.array([[r['selector_weights']+[0.]*(3-len(r['selector_weights']))
                                          for r in traces[f]] for f in FAMILIES])
    result['label'] = np.array([r.label for r in rows], dtype=np.int8)
    for key in ('age', 'switched', 'returned', 'changed_entry', 'context_id', 'revision'):
        result[key] = np.array([getattr(r, key) for r in rows], dtype=np.int32)
    result['cue'] = np.array([r.observation.cue for r in rows])
    result['readings'] = np.array([r.observation.readings for r in rows])
    return result


def run(output):
    cfg = config()
    guard = offline_proof()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    locked = sources()
    write_json(output/'preregistration.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'phase': 'before ALL registered generation, training and selection', 'config': cfg,
        'source_hashes': locked, 'python': sys.version, 'platform': platform.platform(), 'offline_proof': guard})
    for name in locked:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    started = time.perf_counter()
    gate = Gate.load(ROOT/cfg['frozen_gate'])
    assert sha(ROOT/cfg['frozen_gate']) == cfg['frozen_gate_sha256']
    selections, fitted = {}, {}
    for world in cfg['worlds']:
        wc = world_config(cfg, world)
        checkpoint = output/'checkpoints'/world['id']
        checkpoint.mkdir(parents=True)
        x, y = calibration_data(wc)
        experts, history = fit_experts(x, y, wc)
        experts.save(checkpoint/'experts.npz')
        training = {'experts': {'loss_by_epoch': history, 'data_sha256': digest_arrays(x, y)}}
        routers = {}
        for seed in wc['recurrent_seeds']:
            routers[seed], training[str(seed)] = train_router(experts, seed, wc)
            routers[seed].save(checkpoint/f'recurrent_{seed}.npz')
        write_json(checkpoint/'training.json', training)
        selection = select(experts, gate, routers, wc)
        chosen = routers[selection['recurrent']['seed']]
        chosen.save(checkpoint/'recurrent_selected.npz')
        selections[world['id']] = selection
        fitted[world['id']] = experts, chosen
        print(world['id'], 'fit and validation selection locked', flush=True)
    write_json(output/'selection.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'phase': 'ALL world selections locked before ANY held-out generation', 'worlds': selections})
    selection_hash = sha(output/'selection.json')
    write_json(output/'test-generation.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'phase': 'immediately before first held-out stream', 'selection_sha256': selection_hash})
    fit_seconds = time.perf_counter()-started
    started = time.perf_counter()
    records = []
    for world in cfg['worlds']:
        wc = world_config(cfg, world)
        experts, router = fitted[world['id']]
        frozen = [a.copy() for a in (experts.weights, *router.params.values(), gate.weights, gate.mean, gate.scale)]
        for case in cfg['cases']:
            for seed in wc['test_seeds']:
                rows = episode(seed, cfg['test_steps'], case, wc)
                store = Memory(output/'feedback.sqlite') if world == cfg['worlds'][0] and case == 'repeated_changes' and seed == wc['test_seeds'][0] else None
                try:
                    scores, traces = evaluate(rows, experts, gate, selections[world['id']], wc, router,
                                              feedback=case != 'no_feedback', persistent=store)
                finally:
                    if store is not None:
                        store.close()
                if case == 'no_feedback':
                    for family in ('stable_only', 'fast_only', 'recency'):
                        assert [[r[k] for k in ('probability', 'gated_probability', 'route')] for r in traces['two_speed']] == [
                            [r[k] for k in ('probability', 'gated_probability', 'route')] for r in traces[family]]
                records.extend({'world': world['id'], 'case': case, 'seed': seed, 'family': f, **scores[f]} for f in FAMILIES)
                path = output/'traces'/case/f"{world['id']}_{seed}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays(rows, traces))
            print(world['id'], case, 'evaluated', flush=True)
        assert all(np.array_equal(a, b) for a, b in zip(frozen, (experts.weights, *router.params.values(), gate.weights, gate.mean, gate.scale)))
    summary = summarize(records, cfg)
    paired = comparisons(records, cfg)
    scientific = gates(summary, paired, cfg)
    assert sources() == locked and sha(output/'selection.json') == selection_hash
    result = {'config': cfg, 'families': FAMILIES, 'runs': records, 'summary': summary, 'paired': paired,
              'gates': scientific, 'selection_sha256': selection_hash,
              'seconds': {'fit_and_selection': fit_seconds, 'held_out': time.perf_counter()-started},
              'offline_proof': guard, 'interval_scope': 'paired bootstrap of six independent world means; eight dependent streams per world averaged first; not simultaneous intervals',
              'budgets': {'searched_feedback_records': cfg['memory_capacity'], 'stable_view_records': cfg['memory_capacity']-cfg['fast_capacity'],
                          'fast_view_records': cfg['fast_capacity'], 'stable_neighbors': cfg['neighbors'], 'fast_neighbors': cfg['fast_neighbors'],
                          'selector_cue_slots': cfg['selector_capacity'], 'cache_cue_slots': cfg['cache_capacity'],
                          'recurrent_parameters': fitted[cfg['worlds'][0]['id']][1].parameter_count,
                          'shared_expert_parameters': 36, 'audit_history': 'unbounded and excluded from bounded prediction search',
                          'equal_compute_or_total_capacity': False}}
    write_json(output/'results.json', result)
    report(output, result)
    write_json(output/'artifact-hashes.json', {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps(scientific, indent=2), flush=True)


def report(output, result):
    cfg = result['config']
    lines = ['# Two-speed memory: six independent synthetic worlds', '',
             '**Qualification: '+('PASS' if result['gates']['two_speed_qualified'] else 'FAIL')+'**.', '',
             'All thresholds were frozen before registered generation. Raw forecasts decide every scientific gate; the unchanged confidence companion is descriptive only.', '',
             '| Component | Result |', '|---|---|']
    lines += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in result['gates'].items() if k != 'components']
    lines += ['', 'Differences below are two-speed minus the comparator; lower is better. Intervals resample six independent world means, after averaging all eight stream seeds and relevant cases within each world. These are exploratory pointwise intervals with only six world clusters.', '',
              '| Endpoint | Comparator | Difference | 95% interval |', '|---|---|---:|---:|']
    for name, p in result['paired'].items():
        endpoint, right = name.split('/')
        if right in ('stable_only', 'fast_only', 'recency', 'context_cache', 'adaptive_window', 'recent_uniform', 'fading') and endpoint in ('stable_loss', 'stable_return_loss', 'changed_loss', 'change_delay', 'balanced_loss'):
            lines.append(f"| {endpoint} | {right} | {p['mean']:.6f} | [{p['ci95'][0]:.6f}, {p['ci95'][1]:.6f}] |")
    lines += ['', '## Calibration and controls', '', '| Policy | Random accuracy | Raw loss | Gated loss | Raw ECE | Raw confidently wrong |', '|---|---:|---:|---:|---:|---:|']
    for f in FAMILIES:
        r = result['summary']['random_labels'][f]
        lines.append(f"| {f} | {r['raw']['accuracy']:.4%} | {r['raw']['log_loss']:.6f} | {r['gated']['log_loss']:.6f} | {r['raw']['ece']:.6f} | {r['raw']['confidently_wrong_rate']:.4%} |")
    lines += ['', '## Boundaries and budgets', '',
              'The stable and fast stores are disjoint logical views of the same prior-feedback index: older 192 and latest 64 records. Prediction searches at most 256 rows; the persistent audit log and its RAM replay index retain full history. This is not a hard total-RAM cap. Stable routing uses up to 64 matching records; fast uses up to 16 with half-life two. Stable-only and fast-only intentionally have different branch capacities. Adaptive-window uses three candidate sizes (8/32/64 matching records) under the same 256-row search cap. The selectors store at most eight measured-cue prototypes with two/three EMA forecast losses. Context cache and recurrent budgets differ; equal compute is not established.', '',
              'The adaptive-window method is a prequential mixture over window sizes, not a reproduction of ADWIN. The dual-store direction is established by [SAM-kNN](https://www.ijcai.org/Proceedings/2017/0690.pdf); this study does not reproduce SAM-kNN or establish novelty.', '',
              'Policies receive only six measured cue values and eight readings, their own prior released feedback and frozen settings. No hidden context IDs, switch ages, generator seeds, current labels or future feedback enter prediction. SQLite reload is restricted to step < cutoff. Identity cue embedding is not semantic memory; the four logistic tasks are fitted and known. New coefficient worlds vary task geometry, not task family.', '',
              'No API keys, downloads, peptide weights, model inference services, QPU jobs or paid compute. The Python socket guard is not an OS sandbox. Earlier results and v0.1.0 remain frozen.', '',
              'See results.json for every metric, world, stream, candidate and gate; protocol and source hashes are in preregistration.json. Replay evidence is reported separately in verification.json once completed.']
    (Path(output)/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


def verify(output):
    output = Path(output)
    cfg = config()
    proof = offline_proof()
    registration = json.loads((output/'preregistration.json').read_text(encoding='utf-8'))
    assert cfg == registration['config'] and sources() == registration['source_hashes']
    expected = json.loads((output/'results.json').read_text(encoding='utf-8'))
    locked_selection = json.loads((output/'selection.json').read_text(encoding='utf-8'))
    assert sha(output/'selection.json') == expected['selection_sha256']
    for name, digest in json.loads((output/'artifact-hashes.json').read_text(encoding='utf-8')).items():
        assert sha(output/name) == digest, name
    gate = Gate.load(ROOT/cfg['frozen_gate'])
    records, count = [], 0
    for world in cfg['worlds']:
        wc = world_config(cfg, world)
        checkpoint = output/'checkpoints'/world['id']
        x, y = calibration_data(wc)
        experts, history = fit_experts(x, y, wc)
        saved = Ensemble.load(checkpoint/'experts.npz')
        assert np.array_equal(experts.weights, saved.weights)
        training = json.loads((checkpoint/'training.json').read_text(encoding='utf-8'))
        assert training['experts'] == {'loss_by_epoch': history, 'data_sha256': digest_arrays(x, y)}
        routers = {}
        for seed in wc['recurrent_seeds']:
            routers[seed], info = train_router(experts, seed, wc)
            assert info == training[str(seed)]
            old = load_model(checkpoint/f'recurrent_{seed}.npz')
            assert all(np.array_equal(v, old.params[k]) for k, v in routers[seed].params.items())
        selection = select(experts, gate, routers, wc)
        assert selection == locked_selection['worlds'][world['id']]
        router = routers[selection['recurrent']['seed']]
        for case in cfg['cases']:
            for seed in wc['test_seeds']:
                rows = episode(seed, cfg['test_steps'], case, wc)
                s, traces = evaluate(rows, experts, gate, selection, wc, router, feedback=case != 'no_feedback')
                current = arrays(rows, traces)
                with np.load(output/'traces'/case/f"{world['id']}_{seed}.npz", allow_pickle=False) as z:
                    assert set(z.files) == set(current)
                    assert all(np.array_equal(z[k], v) for k, v in current.items())
                records.extend({'world': world['id'], 'case': case, 'seed': seed, 'family': f, **s[f]} for f in FAMILIES)
                count += cfg['test_steps']*len(FAMILIES)
        print(world['id'], 'refit, reselection and every trace replay matched', flush=True)
    summary = summarize(records, cfg)
    paired = comparisons(records, cfg)
    assert records == expected['runs'] and summary == expected['summary'] and paired == expected['paired']
    assert gates(summary, paired, cfg) == expected['gates']
    # Reopen a completed log containing future rows, rebuild every policy using
    # only released history < cutoff, and compare the next committed prediction.
    wc = world_config(cfg, cfg['worlds'][0])
    selection = locked_selection['worlds'][cfg['worlds'][0]['id']]
    cp = output/'checkpoints'/cfg['worlds'][0]['id']
    experts, router = Ensemble.load(cp/'experts.npz'), load_model(cp/'recurrent_selected.npz')
    rows = episode(wc['test_seeds'][0], cfg['test_steps'], 'repeated_changes', wc)
    store = Memory(output/'feedback.sqlite')
    cutoff = cfg['resume_step']
    history = store.history('episode', before_step=cutoff)
    assert len(history) == cutoff and len(store.history('episode', before_step=cfg['test_steps'])) == cfg['test_steps']
    with np.load(output/'traces/repeated_changes'/f"{cfg['worlds'][0]['id']}_{wc['test_seeds'][0]}.npz", allow_pickle=False) as z:
        for i, f in enumerate(FAMILIES):
            policy = Policy(f, experts, gate, selection, public_config(wc), store, clone_router(router) if f == 'recurrent' else None)
            for _, obs, label, _ in history:
                policy.predict(obs)
                policy.observe(obs, label)
            pred, _ = policy.predict(rows[cutoff].observation)
            for key in PREDICTION_FIELDS:
                value = pred[key]
                if key == 'selector_weights':
                    value = value+[0.]*(3-len(value))
                elif value is None:
                    value = -1
                assert np.array_equal(z[key][i, cutoff], value), (f, key)
    store.close()
    receipt = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'expert_and_all_recurrent_refits_exact': True, 'all_validation_candidates_and_selections_exact': True,
               'all_trace_arrays_metrics_intervals_gates_exact': True, 'raw_predictions_replayed': count,
               'companion_predictions_replayed': count, 'policy_history_reloads_exact': len(FAMILIES),
               'resume_cutoff': cutoff, 'future_rows_excluded': cfg['test_steps']-cutoff,
               'offline_proof': proof, 'scientific_qualification': expected['gates']['two_speed_qualified']}
    write_json(output/'verification.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'verify'))
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    (run if a.action == 'run' else verify)(a.output)
