"""Paired worlds, validation-only selection and world-clustered intervals."""
import numpy as np
from confidence_lab.benchmark import digest_arrays
from forget_lab.world import episode
from forget_lab.benchmark import scores as prior_scores
from forget_lab.memory import Memory
from forget_lab.training import clone_router
from sensor_lab.public import Observation as SensorObservation
from recall_recovery.benchmark import mean_dict
from .policy import Policy, FAMILIES, public_config


def evaluate(rows, experts, gate, selection, cfg, router=None, *, families=FAMILIES, feedback=True, persistent=None):
    store = persistent if persistent is not None else Memory()
    policies = {f: Policy(f, experts, gate, selection, public_config(cfg), store,
                         clone_router(router) if f == 'recurrent' else None) for f in families}
    traces = {f: [] for f in families}
    try:
        for step, row in enumerate(rows):
            observations = row.observation
            predictions = {f: p.predict(observations)[0] for f, p in policies.items()}
            # All forecasts committed before labels/private scoring annotations.
            for f, pred in predictions.items():
                assert pred['retrieved_max_step'] is None or pred['retrieved_max_step'] < step
                traces[f].append({'step': step, **pred, 'label': row.label, 'age': row.age,
                    'switched': row.switched, 'returned': row.returned, 'changed_entry': row.changed_entry,
                    'effective_neighbors': float(pred['retrieved_count'])})
            if feedback:
                for p in policies.values():
                    p.observe(observations, row.label)
                outputs = experts.predict(SensorObservation(observations.readings))
                store.append('episode', step, observations, row.label, ((outputs >= .5) != row.label).astype(float))
            else:
                for p in policies.values():
                    p.discard()
        records = {f: prior_scores(t, cfg) for f, t in traces.items()}
        for f, p in policies.items():
            # Prior scorer's count-based placeholder is not a weighted effective N.
            records[f].pop('mean_effective_neighbors')
            records[f]['selector_slots_used'] = len(p.selector.entries)
            records[f]['maximum_scanned_records'] = max(r['scanned_records'] for r in traces[f])
    finally:
        if persistent is None:
            store.close()
    return records, traces


def initial_selection(cfg):
    setting = {'rate': cfg['selector_rates'][0], 'temperature': cfg['selector_temperatures'][0]}
    return {'two_speed': dict(setting), 'adaptive_window': dict(setting),
            'fading': {'half_life': cfg['fading_half_lives'][0]},
            'cache': {'capacity': cfg['cache_capacity'], 'radius': cfg['retrieval_radius'],
                      'rate': cfg['cache_rates'][0], 'blend': 1.},
            'recurrent': {'seed': cfg['recurrent_seeds'][0], 'temperature': 1.}}


def select(experts, gate, routers, cfg):
    setting = initial_selection(cfg)
    worlds = {(c, s): episode(s, cfg['validation_steps'], c, cfg)
              for c in cfg['selection_cases'] for s in cfg['validation_seeds']}
    grids = {k: [] for k in ('two_speed', 'adaptive_window', 'fading', 'cache', 'recurrent')}
    def candidate(key, family, choice, router=None):
        stable, changed = [], []
        for (case, seed), rows in worlds.items():
            s = evaluate(rows, experts, gate, {**setting, key: choice}, cfg, router, families=(family,))[0][family]
            (stable if case in cfg['stable_cases'] else changed).append(
                s['return_first16' if case in cfg['stable_cases'] else 'changed_entry_first16']['log_loss'])
        grids[key].append({**choice, 'validation_objective': float(.5*(np.mean(stable)+np.mean(changed))),
                           'validation_stable_loss': float(np.mean(stable)), 'validation_changed_loss': float(np.mean(changed))})
    for key in ('two_speed', 'adaptive_window'):
        for rate in cfg['selector_rates']:
            for temp in cfg['selector_temperatures']:
                candidate(key, key, {'rate': rate, 'temperature': temp})
    for half in cfg['fading_half_lives']:
        candidate('fading', 'fading', {'half_life': half})
    for rate in cfg['cache_rates']:
        candidate('cache', 'context_cache', {**setting['cache'], 'rate': rate})
    for seed, router in routers.items():
        for temp in cfg['recurrent_temperatures']:
            candidate('recurrent', 'recurrent', {'seed': seed, 'temperature': temp}, router)
    indices = {k: min(range(len(v)), key=lambda i: v[i]['validation_objective']) for k, v in grids.items()}
    selected = {k: {a: b for a, b in grids[k][i].items() if not a.startswith('validation_')} for k, i in indices.items()}
    return {**selected, 'candidate_grids': grids, 'selected_indices': indices,
            'validation_data_sha256': digest_arrays([[*r.observation.cue, *r.observation.readings, r.label]
                                                     for rows in worlds.values() for r in rows])}


def summarize(records, cfg):
    result = {}
    for case in cfg['cases']:
        result[case] = {}
        for family in FAMILIES:
            rows = [r for r in records if r['case'] == case and r['family'] == family]
            result[case][family] = {key: mean_dict([r[key] for r in rows if r[key] is not None])
                                   if any(r[key] is not None for r in rows) else None
                                   for key in ('raw', 'gated', 'return_first16', 'changed_entry_first16', 'changed_entry_first8', 'change_delay')}
    return result


def interval(values, cfg):
    """Resample independent world means, never individual prediction rows."""
    values = np.asarray(values)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws = values[rng.integers(len(values), size=(cfg['bootstrap_replicates'], len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'world_means': values.tolist(), 'independent_worlds': len(values)}


def comparisons(records, cfg):
    index = {(r['world'], r['case'], r['seed'], r['family']): r for r in records}
    endpoints = {'stable_loss': (cfg['stable_cases'], 'raw', 'log_loss'),
                 'stable_return_loss': (cfg['stable_cases'], 'return_first16', 'log_loss'),
                 'changed_loss': (cfg['primary_cases'], 'changed_entry_first16', 'log_loss'),
                 'changed_brier': (cfg['primary_cases'], 'changed_entry_first16', 'brier'),
                 'change_delay': (cfg['primary_cases'], 'change_delay', 'penalized_steps'),
                 'quality_brier': (cfg['selection_cases'], 'raw', 'brier'),
                 'quality_ece': (cfg['selection_cases'], 'raw', 'ece'),
                 'quality_confident_errors': (cfg['selection_cases'], 'raw', 'confidently_wrong_rate')}
    output = {}
    for right in FAMILIES[1:]:
        for name, (cases, metric, key) in endpoints.items():
            means = []
            for world in cfg['worlds']:
                means.append(np.mean([index[world['id'], c, s, 'two_speed'][metric][key]-index[world['id'], c, s, right][metric][key]
                                      for c in cases for s in world['test_seeds']]))
            output[name+'/'+right] = interval(means, cfg)
        a, b = output['stable_return_loss/'+right], output['changed_loss/'+right]
        output['balanced_loss/'+right] = interval(.5*(np.array(a['world_means'])+np.array(b['world_means'])), cfg)
    return output


def gates(summary, paired, cfg):
    t = cfg['thresholds']
    components = {'adaptation': {}, 'retention': {}, 'quality': {}, 'incremental_value': {}, 'causal_controls': {}}
    def check(group, endpoint, right, tolerance):
        p = paired[endpoint+'/'+right]
        components[group][endpoint+'_vs_'+right] = bool(p['mean'] <= tolerance and p['ci95'][1] <= tolerance)
    check('adaptation', 'changed_loss', 'stable_only', -t['changed_gain'])
    check('adaptation', 'change_delay', 'stable_only', -t['delay_gain'])
    for right in ('recency', 'context_cache', 'adaptive_window', 'fading'):
        check('adaptation', 'changed_loss', right, t['changed_noninferiority'])
        check('adaptation', 'change_delay', right, t['delay_noninferiority'])
    for right in ('stable_only', 'recent_uniform', 'context_cache'):
        for endpoint in ('stable_loss', 'stable_return_loss'):
            check('retention', endpoint, right, t['stable_noninferiority'])
    for right in ('recency', 'context_cache', 'adaptive_window'):
        for endpoint, tolerance in (('quality_brier', t['brier_noninferiority']),
                                    ('quality_ece', t['ece_noninferiority']),
                                    ('quality_confident_errors', t['confident_error_noninferiority'])):
            check('quality', endpoint, right, tolerance)
    check('incremental_value', 'balanced_loss', 'adaptive_window', -t['balanced_gain'])
    check('causal_controls', 'balanced_loss', 'shuffled_two_speed', -t['shuffle_gain'])
    for family in FAMILIES:
        components['causal_controls']['chance_'+family] = bool(abs(summary['random_labels'][family]['raw']['accuracy']-.5) <= t['chance_tolerance'])
    result = {k: all(v.values()) for k, v in components.items()}
    result['two_speed_qualified'] = all(result.values())
    result['components'] = components
    return result
