"""Validation selection and first changed-association visit endpoints."""
import numpy as np
from sensor_lab.benchmark import metrics
from confidence_lab.benchmark import bootstrap,digest_arrays
from sensor_lab.public import Observation as SensorObservation
from recall_recovery.benchmark import initial_selection as original_selection,mean_dict
from .policy import Policy,FAMILIES,NEW_FAMILIES
from .memory import Memory
from .training import clone_router
from .world import episode


def score(trace,field='probability'):
    return metrics([{**r,'probability':r[field]} for r in trace]) if trace else None


def response(trace,cfg):
    starts = [i for i,r in enumerate(trace) if r['switched']]
    penalties,recovered = [],0
    window,horizon = cfg['recovery_window'],cfg['recovery_horizon']
    for index,start in enumerate(starts):
        if not trace[start]['changed_entry']:
            continue
        end = starts[index+1] if index+1<len(starts) else len(trace)
        if end-start<window:
            continue
        rows = trace[start:min(end,start+horizon)]
        loss = np.array([-np.log(r['probability'] if r['label'] else 1-r['probability']) for r in rows])
        endpoint = next((i+window-1 for i in range(len(loss)-window+1)
            if loss[i:i+window].mean()<=cfg['recovery_loss_threshold']),None)
        penalties.append(horizon if endpoint is None else endpoint)
        recovered += endpoint is not None
    return {'events':len(penalties),'recovered':recovered,
        'penalized_steps':float(np.mean(penalties)) if penalties else None,
        'recovery_fraction':recovered/len(penalties) if penalties else None}


def scores(trace,cfg):
    stable = [r for r in trace if r['returned'] and not r['changed_entry'] and r['age']<cfg['return_window']]
    changed = [r for r in trace if r['changed_entry'] and r['age']<cfg['return_window']]
    early = [r for r in changed if r['age']<8]
    return {'raw':score(trace),'gated':score(trace,'gated_probability'),
        'return_first16':score(stable),'changed_entry_first16':score(changed),'changed_entry_first8':score(early),
        'return_scored_steps':len(stable),'changed_scored_steps':len(changed),'change_delay':response(trace,cfg),
        'mean_retrieved_count':float(np.mean([r['retrieved_count'] for r in trace])),
        'mean_effective_neighbors':float(np.mean([r['effective_neighbors'] for r in trace]))}


def evaluate(rows,experts,gate,selection,cfg,router,*,feedback=True,persistent=None,families=FAMILIES):
    store = persistent if persistent is not None else Memory()
    policies = {f:Policy(f,experts,gate,selection,cfg,store,
        clone_router(router) if f=='recurrent' else None) for f in families}
    traces,features = {f:[] for f in families},{f:[] for f in families}
    need_store = persistent is not None or any(f in (*NEW_FAMILIES,'memory','shuffled_memory') for f in families)
    try:
        for step,row in enumerate(rows):
            predictions = {}
            for family,policy in policies.items():
                predictions[family],x = policy.predict(row.observation)
                features[family].append(x)
            # Current label and every private scoring field are read after all commits.
            for family,prediction in predictions.items():
                assert prediction['retrieved_max_step'] is None or prediction['retrieved_max_step']<step
                traces[family].append({'step':step,'cue':list(row.observation.cue),'readings':list(row.observation.readings),
                    **prediction,'label':row.label,'context_id':row.context_id,'age':row.age,'switched':row.switched,
                    'returned':row.returned,'mapping_changed':row.mapping_changed,'revision':row.revision,
                    'changed_entry':row.changed_entry})
            if feedback:
                for policy in policies.values():
                    policy.observe(row.observation,row.label)
                if need_store:
                    outputs = experts.predict(SensorObservation(row.observation.readings))
                    store.append('episode',step,row.observation,row.label,((outputs>=.5)!=row.label).astype(float))
            else:
                for policy in policies.values():
                    policy.discard()
        records = {f:{**scores(t,cfg),'gate_features_sha256':digest_arrays(features[f])} for f,t in traces.items()}
    finally:
        if persistent is None:
            store.close()
    return records,traces


def initial_selection(cfg):
    return {**original_selection(cfg),'forgetting':{'capacity':cfg['memory_capacity'],'neighbors':cfg['forget_neighbors'],
        'radius':cfg['forget_radius'],'blend':cfg['forget_blend'],'half_life':cfg['fading_half_lives'][0]}}


def select(experts,gate,routers,cfg):
    settings = initial_selection(cfg)
    worlds = {(c,s):episode(s,cfg['validation_steps'],c,cfg) for c in cfg['selection_cases'] for s in cfg['validation_seeds']}
    grids = {k:[] for k in ('memory','cache','recurrent','forgetting')}
    def objective(family,selection,router):
        stable,changed = [],[]
        for (case,seed),rows in worlds.items():
            s = evaluate(rows,experts,gate,selection,cfg,router,families=(family,))[0][family]
            (stable if case in cfg['stable_cases'] else changed).append(
                s['return_first16' if case in cfg['stable_cases'] else 'changed_entry_first16']['log_loss'])
        w = cfg['selection_stable_weight']
        return {'validation_objective':float(w*np.mean(stable)+(1-w)*np.mean(changed)),
            'validation_stable_entry_loss':float(np.mean(stable)),'validation_changed_entry_loss':float(np.mean(changed))}
    for neighbors in cfg['memory_neighbors']:
        for radius in cfg['memory_radii']:
            for blend in cfg['memory_blends']:
                setting = {'capacity':cfg['memory_capacity'],'neighbors':neighbors,'radius':radius,'blend':blend}
                grids['memory'].append({**setting,**objective('memory',{**settings,'memory':setting},None)})
    print('Original-memory validation complete',flush=True)
    for radius in cfg['cache_radii']:
        for rate in cfg['cache_rates']:
            setting = {'capacity':cfg['cache_capacity'],'radius':radius,'rate':rate,'blend':cfg['cache_blend']}
            grids['cache'].append({**setting,**objective('context_cache',{**settings,'cache':setting},None)})
    print('Cache validation complete',flush=True)
    for seed in cfg['recurrent_seeds']:
        for temperature in cfg['recurrent_temperatures']:
            setting = {'seed':seed,'temperature':temperature}
            grids['recurrent'].append({**setting,**objective('recurrent',{**settings,'recurrent':setting},routers[seed])})
    print('Recurrent validation complete',flush=True)
    for half_life in cfg['fading_half_lives']:
        setting = {**settings['forgetting'],'half_life':half_life}
        grids['forgetting'].append({**setting,**objective('forgetting',{**settings,'forgetting':setting},None)})
    print('Fading-memory validation complete',flush=True)
    selected = {k:min(range(len(grid)),key=lambda i:grid[i]['validation_objective']) for k,grid in grids.items()}
    result = {k:{a:b for a,b in grids[k][i].items() if not a.startswith('validation_')} for k,i in selected.items()}
    result.update(candidate_grids=grids,selected_indices=selected,validation_examples=len(worlds)*cfg['validation_steps'],
        selection_cases=cfg['selection_cases'],selection_stable_weight=cfg['selection_stable_weight'])
    result['validation_data_sha256'] = digest_arrays([[*r.observation.cue,*r.observation.readings,r.label] for rows in worlds.values() for r in rows])
    return result


METRICS = ('raw','gated','return_first16','changed_entry_first16','changed_entry_first8','change_delay')
COMPARATORS = ('memory','recent_uniform','shuffled_forgetting','recency','context_cache','recurrent')
ENDPOINTS = (('raw','log_loss'),('raw','brier'),('raw','confidently_wrong_rate'),
    ('return_first16','log_loss'),('changed_entry_first16','log_loss'),('changed_entry_first16','brier'),
    ('changed_entry_first16','confidently_wrong_rate'),('changed_entry_first8','log_loss'),('change_delay','penalized_steps'))


def summarize(records):
    result = {}
    for case in sorted({r['case'] for r in records}):
        result[case] = {}
        for family in FAMILIES:
            rows = [r for r in records if r['case']==case and r['family']==family]
            s = {}
            for metric in METRICS:
                values = [r[metric] for r in rows if r[metric] is not None]
                s[metric] = mean_dict(values) if values else None
            for key in ('return_scored_steps','changed_scored_steps'):
                s[key] = sum(r[key] for r in rows)
            for key in ('mean_retrieved_count','mean_effective_neighbors'):
                s[key] = float(np.mean([r[key] for r in rows]))
            result[case][family] = s
    return result


def comparisons(records,cfg):
    index = {(r['case'],r['seed'],r['family']):r for r in records}
    output = {}
    for right in COMPARATORS:
        for metric,key in ENDPOINTS:
            parts = {}
            for case in cfg['cases']:
                a = [index[case,s,'forgetting'][metric] for s in cfg['test_seeds']]
                b = [index[case,s,right][metric] for s in cfg['test_seeds']]
                if any(v is None or v[key] is None for v in a+b):
                    continue
                parts[case] = [x[key]-y[key] for x,y in zip(a,b)]
                output[f'{case}/{metric}_{key}/forgetting_minus_{right}'] = bootstrap(parts[case],cfg)
            for group,cases in (('aggregate_change',cfg['primary_cases']),('aggregate_stable',cfg['stable_cases'])):
                if all(c in parts for c in cases):
                    differences = np.array([parts[c] for c in cases]).mean(axis=0)
                    output[f'{group}/{metric}_{key}/forgetting_minus_{right}'] = bootstrap(differences,cfg)
    return output


def gates(summary,paired,cfg):
    value,speed,retention,quality,ablation = {},{},{},{},{}
    def pair(group,metric,right):
        return paired[f'{group}/{metric}/forgetting_minus_{right}']
    for right,minimum in cfg['minimum_entry_improvements'].items():
        p = pair('aggregate_change','changed_entry_first16_log_loss',right)
        value[right+'_minimum_gain'] = p['mean']<=-minimum
        value[right+'_upper_below_zero'] = p['ci95'][1]<0
    for case in cfg['primary_cases']:
        for right in ('recency','context_cache'):
            p = pair(case,'changed_entry_first16_log_loss',right)
            value[case+'_'+right+'_no_hidden_case_harm'] = p['mean']<=cfg['maximum_stable_loss_degradation'] and p['ci95'][1]<=cfg['maximum_stable_loss_degradation']
            for metric,tolerance in (('brier',cfg['maximum_entry_brier_excess']),('confidently_wrong_rate',cfg['maximum_confident_error_excess'])):
                p = pair(case,'changed_entry_first16_'+metric,right)
                quality[case+'_'+right+'_'+metric] = p['mean']<=tolerance and p['ci95'][1]<=tolerance
    p = pair('aggregate_change','change_delay_penalized_steps','memory')
    speed['memory_minimum_two_steps'] = p['mean']<=-cfg['minimum_change_delay_reduction']
    speed['memory_upper_below_zero'] = p['ci95'][1]<0
    for right in ('recency','context_cache'):
        p = pair('aggregate_change','change_delay_penalized_steps',right)
        speed[right+'_not_slower'] = p['mean']<=cfg['maximum_delay_excess_over_fast_baselines'] and p['ci95'][1]<=cfg['maximum_delay_excess_over_fast_baselines']
    for case in cfg['stable_cases']:
        for endpoint in ('raw_log_loss','return_first16_log_loss'):
            p = pair(case,endpoint,'memory')
            retention[case+'_'+endpoint] = p['mean']<=cfg['maximum_stable_loss_degradation'] and p['ci95'][1]<=cfg['maximum_stable_loss_degradation']
        p = pair(case,'raw_confidently_wrong_rate','memory')
        retention[case+'_confident_errors'] = p['mean']<=cfg['maximum_confident_error_excess'] and p['ci95'][1]<=cfg['maximum_confident_error_excess']
    for right,minimum in (('recent_uniform',cfg['minimum_fading_contribution']),('shuffled_forgetting',cfg['minimum_shuffle_improvement'])):
        p = pair('aggregate_change','changed_entry_first16_log_loss',right)
        ablation[right+'_minimum_gain'] = p['mean']<=-minimum
        ablation[right+'_upper_below_zero'] = p['ci95'][1]<0
    for family in FAMILIES:
        quality[family+'_random_accuracy'] = abs(summary['random_labels'][family]['raw']['accuracy']-.5)<=cfg['chance_tolerance']
    p = pair('random_labels','raw_log_loss','recency')
    quality['random_loss_vs_recency'] = p['mean']<=cfg['maximum_random_loss_above_recency'] and p['ci95'][1]<=cfg['maximum_random_loss_above_recency']
    groups = {'entry_value':value,'speed':speed,'stable_retention':retention,'quality':quality,'fading_ablation':ablation}
    result = {k:all(checks.values()) for k,checks in groups.items()}
    result['forgetting_qualified'] = all(result.values())
    result['gated_random_loss_control'] = bool(summary['random_labels']['forgetting']['gated']['log_loss']<=float(np.log(2))+cfg['maximum_random_loss_above_uniform'])
    result['components'] = groups
    return result
