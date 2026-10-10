"""Causal evaluation, validation-only selection and paired scientific gates."""
import numpy as np
from sensor_lab.benchmark import metrics as base_metrics
from confidence_lab.benchmark import bootstrap,digest_arrays
from sensor_lab.public import Observation as SensorObservation
from .policy import Policy,FAMILIES
from .memory import Memory
from .training import clone_router
from .world import episode


def score(trace,field='probability'):
    return base_metrics([{**r,'probability':r[field]} for r in trace]) if trace else None


def response(trace,cfg):
    starts = [i for i,r in enumerate(trace) if r['switched']]
    penalties,recovered = [],0
    window,horizon = cfg['recovery_window'],cfg['recovery_horizon']
    for index,start in enumerate(starts):
        if not trace[start]['returned']:
            continue
        end = starts[index+1] if index+1<len(starts) else len(trace)
        if end-start<window:
            continue
        rows = trace[start:min(end,start+horizon)]
        values = np.array([-np.log(r['probability'] if r['label'] else 1-r['probability']) for r in rows])
        endpoint = next((i+window-1 for i in range(len(values)-window+1)
                         if values[i:i+window].mean()<=cfg['recovery_loss_threshold']),None)
        penalties.append(horizon if endpoint is None else endpoint)
        recovered += endpoint is not None
    return {'events':len(penalties),'recovered':recovered,
            'penalized_steps':float(np.mean(penalties)) if penalties else None,
            'recovery_fraction':recovered/len(penalties) if penalties else None}


def scores(trace,cfg):
    returned = [r for r in trace if r['returned'] and r['age']<cfg['return_window']]
    first = [r for r in trace if not r['returned'] and r['age']<cfg['return_window']]
    stale = [r for r in trace if r['mapping_changed'] and r['age']<cfg['return_window']]
    return {'raw':score(trace),'gated':score(trace,'gated_probability'),
            'return_first16':score(returned),'return_scored_steps':len(returned),
            'first_visit_first16':score(first),'stale_entry_first16':score(stale),
            'return_delay':response(trace,cfg),
            'mean_retrieved_count':float(np.mean([r['retrieved_count'] for r in trace]))}


def evaluate(rows,experts,gate,selection,cfg,router,*,feedback=True,persistent=None,families=FAMILIES):
    store = persistent if persistent is not None else Memory()
    policies = {f:Policy(f,experts,gate,selection,cfg,store,
                        clone_router(router) if f=='recurrent' else None) for f in families}
    trace = {f:[] for f in families}
    features = {f:[] for f in families}
    need_store = persistent is not None or any(f in ('memory','shuffled_memory') for f in families)
    try:
        for step,row in enumerate(rows):
            predictions = {}
            for family,policy in policies.items():
                prediction,x = policy.predict(row.observation)
                predictions[family] = prediction
                features[family].append(x)
            # Current answers and scoring metadata are read only AFTER all commits.
            for family,prediction in predictions.items():
                assert prediction['retrieved_max_step'] is None or prediction['retrieved_max_step']<step
                trace[family].append({'step':step,'cue':list(row.observation.cue),
                    'readings':list(row.observation.readings),**prediction,'label':row.label,
                    'context_id':row.context_id,'age':row.age,'switched':row.switched,
                    'returned':row.returned,'mapping_changed':row.mapping_changed})
            if feedback:
                for policy in policies.values():
                    policy.observe(row.observation,row.label)
                if need_store:
                    outputs = experts.predict(SensorObservation(row.observation.readings))
                    store.append('episode',step,row.observation,row.label,((outputs>=.5)!=row.label).astype(float))
            else:
                for policy in policies.values():
                    policy.discard()
        records = {f:{**scores(t,cfg),'gate_features_sha256':digest_arrays(features[f])} for f,t in trace.items()}
    finally:
        if persistent is None:
            store.close()
    return records,trace


def initial_selection(cfg):
    return {'memory':{'capacity':cfg['memory_capacity'],'neighbors':cfg['memory_neighbors'][0],
            'radius':cfg['memory_radii'][0],'blend':cfg['memory_blends'][0]},
            'cache':{'capacity':cfg['cache_capacity'],'radius':cfg['cache_radii'][0],
            'rate':cfg['cache_rates'][0],'blend':cfg['cache_blend']},
            'recurrent':{'seed':cfg['recurrent_seeds'][0],'temperature':1.}}


def select(experts,gate,routers,cfg):
    settings = initial_selection(cfg)
    worlds = [episode(s,cfg['validation_steps'],c,cfg) for c in cfg['selection_cases'] for s in cfg['validation_seeds']]
    grids = {'memory':[],'cache':[],'recurrent':[]}
    def loss(family,selection,router):
        return float(np.mean([evaluate(rows,experts,gate,selection,cfg,router,families=(family,))[0][family]['raw']['log_loss'] for rows in worlds]))
    for neighbors in cfg['memory_neighbors']:
        for radius in cfg['memory_radii']:
            for blend in cfg['memory_blends']:
                setting = {'capacity':cfg['memory_capacity'],'neighbors':neighbors,'radius':radius,'blend':blend}
                grids['memory'].append({**setting,'validation_log_loss':loss('memory',{**settings,'memory':setting},None)})
    print('Memory validation candidates complete',flush=True)
    for radius in cfg['cache_radii']:
        for rate in cfg['cache_rates']:
            setting = {'capacity':cfg['cache_capacity'],'radius':radius,'rate':rate,'blend':cfg['cache_blend']}
            grids['cache'].append({**setting,'validation_log_loss':loss('context_cache',{**settings,'cache':setting},None)})
    print('Context-cache validation candidates complete',flush=True)
    for seed in cfg['recurrent_seeds']:
        for temperature in cfg['recurrent_temperatures']:
            setting = {'seed':seed,'temperature':temperature}
            grids['recurrent'].append({**setting,'validation_log_loss':loss('recurrent',{**settings,'recurrent':setting},routers[seed])})
    print('Recurrent validation candidates complete',flush=True)
    selected = {k:min(range(len(grid)),key=lambda i:grid[i]['validation_log_loss']) for k,grid in grids.items()}
    result = {k:{a:b for a,b in grids[k][i].items() if a!='validation_log_loss'} for k,i in selected.items()}
    result.update(candidate_grids=grids,selected_indices=selected,
                  validation_examples=len(worlds)*cfg['validation_steps'],selection_cases=cfg['selection_cases'])
    # Digest public observations and labels for selection provenance only.
    result['validation_data_sha256'] = digest_arrays([[*r.observation.cue,*r.observation.readings,r.label] for rows in worlds for r in rows])
    return result


def mean_dict(values):
    return {k:float(np.mean([v[k] for v in values if v[k] is not None]))
            if any(v[k] is not None for v in values) else None for k in values[0]}


def summarize(records):
    output = {}
    for case in sorted({r['case'] for r in records}):
        output[case] = {}
        for family in FAMILIES:
            rows = [r for r in records if r['case']==case and r['family']==family]
            result = {}
            for metric in ('raw','gated','return_first16','first_visit_first16','stale_entry_first16','return_delay'):
                values = [r[metric] for r in rows if r[metric] is not None]
                result[metric] = mean_dict(values) if values else None
            result['return_scored_steps'] = sum(r['return_scored_steps'] for r in rows)
            result['mean_retrieved_count'] = float(np.mean([r['mean_retrieved_count'] for r in rows]))
            output[case][family] = result
    return output


def comparisons(records,cfg):
    index = {(r['case'],r['seed'],r['family']):r for r in records}
    output = {}
    for right in ('memory_disabled','shuffled_memory','recency','recurrent','context_cache'):
        for group,key in (('raw','log_loss'),('gated','log_loss'),('return_first16','log_loss'),('return_delay','penalized_steps')):
            parts = {}
            for case in cfg['cases']:
                a = [index[case,s,'memory'][group] for s in cfg['test_seeds']]
                b = [index[case,s,right][group] for s in cfg['test_seeds']]
                if any(v is None or v[key] is None for v in a+b):
                    continue
                differences = [x[key]-y[key] for x,y in zip(a,b)]
                parts[case] = differences
                output[f'{case}/{group}_{key}/memory_minus_{right}'] = bootstrap(differences,cfg)
            aggregate = np.array([parts[c] for c in cfg['primary_cases']]).mean(axis=0)
            output[f'aggregate/{group}_{key}/memory_minus_{right}'] = bootstrap(aggregate,cfg)
    return output


def gates(summary,paired,cfg):
    recall,speed,quality = {},{},{}
    for right in ('memory_disabled','shuffled_memory','recency','recurrent'):
        p = paired[f'aggregate/return_first16_log_loss/memory_minus_{right}']
        recall[right+'_improvement'] = p['mean']<=-cfg['minimum_return_log_loss_improvement']
        recall[right+'_upper_below_zero'] = p['ci95'][1]<0
    for right in ('recency','recurrent'):
        p = paired[f'aggregate/return_delay_penalized_steps/memory_minus_{right}']
        speed[right+'_two_steps_faster'] = p['mean']<=-cfg['minimum_delay_reduction']
        speed[right+'_upper_below_zero'] = p['ci95'][1]<0
    for case in cfg['quality_cases']:
        a,b = summary[case]['memory']['raw'],summary[case]['recency']['raw']
        tolerance = cfg['maximum_stale_loss_degradation'] if case=='mapping_change' else cfg['maximum_quality_loss_degradation']
        quality[case+'_loss_preserved'] = a['log_loss']-b['log_loss']<=tolerance
        quality[case+'_brier_preserved'] = a['brier']-b['brier']<=cfg['maximum_quality_brier_degradation']
        quality[case+'_confident_errors_preserved'] = a['confidently_wrong_rate']-b['confidently_wrong_rate']<=cfg['maximum_quality_confident_error_excess']
    for family in FAMILIES:
        quality[family+'_random_accuracy_near_chance'] = abs(summary['random_labels'][family]['raw']['accuracy']-.5)<=cfg['chance_tolerance']
    p = paired['aggregate/raw_log_loss/memory_minus_context_cache']
    value = p['mean']<=-cfg['minimum_cache_log_loss_improvement'] and p['ci95'][1]<0
    return {'functional_recall':all(recall.values()),'faster_return_adaptation':all(speed.values()),
            'quality_preserved':all(quality.values()),
            'memory_qualified':all(recall.values()) and all(speed.values()) and all(quality.values()),
            'extra_value_over_context_cache':value,
            'gated_random_loss_control':summary['random_labels']['memory']['gated']['log_loss']-np.log(2)<=cfg['maximum_random_loss_above_uniform'],
            'components':{'recall':recall,'speed':speed,'quality':quality}}
