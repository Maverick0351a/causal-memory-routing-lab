"""Offline selection, predict-before-feedback evaluation and paired inference."""
import numpy as np
from archlab.controllers import Adam
from confidence_lab.benchmark import bootstrap, digest_arrays, losses
from .public import Adapter, logistic_objective
from .policy import FAMILIES, Policy, OnlineCalibration, logit
from .world import episode


def collect_validation(ensemble,cfg):
    streams = []
    for case in cfg['selection_cases']:
        for seed in cfg['validation_seeds']:
            adapter = Adapter(ensemble)
            bases,labels,features = [],[],[]
            for row in episode(seed,cfg['validation_steps'],case,cfg):
                base,x = adapter.predict(row.observation)
                bases.append(base)
                features.append(x)
                labels.append(row.label)
                adapter.observe(row.observation,row.label)
            streams.append((np.array(bases),np.array(labels),np.array(features)))
    return streams


def select(ensemble,cfg):
    streams = collect_validation(ensemble,cfg)
    bases = np.concatenate([s[0] for s in streams])
    labels = np.concatenate([s[1] for s in streams])
    design = np.column_stack(([logit(p) for p in bases],np.ones(len(bases))))
    params = {'weights':np.array([1.,0.])}
    adam = Adam(params,rate=cfg['calibration_learning_rate'])
    for _ in range(cfg['calibration_epochs']):
        _,gradient = logistic_objective(params['weights'],design,labels,0.)
        adam.step(params,{'weights':gradient})
        params['weights'][:] = np.clip(params['weights'],-4.,4.)
    initial = params['weights'].tolist()
    candidates = []
    for rate in cfg['online_rates']:
        for decay in cfg['online_decays']:
            total = 0.
            for base,y,_ in streams:
                online = OnlineCalibration(initial,rate,decay)
                for p,label in zip(base,y):
                    prediction = online.predict(p)
                    total -= np.log(prediction if label else 1-prediction)
                    online.observe(p,int(label))
            candidates.append({'rate':rate,'decay':decay,'log_loss':float(total/len(labels))})
    best = min(range(len(candidates)),key=lambda i:candidates[i]['log_loss'])
    return {'fixed_weights':initial,'fixed_validation_log_loss':float(logistic_objective(
                params['weights'],design,labels,0.)[0]),
            'online':{k:v for k,v in candidates[best].items() if k!='log_loss'},
            'online_validation_grid':candidates,'selected_online_index':best,
            'validation_examples':len(labels),'validation_data_sha256':digest_arrays(
                bases,labels,np.concatenate([s[2] for s in streams])),
            'selection_cases':cfg['selection_cases']}


def metrics(trace):
    if not trace:
        return None
    p,y = np.array([r['probability'] for r in trace]),np.array([r['label'] for r in trace])
    correct = (p>=.5)==y
    confidence = np.maximum(p,1-p)
    high = confidence>=.9
    ece = 0.
    for i in range(10):
        mask = (confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1)
        if mask.any():
            ece += mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return {'accuracy':float(correct.mean()),'log_loss':float(losses(p,y).mean()),
            'brier':float(np.mean((p-y)**2)),'ece':float(ece),
            'confidently_wrong_rate':float((high&~correct).mean()),
            'high_confidence_coverage':float(high.mean()),
            'high_confidence_error_rate':float((~correct[high]).mean()) if high.any() else None}


def slices(trace):
    masks = {'signal':lambda r:r['mode']=='signal','noise':lambda r:r['mode']=='noise',
             'signal_entry_first16':lambda r:r['mode']=='signal' and r['age']<16 and r['step']-r['age']>0,
             'noise_entry_first16':lambda r:r['mode']=='noise' and r['age']<16 and r['step']-r['age']>0,
             'signal_late32':lambda r:r['mode']=='signal' and r['age']>=32,
             'noise_late32':lambda r:r['mode']=='noise' and r['age']>=32}
    return {k:{'steps':len(rows),'metrics':metrics(rows)} for k,mask in masks.items()
            for rows in [[r for r in trace if mask(r)]]}


def delay(trace,cfg):
    """Descriptive loss response; startup excluded, next switch truncates search."""
    starts = [i for i,r in enumerate(trace) if r['switched']]
    output = {}
    window,horizon = cfg['delay_window'],cfg['delay_horizon']
    for mode in ('signal','noise'):
        penalties,recovered = [],0
        for index,start in enumerate(starts):
            end = starts[index+1] if index+1<len(starts) else len(trace)
            if trace[start]['mode']!=mode or end-start<window:
                continue
            rows = trace[start:min(end,start+horizon)]
            threshold = cfg['signal_recovery_loss'] if mode=='signal' else cfg['noise_recovery_loss']
            values = losses([r['probability'] for r in rows],[r['label'] for r in rows])
            endpoint = next((i+window-1 for i in range(len(values)-window+1)
                             if values[i:i+window].mean()<=threshold),None)
            penalties.append(horizon if endpoint is None else endpoint)
            recovered += endpoint is not None
        output[mode] = {'events':len(penalties),'recovered':recovered,
            'penalized_steps':float(np.mean(penalties)) if penalties else None,
            'recovery_fraction':recovered/len(penalties) if penalties else None}
    return output


def evaluate(rows,ensemble,gate,selection,cfg,*,feedback=True,persistent=None,namespace=''):
    adapter = Adapter(ensemble)
    policies = {f:Policy(f,gate,selection) for f in FAMILIES}
    traces = {f:[] for f in FAMILIES}
    feature_rows = []
    for step,row in enumerate(rows):
        base,features = adapter.predict(row.observation)
        feature_rows.append(features)
        predictions = {f:p.predict(base,features) for f,p in policies.items()}
        # ALL predictions are committed before accessing this row's answer/metadata.
        for f,prediction in predictions.items():
            traces[f].append({'step':step,'observation':list(row.observation.readings),**prediction,
                              'label':row.label,'mode':row.mode,'age':row.age,'switched':row.switched})
        if feedback:
            for p in policies.values():
                p.observe(row.label)
            adapter.observe(row.observation,row.label)
            if persistent is not None:
                persistent.append(namespace,step,row.observation,row.label)
        else:
            adapter.discard()
            for p in policies.values():
                p.discard()
    digest = digest_arrays(feature_rows)
    records = {f:{'metrics':metrics(t),'slices':slices(t),'delay':delay(t,cfg),
                  'features_sha256':digest} for f,t in traces.items()}
    return records,traces


def average(records):
    def means(values):
        return {k:float(np.mean([v[k] for v in values if v[k] is not None]))
                if any(v[k] is not None for v in values) else None for k in values[0]}
    result = {}
    for case in sorted({r['case'] for r in records}):
        result[case] = {}
        for family in FAMILIES:
            rows = [r for r in records if r['case']==case and r['family']==family]
            parts = {}
            for key in rows[0]['slices']:
                available = [r['slices'][key]['metrics'] for r in rows if r['slices'][key]['metrics'] is not None]
                parts[key] = {'steps':sum(r['slices'][key]['steps'] for r in rows),
                              'metrics':means(available) if available else None}
            delays = {mode:means([r['delay'][mode] for r in rows]) for mode in ('signal','noise')}
            result[case][family] = {**means([r['metrics'] for r in rows]),'slices':parts,'delay':delays}
    return result


def comparisons(records,cfg):
    indexed = {(r['case'],r['seed'],r['family']):r['metrics'] for r in records}
    output = {}
    for left,right in (('frozen_gate','online_calibration'),('frozen_gate','fixed_calibration'),
                       ('online_calibration','fixed_calibration'),('frozen_gate','base')):
        per_case = {}
        for case in cfg['cases']:
            diffs = [indexed[case,s,left]['log_loss']-indexed[case,s,right]['log_loss'] for s in cfg['test_seeds']]
            per_case[case] = diffs
            output[f'{case}/{left}_minus_{right}'] = bootstrap(diffs,cfg)
        aggregate = np.array([per_case[c] for c in cfg['primary_cases']]).mean(axis=0)
        output[f'aggregate/{left}_minus_{right}'] = bootstrap(aggregate,cfg)
    return output


def scientific_gates(summary,paired,cfg):
    pair = paired['aggregate/frozen_gate_minus_online_calibration']
    value = {'minimum_improvement':pair['mean']<=-cfg['minimum_log_loss_improvement'],
             'paired_upper_below_zero':pair['ci95'][1]<0}
    quality = {}
    for case in cfg['primary_cases']:
        a,b = summary[case]['frozen_gate'],summary[case]['online_calibration']
        value[case+'_loss_preserved'] = a['log_loss']-b['log_loss']<=cfg['maximum_case_degradation']
        quality[case+'_brier_preserved'] = a['brier']-b['brier']<=cfg['maximum_brier_degradation']
        quality[case+'_confident_errors_preserved'] = a['confidently_wrong_rate']-b['confidently_wrong_rate']<=cfg['maximum_confident_error_excess']
    for case in ('mixed','short_mixed'):
        for key in ('signal_entry_first16','noise_entry_first16'):
            a = summary[case]['frozen_gate']['slices'][key]['metrics']['log_loss']
            b = summary[case]['online_calibration']['slices'][key]['metrics']['log_loss']
            quality[case+'/'+key] = a-b<=cfg['maximum_transition_degradation']
    quality['stable_loss_preserved'] = summary['stable']['frozen_gate']['log_loss']-summary['stable']['fixed_calibration']['log_loss']<=cfg['maximum_stable_degradation']
    random = summary['random_labels']
    quality['random_loss_near_uniform'] = random['frozen_gate']['log_loss']-random['uniform']['log_loss']<=cfg['maximum_random_excess']
    quality['random_confident_errors_halved'] = random['frozen_gate']['confidently_wrong_rate']<=cfg['maximum_random_confident_error_ratio']*random['base']['confidently_wrong_rate']
    controls = {f+'_random_accuracy_near_chance':abs(random[f]['accuracy']-.5)<=cfg['chance_tolerance'] for f in FAMILIES}
    return {'cross_task_value':all(value.values()),'cross_task_quality':all(quality.values()),
            'negative_controls':all(controls.values()),
            'qualified_on_sensor_family':all(value.values()) and all(quality.values()) and all(controls.values()),
            'components':{'value':value,'quality':quality,'controls':controls}}
