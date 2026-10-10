"""Derive descriptive charts from released archives; never rerun/select a policy.

Run from the repo: python -m release_tools.build_charts --archives ../dataset/artifacts
The JSON output is embedded into the static explorer during release assembly.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import tarfile
import numpy as np
from .integrity import sha

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(rb'(?i)(?:[a-z]:[\\/]Users[\\/]|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')


def series(rows, prediction, kind):
    p = np.asarray([r[prediction] for r in rows], dtype=float)
    y = np.asarray([r['_label'] for r in rows])
    correct = (p >= .5) == y
    confidence = np.maximum(p, 1-p)
    clipped = np.clip(p, 1e-12, 1-1e-12)
    loss = -(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    bins = np.minimum((confidence*10).astype(int), 9)
    reliability = np.zeros((10, 3))
    for b in range(10):
        mask = bins == b
        reliability[b] = [mask.sum(), confidence[mask].sum(), correct[mask].sum()]
    adaptation = {}
    for event in ('switch', 'changed', 'return'):
        ages = np.asarray([r['_'+event] for r in rows])
        values = np.zeros((64, 3))
        for a in range(64):
            mask = ages == a
            values[a] = [mask.sum(), loss[mask].sum(), correct[mask].sum()]
        adaptation[event] = values
    return reliability, adaptation


def build(archives):
    registry = json.loads((ROOT/'research-index.json').read_text())
    index = json.loads((ROOT/'artifacts-index.json').read_text())
    chart = {'version': 1, 'notes': 'Descriptive pooled predictions; no independent-sample intervals. '
             'Switch curves stop at the next recorded switch; initial blocks excluded. '
             'Model fits sharing a stream are dependent. Confidence bins are [0,.1),...,[.9,1].', 'studies': {}}
    audit = {'archives': {}, 'trace_rows': 0, 'privacy_pattern_hits': 0}
    for study in registry:
        name = study['study']
        result_file = ROOT/'results'/name/'results.json'
        cases = {}
        if result_file.exists():
            result = json.loads(result_file.read_text())
            for r in result.get('runs', []):
                case = r.get('case', r.get('condition'))
                policy = r['family']
                slot = cases.setdefault(case, {}).setdefault(policy, {'seeds': [], 'curves': {}})
                endpoints = {k:v for k,v in r.items() if isinstance(v,dict) and 'log_loss' in v}
                if 'metrics' in endpoints: endpoints['whole'] = endpoints.pop('metrics')
                slot['seeds'].append({'seed':r['seed'], 'model':r.get('model', policy), 'values':endpoints})
        archive = archives/(name+'.tar.gz')
        assert sha(archive) == index[name]['sha256']
        pooled = {}
        scanned = 0
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                f = tar.extractfile(member)
                if not member.name.endswith('.jsonl'):
                    for chunk in iter(lambda:f.read(1048576), b''):
                        assert not FORBIDDEN.search(chunk), 'Private material in archive'
                    continue
                decoded = []
                for line in f:
                    assert not FORBIDDEN.search(line), 'Private material in trace'
                    decoded.append(json.loads(line))
                scanned += len(decoded)
                parts = member.name.split('/')
                case = parts[1]
                case_slot = cases.setdefault(case,{})
                # Switch metadata is used for this retrospective evaluator chart only.
                last = None
                types = {'changed':False, 'return':False}
                grouped = defaultdict(list)
                for r in decoded:
                    switched = r.get('switched',False) or r.get('rule_switched',False) or r.get('mode_switched',False)
                    if switched:
                        last = r['step']
                        types = {'changed':r.get('changed_entry',False), 'return':r.get('returned',False)}
                    age = r['step']-last if last is not None else -1
                    common = {'_label':r['label'], '_switch':age,
                              '_changed':age if types['changed'] else -1,
                              '_return':age if types['return'] else -1}
                    if 'predictions' in r:
                        for policy, pred in r['predictions'].items(): grouped[policy].append({**common, **pred})
                    else:
                        model = parts[-1].removesuffix('.jsonl')
                        # Remove only a final numeric model seed, not policy words.
                        policy = re.sub(r'_\d+$', '', model)
                        grouped[policy].append({**common, **r})
                for policy, rows in grouped.items():
                    slot = case_slot.setdefault(policy,{'seeds':[], 'curves':{}})
                    for endpoint, field in [('raw','probability'), ('gated','gated_probability')]:
                        if field not in rows[0]: continue
                        rel, adapt = series(rows,field,endpoint)
                        key = (case,policy,endpoint)
                        if key not in pooled: pooled[key] = [rel,adapt]
                        else:
                            pooled[key][0] += rel
                            for event in adapt: pooled[key][1][event] += adapt[event]
        for (case,policy,endpoint),(rel,adapt) in pooled.items():
            convert = lambda a:[[int(n),float(x/n) if n else None,float(y/n) if n else None] for n,x,y in a]
            cases[case][policy]['curves'][endpoint] = {'reliability':convert(rel), **{k:convert(v) for k,v in adapt.items()}}
        chart['studies'][name] = cases
        audit['trace_rows'] += scanned
        audit['archives'][name] = {'sha256':index[name]['sha256'], 'rows':scanned, 'safe_member_paths':True}
        print(name,scanned,'trace rows aggregated and scanned',flush=True)
    return chart,audit


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'site/charts.json')
    a=p.parse_args();chart,audit=build(a.archives)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(chart,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    (ROOT/'provenance/chart-derivation.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
