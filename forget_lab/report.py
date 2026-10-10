"""Self-contained local report; every pass and failure remains visible."""
import argparse
import html
import json
from pathlib import Path
from .policy import FAMILIES


def generate(output):
    output = Path(output)
    r = json.loads((output/'results.json').read_text(encoding='utf-8'))
    cfg,g = r['config'],r['gates']
    status = lambda x:'PASS' if x else 'FAIL'
    lines = ['# Explicit forgetting on changing associations','']
    lines += [f"- {name}: **{status(passed)}**" for name,passed in g.items() if name!='components']
    lines += ['', 'All qualification gates use raw predictions. The confidence gate is a descriptive companion.',
        'Changed-entry windows contain only the FIRST visit to each cue after each mapping revision, with age <16. Later recovered visits cannot dilute this endpoint.',
        'Revision IDs, context IDs, changed-entry flags, answers and switches are evaluator-only and read after every policy predicts.','',
        '| Case | Policy | Raw loss | Gated loss | Raw accuracy | Brier | ECE | Confidently wrong fraction | Changed-entry16 loss | Response steps |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x:'n/a' if x is None else f'{x:.5f}'
    for case in cfg['cases']:
        for family in FAMILIES:
            s = r['summary'][case][family]
            a = s['raw']
            c = s['changed_entry_first16']
            lines.append(f"| {case} | {family} | {a['log_loss']:.5f} | {s['gated']['log_loss']:.5f} | {a['accuracy']:.3%} | {a['brier']:.5f} | {a['ece']:.5f} | {a['confidently_wrong_rate']:.3%} | {fmt(c['log_loss'] if c else None)} | {fmt(s['change_delay']['penalized_steps'])} |")
    lines += ['', '## Paired primary comparisons','',
        '| Endpoint | Forgetting minus comparator | Difference | Paired-seed 95% interval |',
        '|---|---|---:|---|']
    for endpoint in ('changed_entry_first16_log_loss','changed_entry_first8_log_loss','change_delay_penalized_steps'):
        for right in ('memory','recent_uniform','shuffled_forgetting','recency','context_cache','recurrent'):
            p = r['paired'][f'aggregate_change/{endpoint}/forgetting_minus_{right}']
            lines.append(f"| {endpoint} | {right} | {p['mean']:+.6f} | [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}] |")
    lines += ['', 'Primary cases are once-changing, repeatedly changing and repeatedly changing with noisy cues. Average cases WITHIN each of24 seeds before bootstrapping10000 seed-block resamples.',
        'Response is first endpoint of an8-label raw-loss window <=.55, capped64 and truncated at next switch/end; unrecovered=64. This measures feedback steps, not wall time or a change detector.',
        'All candidate families use the SAME validation objective: .5 stable return-first16 loss + .5 changed-entry-first16 loss. Recurrent training includes stable and changing streams.',
        'Intervals condition on one coefficient world, expert fit and validation selection, exclude fit/selection uncertainty and are not simultaneous confidence bands.','',
        '## Every frozen component','']
    for group,checks in g['components'].items():
        lines += ['### '+group,'']+[f'- {name}: {status(passed)}' for name,passed in checks.items()]+['']
    lines += ['## Method and scope','',
        'Fading retrieval searches the latest1024 eligible public-feedback records, keeps the newest64 cue matches within radius1.25, and weights their ranks by2^(-rank/half_life). Validation selects half-life2,8 or32 matching-feedback observations. No hidden boundary or reset signal is used.',
        'recent_uniform uses those same records, radius, blend, count warmup and ordering with uniform weights; it isolates fading from simply preferring recent matches. Shuffled forgetting causally permutes key-feedback associations with the same key multiset, counts and effective sample sizes.',
        'Original memory retains its distance-nearest retrieval algorithm, cache retains its centroid/error-EWMA algorithm; both are reselected on fresh validation. Disabled memory equals recency as an intentional duplicate.',
        'Audit history is retained; forgetting changes prediction weights rather than physically deleting stored evidence. Effective neighbor counts are recorded. Search, cache and recurrent budgets differ.',
        json.dumps(r['budgets'],indent=2),
        'The embedding remains identity encoding of six measured cue values over four known fitted tasks. This is a synthetic concept-drift experiment, not semantic memory, new-task learning, novelty proof, SAM-kNN/ADWIN/Titans reproduction or general architecture superiority.',
        'Sources and exact distinctions are in source/experiments/forgetting-20261010/RELATED_RESEARCH.md. Fresh-process refit/reselection/replay and all-policy strict prior-feedback resumes are in verification.json.',
        'No keys, downloaded models, API calls, services or QPU jobs are required; the Python process guard is not an OS sandbox. Earlier results, failed gates and live defaults remain unchanged.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    data = json.dumps(r['summary']).replace('<','\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forgetting experiment</title><style>body{background:#142031;color:#eaf0f9;font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 18px}select{font:inherit;padding:7px;margin:8px;background:#283b55;color:inherit}.bar{height:17px;background:#8ab4f8;border-radius:4px}.forgetting{background:#eab566}article{margin:14px 0}small{color:#b4c4dc}table{width:100%;border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #34465d}</style><h1>Does fading memory help after rules change?</h1><p>STATUS</p><label>Case<select id="case"></select></label><label>Endpoint<select id="endpoint"><option value="changed_entry_first16">First16 after changed association returns</option><option value="changed_entry_first8">First8 after changed association returns</option><option value="return_first16">Unchanged return first16</option><option value="raw">Whole stream raw</option><option value="gated">Whole stream confidence companion</option></select></label><label>Metric<select id="metric"><option value="log_loss">Log loss</option><option value="brier">Brier</option><option value="accuracy">Accuracy</option><option value="confidently_wrong_rate">Confident mistakes</option></select></label><div id="bars"></div><table><thead><tr><th>Policy</th><th>Changed entry loss</th><th>Response steps</th></tr></thead><tbody id="scores"></tbody></table><p><small>All qualification uses raw predictions. Lower loss/Brier/confident mistakes is better. Causal fading changes weights; the audit history remains. Capacities differ. See RESULTS.md for all failed gates.</small></p><script>const data=DATA,c=document.getElementById('case'),e=document.getElementById('endpoint'),m=document.getElementById('metric');Object.keys(data).forEach(x=>c.add(new Option(x,x)));c.value='repeated_changes';const fmt=v=>v===null||v===undefined?'n/a':v.toFixed(5);function render(){const rows=Object.entries(data[c.value]),vals=rows.map(([f,s])=>s[e.value]?.[m.value]??null),max=Math.max(...vals.filter(x=>x!==null),1e-9);document.getElementById('bars').innerHTML=rows.map(([f,s],i)=>`<article>${f}: <b>${fmt(vals[i])}</b><div class="bar ${f==='forgetting'?'forgetting':''}" style="width:${vals[i]===null?0:100*vals[i]/max}%"></div></article>`).join('');document.getElementById('scores').innerHTML=rows.map(([f,s])=>`<tr><td>${f}</td><td>${fmt(s.changed_entry_first16?.log_loss)}</td><td>${fmt(s.change_delay.penalized_steps)}</td></tr>`).join('')}c.onchange=e.onchange=m.onchange=render;render();</script></html>'''
    message = ' | '.join(f'{k}: {status(g[k])}' for k in ('entry_value','speed','stable_retention','quality','fading_ablation','forgetting_qualified'))
    (output/'comparison.html').write_text(page.replace('STATUS',html.escape(message)).replace('DATA',data),encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    generate(parser.parse_args().run)
