"""Reviewable metrics, failed gates and self-contained interactive comparison."""
import argparse
import html
import json
from pathlib import Path
from .policy import FAMILIES


def generate(output):
    output = Path(output)
    result = json.loads((output/'results.json').read_text())
    cfg,g = result['config'],result['gates']
    status = lambda x:'PASS' if x else 'FAIL'
    rows = ['# Recurring-context memory experiment','',
        f"Functional recall: **{status(g['functional_recall'])}**.",
        f"Faster return adaptation: **{status(g['faster_return_adaptation'])}**.",
        f"Novel/stale/random quality: **{status(g['quality_preserved'])}**.",
        f"Overall memory qualification: **{status(g['memory_qualified'])}**.",
        f"Extra value over conventional context cache: **{status(g['extra_value_over_context_cache'])}**.",
        f"Descriptive gated random-loss control: **{status(g['gated_random_loss_control'])}**.",'',
        'All qualification endpoints use raw probabilities; confidence shrinkage cannot rescue failed recall, speed or relative quality.',
        'New random cue-to-task mappings are sampled in every stream. Current answers and private context IDs never enter inference or memory.',
        'Memory-disabled and recency are intentionally the same policy; they are one independent baseline, with a duplicate equality check.','',
        '| Case | Policy | Raw loss | Gated loss | Raw Brier | Accuracy | Raw confident errors | Gated confident errors |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for case in cfg['cases']:
        for family in FAMILIES:
            s = result['summary'][case][family]
            a,b = s['raw'],s['gated']
            rows.append(f"| {case} | {family} | {a['log_loss']:.5f} | {b['log_loss']:.5f} | {a['brier']:.5f} | {a['accuracy']:.3%} | {a['confidently_wrong_rate']:.3%} | {b['confidently_wrong_rate']:.3%} |")
    rows += ['', '## Primary return comparisons','',
        '| Endpoint | Memory minus baseline | Mean difference | Paired-seed 95% interval |',
        '|---|---|---:|---|']
    for group in ('return_first16_log_loss','return_delay_penalized_steps','raw_log_loss'):
        for right in ('recency','shuffled_memory','recurrent','context_cache'):
            p = result['paired'][f'aggregate/{group}/memory_minus_{right}']
            rows.append(f"| {group} | {right} | {p['mean']:+.6f} | [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}] |")
    rows += ['', 'Primary cases are recurring/noisy cues, averaged within each seed before bootstrapping 24 seed blocks. Steps and cases are not independent replicates.',
        'Require first16 loss improvement >=.02 nats versus recency/disabled, shuffle and recurrent; return delay >=2 steps faster versus recency and recurrent. Every paired upper bound must be below zero.',
        'Extra value gate versus the simple cache requires >=.01 nats overall raw loss improvement. It is separate from whether memory helps versus context-blind learning.',
        'Intervals condition on one expert fit and one selection and exclude fit/selection uncertainty. They are not simultaneous confidence bands.','',
        '## Return behavior','',
        '| Case | Policy | First16 return loss | Penalized response steps | Recovered fraction | Total return steps |',
        '|---|---|---:|---:|---:|---:|']
    for case in cfg['primary_cases']:
        for family in FAMILIES:
            s = result['summary'][case][family]
            rows.append(f"| {case} | {family} | {s['return_first16']['log_loss']:.5f} | {s['return_delay']['penalized_steps']:.3f} | {s['return_delay']['recovery_fraction']:.3%} | {s['return_scored_steps']} |")
    rows += ['', 'Response uses the first8-label window below .55 log loss. Startup/first visits excluded; search truncates at next switch/end; unrecovered events score64. It is a stochastic response metric, not a change detector.','',
        '## Frozen components','']
    for group,checks in g['components'].items():
        rows += ['### '+group,'']+[f'- {name}: {status(passed)}' for name,passed in checks.items()]+['']
    rows += ['## Budgets and scope','',
        json.dumps(result['budgets'],indent=2),
        'These methods have different capacity and compute. Memory searches at most1024 released feedback records; transcript and RAM audit index retain the complete2048-step episode. Cache holds8 summaries. The recurrent network has1036 weights and24 hidden values, and extra offline training.',
        'Three recurrent candidates each see98304 examples and3072 updates; selection scores8 memory,4 cache and9 recurrent candidates on independent validation streams. Full curves and every checkpoint are saved.',
        'Confidence checkpoint/normalization are frozen. Companion gating changes confidence, not recognition; primary recall and speed use raw probabilities.',
        'Memory search keys are an identity encoding of six measured cue values. This does not establish semantic retrieval, real-world sensor competence or algorithmic novelty.',
        'No-feedback retains learned experts and the neural prior; only feedback-driven state/retrieval is removed. Chance is required only on independent random labels.',
        'Every raw/gated score, ECE, confidence coverage and conditional high-confidence error is retained in results.json. Source/selection hashes precede held-out generation.',
        'Fresh-process refit/reselection/replay and strict prior-feedback SQLite resumes are in verification.json. Existing artifacts/defaults stay preserved. No downloads, keys, services or QPU jobs; Python guard is not an OS sandbox.']
    (output/'RESULTS.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    data = json.dumps(result['summary']).replace('<','\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recurring memory experiment</title>
<style>body{font:16px system-ui;background:#121b29;color:#e6edf8;max-width:1050px;margin:30px auto;padding:0 20px}select{font:inherit;padding:8px;background:#26364c;color:inherit;margin:10px 15px 15px 0}.bar{height:20px;border-radius:4px;background:#7aa7ec}.memory{background:#edb96e}article{margin:14px 0}small{color:#adbed5}table{width:100%;border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #34465d}</style>
<h1>Does memory help when a context returns?</h1><p>Local synthetic experiment · Frozen seeds and confidence gate</p><p>STATUS</p>
<label>Case <select id="case"></select></label><label>Output <select id="kind"><option value="raw">Raw</option><option value="gated">Frozen gate companion</option></select></label><label>Metric <select id="metric"><option value="log_loss">Log loss</option><option value="brier">Brier</option><option value="confidently_wrong_rate">Confident mistakes</option><option value="accuracy">Accuracy</option></select></label>
<div id="bars"></div><table><thead><tr><th>Policy</th><th>Return first16 loss</th><th>Response steps</th><th>Accuracy</th></tr></thead><tbody id="scores"></tbody></table>
<p><small>All qualification gates use raw predictions. Lower loss/Brier/confident mistakes is better. Disabled memory equals recency. Capacity and compute differ. See RESULTS.md for all failed gates and uncertainty limits.</small></p>
<script>const data=DATA;const cases=document.getElementById('case'),kind=document.getElementById('kind'),metric=document.getElementById('metric');Object.keys(data).forEach(c=>cases.add(new Option(c,c)));cases.value='recurring';function fmt(v){return v===null?'n/a':v.toFixed(5)}function render(){const rows=Object.entries(data[cases.value]),k=kind.value,m=metric.value,max=Math.max(...rows.map(([f,v])=>v[k][m]),1e-9);document.getElementById('bars').innerHTML=rows.map(([f,v])=>`<article>${f}: <b>${fmt(v[k][m])}</b><div class="bar ${f==='memory'?'memory':''}" style="width:${100*v[k][m]/max}%"></div></article>`).join('');document.getElementById('scores').innerHTML=rows.map(([f,v])=>`<tr><td>${f}</td><td>${fmt(v.return_first16?.log_loss??null)}</td><td>${fmt(v.return_delay.penalized_steps)}</td><td>${(v[k].accuracy*100).toFixed(2)}%</td></tr>`).join('')}cases.onchange=kind.onchange=metric.onchange=render;render();</script></html>'''
    message = ' · '.join(f'{key}: {status(g[key])}' for key in ('functional_recall','faster_return_adaptation','quality_preserved','memory_qualified','extra_value_over_context_cache'))
    (output/'comparison.html').write_text(page.replace('STATUS',html.escape(message)).replace('DATA',data),encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    generate(parser.parse_args().run)
