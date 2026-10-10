"""Saved report and self-contained interactive metric comparison."""
import argparse
import html
import json
from pathlib import Path
from .policy import FAMILIES


def generate(output):
    output = Path(output)
    r = json.loads((output/'results.json').read_text())
    cfg,gates = r['config'],r['gates']
    primary = r['paired']['aggregate/frozen_gate_minus_online_calibration']
    rows = ['# Continuous sensor confidence transfer','',
        f"Cross-task predictive value: **{'PASS' if gates['cross_task_value'] else 'FAIL'}**.",
        f"Cross-task quality: **{'PASS' if gates['cross_task_quality'] else 'FAIL'}**.",
        f"Random-label controls: **{'PASS' if gates['negative_controls'] else 'FAIL'}**.",
        f"Sensor-family qualification: **{'PASS' if gates['qualified_on_sensor_family'] else 'FAIL'}**.",'',
        f"Primary gate minus online loss: {primary['mean']:+.6f} nats, paired-seed95% interval [{primary['ci95'][0]:+.6f}, {primary['ci95'][1]:+.6f}].",
        'Required: <=-.01 nats, upper interval<0, no stress case worse by>.02. All failed components are retained.','',
        'Synthetic fault propensity from eight continuous normal readings;36 learned base parameters. This is not measured sensor data or a hardware diagnostic.',
        'The gate checkpoint and its normalization are unchanged. Online Platt calibration directly updates slope/intercept from past released feedback.','',
        f"Validation-selected online rate={r['selection']['online']['rate']}, decay={r['selection']['online']['decay']}; fixed slope/intercept={r['selection']['fixed_weights']}.",'',
        '| Case | Policy | Log loss | Brier | Accuracy | ECE | Confidently wrong | High-conf coverage |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for case in cfg['cases']:
        for family in FAMILIES:
            s = r['summary'][case][family]
            rows.append(f"| {case} | {family} | {s['log_loss']:.5f} | {s['brier']:.5f} | {s['accuracy']:.3%} | {s['ece']:.5f} | {s['confidently_wrong_rate']:.3%} | {s['high_confidence_coverage']:.3%} |")
    rows += ['', '## Paired loss comparisons','',
        '| Case | Difference | Mean | Paired-seed95% interval |','|---|---|---:|---|']
    for case in ('aggregate',*cfg['cases']):
        for pair in ('frozen_gate_minus_online_calibration','frozen_gate_minus_fixed_calibration','online_calibration_minus_fixed_calibration'):
            p = r['paired'][f'{case}/{pair}']
            rows.append(f"| {case} | {pair} | {p['mean']:+.6f} | [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}] |")
    rows += ['', 'The primary aggregate averages four stress cases within each seed, then resamples24 seeds. Cases and individual steps are not independent replicates.',
        'Intervals condition on one learned ensemble and one selection; they exclude training/selection uncertainty. Per-case intervals are descriptive.','',
        '## Transitions and response delay','',
        '| Case | Policy | Entry | First16 loss | Descriptive response steps | Recovered fraction |',
        '|---|---|---|---:|---:|---:|']
    def fmt(v):
        return 'n/a' if v is None else f'{v:.5f}'
    for case in ('mixed','short_mixed','rotating'):
        for family in ('base','fixed_calibration','frozen_gate','online_calibration'):
            s = r['summary'][case][family]
            for mode in ('signal','noise'):
                score = s['slices'][mode+'_entry_first16']['metrics']
                d = s['delay'][mode]
                rows.append(f"| {case} | {family} | {mode} | {fmt(score['log_loss'] if score else None)} | {fmt(d['penalized_steps'])} | {fmt(d['recovery_fraction'])} |")
    rows += ['', 'Startup excluded. Search ends at the next switch/end. Unrecovered eligible events score96; report is an equal-stream mean. Loss-window response is stochastic and is not a change detector.',
        '', '## Every frozen gate component','']
    for group,values in gates['components'].items():
        rows += ['### '+group,'']+[f"- {key}: {'PASS' if value else 'FAIL'}" for key,value in values.items()]+['']
    rows += ['All predictions precede feedback; hidden coefficients/modes/ages/switches are scoring-only. The identical public observation streams prevent an artificial mode marker.',
        'No-feedback retains the frozen learned predictor and therefore can still recognize signal. It withholds all online feedback, including adapter summaries.',
        'Gate shrinkage preserves base class sign apart from rounding ties; accuracy equality is structural. Platt calibration can adjust bias and invert sign, so any recognition difference belongs to that baseline.',
        'Confidently wrong is a fraction of all predictions; coverage and conditional high-confidence error are also retained in results.json.',
        'All source/checkpoints/candidate grids/traces and failed gates are saved. Fresh-process verification and causal SQLite resume are in verification.json.',
        'Prior research defaults and live assistant configuration remain unchanged. No downloads, API keys, pretrained peptide weights, network services or quantum jobs were used. Python socket guard is not an OS sandbox.']
    (output/'RESULTS.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    data = json.dumps(r['summary']).replace('<','\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sensor transfer results</title>
<style>body{font:16px system-ui;background:#101722;color:#e3e9f5;max-width:1050px;margin:35px auto;padding:0 20px}select{font:inherit;background:#243348;color:inherit;padding:8px;margin:10px 15px 20px 0}article{margin:15px 0}.bar{height:22px;background:#75a7f7;border-radius:3px}.gate{background:#efba5a}small{color:#a8b8cd}label{display:inline-block}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #334258}h1{font-size:27px}</style>
<h1>Continuous sensor confidence transfer</h1><p>LOCAL SYNTHETIC EXPERIMENT · Frozen gate · Fresh seeds</p><p>STATUS</p>
<label>Case <select id="case"></select></label><label>Metric <select id="metric"><option value="log_loss">Log loss</option><option value="brier">Brier</option><option value="confidently_wrong_rate">Confidently wrong</option><option value="accuracy">Accuracy</option><option value="ece">ECE</option></select></label>
<div id="bars"></div><table><thead><tr><th>Policy</th><th>Log loss</th><th>Brier</th><th>Accuracy</th><th>Confidently wrong</th></tr></thead><tbody id="scores"></tbody></table>
<p><small>Log loss, Brier, ECE and confident errors: lower is better. Accuracy: higher is better. Equal-stream means. Gate shrinkage retains the base class. Full frozen gates, paired intervals and transition limitations are in RESULTS.md.</small></p>
<script>const data=DATA;const cases=document.getElementById('case'),metric=document.getElementById('metric');Object.keys(data).forEach(c=>cases.add(new Option(c,c)));cases.value='mixed';function render(){const rows=Object.entries(data[cases.value]),m=metric.value,max=Math.max(...rows.map(([f,v])=>v[m]),1e-8);document.getElementById('bars').innerHTML=rows.map(([f,v])=>`<article>${f}: <b>${v[m].toFixed(5)}</b><div class="bar ${f==='frozen_gate'?'gate':''}" style="width:${100*v[m]/max}%"></div></article>`).join('');document.getElementById('scores').innerHTML=rows.map(([f,v])=>`<tr><td>${f}</td><td>${v.log_loss.toFixed(5)}</td><td>${v.brier.toFixed(5)}</td><td>${(100*v.accuracy).toFixed(2)}%</td><td>${(100*v.confidently_wrong_rate).toFixed(3)}%</td></tr>`).join('');}cases.onchange=metric.onchange=render;render();</script></html>'''
    status = f"Predictive value: {'PASS' if gates['cross_task_value'] else 'FAIL'} · Quality: {'PASS' if gates['cross_task_quality'] else 'FAIL'} · Qualification: {'PASS' if gates['qualified_on_sensor_family'] else 'FAIL'}"
    page = page.replace('STATUS',html.escape(status)).replace('DATA',data)
    (output/'comparison.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    generate(parser.parse_args().run)
