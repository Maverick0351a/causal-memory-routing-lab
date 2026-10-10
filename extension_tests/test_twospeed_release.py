"""Additive release checks, separate from the frozen scientific test suite."""
import json
from pathlib import Path
import re
import numpy as np
from release_tools.integrity import sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'two-speed-v1'


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def test_every_registered_source_and_protocol_still_matches():
    proof = read('results/'+STUDY+'/preregistration.json')
    for name, digest in proof['source_hashes'].items():
        assert sha(ROOT/name) == digest, name
    assert proof['phase'].startswith('before ALL registered')
    assert proof['config'] == read('experiments/two-speed-20261010/protocol.json')


def test_registered_counts_full_replay_and_qualification_preserved():
    result = read('results/'+STUDY+'/results.json')
    receipt = read('results/'+STUDY+'/verification.json')
    assert len(result['runs']) == 6*8*8*11
    assert receipt['status'] == 'PASS'
    assert receipt['raw_predictions_replayed'] == receipt['companion_predictions_replayed'] == 6*8*8*11*1024
    assert receipt['policy_history_reloads_exact'] == 11 and receipt['future_rows_excluded'] == 512
    assert receipt['scientific_qualification'] == result['gates']['two_speed_qualified']
    assert result['gates']['two_speed_qualified'] == all(all(v.values()) for v in result['gates']['components'].values())


def test_new_chart_seed_means_and_counts_match_every_endpoint():
    result = read('results/'+STUDY+'/results.json')
    chart = read('site/two-speed-charts.json')['studies'][STUDY]
    for case,policies in chart.items():
        for family, data in policies.items():
            assert len(data['seeds']) == 48 and len({r['model'] for r in data['seeds']}) == 6
            for endpoint in ('raw','gated','return_first16','changed_entry_first16','changed_entry_first8'):
                values = [r['values'][endpoint]['log_loss'] for r in data['seeds'] if endpoint in r['values']]
                summary = result['summary'][case][family][endpoint]
                if summary is not None:
                    np.testing.assert_allclose(np.mean(values),summary['log_loss'],atol=1e-12,rtol=0)
            for endpoint in ('raw','gated'):
                assert sum(r[0] for r in data['curves'][endpoint]['reliability']) == 48*1024
                if case in ('recurring','noisy_cues'):
                    assert sum(r[0] for r in data['curves'][endpoint]['changed']) == 0


def test_tradeoff_world_points_are_world_means_not_stream_pseudoreplicates():
    result = read('results/'+STUDY+'/results.json')
    chart = read('site/two-speed-charts.json')['tradeoffs'][STUDY]
    for family, points in chart.items():
        assert len(points) == 6 and len({p['world'] for p in points}) == 6
        for point in points:
            rows = [r for r in result['runs'] if r['world']==point['world'] and r['family']==family]
            stable = [r['raw']['log_loss'] for r in rows if r['case'] in result['config']['stable_cases']]
            changed = [r['changed_entry_first16']['log_loss'] for r in rows if r['case'] in result['config']['primary_cases']]
            assert abs(point['stable_loss']-np.mean(stable)) < 1e-12
            assert abs(point['changed_loss']-np.mean(changed)) < 1e-12


def test_explorer_keeps_every_old_study_and_exact_new_data():
    html = (ROOT/'site/index.html').read_text(encoding='utf-8')
    studies = json.loads(re.search(r'<script id="research-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    charts = json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert studies[:-1] == read('research-index.json')
    assert studies[-1] == read('two-speed-index.json')
    old = read('site/charts.json')['studies']
    assert all(charts['studies'][k] == v for k,v in old.items())
    assert charts['studies'][STUDY] == read('site/two-speed-charts.json')['studies'][STUDY]
    assert 'tradeoff-panel' in html and 'tradeoff-world' in html and 'Download SVG' in html
