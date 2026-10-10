"""Chart aggregates must preserve endpoints and keep empty events empty."""
import json
from pathlib import Path
import numpy as np
from release_tools.build_charts import series

ROOT=Path(__file__).resolve().parents[1]


def test_confidence_bins_include_exact_one_and_tied_half():
    rows=[{'probability':p,'_label':y,'_switch':a,'_changed':-1,'_return':-1}
          for p,y,a in [(1.,1,0),(.5,0,-1),(0.,0,1)]]
    reliability,curves=series(rows,'probability','raw')
    assert reliability[:,0].sum()==3
    assert np.array_equal(reliability[9],[2,2,2])
    assert np.array_equal(reliability[5],[1,.5,0])
    assert curves['switch'][:,0].sum()==2  # no initial/unmarked step
    assert curves['changed'].sum()==curves['return'].sum()==0


def test_public_seed_chart_means_match_registered_summaries():
    charts=json.loads((ROOT/'site/charts.json').read_text())['studies']
    registry=json.loads((ROOT/'research-index.json').read_text())
    checked=0
    for study in registry:
        for case,policies in study['summary'].items():
            for policy,summary in policies.items():
                seed_rows=charts[study['study']][case][policy]['seeds']
                endpoints={'whole':summary} if 'log_loss' in summary else summary
                for endpoint,metrics in endpoints.items():
                    if not isinstance(metrics,dict) or 'log_loss' not in metrics: continue
                    values=[r['values'][endpoint]['log_loss'] for r in seed_rows if endpoint in r['values']]
                    assert values
                    assert np.isclose(np.mean(values),metrics['log_loss'],rtol=0,atol=1e-12)
                    checked+=1
    assert checked>200


def test_reliability_counts_and_archive_inventory():
    data=json.loads((ROOT/'site/charts.json').read_text())
    proof=json.loads((ROOT/'provenance/chart-derivation.json').read_text())
    assert len(proof['archives'])==9 and proof['privacy_pattern_hits']==0
    for case,policies in data['studies']['forgetting-v1'].items():
        for policy,row in policies.items():
            for endpoint in ('raw','gated'):
                assert sum(r[0] for r in row['curves'][endpoint]['reliability'])==24*2048
                assert all(0<=r[1]<=1 and 0<=r[2]<=1 for r in row['curves'][endpoint]['reliability'] if r[0])
