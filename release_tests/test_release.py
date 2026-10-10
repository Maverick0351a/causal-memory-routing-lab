import json
from pathlib import Path
import pytest
from release_tools.integrity import verify_release,sha

ROOT=Path(__file__).resolve().parents[1]


def test_public_integrity_scope_is_explicit():
    proof=verify_release(ROOT)
    assert proof['release_files_match']
    assert 'original host preservation is historical' in proof['scope']
    assert 'vendor_clean' not in proof and 'original_project_files_unchanged' not in proof


def test_every_unchanged_copied_source_matches_original_digest():
    entries=json.loads((ROOT/'provenance/source-map.json').read_text())
    for path,info in entries.items():
        assert sha(ROOT/path)==info['public_sha256']
        if not info['adaptations']: assert info['public_sha256']==info['original_sha256']


def test_scientific_ast_equivalence_receipt_covers_every_package():
    proof=json.loads((ROOT/'provenance/scientific-ast-equivalence.json').read_text())
    assert len(proof)>30 and all(proof.values())
    assert any(x.startswith('forget_lab/') for x in proof)


@pytest.mark.parametrize('run,gate',[('benchmark-v2','comparative'),('matched-routing-v1','ar_predictive_value'),
                                  ('sensor-transfer-v1','qualified_on_sensor_family'),('forgetting-v1','forgetting_qualified')])
def test_failed_scientific_gates_remain_failed(run,gate):
    result=json.loads((ROOT/'results'/run/'results.json').read_text())
    assert result['gates'][gate] is False


def test_incomplete_export_and_historical_issue_retained():
    entries={r['study']:r for r in json.loads((ROOT/'research-index.json').read_text())}
    assert entries['benchmark-v1']['status']=='historical metric issue'
    assert entries['recurring-memory-v1']['status']=='incomplete export failure'
    assert entries['recurring-memory-v1']['verification_status'] is None


def test_archive_inventory_and_evaluation_boundary():
    index=json.loads((ROOT/'artifacts-index.json').read_text())
    assert len(index)==9 and index['forgetting-v1']['trace_files']==192
    assert all(len(v['sha256'])==64 and v['bytes']>0 for v in index.values())
    text=(ROOT/'DATA_BOUNDARY.md').read_text()
    assert 'evaluator-only' in text and 'step < cutoff' in text and 'exposed' in text
