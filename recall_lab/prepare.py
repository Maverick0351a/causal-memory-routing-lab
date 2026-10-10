"""Preserve every previous lab artifact before this isolated follow-up."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/recurring-memory-20261010'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'preservation.json').exists():
    raise RuntimeError('Preservation manifest already exists')
excluded = {'.venv','vendor','.git','__pycache__','.pytest_cache','recall_lab','recall_tests','recurring-memory-20261010'}
paths = [p for p in ROOT.rglob('*') if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts) and p!=ROOT/'AGENTS.md']
manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(OUT/'preservation.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
(OUT/'source-before').mkdir()
shutil.copyfile(ROOT/'AGENTS.md',OUT/'source-before/AGENTS.md')
print('Preserved',len(manifest),'prior lab files; AGENTS.md backed up')
