"""Create immutable preservation evidence, once, before any old-file edits."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/confidence-transfer-20261010"
if (OUT/"preservation.json").exists():
    raise RuntimeError("Refuse to overwrite preservation manifest")
paths = []
for folder in ("archlab","confidence_lab","scripts","tests","audit","results",
               "experiments/matched-routing-20261010","experiments/confidence-gating-20261010"):
    paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
paths.extend(ROOT/name for name in ("Run-Lab.ps1","README.md","PROTOCOL.md","Setup.ps1",
                                   "requirements-dev.in",".gitignore"))
manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
(OUT/"preservation.json").write_text(json.dumps(manifest,indent=2)+"\n")
(OUT/"source-before").mkdir(exist_ok=False)
shutil.copyfile(ROOT/"AGENTS.md",OUT/"source-before/AGENTS.md")
print("Preserved",len(manifest),"old lab files; AGENTS.md backed up")
