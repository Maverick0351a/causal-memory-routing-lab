"""Public integrity is separate from historical checks of private host files."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def verify_release(root):
    root=Path(root)
    expected=json.loads((root/'provenance/portable-source-manifest.json').read_text(encoding='utf-8'))
    for name,digest in expected.items():
        if sha(root/name)!=digest: raise RuntimeError('Released source/input changed: '+name)
    return {'release_files_match':True,'checked_release_files':len(expected),
            'scope':'public release only; original host preservation is historical'}


def source_hashes(root):
    root=Path(root)
    verify_release(root)
    manifest=root/'provenance/portable-source-manifest.json'
    return {**json.loads(manifest.read_text(encoding='utf-8')),
            'provenance/portable-source-manifest.json':sha(manifest)}
