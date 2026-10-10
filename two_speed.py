"""Offline entry point for the separately registered two-speed memory study."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('test', 'run', 'verify'))
    parser.add_argument('--output')
    args, extra = parser.parse_known_args()
    env = dict(os.environ)
    for key in list(env):
        if any(t in key.upper() for t in ('API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN')) or key in ('HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN'):
            env.pop(key, None)
    env.update(ARCHLAB_OFFLINE='1', PYTHONUTF8='1', LANGCHAIN_TRACING_V2='false', LANGSMITH_TRACING='false',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(ROOT/'scripts/offline_guard')+os.pathsep+str(ROOT))
    if args.action == 'test':
        command = ['-m', 'pytest', 'tests', 'transfer_tests', 'sensor_tests', 'recall_tests',
                   'recall_recovery_tests', 'forget_tests', 'release_tests', 'twospeed_tests', '-q', *extra]
    else:
        if not args.output:
            parser.error('--output is required for run or verify')
        command = ['-m', 'twospeed_lab.experiment', args.action, '--output', args.output, *extra]
    raise SystemExit(subprocess.call([sys.executable, *command], cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
