"""Offline public entry point. Installation and Hub downloads are explicit steps."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
MODULES={'matched':'archlab.matched','confidence':'confidence_lab.experiment',
         'transfer':'transfer_lab.experiment','sensor':'sensor_lab.experiment',
         'memory':'recall_recovery.experiment','forgetting':'forget_lab.experiment'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('test')
    run=sub.add_parser('run')
    run.add_argument('study',choices=MODULES)
    run.add_argument('--output',required=True)
    replay=sub.add_parser('verify-published-forgetting')
    replay.add_argument('--archive',required=True,help='Local forgetting-v1.tar.gz from the dataset')
    replay.add_argument('--receipt',required=True,help='New receipt filename; existing receipts are preserved')
    args,extra=p.parse_known_args()
    if extra[:1]==['--']: extra=extra[1:]
    env=dict(os.environ)
    for key in list(env):
        if any(t in key.upper() for t in ('API_KEY','ACCESS_TOKEN','AUTH_TOKEN')) or key in ('HF_TOKEN','HUGGINGFACE_HUB_TOKEN'):
            env.pop(key,None)
    env.update(ARCHLAB_OFFLINE='1',PYTHONUTF8='1',LANGCHAIN_TRACING_V2='false',LANGSMITH_TRACING='false',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(ROOT/'scripts/offline_guard')+os.pathsep+str(ROOT))
    if args.action=='test':
        command=['-m','pytest','tests','transfer_tests','sensor_tests','recall_tests','recall_recovery_tests','forget_tests','release_tests','-q',*extra]
    elif args.action=='run': command=['-m',MODULES[args.study],'--output',args.output,*extra]
    else: command=['-m','release_tools.replay','--archive',args.archive,'--receipt',args.receipt,*extra]
    raise SystemExit(subprocess.call([sys.executable,*command],cwd=ROOT,env=env))


if __name__=='__main__': main()
