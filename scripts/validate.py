"""Durable validation entrypoint: isolated environment, tests, benchmarks, artifacts."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','full'],default='smoke')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if sys.prefix==sys.base_prefix:
        parser.error('Use an isolated venv: python -m venv .venv; .venv/bin/python -m pip install -r requirements.lock.txt')
    for line in (ROOT/'requirements.lock.txt').read_text().splitlines():
        if '==' in line:
            name,version=line.split('==')
            if importlib.metadata.version(name)!=version:
                parser.error(f'Locked dependency mismatch: {name}; install requirements.lock.txt')
    output=args.output or ROOT/'validation_results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(MPLBACKEND='Agg',MPLCONFIGDIR=str(output/'matplotlib-cache'),
                                    PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
                                    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    stages=[]
    def run(name,command):
        log=output/f'{name}.log'
        with log.open('w') as f:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        stages.append({'stage':name,'exit_code':result.returncode,'log':str(log)})
        (output/'validation.json').write_text(json.dumps({'profile':args.profile,'state':'running','stages':stages},indent=2)+'\n')
        print(f'{name}: {"PASS" if result.returncode==0 else "FAIL"}; {log}',flush=True)
        return result.returncode==0
    ok=run('tests',[sys.executable,'-m','pytest','tests','-q','-p','no:cacheprovider',f'--junitxml={output / "junit.xml"}'])
    if ok:
        common=[sys.executable,'scripts/run_benchmarks.py','--workers',str(args.workers)]
        resume=['--resume'] if args.resume else []
        if args.profile=='smoke':
            ok=run('smoke',common+['--seeds','3','--scenarios','nominal','gps_loss_5s','gps_loss_10s',
                                 'gps_outliers','gps_bias','noise_mismatch','deterministic_recovery',
                                 '--output',str(output/'smoke')]+resume)
        else:
            ok=run('comparison',common+['--seeds','100','--output',str(output/'comparison')]+resume)
            ok=run('gps_only',common+['--seeds','20','--no-heading','--scenarios','nominal','gps_loss_5s',
                                    'gps_loss_10s','--output',str(output/'gps_only')]+resume) and ok
            ok=run('representative',common+['--seeds','1','--save-traces','--scenarios','nominal','gps_loss_5s',
                          'gps_loss_10s','gps_outliers','gps_bias','noise_mismatch','deterministic_recovery',
                          '--output',str(output/'representative')]+resume) and ok
    (output/'validation.json').write_text(json.dumps({'profile':args.profile,'state':'complete' if ok else 'failed',
                                                      'stages':stages},indent=2)+'\n')
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
