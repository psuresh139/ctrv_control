"""Resumable, paired-seed LQR/MPC benchmarks with strict durable exports."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ctrv.simulation import SCENARIOS, prepare_reference, run_simulation
from ctrv.mpc import MPCConfig
from ctrv.limits import DEFAULT_LIMITS

BASE_SCENARIOS=['nominal','gps_loss_5s','gps_loss_10s','gps_outliers','gps_bias','noise_mismatch']


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def provenance():
    digest=hashlib.sha256()
    paths=sorted([*ROOT.glob('ctrv/*.py'),*ROOT.glob('scripts/*.py'),*ROOT.glob('tests/*.py'),
                  ROOT/'requirements.lock.txt',*ROOT.glob('.github/workflows/*.yml')])
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
    def git(*args):
        r=subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,text=True)
        return r.stdout.strip() if r.returncode==0 else None
    dependencies={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}
    return {'source_sha256':digest.hexdigest(),'git_commit':git('rev-parse','HEAD'),
            'git_status':git('status','--porcelain'),'python':sys.version,'platform':platform.platform(),
            'dependencies':dict(sorted(dependencies.items())),
            'thread_settings':{name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}


def fingerprint(config,environment):
    selected={'config':config,'source_sha256':environment['source_sha256'],
              'python':environment['python'],'platform':environment['platform'],
              'dependencies':environment['dependencies'],'thread_settings':environment.get('thread_settings',{})}
    return hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()


def interval(values,rng):
    values=np.asarray(values,float)
    if len(values)<2:return None
    means=rng.choice(values,size=(2000,len(values)),replace=True).mean(axis=1)
    return np.percentile(means,[2.5,97.5]).tolist()


def aggregate(rows,bootstrap_seed=2026):
    rng=np.random.default_rng(bootstrap_seed);summary=[]
    groups=sorted({(r['scenario'],r['estimator'],r.get('controller','lqr')) for r in rows})
    keys=['cross_track_rmse_m','cross_track_p95_m','cross_track_max_m','estimation_position_rmse_m',
          'outage_estimation_rmse_m','physical_limit_violation_pct','saturation_pct','nis_normalized_mean',
          'nees_mean','solver_fallback_pct','normalized_slack_max','normalized_slack_mean',
          'controller_time_p50_ms','controller_time_p95_ms','controller_time_max_ms','deadline_miss_pct']
    for scenario,estimator,controller in groups:
        group=[r for r in rows if (r['scenario'],r['estimator'],r.get('controller','lqr'))==(scenario,estimator,controller)]
        valid=[r for r in group if r['status']=='ok']
        result=dict(scenario=scenario,estimator=estimator,controller=controller,runs=len(group),failed_runs=len(group)-len(valid))
        for flag in ('completed','stayed_on_track','success'):
            result[flag+'_rate']=sum(r.get(flag,False) for r in group)/len(group)
        for key in keys:
            values=[r[key] for r in valid if r.get(key) is not None]
            result[key+'_mean']=float(np.mean(values)) if values else None
            result[key+'_ci95']=interval(values,rng)
        summary.append(result)
    return summary


def paired_comparisons(rows):
    rng=np.random.default_rng(2026);results=[]
    for scenario,estimator in sorted({(r['scenario'],r['estimator']) for r in rows}):
        lookup={(r.get('controller','lqr'),r['seed']):r for r in rows
                if r['scenario']==scenario and r['estimator']==estimator}
        seeds=sorted({seed for controller,seed in lookup if ('lqr',seed) in lookup and ('mpc',seed) in lookup})
        result={'scenario':scenario,'estimator':estimator,'matched_seeds':len(seeds),'difference':'mpc minus lqr'}
        for key in ['cross_track_rmse_m','physical_limit_violation_pct','completed','success']:
            values=[]
            for seed in seeds:
                lqr,mpc=lookup['lqr',seed],lookup['mpc',seed]
                if key in ('completed','success'):
                    values.append(float(mpc.get(key,False))-float(lqr.get(key,False)))
                elif lqr['status']=='ok' and mpc['status']=='ok':
                    values.append(mpc[key]-lqr[key])
            result[key+'_valid_pairs']=len(values)
            result[key+'_difference_mean']=float(np.mean(values)) if values else None
            result[key+'_ci95']=interval(values,rng)
        results.append(result)
    return results


_WORK_PREPARED=None

def run_group(job):
    global _WORK_PREPARED
    if _WORK_PREPARED is None:_WORK_PREPARED=prepare_reference()
    name,estimator,controller,seeds,heading,output,save_traces,run_fingerprint=job
    rows=[]
    for seed in seeds:
        identity=f'{name}_{estimator}_{controller}_{seed}'
        checkpoint=Path(output)/'checkpoints'/f'{identity}.json'
        if checkpoint.exists():
            payload=json.loads(checkpoint.read_text())
            if payload['fingerprint']!=run_fingerprint:raise ValueError('Checkpoint fingerprint mismatch')
            row=payload['row']
            if save_traces and row['status']=='ok' and not (Path(output)/f'{identity}.npz').exists():
                raise ValueError('Checkpoint trace is missing')
            rows.append(row);continue
        row={'scenario':name,'estimator':estimator,'controller':controller,'seed':seed}
        try:
            r=run_simulation(seed=seed,estimator=estimator,controller=controller,scenario=SCENARIOS[name],
                             heading_sigma_deg=heading,prepared=_WORK_PREPARED,debug=False)
            if (not np.all(np.isfinite(r['true_states'])) or not np.all(np.isfinite(r['est_states']))
                    or not np.all(np.isfinite(r['covariances'])) or np.min(np.linalg.eigvalsh(r['covariances']))<=0):
                raise ValueError('Invalid state or covariance')
            controls=r['controls']; L=DEFAULT_LIMITS
            if np.any(controls<[-L.max_braking-1e-8,-L.max_yaw_accel-1e-8]) or np.any(controls>[L.max_accel+1e-8,L.max_yaw_accel+1e-8]):
                raise ValueError('Actuator bounds violated')
            for value in r['metrics'].values():
                if value is not None and not np.isfinite(value):raise ValueError('Nonfinite metric')
            row.update(status='ok',**r['metrics'])
            row['config']=r['config']
            row['solver_status_counts']={status:sum(d['status']==status for d in r['controller_diagnostics'])
                                         for status in sorted({d['status'] for d in r['controller_diagnostics']})}
            if save_traces:
                target=Path(output)/f'{identity}.npz';tmp=target.with_suffix('.npz.tmp')
                diagnostics=r['controller_diagnostics']
                with tmp.open('wb') as f:
                    np.savez_compressed(f,**{k:r[k] for k in ('times','true_states','est_states','ref_states','controls',
                                        'covariances','nis_log','nis_dof','gps_available','gate_log','limit_violations')},
                                        solver_status=np.array([d['status'] for d in diagnostics]),
                                        solver_iterations=np.array([d['iterations'] for d in diagnostics]),
                                        solver_primal_residual=np.array([d['primal_residual'] if d['primal_residual'] is not None else np.nan for d in diagnostics]),
                                        solver_dual_residual=np.array([d['dual_residual'] if d['dual_residual'] is not None else np.nan for d in diagnostics]),
                                        solver_slack_mean=np.array([d['slack_mean'] if d['slack_mean'] is not None else np.nan for d in diagnostics]),
                                        solver_fallback=np.array([d['fallback'] for d in diagnostics]),
                                        solver_slack_max=np.array([d['slack_max'] if d['slack_max'] is not None else np.nan for d in diagnostics]),
                                        controller_time_s=np.array([d['computation_s'] for d in diagnostics]))
                    f.flush();os.fsync(f.fileno())
                os.replace(tmp,target)
        except Exception as exc:
            row.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        atomic_json(checkpoint,{'fingerprint':run_fingerprint,'row':row});rows.append(row)
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,default=100)
    parser.add_argument('--start-seed',type=int,default=0)
    parser.add_argument('--scenarios',nargs='+',choices=list(SCENARIOS),default=BASE_SCENARIOS)
    parser.add_argument('--estimators',nargs='+',choices=['ekf','ukf'],default=['ekf','ukf'])
    parser.add_argument('--controllers',nargs='+',choices=['lqr','mpc'],default=['lqr','mpc'])
    parser.add_argument('--no-heading',action='store_true')
    parser.add_argument('--save-traces',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmark_results')
    args=parser.parse_args(argv)
    if min(args.seeds,args.workers)<1 or args.start_seed<0:parser.error('Invalid seeds/workers')
    for values in (args.scenarios,args.estimators,args.controllers):
        if len(values)!=len(set(values)):parser.error('Duplicate benchmark selections are not allowed')
    args.output.mkdir(parents=True,exist_ok=True)
    environment=provenance()
    config={'seeds':list(range(args.start_seed,args.start_seed+args.seeds)),'scenarios':args.scenarios,
            'estimators':args.estimators,'controllers':args.controllers,'heading_sigma_deg':None if args.no_heading else 5.,
            'save_traces':args.save_traces,'mpc_config':asdict(MPCConfig()),'limits':asdict(DEFAULT_LIMITS)}
    run_fingerprint=fingerprint(config,environment);manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        if not args.resume:parser.error('Output exists; choose another directory or --resume')
        old=json.loads(manifest_path.read_text())
        if old['fingerprint']!=run_fingerprint:parser.error('Resume source/config/environment fingerprint mismatch')
    elif args.resume and any((args.output/'checkpoints').glob('*.json')):
        parser.error('Cannot resume checkpoints without a manifest')
    ref=prepare_reference()[0]
    manifest={'schema_version':2,'fingerprint':run_fingerprint,'provenance':environment,'configuration':config,
              'bootstrap_seed':2026,'workers':args.workers,'reference_duration_s':float(ref['times'][-1]),
              'reference_arc_length_m':float(ref['arc_length']),'state':'running',
              'success_definition':'>=98% final progress; endpoint/cross-track <=5m; zero sampled physical violations',
              'model_limitations':'Kinematic CTRV; linearized soft envelope constraints are not nonlinear safety guarantees.'}
    atomic_json(manifest_path,manifest)
    jobs=[(name,estimator,controller,config['seeds'],config['heading_sigma_deg'],str(args.output),args.save_traces,run_fingerprint)
          for name in args.scenarios for estimator in args.estimators for controller in args.controllers]
    rows=[]
    if args.workers==1:
        for job in jobs:
            rows.extend(run_group(job));print(f'{job[0]} / {job[1]} / {job[2]}: {args.seeds} seeds finished',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_group,job):job for job in jobs}
            for future in as_completed(futures):
                job=futures[future];rows.extend(future.result())
                print(f'{job[0]} / {job[1]} / {job[2]}: {args.seeds} seeds finished',flush=True)
    rows.sort(key=lambda r:(r['scenario'],r['estimator'],r['controller'],r['seed']))
    summaries=aggregate(rows);pairs=paired_comparisons(rows)
    for filename,data in [('runs.json',rows),('summary.json',summaries),('paired_comparisons.json',pairs)]:
        atomic_json(args.output/filename,data)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/'runs.csv.tmp').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    os.replace(args.output/'runs.csv.tmp',args.output/'runs.csv')
    report=['# LQR versus MPC under GPS failure','',
            f"Heading sensing: {config['heading_sigma_deg']} degrees; seeds: {args.seeds} per group.",'',
            '| Scenario | Estimator | Controller | Runs | RMSE (m) | Completion | Violations (% steps) | Fallback (% steps) |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(value):return f'{value:.2f}' if value is not None else '—'
    for s in summaries:
        report.append(f"| {s['scenario']} | {s['estimator']} | {s['controller']} | {s['runs']} | {fmt(s['cross_track_rmse_m_mean'])} | {s['completed_rate']:.0%} | {fmt(s['physical_limit_violation_pct_mean'])} | {fmt(s['solver_fallback_pct_mean'])} |")
    report.extend(['','Paired MPC-minus-LQR differences and seed-bootstrap 95% intervals are in paired_comparisons.json.',
                   'Completion is separate from success. Solver fallback and physical violations are reported without exclusion.',
                   'Soft predicted constraints and state-dependent Gaussian disturbances do not guarantee actual feasibility.',
                   'Controller timings include QP assembly/update/solve; template construction contributes on its first use.',
                   'Cross-platform timing and bitwise reproducibility are not promised. NIS includes rejected observations.',
                   'Checkpoint/source/dependency fingerprints enforce compatible resumes; failed runs remain in denominators.'])
    (args.output/'report.md').write_text('\n'.join(report)+'\n')
    failures=sum(r['status']!='ok' for r in rows)
    manifest.update(state='failed' if failures else 'complete',runs=len(rows),execution_failures=failures)
    atomic_json(manifest_path,manifest)
    return int(failures>0)


if __name__=='__main__':raise SystemExit(main())
