"""Durable paired ablations for uncertainty, actuator dynamics and friction."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_benchmarks import atomic_json, provenance, fingerprint, aggregate, interval
from ctrv.simulation import prepare_reference
from ctrv.vehicle_control import build_vehicle_path, Q_VEHICLE, R_VEHICLE
from ctrv.mpc import MPCConfig
from ctrv.vehicle_simulation import run_vehicle_simulation, POLICIES
from ctrv.vehicle import VehicleConfig
from ctrv.vehicle_estimation import VehicleNoise
from ctrv.limits import DEFAULT_LIMITS
from dataclasses import asdict, replace

CASES={'nominal':('nominal',.9,None),'gps_loss_5s':('gps_loss_5s',.9,None),
       'gps_loss_10s':('gps_loss_10s',.9,None),'wet':('nominal',.45,None),
       'friction_patch':('nominal',.9,.25),'noise_mismatch':('noise_mismatch',.9,None),
       'deterministic_recovery':('deterministic_recovery',.9,None),
       'gps_bias':('gps_bias',.9,None),'gps_outliers':('gps_outliers',.9,None),
       'actuator_lag':('nominal',.9,None)}
_PREPARED={}


def run_group(job):
    name,estimator,policy,seeds,heading,duration,output,save_traces,digest=job
    sensor,mu,patch=CASES[name]
    rows=[]
    for seed in seeds:
        identity=f'{name}_{estimator}_{policy}_{seed}';root=Path(output)
        checkpoint=root/'checkpoints'/f'{identity}.json';trace=root/'traces'/f'{identity}.npz'
        if checkpoint.exists():
            payload=json.loads(checkpoint.read_text())
            if payload['fingerprint']!=digest:raise ValueError('Vehicle checkpoint fingerprint mismatch')
            if save_traces and payload['row']['status']=='ok' and not trace.exists():raise ValueError('Vehicle trace missing')
            rows.append(payload['row']);continue
        row={'scenario':name,'estimator':estimator,'controller':policy,'seed':seed}
        try:
            if mu not in _PREPARED:
                ref,_,_=prepare_reference();_PREPARED[mu]=(ref,build_vehicle_path(ref,mu))
            vehicle=replace(VehicleConfig(),acceleration_tau_s=.5,steering_tau_s=.3) if name=='actuator_lag' else VehicleConfig()
            result=run_vehicle_simulation(vehicle_config=vehicle,seed=seed,estimator=estimator,policy=policy,scenario=sensor,
                        mu=mu,patch_mu=patch,heading_sigma_deg=heading,duration=duration,prepared=_PREPARED[mu])
            for value in result['metrics'].values():
                if value is not None and not np.isfinite(value):raise ValueError('Nonfinite vehicle metric')
            row.update(status='ok',**result['metrics']);row['config']=result['config']
            row['solver_status_counts']={status:sum(d['status']==status for d in result['controller_diagnostics'])
                    for status in sorted({d['status'] for d in result['controller_diagnostics']})}
            if save_traces:
                trace.parent.mkdir(parents=True,exist_ok=True);temporary=trace.with_suffix('.npz.tmp')
                diagnostics=result['controller_diagnostics']
                arrays={key:result[key] for key in ['times','true_states','estimated_states','covariances','commands','applied_commands','speed_caps','estimated_progress']}
                arrays.update({key:np.array([d[key] if d[key] is not None else np.nan for d in diagnostics])
                    for key in ['iterations','primal_residual','dual_residual','fallback','slack_max','slack_mean','tightening_max','computation_s']})
                arrays['friction_mu']=np.array([d['friction_mu'] for d in result['vehicle_diagnostics']])
                arrays['tire_utilization']=np.array([d['tire_utilization_max'] for d in result['vehicle_diagnostics']])
                with temporary.open('wb') as f:
                    np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
                os.replace(temporary,trace)
        except Exception as exc:
            row.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        atomic_json(checkpoint,{'fingerprint':digest,'row':row});rows.append(row)
    return rows


def paired_ablations(rows):
    rng=np.random.default_rng(2026);results=[]
    for scenario,estimator in sorted({(r['scenario'],r['estimator']) for r in rows}):
        lookup={(r['controller'],r['seed']):r for r in rows if (r['scenario'],r['estimator'])==(scenario,estimator)}
        for policy in ['tightening','speed','uncertainty']:
            seeds=sorted(seed for controller,seed in lookup if controller==policy and ('mpc',seed) in lookup)
            if not seeds:continue
            pair={'scenario':scenario,'estimator':estimator,'difference':policy+' minus mpc','matched_seeds':len(seeds)}
            for key in ['cross_track_rmse_m','physical_limit_violation_pct','completed','success','completion_time_s','solver_fallback_pct']:
                values=[]
                for seed in seeds:
                    base,new=lookup['mpc',seed],lookup[policy,seed]
                    if key in ('completed','success'):values.append(float(new.get(key,False))-float(base.get(key,False)))
                    elif base['status']=='ok' and new['status']=='ok' and base.get(key) is not None and new.get(key) is not None:
                        values.append(new[key]-base[key])
                pair[key+'_valid_pairs']=len(values)
                pair[key+'_difference_mean']=float(np.mean(values)) if values else None
                pair[key+'_ci95']=interval(values,rng)
            results.append(pair)
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,default=20);parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--scenarios',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--estimators',nargs='+',choices=['ekf','ukf'],default=['ekf','ukf'])
    parser.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES))
    parser.add_argument('--duration',type=float);parser.add_argument('--no-heading',action='store_true')
    parser.add_argument('--save-traces',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    if args.seeds<1 or args.workers<1:parser.error('Positive seeds and workers required')
    config={'seeds':list(range(args.seeds)),'scenarios':args.scenarios,'estimators':args.estimators,
            'policies':args.policies,'workers':args.workers,'duration':args.duration,'heading_sigma_deg':None if args.no_heading else 5.,
            'save_traces':args.save_traces,'cases':CASES,'vehicle':asdict(VehicleConfig()),
            'noise':asdict(VehicleNoise()),'limits':asdict(DEFAULT_LIMITS),
            'mpc':asdict(MPCConfig(rate_weights=(.5,50.))),'Q':Q_VEHICLE.tolist(),'R':R_VEHICLE.tolist(),
            'policy_configurations':{name:asdict(value) if value is not None else None for name,value in POLICIES.items()},
            'dt':.1,'track_half_width':5.,'speed_measure':'ground-speed magnitude',
            'acceleration_measure':'body-frame physical acceleration','default_time_budget_multiplier':2.,
            'case_vehicle_overrides':{'actuator_lag':{'acceleration_tau_s':.5,'steering_tau_s':.3}}}
    env=provenance();digest=fingerprint(config,env);args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        if not args.resume:parser.error('Existing output requires --resume')
        if json.loads(manifest_path.read_text())['fingerprint']!=digest:parser.error('Vehicle resume source/config/dependency/seed fingerprint mismatch')
    manifest={'schema_version':1,'configuration':config,'provenance':env,'fingerprint':digest,
              'state':'running','expected_runs':args.seeds*len(args.scenarios)*len(args.estimators)*len(args.policies),
              'model_limitations':'Synthetic bicycle, fixed axle loads, actuator lag, combined tire limits; local Gaussian soft constraints; no hardware safety claim.'}
    atomic_json(manifest_path,manifest)
    jobs=[(s,e,p,config['seeds'],config['heading_sigma_deg'],args.duration,str(args.output),args.save_traces,digest)
          for s in args.scenarios for e in args.estimators for p in args.policies]
    rows=[]
    if args.workers==1:
        for job in jobs:
            rows.extend(run_group(job));print(f'{job[0]} / {job[1]} / {job[2]} complete',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_group,job):job for job in jobs}
            for future in as_completed(futures):
                job=futures[future];rows.extend(future.result());print(f'{job[0]} / {job[1]} / {job[2]} complete',flush=True)
    rows.sort(key=lambda r:(r['scenario'],r['estimator'],r['controller'],r['seed']))
    summary=aggregate(rows)
    extra=['completion_time_s','tightening_max','speed_cap_mean_mps','speed_max_mps','tire_utilization_max',
           'nees_normalized_mean','innovation_gate_pct','position_interval_coverage_pct','heading_interval_coverage_pct','speed_violation_pct','yaw_rate_violation_pct','yaw_acceleration_violation_pct','longitudinal_acceleration_violation_pct','lateral_acceleration_violation_pct',
           'steering_violation_pct','steering_rate_violation_pct','track_corridor_violation_pct']
    for group in summary:
        valid=[r for r in rows if r['status']=='ok' and all(r[k]==group[k] for k in ['scenario','estimator','controller'])]
        for key in extra:
            values=[r[key] for r in valid if r.get(key) is not None]
            group[key+'_mean']=float(np.mean(values)) if values else None
    for filename,data in [('runs.json',rows),('summary.json',summary),('paired_comparisons.json',paired_ablations(rows))]:atomic_json(args.output/filename,data)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/'runs.csv.tmp').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        f.flush();os.fsync(f.fileno())
    os.replace(args.output/'runs.csv.tmp',args.output/'runs.csv')
    report=['# Uncertainty and synthetic vehicle ablations','',
            'Policies share vehicle parameters, sensor draws, command disturbances and friction schedule. Historical CTRV results remain a separate baseline.','',
            '| Scenario | Estimator | Policy | RMSE (m) | Completion | Violations (% steps) | Fallback (% steps) |',
            '|---|---|---|---:|---:|---:|---:|']
    for group in summary:
        def number(key):
            value=group.get(key);return f'{value:.2f}' if value is not None else 'N/A'
        report.append(f"| {group['scenario']} | {group['estimator']} | {group['controller']} | {number('cross_track_rmse_m_mean')} | {100*group['completed_rate']:.0f}% | {number('physical_limit_violation_pct_mean')} | {number('solver_fallback_pct_mean')} |")
    report.extend(['','Paired seed-bootstrap 95% intervals are in paired_comparisons.json. Completion-time comparisons include only pairs where both completed; valid-pair counts are explicit.',
                   'Per-component physical violations, slack, covariance consistency and timing remain visible. Confidence margins are uncalibrated local Gaussian surrogates; softened constraints do not establish chance feasibility.',
                   'Friction is declared available equally to all policies. A low-friction patch occurs at 15–25 seconds. Fixed normal loads, synthetic tire parameters and low-speed slip regularization limit physical fidelity.',
                   'The shared time budget is twice the historical reference duration; completion time exposes the cost of slowing. Timing is descriptive for this machine.'])
    (args.output/'report.md').write_text('\n'.join(report)+'\n')
    failures=sum(r['status']!='ok' for r in rows)
    end_fingerprint=fingerprint(config,provenance())
    unchanged=end_fingerprint==digest
    manifest.update(state='complete' if failures==0 and len(rows)==manifest['expected_runs'] and unchanged else 'failed',completed_runs=len(rows),execution_failures=failures,
                    end_fingerprint=end_fingerprint,source_and_environment_unchanged=unchanged)
    atomic_json(manifest_path,manifest)
    return 0 if manifest['state']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
