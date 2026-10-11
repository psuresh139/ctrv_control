"""Conditional command uncertainty calibration on saved development traces.

Truth is used offline to hold the starting state fixed. This does not measure
full estimator or closed-loop chance-constraint calibration. Model diffusion is
excluded to isolate command disturbances; experiment noise is not retuned.
"""
import argparse,json,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ctrv.vehicle import VehicleConfig,step_vehicle
from ctrv.vehicle_estimation import VehicleNoise,vehicle_linearization,vehicle_process_covariance
from ctrv.vehicle_control import envelope_gradients
from ctrv.bounded_uncertainty import command_moments,acceleration_quantiles
from ctrv.limits import DEFAULT_LIMITS
from scripts.run_benchmarks import atomic_json,provenance


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--samples',type=int,default=4096)
 a=p.parse_args(argv)
 if a.samples<1000:p.error('At least 1000 samples required')
 rng=np.random.default_rng(2026);records=[]
 for scenario in ['nominal','gps_loss_10s']:
  for seed in a.seeds:
   identity=f'{scenario}_ukf_uncertainty_{seed}'
   checkpoint=a.traces.parent/'checkpoints'/f'{identity}.json'
   config=json.loads(checkpoint.read_text())['row']['config']
   vehicle=VehicleConfig(**config['vehicle'])
   noise=replace(VehicleNoise(**config['noise']),longitudinal_model_psd=0.,lateral_model_psd=0.,yaw_model_psd=0.)
   with np.load(a.traces/f'{identity}.npz',allow_pickle=False) as d:
    for time in [10.,30.,60.]:
     k=min(int(time/.1),len(d['commands'])-1);x=d['true_states'][k];u=d['commands'][k];mu=float(d['friction_mu'][k])
     commands=u+rng.normal(size=(a.samples,2))*np.sqrt(np.diag(noise.command_covariance(.1)))
     states,physics=step_vehicle(np.broadcast_to(x,(a.samples,8)),commands,.1,mu,vehicle,DEFAULT_LIMITS)
     nominal,A,B=vehicle_linearization(x,u,.1,mu,vehicle,DEFAULT_LIMITS)
     oldQ=vehicle_process_covariance(B,.1,noise);base,G=envelope_gradients(nominal,DEFAULT_LIMITS,mu,vehicle)
     physical=physics['actual_acceleration'];sigma=np.sqrt(max(0.,G[4]@oldQ@G[4]))
     from scipy.stats import norm
     old_interval=base[4]+np.array([-1,1])*norm.ppf(.99)*sigma
     row=dict(scenario=scenario,seed=seed,time_s=float(d['times'][k]),sampled_actuator_mean=float(states[:,6].mean()),sampled_actuator_variance=float(states[:,6].var()),sampled_acceleration_quantiles=np.quantile(physical,[.01,.99]).tolist(),linearized_acceleration_coverage_pct=float(100*np.mean((physical>=old_interval[0])&(physical<=old_interval[1]))),orders=[],resolutions=[])
     for order in [5,9,17]:
      shift,Q=command_moments(x,u,.1,mu,vehicle,DEFAULT_LIMITS,noise,order=order)
      row['orders'].append(dict(order=order,actuator_mean=float(nominal[6]+shift[6]),actuator_variance=float(Q[6,6])))
      if order==9:
       for power in [8,9,11]:
        lo,hi=acceleration_quantiles(nominal+shift,Q,.99,mu,vehicle,DEFAULT_LIMITS,sample_power=power)
        row['resolutions'].append(dict(samples=2**power,quantile_low=float(lo),quantile_high=float(hi),coverage_pct=float(100*np.mean((physical>=lo)&(physical<=hi))),lower_tail_pct=float(100*np.mean(physical<lo)),upper_tail_pct=float(100*np.mean(physical>hi))))
     records.append(row)
 a.output.mkdir(parents=True,exist_ok=True)
 atomic_json(a.output/'calibration.json',dict(scope='Conditional command-only development snapshots, not joint state/noise or closed-loop confidence validation',provenance=provenance(),sample_seed=2026,samples_per_snapshot=a.samples,nominal_central_coverage_pct=98.,records=records))
 lines=['# Conditional uncertainty calibration','', 'Starting states are fixed from saved development traces; true states are used only offline. Command noise levels and physical limits are retained. State-estimation uncertainty and artificial model diffusion are excluded from this conditional check. The nominal central interval is 98%, from 0.01 and 0.99 quantiles.','', '| Snapshot | Linearized coverage % | 256-point coverage % | 512-point coverage % | 2048-point coverage % |','|---|---:|---:|---:|---:|']
 for r in records:lines.append(f"| {r['scenario']} seed {r['seed']} t={r['time_s']:.1f} | {r['linearized_acceleration_coverage_pct']:.2f} | "+' | '.join(f"{v['coverage_pct']:.2f}" for v in r['resolutions'])+' |')
 lines+=['','Mean empirical coverage at 512 points: '+f"{np.mean([r['resolutions'][1]['coverage_pct'] for r in records]):.2f}%.",'','Order 5/9/17 moment estimates, asymmetric tail frequencies, sampled physical quantiles and 256/512/2048 resolution sensitivity are in calibration.json. High or low coverage is evidence of miscalibration, not a guarantee. These conditional checks cannot validate the projected Gaussian estimator posterior or reference-local optimized constraints. No parameter is selected from these results to favor confirmation.']
 (a.output/'report.md').write_text('\n'.join(lines)+'\n')
 print(lines[-3]);return 0

if __name__=='__main__':raise SystemExit(main())
