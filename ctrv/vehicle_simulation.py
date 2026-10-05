"""Paired synthetic vehicle experiments, independent of historical CTRV runs.

All policies share GPS/heading draws, command disturbances, friction schedule,
path geometry, vehicle parameters and time budget. Path progress and speed
are computed from estimates only; true states are used solely for measurements
and evaluation. The surface coefficient is declared available to all policies.
"""
from dataclasses import asdict
from time import perf_counter
import numpy as np
from scipy.stats import chi2, norm
from ctrv.constants import DT, P_INIT
from ctrv.model import wrap_to_pi
from ctrv.simulation import SCENARIOS, prepare_reference, path_errors
from ctrv.limits import DEFAULT_LIMITS
from ctrv.vehicle import VehicleConfig, step_vehicle
from ctrv.vehicle_estimation import VehicleNoise,vehicle_filter_step,validate_vehicle_covariance
from ctrv.vehicle_control import build_vehicle_path,rolling_reference,VehicleMPC
from ctrv.uncertainty import UncertaintyPolicy

POLICIES={'mpc':None,'tightening':UncertaintyPolicy(speed_control=False),
          'speed':UncertaintyPolicy(tightening=False),'uncertainty':UncertaintyPolicy()}


def run_vehicle_simulation(seed=42,estimator='ukf',policy='uncertainty',scenario='nominal',
                           vehicle_config=VehicleConfig(),noise=VehicleNoise(),limits=DEFAULT_LIMITS,
                           mu=.9,patch_mu=None,heading_sigma_deg=5.,duration=None,prepared=None):
    if policy not in POLICIES or estimator not in ('ekf','ukf'):
        raise ValueError('Unknown vehicle policy or estimator')
    if not np.isfinite(mu) or mu<=0 or (patch_mu is not None and (not np.isfinite(patch_mu) or patch_mu<=0)):
        raise ValueError('Positive finite surface friction required')
    if abs(limits.wheelbase-vehicle_config.front_distance_m-vehicle_config.rear_distance_m)>1e-9:
        raise ValueError('Vehicle axle spacing must match configured wheelbase')
    scenario=SCENARIOS[scenario] if isinstance(scenario,str) else scenario
    if prepared is None:
        reference,_,_=prepare_reference(limits=limits)
        path=build_vehicle_path(reference,mu,vehicle_config,limits)
    else:reference,path=prepared
    duration=2*reference['times'][-1] if duration is None else duration
    if not np.isfinite(duration) or duration<=0:raise ValueError('Positive finite duration required')
    steps=max(1,int(np.ceil(duration/DT-1e-12)));times=np.arange(steps+1)*DT
    streams=np.random.SeedSequence(seed).spawn(3)
    command_draws=np.random.default_rng(streams[0]).standard_normal((steps,2))
    sensor_draws=np.random.default_rng(streams[1]).standard_normal((steps,3))
    faults=np.random.default_rng(streams[2]);outliers=faults.random(steps)<scenario.gps_outlier_probability
    outlier_draws=faults.standard_normal((steps,2))*scenario.gps_outlier_sigma
    true=np.empty((steps+1,8));estimated=np.empty_like(true);covariance=np.empty((steps+1,8,8))
    true[0]=path['states'][0]
    estimated[0]=path['states'][0]+np.array([2.,2.,.5,.1,.05,.05,0.,0.])
    estimated[0,3]=wrap_to_pi(estimated[0,3])
    covariance[0]=np.zeros((8,8));covariance[0,:5,:5]=P_INIT
    covariance[0,5:,5:]=np.diag([.25,.25,np.deg2rad(2.)**2])
    controller=VehicleMPC(vehicle_config=vehicle_config,limits=limits,noise=noise)
    selected=POLICIES[policy];commands=[];applied_commands=[];diagnostics=[];physics=[];caps=[];progresses=[];innovations=[]
    progress=0.;index=0;previous=path['states'][0,[6,7]];completed_at=None
    for k in range(steps):
        tick=perf_counter()
        surface=patch_mu if patch_mu is not None and 15<=times[k]<25 else mu
        # Local nearest-path search avoids distant branch jumps. Projection
        # is monotonic, while lateral error remains in the tracking objective.
        end=min(len(path['arc']),index+40)
        distance=np.sum((path['states'][index:end,:2]-estimated[k,:2])**2,axis=1)
        index+=int(np.argmin(distance));progress=max(progress,float(path['arc'][index]))
        nominal=float(path['states'][index,2]);cap=limits.max_speed
        if selected is not None and selected.speed_control:
            cap,_,_=selected.speed_cap(estimated[k],covariance[k],nominal,path['curvature'][index],surface)
        X,U=rolling_reference(path,progress,cap,controller.config.horizon,limits,vehicle_config,surface)
        command,diagnostic=controller.command(estimated[k],covariance[k],X,U,previous,surface,selected,
                                              speed_cap=cap if selected is not None and selected.speed_control else None)
        diagnostic['computation_s']=perf_counter()-tick
        previous=command;commands.append(command);diagnostics.append(diagnostic);caps.append(cap);progresses.append(progress)
        perturbation=np.sqrt(np.diag(noise.command_covariance(DT))*scenario.plant_noise_scale)*command_draws[k]
        applied=np.clip(command+perturbation,[-limits.max_braking,-limits.max_steering],[limits.max_accel,limits.max_steering])
        applied_commands.append(applied)
        true[k+1],physical=step_vehicle(true[k],applied,DT,surface,vehicle_config,limits)
        physics.append(physical)
        observation_time=times[k+1]
        gps=not scenario.gps_outage_start<=observation_time<scenario.gps_outage_start+scenario.gps_outage_duration
        rows=[];z=[];variances=[]
        if gps:
            observation=true[k+1,:2]+sensor_draws[k,:2]*np.sqrt(scenario.sensor_noise_scale)+[scenario.gps_bias_x,scenario.gps_bias_y]
            if outliers[k]:observation+=outlier_draws[k]
            rows.extend(np.eye(8)[:2]);z.extend(observation);variances.extend([1.,1.])
        if heading_sigma_deg is not None:
            sigma=np.deg2rad(heading_sigma_deg)
            rows.append(np.eye(8)[3]);z.append(wrap_to_pi(true[k+1,3]+sigma*sensor_draws[k,2]*np.sqrt(scenario.sensor_noise_scale)))
            variances.append(sigma**2)
        H=np.array(rows).reshape(-1,8)
        estimated[k+1],covariance[k+1],info=vehicle_filter_step(estimated[k],covariance[k],command,DT,surface,
            np.array(z) if rows else None,H,np.diag(variances),estimator,noise,vehicle_config,limits,
            gate=float(chi2.ppf(.99,len(rows))) if rows else None)
        innovations.append(info)
        # Completion is evaluated from truth; it does not steer the controller.
        if np.linalg.norm(true[k+1,:2]-path['states'][-1,:2])<=5.:
            _,actual_progress,_=path_errors(true[k+1:k+2,:2],reference['states'])
            if actual_progress[0]>=.98:
                completed_at=observation_time
                break
    count=len(commands);times=times[:count+1];true=true[:count+1];estimated=estimated[:count+1];covariance=covariance[:count+1]
    commands=np.array(commands);caps=np.array(caps)
    cross,actual_progress,_=path_errors(true[1:,:2],reference['states'])
    lateral=np.abs([d['actual_lateral_acceleration'] for d in physics]);speed=np.hypot(true[1:,2],true[1:,5])
    components={'speed':speed>limits.max_speed+1e-7,
                'yaw_rate':np.abs(true[1:,4])>limits.max_yaw_rate+1e-7,
                'yaw_acceleration':np.abs([d['actual_yaw_acceleration'] for d in physics])>limits.max_yaw_accel+1e-7,
                'longitudinal_acceleration':(np.array([d['actual_acceleration'] for d in physics])>limits.max_accel+1e-7)|(np.array([d['actual_acceleration'] for d in physics]) < -limits.max_braking-1e-7),
                'lateral_acceleration':lateral>limits.max_lateral_accel+1e-7,
                'steering':np.abs(true[1:,7])>limits.max_steering+1e-7,
                'steering_rate':np.abs(np.diff(true[:,7]))/DT>limits.max_steering_rate+1e-7,
                'track_corridor':np.abs(cross)>5.}
    union=np.any(np.array(list(components.values())),axis=0)
    error=true[1:]-estimated[1:];error[:,3]=wrap_to_pi(error[:,3])
    nees=np.einsum('ij,ij->i',error,np.linalg.solve(covariance[1:],error[...,None])[...,0])
    timing=np.array([d['computation_s'] for d in diagnostics]);slacks=[d['slack_max'] for d in diagnostics if d['slack_max'] is not None]
    slack_means=[d['slack_mean'] for d in diagnostics if d['slack_mean'] is not None]
    innovation_values=[d['nis']/d['dof'] for d in innovations if d['nis'] is not None and d['dof']>0]
    angles=estimated[1:,3]+np.arctan2(estimated[1:,5],np.maximum(estimated[1:,2],.5))
    normals=np.column_stack([-np.sin(angles),np.cos(angles)])
    position_variance=np.einsum('ij,ijk,ik->i',normals,covariance[1:,:2,:2],normals)
    position_error=np.abs(np.einsum('ij,ij->i',normals,error[:,:2]))
    quantile=norm.ppf(.99)
    metrics={'cross_track_rmse_m':float(np.sqrt(np.mean(cross**2))),
             'estimation_position_rmse_m':float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
             'completed':completed_at is not None,'completion_time_s':completed_at,
             'progress_final':float(actual_progress[-1]),'endpoint_error_m':float(np.linalg.norm(true[-1,:2]-path['states'][-1,:2])),
             'physical_limit_violation_pct':float(100*union.mean()),
             'success':bool(completed_at is not None and not union.any()),
             'solver_fallback_pct':float(100*np.mean([d['fallback'] for d in diagnostics])),
             'normalized_slack_max':float(max(slacks)) if slacks else None,
             'normalized_slack_mean':float(np.mean(slack_means)) if slack_means else None,
             'tightening_max':float(max(d['tightening_max'] for d in diagnostics)),
             'speed_cap_mean_mps':float(caps.mean()),'speed_max_mps':float(speed.max()),
             'longitudinal_speed_max_mps':float(true[1:,2].max()),
             'nees_mean':float(nees.mean()),'nees_normalized_mean':float(nees.mean()/8),
             'nis_normalized_mean':float(np.mean(innovation_values)) if innovation_values else None,
             'position_interval_coverage_pct':float(100*np.mean(position_error<=quantile*np.sqrt(np.maximum(0.,position_variance)))),
             'heading_interval_coverage_pct':float(100*np.mean(np.abs(error[:,3])<=quantile*np.sqrt(covariance[1:,3,3]))),
             'innovation_gate_pct':float(100*np.mean([d['gated'] for d in innovations])),
             'tire_utilization_max':float(max(d['tire_utilization_max'] for d in physics)),
             'controller_time_p50_ms':float(1000*np.percentile(timing,50)),
             'controller_time_p95_ms':float(1000*np.percentile(timing,95)),
             'controller_time_max_ms':float(1000*timing.max()),
             'deadline_miss_pct':float(100*np.mean(timing>DT))}
    metrics.update({name+'_violation_pct':float(100*mask.mean()) for name,mask in components.items()})
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(estimated)):raise RuntimeError('Nonfinite vehicle trajectory')
    if not np.all(commands>=np.array([-limits.max_braking,-limits.max_steering])-1e-7) or not np.all(commands<=np.array([limits.max_accel,limits.max_steering])+1e-7):
        raise RuntimeError('Vehicle command exceeded hard actuator envelope')
    applied_commands=np.array(applied_commands)
    if not np.all(applied_commands>=np.array([-limits.max_braking,-limits.max_steering])-1e-7) or not np.all(applied_commands<=np.array([limits.max_accel,limits.max_steering])+1e-7):
        raise RuntimeError('Applied vehicle demand exceeded hard actuator envelope')
    for P in covariance:validate_vehicle_covariance(P)
    return {'metrics':metrics,'true_states':true,'estimated_states':estimated,'covariances':covariance,
            'commands':commands,'applied_commands':np.array(applied_commands),'times':times,'speed_caps':caps,'estimated_progress':np.array(progresses),
            'controller_diagnostics':diagnostics,'vehicle_diagnostics':physics,'innovations':innovations,
            'config':{'seed':seed,'estimator':estimator,'policy':policy,'scenario':asdict(scenario),
                      'vehicle':asdict(vehicle_config),'noise':asdict(noise),'limits':asdict(limits),
                      'mu':mu,'patch_mu':patch_mu,'heading_sigma_deg':heading_sigma_deg,'duration':duration}}
