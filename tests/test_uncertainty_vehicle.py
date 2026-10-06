import numpy as np
from ctrv.uncertainty import UncertaintyPolicy,propagate_covariances,tighten_bounds
from ctrv.vehicle import step_vehicle,tire_forces,VehicleConfig
from ctrv.simulation import prepare_reference
from ctrv.mpc import MPCController


def test_speed_reduces_with_uncertainty_and_friction():
 p=UncertaintyPolicy();x=np.array([0.,0.,10.,0.,0.])
 caps=[p.speed_cap(x,np.eye(5)*scale,10.,.1,1.)[0] for scale in [.001,.1,1.,10.]]
 assert all(a>=b for a,b in zip(caps,caps[1:]))
 assert p.speed_cap(x,np.zeros((5,5)),10.,.1,.1)[0]<p.speed_cap(x,np.zeros((5,5)),10.,.1,1.)[0]


def test_tightening_zero_covariance_and_monotonic_margin():
 r,A,B=prepare_reference();m=MPCController(r,A,B);upper=m.template(0)[6]
 outputs=[]
 for scale in [0.,.01,.1]:
  adjusted=upper.copy();margin,_=tighten_bounds(m,0,adjusted,np.eye(5)*scale,0.,0.,UncertaintyPolicy(),15.)
  outputs.append(margin)
  if scale==0:np.testing.assert_array_equal(adjusted,upper)
  else:assert np.all(adjusted<=upper)
 assert outputs[0]==0 and outputs[1]<outputs[2]


def test_straight_actuator_response_and_rate_limits():
 x=np.zeros(8);x[2]=5.
 y,d=step_vehicle(x,[3.,.4],.1)
 assert 0<y[6]<3 and 0<y[7]<=np.deg2rad(45)*.1+1e-8
 assert abs(y[6]-x[6])<=1.+1e-8
 assert np.all(np.isfinite(y)) and d['tire_utilization_max']<=1.+1e-10


def test_friction_circle_combined_acceleration():
 x=np.array([0.,0.,15.,0.,.3,0.,3.,.2])
 for mu in [.2,.5,1.]:
  fx,fy,util=tire_forces(x,mu)
  assert np.max(util)<=1.+1e-10
 low=tire_forces(x,.2);high=tire_forces(x,1.)
 assert np.linalg.norm(np.r_[low[0],low[1]])<np.linalg.norm(np.r_[high[0],high[1]])


def test_integration_convergence_and_standstill():
 x=np.array([0.,0.,8.,0.,0.,0.,0.,0.])
 a,_=step_vehicle(x,[1.,.1],.1,config=VehicleConfig(integration_step_s=.01))
 b,_=step_vehicle(x,[1.,.1],.1,config=VehicleConfig(integration_step_s=.005))
 np.testing.assert_allclose(a,b,atol=1e-5)
 y,_=step_vehicle(np.zeros(8),[-3.,.1],1.)
 assert y[2]>=0 and np.all(np.isfinite(y))
 np.testing.assert_array_equal(y[:2],np.zeros(2))


def test_vehicle_filter_angle_chart_and_prediction_only():
 from ctrv.vehicle_estimation import vehicle_filter_step, VehicleNoise
 from ctrv.model import wrap_to_pi
 import pytest
 x=np.zeros(8);x[2]=8.;x[3]=np.pi-.001
 P=np.diag([1.,1.,.1,.01,.001,.01,.01,.001])
 H=np.eye(8)[[0,1,3]];R=np.diag([.1,.1,.001])
 for estimator in ['ekf','ukf']:
  predicted,Ppred,info=vehicle_filter_step(x,P,[0.,0.],.1,.9,estimator=estimator)
  assert info['nis'] is None and np.linalg.eigvalsh(Ppred).min()>0
  z=H@predicted;z[2]=wrap_to_pi(z[2]+.002)
  posterior,Ppost,info=vehicle_filter_step(x,P,[0.,0.],.1,.9,z,H,R,estimator)
  assert abs(wrap_to_pi(posterior[3]-predicted[3]))<.003
  assert info['nis']<.01 and not info['gated']
  assert np.trace(Ppost)<np.trace(Ppred)
  with pytest.raises(ValueError):vehicle_filter_step(x,np.eye(5),[0.,0.],.1,.9)


def test_vehicle_linearization_predicts_local_perturbations():
 from ctrv.vehicle_estimation import vehicle_linearization, vehicle_prediction, vehicle_process_covariance
 x=np.array([0.,0.,8.,np.pi-.001,.1,.05,.4,.04]);u=np.array([.5,.06])
 nominal,A,B=vehicle_linearization(x,u,.1,.9)
 dx=np.array([.001,-.002,.001,.0001,.0001,.0002,.001,.0001]);du=np.array([.001,.0001])
 actual=vehicle_prediction(x+dx,u+du,.1,.9)
 expected=nominal+A@dx+B@du
 actual[3]=expected[3]+(actual[3]-expected[3]+np.pi)%(2*np.pi)-np.pi
 np.testing.assert_allclose(actual,expected,atol=1e-6)
 Q=vehicle_process_covariance(B,.1)
 assert np.linalg.eigvalsh(Q).min()>-1e-12
 assert Q[6,6]>0 and Q[7,7]>0


def test_vehicle_estimator_rejects_outlier_without_reading_truth():
 from ctrv.vehicle_estimation import vehicle_filter_step
 x=np.zeros(8);x[2]=5.;P=np.eye(8)*.01
 predicted,Ppred,_=vehicle_filter_step(x,P,[0.,0.],.1,.9)
 posterior,Ppost,info=vehicle_filter_step(x,P,[0.,0.],.1,.9,np.array([100.,100.]),np.eye(8)[:2],np.eye(2),gate=9.21)
 assert info['gated'] and info['nis']>9.21
 np.testing.assert_array_equal(predicted,posterior)
 np.testing.assert_array_equal(Ppred,Ppost)


def test_batched_vehicle_map_and_linearization_match_scalar():
 from ctrv.vehicle_estimation import vehicle_linearization
 states=np.array([[0.,0.,8.,.1,.1,.05,.2,.04],[1.,2.,6.,3.14,-.1,-.05,.1,-.03]])
 commands=np.array([[.4,.06],[-.2,-.04]])
 mapped,A,B=vehicle_linearization(states,commands,.1,.8)
 for i in range(2):
  scalar=vehicle_linearization(states[i],commands[i],.1,.8)
  for batch,single in zip([mapped[i],A[i],B[i]],scalar):np.testing.assert_allclose(batch,single,atol=1e-12)


def test_actuator_mpc_nominal_warm_start_and_failure():
 from ctrv.vehicle_control import VehicleMPC, equilibrium_state
 from ctrv.vehicle_estimation import VehicleNoise
 from ctrv.mpc import MPCConfig
 import pytest
 state=equilibrium_state(np.array([0.,0.,8.,np.pi-.01,0.]),0.,.9)
 X=np.repeat(state[None,:],5,axis=0)
 X[:,0]=np.arange(5)*.1*8*np.cos(state[3]);X[:,1]=np.arange(5)*.1*8*np.sin(state[3])
 U=X[:-1][:,[6,7]]
 m=VehicleMPC(config=MPCConfig(horizon=4),noise=VehicleNoise(0.,0.,0.,0.,0.))
 for _ in range(2):
  command,d=m.command(state,np.eye(8)*1e-8,X,U,U[0],policy=UncertaintyPolicy())
  assert d['solved'] and not d['fallback'] and d['slack_max']<1e-5
 assert d['warm_start']
 def fail(**kwargs):raise RuntimeError('injected solver failure')
 m.workspaces[4].solve=fail
 command,d=m.command(state,np.eye(8)*1e-8,X,U,U[0])
 assert d['fallback'] and d['status']=='solver error: RuntimeError'
 assert -5<=command[0]<=3 and abs(command[1])<=np.deg2rad(35)


def test_vehicle_governor_policies_keep_physics_and_seed_configuration():
 from ctrv.vehicle_simulation import run_vehicle_simulation
 baseline=run_vehicle_simulation(seed=7,policy='mpc',estimator='ekf',duration=.3)
 governed=run_vehicle_simulation(seed=7,policy='speed',estimator='ekf',duration=.3)
 for key in ['seed','estimator','scenario','vehicle','noise','limits','mu','patch_mu','heading_sigma_deg','duration']:
  assert baseline['config'][key]==governed['config'][key]
 assert np.mean(governed['speed_caps'])<np.mean(baseline['speed_caps'])
 assert governed['metrics']['tightening_max']==0.
 assert baseline['true_states'].shape[1]==8
 assert all(d['tire_utilization_max']<=1.+1e-10 for d in governed['vehicle_diagnostics'])


def test_actuator_mpc_slack_and_speed_constraint_activation():
 from ctrv.vehicle_control import VehicleMPC
 from ctrv.vehicle_estimation import VehicleNoise
 from ctrv.mpc import MPCConfig
 state=np.zeros(8);state[2]=8.
 X=np.repeat(state[None,:],5,axis=0);X[:,0]=np.arange(5)*.8;U=np.zeros((4,2))
 P=np.diag([25.,25.,.01,.001,.0001,.001,.001,.0001])
 m=VehicleMPC(config=MPCConfig(horizon=4),noise=VehicleNoise(0.,0.,0.,0.,0.))
 command,d=m.command(state,P,X,U,U[0],policy=UncertaintyPolicy(),speed_cap=3.)
 assert d['solved'] and d['tightening_max']>2. and d['slack_max']>1.
 assert command[0]<-.1


def test_uncertainty_policy_rejects_nonfinite_and_invalid_configuration():
 import pytest
 for kwargs in [{'position_scale_m':np.nan},{'confidence':1.},{'friction_reserve':1.1},{'tightening':'yes'}]:
  with pytest.raises(ValueError):UncertaintyPolicy(**kwargs)


def test_vehicle_timing_includes_reference_generation(monkeypatch):
 import ctrv.vehicle_simulation as simulation
 clock=[0.]
 original=simulation.rolling_reference
 def counted_reference(*args,**kwargs):
  reference=original(*args,**kwargs);clock[0]+=.5
  return reference
 monkeypatch.setattr(simulation,'perf_counter',lambda:clock[0])
 monkeypatch.setattr(simulation,'rolling_reference',counted_reference)
 result=simulation.run_vehicle_simulation(policy='mpc',estimator='ekf',duration=.2)
 assert all(d['computation_s']==.5 for d in result['controller_diagnostics'])
 assert result['metrics']['controller_time_p50_ms']==500.
 assert result['metrics']['deadline_miss_pct']==100.


def test_low_speed_vehicle_acceleration_respects_tire_force_budget():
 from ctrv.vehicle import derivative
 config=VehicleConfig()
 for speed in [.05,.2,.5,1.,2.]:
  for mu in [.1,.25,.9]:
   x=np.array([0.,0.,speed,.3,.2,.8,2.,.3])
   rates=derivative(x,[2.,.3],mu,config)
   fx=config.mass_kg*(rates[2]-x[4]*x[5])+config.drag_coefficient*speed**2
   fy=config.mass_kg*(rates[5]+x[4]*speed)
   assert np.hypot(fx,fy)<=mu*config.mass_kg*9.81+1e-8
 x=np.zeros(8);x[6]=-3.;x[7]=.3
 fx,fy,_=tire_forces(x,.25)
 np.testing.assert_array_equal(fx,np.zeros(2));np.testing.assert_array_equal(fy,np.zeros(2))


def test_vehicle_envelopes_use_ground_speed_and_physical_body_acceleration():
 from ctrv.vehicle import derivative
 from ctrv.vehicle_control import envelope,envelope_gradients
 x=np.array([0.,0.,5.,.2,.4,2.,.5,.1]);command=np.array([.5,.1])
 rates=derivative(x,command,.5)
 values,G=envelope_gradients(x,mu=.5)
 assert values[0]==np.hypot(5.,2.)
 np.testing.assert_allclose(G[0,[2,5]],x[[2,5]]/np.hypot(5.,2.))
 assert values[4]==rates[2]-x[4]*x[5]
 y,d=step_vehicle(x,command,.1,mu=.5)
 post_rates=derivative(y,command,.5)
 assert d['actual_acceleration']==post_rates[2]-y[4]*y[5]
 assert d['actual_lateral_acceleration']==post_rates[5]+y[4]*y[2]


def test_vehicle_time_grid_does_not_add_a_float_roundoff_step():
 from ctrv.vehicle_simulation import run_vehicle_simulation
 result=run_vehicle_simulation(policy='mpc',estimator='ekf',duration=.1*3)
 assert len(result['commands'])==3
 assert result['times'][-1]==.1*3


def test_vehicle_reference_equilibria_preserve_path_velocity_on_dry_and_wet_surfaces():
 from ctrv.vehicle import derivative
 from ctrv.vehicle_control import build_vehicle_path
 reference,_,_=prepare_reference()
 for mu in [.9,.45]:
  path=build_vehicle_path(reference,mu)
  states=path['states']
  np.testing.assert_allclose(np.hypot(states[:,2],states[:,5]),reference['states'][:,2],atol=1e-8)
  rates=derivative(states,states[:,[6,7]],mu)
  np.testing.assert_allclose(rates[:,[2,4,5]],0.,atol=1e-6)
  expected=reference['states'][:,2,None]*np.column_stack([np.cos(reference['states'][:,3]),np.sin(reference['states'][:,3])])
  np.testing.assert_allclose(rates[:,:2],expected,atol=1e-8)


def test_covariance_reconstruction_at_large_uncertainty():
 from ctrv.vehicle_estimation import stabilize_covariance, validate_vehicle_covariance
 # Recorded covariance from noise-mismatch EKF seed 18.
 P=np.array([[490528414871.0012, -424861143196.4153, -6.461288839333549e-16, -466798586417.1694, -5676222223321.272, -7890044255482.75, 2.5522413197129236e-14, -2.5352063245587106e-05], [-424861143196.41534, 367984780342.71783, -1.7728326109102845e-16, 404308038138.4892, 4916343660685.885, 6833800286852.889, -2.1372855546954937e-14, 2.1958172904572234e-05], [-6.461288839333548e-16, -1.7728326109102848e-16, 0.002000000000000675, -5.658448308540139e-16, -4.957468589798939e-16, 3.5950129113981486e-16, 7.734579649853698e-17, -5.472505862617765e-19], [-466798586417.16943, 404308038138.48926, -5.658448308540139e-16, 444216713396.915, 5401628997861.403, 7508355058707.879, -2.408097037276691e-14, 2.412565352084271e-05], [-5676222223321.272, 4916343660685.885, -4.95746858979894e-16, 5401628997861.403, 65683246377263.164, 91300815995386.31, -2.879759965793008e-13, 0.0002933695285390097], [-7890044255482.751, 6833800286852.889, 3.5950129113981486e-16, 7508355058707.879, 91300815995386.33, 126909668160204.05, -4.011361586551037e-13, 0.00040778857165629665], [2.5522413197129236e-14, -2.1372855546954937e-14, 7.734579649853698e-17, -2.408097037276691e-14, -2.879759965793008e-13, -4.011361586551037e-13, 3.6027837121655386, 1.4131040163579038e-16], [-2.535206324558711e-05, 2.1958172904572234e-05, -5.472505862617765e-19, 2.412565352084271e-05, 0.0002933695285390097, 0.00040778857165629665, 1.4131040163579038e-16, 4.966534709859452e-05]])
 repaired=stabilize_covariance(P)
 validate_vehicle_covariance(repaired)
 assert np.array_equal(repaired,repaired.T)
 assert np.linalg.eigvalsh(repaired).min()>0
 assert np.isclose(np.linalg.eigvalsh(repaired).max(),np.linalg.eigvalsh((P+P.T)/2).max(),rtol=1e-12)


def test_vehicle_tightening_excludes_physically_enforced_actuator_limits():
 from ctrv.vehicle_control import VehicleMPC
 from ctrv.mpc import MPCConfig
 state=np.zeros(8);state[2]=8.
 X=np.repeat(state[None,:],5,axis=0);X[:,0]=np.arange(5)*.8;U=np.zeros((4,2))
 controller=VehicleMPC(config=MPCConfig(horizon=4))
 _,d=controller.command(state,np.eye(8)*.01,X,U,U[0],policy=UncertaintyPolicy())
 margins=d['tightening_by_quantity']
 assert margins[3]==0. and margins[6]==0.
 assert margins[0]>0. and margins[7]>0.
