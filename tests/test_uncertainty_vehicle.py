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
