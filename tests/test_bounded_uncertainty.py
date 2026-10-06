import numpy as np
from ctrv.bounded_uncertainty import command_moments,acceleration_quantiles
from ctrv.vehicle import VehicleConfig,step_vehicle
from ctrv.vehicle_estimation import VehicleNoise,vehicle_linearization,vehicle_process_covariance,vehicle_filter_step
from ctrv.limits import DEFAULT_LIMITS


def test_bounded_command_moments_against_sampled_saturation():
 x=np.zeros(8);x[2]=8.;u=np.array([2.8,.05]);noise=VehicleNoise(2,.00005,0,0,0)
 shift,Q=command_moments(x,u,.1,.9,VehicleConfig(),DEFAULT_LIMITS,noise)
 rng=np.random.default_rng(1)
 commands=u+rng.normal(size=(20000,2))*np.sqrt(np.diag(noise.command_covariance(.1)))
 samples,_=step_vehicle(np.broadcast_to(x,(len(commands),8)),commands,.1,.9)
 nominal,_,B=vehicle_linearization(x,u,.1,.9)
 assert shift[6]<-.3
 assert abs(shift[6]-(samples-nominal).mean(axis=0)[6])<.06
 assert abs(Q[6,6]/np.var(samples[:,6])-1)<.12
 assert Q[6,6]<vehicle_process_covariance(B,.1,noise)[6,6]
 assert np.linalg.eigvalsh(Q).min()>-1e-10


def test_bounded_moments_zero_noise_and_batch_equivalence():
 x=np.zeros(8);x[2]=8.;u=np.array([.5,.01]);noise=VehicleNoise(0,0,0,0,0)
 shift,Q=command_moments(x,u,.1,.9,VehicleConfig(),DEFAULT_LIMITS,noise)
 assert np.allclose(shift,0.,atol=1e-12) and np.allclose(Q,0.,atol=1e-12)
 S,C=command_moments(np.array([x,x]),np.array([u,u]),.1,.9,VehicleConfig(),DEFAULT_LIMITS,VehicleNoise())
 s,c=command_moments(x,u,.1,.9,VehicleConfig(),DEFAULT_LIMITS,VehicleNoise())
 assert np.allclose(S,[s,s]) and np.allclose(C,[c,c])


def test_acceleration_quantiles_use_bounded_actuator_and_physical_force():
 x=np.zeros(8);x[2]=8.;P=np.zeros((8,8));P[6,6]=100.
 lo,hi=acceleration_quantiles(x,P,.99,.9,VehicleConfig(),DEFAULT_LIMITS)
 drag=VehicleConfig().drag_coefficient*64/VehicleConfig().mass_kg
 assert np.isclose(lo,-5.-drag) and np.isclose(hi,3.-drag)
 assert np.array_equal([lo,hi],acceleration_quantiles(x,P,.99,.9,VehicleConfig(),DEFAULT_LIMITS))
 zero=acceleration_quantiles(x,np.zeros((8,8)),.99,.9,VehicleConfig(),DEFAULT_LIMITS)
 assert np.allclose(zero,-drag)


def test_filter_includes_clipping_mean_shift_and_keeps_baseline():
 x=np.zeros(8);x[2]=8.;u=np.array([2.8,.05]);P=np.eye(8)*1e-5
 for estimator in ['ekf','ukf']:
  old,Po,_=vehicle_filter_step(x,P,u,.1,.9,estimator=estimator,prediction_model='linearized')
  new,Pn,_=vehicle_filter_step(x,P,u,.1,.9,estimator=estimator,prediction_model='bounded')
  assert new[6]<old[6]-.3 and Pn[6,6]<Po[6,6]
  assert np.linalg.eigvalsh(Pn).min()>0.


def test_acceleration_interval_diagnostic_retains_confidence_and_baseline():
 from ctrv.vehicle_control import VehicleMPC
 from ctrv.mpc import MPCConfig
 from ctrv.uncertainty import UncertaintyPolicy
 x=np.zeros(8);x[2]=8.
 X=np.repeat(x[None,:],5,axis=0);X[:,0]=np.arange(5)*.8;U=np.zeros((4,2))
 P=np.eye(8)*1e-8;P[6,6]=100.
 results=[]
 for model in ['linearized','bounded']:
  m=VehicleMPC(config=MPCConfig(horizon=4),noise=VehicleNoise(0,0,0,0,0),prediction_model=model)
  command,d=m.command(x,P,X,U,U[0],policy=UncertaintyPolicy())
  assert np.isfinite(command).all() and len(d['acceleration_quantiles'])==4
  if model=='bounded':
   from ctrv.vehicle_control import envelope
   value=envelope(x)[4]
   expected=max(max(high-value,value-low)/5. for low,high in d['acceleration_quantiles'])
   assert np.isclose(d['tightening_by_quantity'][4],expected)
  results.append(d['unattainable_acceleration_steps'])
 assert results[0]>0 and results[1]<results[0]
