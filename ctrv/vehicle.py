"""Synthetic dynamic bicycle with explicit actuator and combined tire limits.

State [x,y,v_long,heading,yaw_rate,v_lateral,accel_actuator,steering_actuator].
The first five states preserve historical ordering; vehicle filters use all eight.
No measured vehicle
parameters, suspension, wheel dynamics, load transfer or hardware validation.
"""
from dataclasses import dataclass
import numpy as np
from ctrv.model import wrap_to_pi
from ctrv.limits import DEFAULT_LIMITS


@dataclass(frozen=True)
class VehicleConfig:
    mass_kg: float = 1500.
    inertia_kgm2: float = 2500.
    front_distance_m: float = 1.2
    rear_distance_m: float = 1.5
    cornering_front_nprad: float = 80000.
    cornering_rear_nprad: float = 80000.
    acceleration_tau_s: float = .25
    steering_tau_s: float = .15
    acceleration_jerk_mps3: float = 10.
    drag_coefficient: float = .35
    integration_step_s: float = .01
    low_speed_regularization_mps: float = .5

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in self.__dict__.values()):
            raise ValueError('Positive finite vehicle parameters required')


def tire_forces(state,mu,config=VehicleConfig()):
    if not np.isfinite(mu) or mu<=0:raise ValueError('Positive friction required')
    state=np.asarray(state,float)
    actual_v=np.maximum(state[...,2],0.)
    v=np.sqrt(actual_v**2+config.low_speed_regularization_mps**2);r=state[...,4];vy=state[...,5];a=state[...,6];delta=state[...,7]
    L=config.front_distance_m+config.rear_distance_m
    weights=np.array([config.rear_distance_m,config.front_distance_m])/L
    normal=config.mass_kg*9.81*weights
    effective_a=np.where((actual_v<=0.) & (a<0.),0.,a)
    demanded_long=config.mass_kg*effective_a[...,None]*weights
    longitudinal=np.clip(demanded_long,-mu*normal,mu*normal)
    available=np.sqrt(np.maximum(0.,(mu*normal)**2-longitudinal**2))
    slip=np.stack([delta-np.arctan2(vy+config.front_distance_m*r,v),
                   -np.arctan2(vy-config.rear_distance_m*r,v)],axis=-1)
    stiffness=np.array([config.cornering_front_nprad,config.cornering_rear_nprad])
    contact_lateral=np.stack([vy+config.front_distance_m*r,vy-config.rear_distance_m*r],axis=-1)
    contact_speed_sq=actual_v[...,None]**2+contact_lateral**2
    motion_factor=np.sqrt(contact_speed_sq/(contact_speed_sq+config.low_speed_regularization_mps**2))
    lateral=motion_factor*available*np.tanh(stiffness*slip/np.maximum(available,1e-9))
    utilization=np.hypot(longitudinal,lateral)/(mu*normal)
    return longitudinal,lateral,utilization


def derivative(state,command,mu,config=VehicleConfig(),limits=DEFAULT_LIMITS):
    state=np.asarray(state,float);command=np.asarray(command,float)
    longitudinal,lateral,util=tire_forces(state,mu,config)
    v=np.maximum(state[...,2],0.);psi=state[...,3];r=state[...,4];vy=state[...,5];delta=state[...,7]
    fx=longitudinal[...,0]*np.cos(delta)-lateral[...,0]*np.sin(delta)+longitudinal[...,1]
    fy=lateral[...,0]*np.cos(delta)+longitudinal[...,0]*np.sin(delta)+lateral[...,1]
    moment=config.front_distance_m*(lateral[...,0]*np.cos(delta)+longitudinal[...,0]*np.sin(delta))-config.rear_distance_m*lateral[...,1]
    acceleration=fx/config.mass_kg-config.drag_coefficient*v*abs(v)/config.mass_kg+r*vy
    acceleration=np.where((state[...,2]<=0.) & (acceleration<0.),0.,acceleration)
    lateral_rate=fy/config.mass_kg-r*v
    yaw_rate=moment/config.inertia_kgm2
    # Low-speed regularization belongs inside the tire forces. Blending
    # arbitrary kinematic accelerations here would violate the force budget.
    target_a=np.clip(command[...,0],-limits.max_braking,limits.max_accel)
    target_delta=np.clip(command[...,1],-limits.max_steering,limits.max_steering)
    da=np.clip((target_a-state[...,6])/config.acceleration_tau_s,
               -config.acceleration_jerk_mps3,config.acceleration_jerk_mps3)
    dd=np.clip((target_delta-delta)/config.steering_tau_s,-limits.max_steering_rate,limits.max_steering_rate)
    return np.stack([v*np.cos(psi)-vy*np.sin(psi),v*np.sin(psi)+vy*np.cos(psi),
                     acceleration,r,yaw_rate,lateral_rate,da,dd],axis=-1)


def step_vehicle(state,command,dt,mu=1.,config=VehicleConfig(),limits=DEFAULT_LIMITS):
    state=np.asarray(state,float).copy();command=np.asarray(command,float)
    if state.shape[-1:]!=(8,) or command.shape[-1:]!=(2,) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(command)) or not np.isfinite(dt) or dt<=0:
        raise ValueError('Invalid state/command/timestep')
    count=int(np.ceil(dt/config.integration_step_s));h=dt/count;peak=np.zeros(state.shape[:-1])
    for _ in range(count):
        f=lambda x:derivative(x,command,mu,config,limits)
        a=f(state);b=f(state+h*a/2);c=f(state+h*b/2);d=f(state+h*c)
        state+=h*(a+2*b+2*c+d)/6
        state[...,2]=np.maximum(0.,state[...,2]);state[...,3]=wrap_to_pi(state[...,3])
        state[...,6]=np.clip(state[...,6],-limits.max_braking,limits.max_accel)
        state[...,7]=np.clip(state[...,7],-limits.max_steering,limits.max_steering)
        peak=np.maximum(peak,np.max(tire_forces(state,mu,config)[2],axis=-1))
    rates=derivative(state,command,mu,config,limits)
    scalar=lambda v:float(v) if np.ndim(v)==0 else v
    return state,{'friction_mu':mu,'tire_utilization_max':scalar(peak),'actual_acceleration':scalar(rates[...,2]),'actual_yaw_acceleration':scalar(rates[...,4]),
                  'actual_lateral_acceleration':scalar(rates[...,5]+state[...,4]*state[...,2]),
                  'actuator_acceleration':scalar(state[...,6]),'actuator_steering':scalar(state[...,7])}
