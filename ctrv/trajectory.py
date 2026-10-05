"""Arc-length spline reference with curvature-derived yaw rate and retiming.

Segment commands' first column supplies speed ceilings; the legacy yaw-rate
column is ignored. Geometry determines heading and yaw rate. Retiming enforces
limits without clipping controls independently of reference states.
"""
import numpy as np
from scipy.interpolate import make_interp_spline
from ctrv.constants import DT
from ctrv.limits import DEFAULT_LIMITS
from ctrv.model import predict_state_ctrv, wrap_to_pi


def build_reference_from_waypoints(waypoints, commands, dt=DT,
                                  sampling_res=0.1, t_ramp=0.5,
                                  limits=DEFAULT_LIMITS):
    """Generate fixed-dt samples. t_ramp is retained for API compatibility.

    The dense spline's numerical arc length defines travel time. Forward and
    backward passes enforce acceleration/braking; global dilation subsequently
    enforces sampled yaw/steering-rate and lateral limits. Infeasible geometry
    raises ValueError rather than silently changing the track.
    """
    waypoints, commands = np.asarray(waypoints, float), np.asarray(commands, float)
    if (waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 3
            or commands.shape != (len(waypoints)-1, 2)
            or not np.all(np.isfinite(waypoints)) or not np.all(np.isfinite(commands))
            or np.any(commands[:, 0] <= 0) or not np.isfinite(dt) or dt <= 0
            or not np.isfinite(sampling_res) or sampling_res <= 0):
        raise ValueError('Invalid waypoints, commands, or sampling intervals')
    chords = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    if np.any(chords <= 1e-9):
        raise ValueError('Adjacent waypoints must be distinct')
    param = np.r_[0., np.cumsum(chords)]
    tangent0 = (waypoints[1]-waypoints[0])/chords[0]
    tangent1 = (waypoints[-1]-waypoints[-2])/chords[-1]
    spline = make_interp_spline(param, waypoints, k=3,
                               bc_type=([(1, tangent0)], [(1, tangent1)]))
    q = np.linspace(0, param[-1], int(np.ceil(param[-1]/sampling_res))+1)
    points = spline(q)
    deriv = spline.derivative()(q)
    second = spline.derivative(2)(q)
    norm = np.linalg.norm(deriv, axis=1)
    if np.any(norm < 1e-8):
        raise ValueError('Spline contains a stationary tangent')
    curvature = (deriv[:, 0]*second[:, 1]-deriv[:, 1]*second[:, 0])/norm**3
    if np.max(np.abs(curvature)) > limits.max_curvature:
        raise ValueError('Track curvature exceeds steering geometry limit; revise waypoints')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    segments = np.minimum(np.searchsorted(param[1:], q, side='right'), len(commands)-1)
    speed = np.minimum(commands[segments, 0], limits.max_speed)
    kabs = np.maximum(np.abs(curvature), 1e-12)
    speed = np.minimum(speed, np.sqrt(limits.max_lateral_accel/kabs))
    speed = np.minimum(speed, limits.max_yaw_rate/kabs)
    ds = np.diff(arc)
    for i in range(1, len(speed)):
        speed[i] = min(speed[i], np.sqrt(speed[i-1]**2+2*limits.max_accel*ds[i-1]))
    for i in range(len(speed)-2, -1, -1):
        speed[i] = min(speed[i], np.sqrt(speed[i+1]**2+2*limits.max_braking*ds[i]))
    base_time = np.r_[0., np.cumsum(2*ds/(speed[:-1]+speed[1:]))]
    dilation = 1.0
    for _ in range(30):
        # End exactly on the last waypoint at a whole multiple of dt.
        duration = np.ceil(base_time[-1]*dilation/dt)*dt
        factor = duration/base_time[-1]
        times = np.arange(int(round(duration/dt))+1)*dt
        q_fixed = np.interp(times/factor, base_time, q)
        xy = spline(q_fixed)
        d = spline.derivative()(q_fixed)
        dd = spline.derivative(2)(q_fixed)
        kappa = (d[:, 0]*dd[:, 1]-d[:, 1]*dd[:, 0])/np.linalg.norm(d, axis=1)**3
        v = np.interp(times/factor, base_time, speed)/factor
        psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        omega = kappa*v
        a = np.r_[np.diff(v)/dt, 0.]
        alpha = np.r_[np.diff(omega)/dt, 0.]
        delta = np.arctan(limits.wheelbase*kappa)
        ratio = max(np.max(a)/limits.max_accel, -np.min(a)/limits.max_braking,
                    np.max(np.abs(alpha))/limits.max_yaw_accel,
                    np.max(v*v*np.abs(kappa))/limits.max_lateral_accel,
                    (np.max(v)/limits.max_speed)**2,
                    (np.max(np.abs(omega))/limits.max_yaw_rate)**2,
                    (np.max(np.abs(np.diff(delta)))/dt/limits.max_steering_rate)**2)
        states = np.column_stack([xy, v, wrap_to_pi(psi), omega])
        controls = np.column_stack([a, alpha])
        defects = np.array([predict_state_ctrv(states[i], dt, controls[i])-states[i+1]
                            for i in range(len(states)-1)])
        defects[:, 3] = wrap_to_pi(defects[:, 3])
        if ratio <= 1+1e-8 and np.max(np.linalg.norm(defects[:, :2], axis=1)) < 0.02 and np.max(np.abs(defects[:, 3])) < 0.002:
            return {'states': states, 'times': times, 'u_ref': controls,
                    'curvature': kappa, 'steering': delta, 'defects': defects,
                    'arc_length': arc[-1], 'time_dilation': factor}
        dilation = factor*max(1.05, np.sqrt(max(ratio, 1))*1.02)
    raise ValueError('Unable to produce reference within limits and discretization tolerances')
