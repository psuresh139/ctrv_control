"""Reproducible LQR baseline with GPS loss and honest physical-limit metrics."""
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy.stats import chi2
from scipy.spatial import cKDTree
from ctrv.constants import (DT, WAYPOINTS_SOFT, SEGMENT_COMMANDS_SOFT, Q_A_PSD,
                            Q_ALPHA_PSD, P_INIT, Q_LQR, R_LQR)
from ctrv.model import predict_state_ctrv, wrap_to_pi
from ctrv.noise import calculate_Q
from ctrv.ekf import ukf_step, ekf_step, compute_nis_summary
from ctrv.lqr import build_ltv_linearization, ltv_lqr_scheduled_riccati
from ctrv.trajectory import build_reference_from_waypoints
from ctrv.limits import DEFAULT_LIMITS


@dataclass(frozen=True)
class SensorScenario:
    name: str = 'nominal'
    gps_outage_start: float = 15.0
    gps_outage_duration: float = 0.0
    gps_outlier_probability: float = 0.0
    gps_outlier_sigma: float = 20.0
    gps_bias_x: float = 0.0
    gps_bias_y: float = 0.0
    plant_noise_scale: float = 1.0
    sensor_noise_scale: float = 1.0

    def __post_init__(self):
        numeric = [v for k, v in asdict(self).items() if k != 'name']
        if (not np.all(np.isfinite(numeric)) or self.gps_outage_start < 0
                or self.gps_outage_duration < 0 or self.gps_outlier_sigma < 0
                or self.plant_noise_scale < 0 or self.sensor_noise_scale < 0 or not 0 <= self.gps_outlier_probability <= 1):
            raise ValueError('Invalid scenario')


SCENARIOS = {
    'nominal': SensorScenario(),
    'deterministic_recovery': SensorScenario('deterministic_recovery', plant_noise_scale=0., sensor_noise_scale=0.),
    'gps_loss_5s': SensorScenario('gps_loss_5s', gps_outage_duration=5.),
    'gps_loss_10s': SensorScenario('gps_loss_10s', gps_outage_duration=10.),
    'gps_outliers': SensorScenario('gps_outliers', gps_outlier_probability=0.05),
    'gps_bias': SensorScenario('gps_bias', gps_bias_x=3., gps_bias_y=-2.),
    'noise_mismatch': SensorScenario('noise_mismatch', plant_noise_scale=4.),
}


def prepare_reference(waypoints=None, segment_commands=None, limits=DEFAULT_LIMITS):
    ref = build_reference_from_waypoints(
        WAYPOINTS_SOFT if waypoints is None else waypoints,
        SEGMENT_COMMANDS_SOFT if segment_commands is None else segment_commands,
        DT, limits=limits)
    A, B = build_ltv_linearization(ref, DT)
    return ref, A, B


def path_errors(positions, reference):
    """Nearest-polyline distance; also return projected progress and tangent.

    Search segments adjacent to the nearest sampled vertex, avoiding an N²
    distance matrix. This track has no self-intersections.
    """
    path = reference[:, :2]
    _, vertices = cKDTree(path).query(positions)
    starts = path[:-1]
    vectors = np.diff(path, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    candidates = np.clip(vertices[:, None]+np.arange(-2, 2), 0, len(vectors)-1)
    offset = positions[:, None, :]-starts[candidates]
    fraction = np.clip(np.sum(offset*vectors[candidates], axis=2)/np.maximum(lengths[candidates]**2, 1e-16), 0, 1)
    residual = offset-fraction[..., None]*vectors[candidates]
    norms = np.linalg.norm(residual, axis=2)
    j = np.argmin(norms, axis=1)
    rows = np.arange(len(positions))
    index = candidates[rows, j]
    return norms[rows, j], (arc[index]+fraction[rows, j]*lengths[index])/arc[-1], np.arctan2(vectors[index, 1], vectors[index, 0])


def run_simulation(q_a=Q_A_PSD, q_alpha=Q_ALPHA_PSD, seed=42, debug=True,
                   use_gate=True, estimator='ukf', Q_lqr=None, R_lqr=None,
                   window=60, heading_sigma_deg=5.0, waypoints=None,
                   segment_commands=None, scenario=None, plant_q_a=Q_A_PSD,
                   plant_q_alpha=Q_ALPHA_PSD, gps_sigma=1.0,
                   limits=DEFAULT_LIMITS, prepared=None, track_half_width=5.0,
                   controller="lqr", mpc_config=None, uncertainty_policy=None,friction_mu=.9):
    """Fixed-time tracking baseline. window is retained for API compatibility.

    GPS disappears during outage; independent heading remains if configured.
    True process noise and sensor noise are fixed independently of filter Q.
    Acceleration/yaw-acceleration are hard actuator bounds; lateral acceleration,
    speed and bicycle-equivalent steering limits are measured, not hidden by
    projecting the state. This is a kinematic CTRV plant, not a tire-slip model.
    """
    if uncertainty_policy is not None and controller != "mpc":
        raise ValueError("Uncertainty policy requires MPC")
    if not np.isfinite(friction_mu) or friction_mu<=0:
        raise ValueError("Positive finite declared friction required")
    if controller not in ('lqr', 'mpc'):
        raise ValueError('controller must be lqr or mpc')
    if estimator not in ('ekf', 'ukf'):
        raise ValueError('estimator must be ekf or ukf')
    if (gps_sigma <= 0 or (heading_sigma_deg is not None and heading_sigma_deg <= 0)
            or min(q_a, q_alpha, plant_q_a, plant_q_alpha) < 0 or track_half_width <= 0):
        raise ValueError('Invalid noise or track parameters')
    scenario = SensorScenario() if scenario is None else scenario
    ref, A, B = prepare_reference(waypoints, segment_commands, limits) if prepared is None else prepared
    Xref, Uref, times = ref['states'], ref['u_ref'], ref['times']
    T = len(times)
    Q = Q_LQR if Q_lqr is None else Q_lqr
    R = R_LQR if R_lqr is None else R_lqr
    K_seq, _ = ltv_lqr_scheduled_riccati(A, B, Xref, Q, R,
                                          Q_turn=Q, R_turn=R)
    mpc = None
    if controller == 'mpc':
        from ctrv.mpc import MPCController
        mpc = MPCController(ref,A,B,mpc_config,limits,track_half_width,Q,R)
    controller_diagnostics=[]
    # Separate streams keep disturbances paired across estimators/scenarios,
    # even when observations vanish or outlier injection changes.
    streams = np.random.SeedSequence(seed).spawn(3)
    plant_rng, sensor_rng, fault_rng = [np.random.default_rng(s) for s in streams]
    plant_draws = plant_rng.standard_normal((T-1, 5))
    sensor_draws = sensor_rng.standard_normal((T-1, 3))
    outlier_flags = fault_rng.random(T-1) < scenario.gps_outlier_probability
    outlier_draws = fault_rng.standard_normal((T-1, 2))*scenario.gps_outlier_sigma
    true = np.empty((T, 5)); estimated = np.empty_like(true)
    covariances = np.empty((T, 5, 5))
    true[0] = Xref[0]
    estimated[0] = Xref[0]+np.array([2., 2., .5, .1, .05])
    estimated[0, 3] = wrap_to_pi(estimated[0, 3])
    covariances[0] = P_INIT
    controls = np.empty((T-1, 2)); raw_controls = np.empty_like(controls)
    nis = np.full(T-1, np.nan); dof = np.zeros(T-1, int)
    gates = np.zeros(T-1, bool); gps_available = np.ones(T-1, bool)
    step_fn = ukf_step if estimator == 'ukf' else ekf_step
    for k in range(T-1):
        tick=perf_counter()
        error = estimated[k]-Xref[k]
        error[3] = wrap_to_pi(error[3])
        raw_controls[k] = Uref[k]-K_seq[k]@error
        bounded=np.clip(raw_controls[k],[-limits.max_braking,-limits.max_yaw_accel],
                        [limits.max_accel,limits.max_yaw_accel])
        if mpc is None:
            controls[k]=bounded
            diagnostic={'status':'lqr','solved':True,'fallback':False,'iterations':0,
                        'primal_residual':None,'dual_residual':None,'slack_max':None,
                        'slack_mean':None,'horizon':0,'warm_start':False}
        else:
            cap=None
            if uncertainty_policy is not None:
                cap,_,_=uncertainty_policy.speed_cap(estimated[k],covariances[k],Xref[k,2],ref['curvature'][k],friction_mu)
            controls[k],diagnostic=mpc.command(k,estimated[k],controls[k-1] if k else Uref[0],bounded,
                covariance=covariances[k] if uncertainty_policy else None,uncertainty_policy=uncertainty_policy,
                q_a=q_a,q_alpha=q_alpha,speed_cap=cap)
            raw_controls[k]=controls[k]
        diagnostic['computation_s']=perf_counter()-tick
        controller_diagnostics.append(diagnostic)
        Qtrue = calculate_Q(DT, plant_q_a, plant_q_alpha, true[k, 3])*scenario.plant_noise_scale
        values, vectors = np.linalg.eigh(Qtrue)
        noise = (vectors*np.sqrt(np.maximum(values, 0)))@plant_draws[k]
        true[k+1] = predict_state_ctrv(true[k], DT, controls[k])+noise
        true[k+1, 3] = wrap_to_pi(true[k+1, 3])
        true[k+1, 2] = max(0., true[k+1, 2])
        observation_time = times[k+1]
        available = not (scenario.gps_outage_start <= observation_time <
                         scenario.gps_outage_start+scenario.gps_outage_duration)
        gps_available[k] = available
        rows, observations, variances = [], [], []
        if available:
            gps = true[k+1, :2]+gps_sigma*sensor_draws[k, :2]*np.sqrt(scenario.sensor_noise_scale)+[scenario.gps_bias_x, scenario.gps_bias_y]
            if outlier_flags[k]:
                gps += outlier_draws[k]
            rows.extend(np.eye(5)[:2]); observations.extend(gps); variances.extend([gps_sigma**2]*2)
        if heading_sigma_deg is not None:
            sigma = np.deg2rad(heading_sigma_deg)
            rows.append(np.eye(5)[3]); variances.append(sigma**2)
            observations.append(wrap_to_pi(true[k+1, 3]+sigma*sensor_draws[k, 2]*np.sqrt(scenario.sensor_noise_scale)))
        dof[k] = len(rows)
        H = np.asarray(rows).reshape(-1, 5)
        z = np.array(observations) if rows else None
        gate = float(chi2.ppf(.99, dof[k])) if use_gate and dof[k] else None
        estimated[k+1], covariances[k+1], info = step_fn(
            estimated[k], covariances[k], z, DT, np.diag(variances), q_a, q_alpha,
            u_k=controls[k], predict_fn=predict_state_ctrv, H_meas=H, innovation_gate=gate)
        nis[k], gates[k] = info['nis'], info['gated']
    cross, progress, tangent = path_errors(true[1:, :2], Xref)
    estimation_error = true[1:]-estimated[1:]
    estimation_error[:, 3] = wrap_to_pi(estimation_error[:, 3])
    nees = np.einsum('ij,ij->i', estimation_error,
                     np.linalg.solve(covariances[1:], estimation_error[..., None])[..., 0])
    speed = true[1:, 2]; yaw = true[1:, 4]
    lateral = np.abs(speed*yaw)
    steering = np.arctan(limits.wheelbase*yaw/np.maximum(speed, .1))
    initial_steering = np.arctan(limits.wheelbase*true[0, 4]/max(true[0, 2], .1))
    steering_rate = np.abs(np.diff(np.r_[initial_steering, steering]))/DT
    saturated = np.any(np.abs(controls-raw_controls) > 1e-10, axis=1)
    if controller == 'mpc':
        saturated=np.any(np.isclose(controls,[-limits.max_braking,-limits.max_yaw_accel],atol=1e-5,rtol=0)
                         | np.isclose(controls,[limits.max_accel,limits.max_yaw_accel],atol=1e-5,rtol=0),axis=1)
    violations = ((speed > limits.max_speed+1e-8) | (lateral > limits.max_lateral_accel+1e-8)
                  | (np.abs(yaw) > limits.max_yaw_rate+1e-8)
                  | (np.abs(steering) > limits.max_steering+1e-8)
                  | (steering_rate > limits.max_steering_rate+1e-8))
    summary = compute_nis_summary(nis, degrees_of_freedom=dof)
    endpoint_error = np.linalg.norm(true[-1, :2]-Xref[-1, :2])
    heading_errors = wrap_to_pi(true[1:, 3]-tangent)
    observed = dof > 0
    outage = ~gps_available
    metrics = {
        'cross_track_rmse_m': float(np.sqrt(np.mean(cross**2))),
        'cross_track_p95_m': float(np.percentile(cross, 95)),
        'cross_track_max_m': float(np.max(cross)),
        'heading_rmse_rad': float(np.sqrt(np.mean(heading_errors**2))),
        'estimation_position_rmse_m': float(np.sqrt(np.mean(np.sum(estimation_error[:, :2]**2, axis=1)))),
        'outage_estimation_rmse_m': float(np.sqrt(np.mean(np.sum(estimation_error[outage, :2]**2, axis=1)))) if np.any(outage) else None,
        'outage_cross_track_max_m': float(np.max(cross[outage])) if np.any(outage) else None,
        'endpoint_error_m': float(endpoint_error), 'final_progress': float(progress[-1]),
        'completed': bool(progress[-1] >= .98 and endpoint_error <= track_half_width),
        'stayed_on_track': bool(np.all(cross <= track_half_width)),
        'physical_limit_violation_pct': float(100*np.mean(violations)),
        'saturation_pct': float(100*np.mean(saturated)),
        'rejection_pct': float(100*np.mean(gates[observed])) if np.any(observed) else None,
        'nis_normalized_mean': summary['normalized_mean'],
        'nees_mean': float(np.mean(nees)),
        'control_energy': float(DT*np.sum(controls**2)),
        'control_rate_rms': float(np.sqrt(np.mean((np.diff(controls, axis=0)/DT)**2))),
        'gps_missing_steps': int(np.sum(outage)),
        'max_lateral_accel_mps2': float(np.max(lateral)),
        'max_speed_mps': float(np.max(speed)),
    }
    durations=np.array([d['computation_s'] for d in controller_diagnostics])
    slacks=[d['slack_max'] for d in controller_diagnostics if d['slack_max'] is not None]
    metrics.update(solver_fallback_pct=float(100*np.mean([d['fallback'] for d in controller_diagnostics])),
                   normalized_slack_max=float(max(slacks)) if slacks else None,
                   normalized_slack_mean=float(np.mean([d['slack_mean'] for d in controller_diagnostics if d['slack_mean'] is not None])) if slacks else None,
                   controller_time_p50_ms=float(np.percentile(durations,50)*1000),
                   controller_time_p95_ms=float(np.percentile(durations,95)*1000),
                   controller_time_max_ms=float(np.max(durations)*1000),
                   deadline_miss_pct=float(100*np.mean(durations>DT)))
    metrics['success'] = bool(metrics['completed'] and metrics['stayed_on_track'] and not np.any(violations))
    if debug:
        print(f"{estimator.upper()} / {scenario.name} / seed={seed}: {metrics}")
    return {'true_states': true, 'est_states': estimated, 'ref_states': Xref,
            'times': times, 'controls': controls, 'raw_controls': raw_controls,
            'pos_errs': cross, 'cov_traces': np.trace(covariances[1:], axis1=1, axis2=2),
            'covariances': covariances, 'nis_log': nis, 'nis_dof': dof,
            'nis_summary': summary, 'gate_log': gates, 'gps_available': gps_available,
            'K_seq': K_seq, 'estimator': estimator, 'metrics': metrics,
            'nees_log': nees, 'limit_violations': violations,
            'uncertainty_policy':asdict(uncertainty_policy) if uncertainty_policy else None, 'friction_mu':friction_mu,
            'controller':controller, 'controller_diagnostics':controller_diagnostics,
            'config': {'seed': seed, 'estimator': estimator, 'controller': controller,
                       'uncertainty_policy':asdict(uncertainty_policy) if uncertainty_policy else None,'friction_mu':friction_mu,
                       'mpc_config': asdict(mpc.config) if mpc else None,
                       'scenario': asdict(scenario), 'limits': asdict(limits),
                       'filter_q_a': q_a, 'filter_q_alpha': q_alpha,
                       'plant_q_a': plant_q_a, 'plant_q_alpha': plant_q_alpha,
                       'gps_sigma': gps_sigma, 'heading_sigma_deg': heading_sigma_deg,
                       'track_half_width': track_half_width, 'dt': DT,
                       'gate_probability': .99 if use_gate else None,
                       'Q_lqr': Q.tolist(), 'R_lqr': R.tolist(),
                       'initial_covariance': P_INIT.tolist(),
                       'initial_estimate_offset': [2., 2., .5, .1, .05]},
            'sat_report': {'% steps saturated': metrics['saturation_pct']}}
