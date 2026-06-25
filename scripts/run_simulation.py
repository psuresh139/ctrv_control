"""
scripts/run_simulation.py

Main closed-loop simulation: EKF + LTV-LQR on COTA-inspired track.

Usage
-----
    python scripts/run_simulation.py
    python scripts/run_simulation.py --save results.png
    python scripts/run_simulation.py --estimator ekf --seed 7

Outputs
-------
  - Console: per-step debug, saturation report, NIS report
  - Plot:    6-panel analysis dashboard
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ctrv.noise import calculate_Q as _calc_Q
import numpy.random as nprng

from ctrv import (
    predict_state_ctrv, wrap_to_pi,
    ekf_step, ukf_step, UKFParams, compute_nis_summary, print_nis_report,
    build_ltv_linearization, ltv_lqr_backward_riccati,
    ltv_lqr_scheduled_riccati, pick_k_nearest,
    build_reference_from_waypoints,
    WAYPOINTS, SEGMENT_COMMANDS, DT,
    WAYPOINTS_SOFT, SEGMENT_COMMANDS_SOFT,
    Q_A_PSD, Q_ALPHA_PSD, R_MEAS, P_INIT,
    Q_LQR, R_LQR, U_BOUNDS, INNOVATION_GATE,
    IDX_X, IDX_Y, IDX_V, IDX_PSI, IDX_PSIDOT,
    N_STATES, N_CONTROL, H_MEAS,
)

# ── Curvature-scheduled LQR costs ────────────────────────────────────────────
# On tight corners (κ ≥ 0.02 rad/m) the controller needs lower yaw-accel
# penalty to execute the turn.  On straights it can afford to be smoother.
Q_LQR_TURN = np.diag([40.0, 40.0, 2.0, 25.0, 12.0])
R_LQR_TURN = np.diag([5.0, 50.0])

# Maximum position error magnitude fed to the LQR feedback law.
# When the vehicle deviates far from the reference, the raw LQR feedback
# can overshoot and destabilize the loop.  Clamping the position error
# ensures the controller works within its linearization regime and
# prevents single-step divergence events from cascading.
MAX_POS_ERR_FEEDBACK = 10.0   # metres


def run_simulation(
    q_a:        float = Q_A_PSD,
    q_alpha:    float = Q_ALPHA_PSD,
    seed:       int   = 42,
    debug:      bool  = True,
    use_gate:   bool  = True,
    estimator:  str   = 'ukf',   # 'ukf' or 'ekf'
    Q_lqr:      np.ndarray = None,
    R_lqr:      np.ndarray = None,
    window:     int   = 60,
    heading_sigma_deg: float = None,
    waypoints:  np.ndarray = None,
    segment_commands: np.ndarray = None,
) -> dict:
    """
    Full closed-loop EKF/UKF + LTV-LQR simulation.

    Parameters
    ----------
    q_a       : longitudinal accel noise PSD  (tune via NIS)
    q_alpha   : yaw accel noise PSD           (tune via NIS)
    seed      : random seed for reproducibility
    debug     : print per-step diagnostics
    use_gate  : enable innovation gating
    estimator : 'ukf' (default) or 'ekf' for comparison
    Q_lqr     : (5,5) LQR state cost; defaults to constants.Q_LQR
    R_lqr     : (2,2) LQR input cost; defaults to constants.R_LQR
    window    : pick_k_nearest forward search window (steps, default 60)
    heading_sigma_deg : float  if not None, add heading measurement with this
                        noise level (degrees).  Typical: 5.0 for cheap IMU,
                        3.0 for BNO055, 1.5 for good IMU.
    waypoints : (N,2) override track waypoints (for softened corners)
    segment_commands : (N-1,2) override segment commands

    Returns
    -------
    results dict (see bottom of function for keys)
    """
    np.random.seed(seed)

    # ── Track waypoints ────────────────────────────────────────────────────
    _wps  = WAYPOINTS if waypoints is None else waypoints
    _cmds = SEGMENT_COMMANDS if segment_commands is None else segment_commands

    # ── Measurement model ─────────────────────────────────────────────────
    if heading_sigma_deg is not None:
        from ctrv.constants import H_MEAS_3DOF
        _H_meas = H_MEAS_3DOF
        _R_meas = np.diag([1.0**2, 1.0**2, np.deg2rad(heading_sigma_deg)**2])
        _n_meas = 3
    else:
        _H_meas = H_MEAS
        _R_meas = R_MEAS
        _n_meas = 2

    # ── Build reference trajectory ────────────────────────────────────────
    ref_traj = build_reference_from_waypoints(_wps, _cmds, DT)
    Xref     = ref_traj['states']
    Uref     = ref_traj['u_ref']
    T        = len(Xref)

    REF_X_INIT = np.array([_wps[0, 0], _wps[0, 1],
                            _cmds[0, 0], 0.0, 0.0])

    # ── Pre-compute LTV-LQR gains (ONCE, not inside loop) ─────────────────
    _Q_lqr = Q_LQR if Q_lqr is None else Q_lqr
    _R_lqr = R_LQR if R_lqr is None else R_lqr

    if debug:
        print("Pre-computing LTV linearization and curvature-scheduled Riccati sweep...")
    A_seq, B_seq = build_ltv_linearization(ref_traj, DT)
    K_seq, P_seq = ltv_lqr_scheduled_riccati(
        A_seq, B_seq, Xref,
        Q_straight=_Q_lqr, R_straight=_R_lqr,
        Q_turn=Q_LQR_TURN,  R_turn=R_LQR_TURN,
        curv_threshold=0.02,
    )
    if debug:
        print(f"  Done. {len(K_seq)} gain matrices computed.")

    # ── Initial conditions ─────────────────────────────────────────────────
    x0_true = REF_X_INIT.copy()
    x0_est  = REF_X_INIT.copy() + np.array([2.0, 2.0, 0.5, 0.1, 0.05])

    x_true = x0_true.copy()
    x_est  = x0_est.copy()
    P_est  = P_INIT.copy()

    # ── Logs ──────────────────────────────────────────────────────────────
    true_states = [x_true.copy()]
    est_states  = [x_est.copy()]
    controls    = []
    pos_errs    = []
    cov_traces  = []
    nis_log     = []
    gate_log    = []

    n_clip_a  = 0
    n_clip_al = 0
    k_prog    = 0

    gate_threshold = INNOVATION_GATE if use_gate else None
    _step_fn = ukf_step if estimator == 'ukf' else ekf_step

    if debug:
        print(f"\n=== Simulation Start ===")
        print(f"Estimator: {estimator.upper()}  T={T}  q_a={q_a}  q_alpha={q_alpha}  gate={gate_threshold}")

    # ── Main loop ─────────────────────────────────────────────────────────
    for k in range(T - 1):

        # ── Reference selection (spatial, forward-only) ───────────────────
        k_spatial = pick_k_nearest(x_est, Xref, k_prog, window=window)
        k_spatial = min(k_spatial, T - 2)

        # Blend spatial and temporal indices to prevent excessive drift.
        # Use whichever is further along, but clamp so the temporal index
        # never forces more than MAX_LAG steps ahead of the spatial match.
        k_temporal = min(k, T - 2)
        MAX_LAG    = 50   # max steps k_use may lag behind time
        k_floor    = max(0, k_temporal - MAX_LAG)
        k_use      = max(k_spatial, k_floor)
        k_use      = min(k_use, T - 2)

        k_prog = k_use

        x_ref = Xref[k_use].copy()
        u_ff  = Uref[k_use].copy()
        K     = K_seq[k_use]

        # ── Control law: feedforward + LQR feedback ───────────────────────
        e = (x_est - x_ref).copy()
        e[IDX_PSI] = wrap_to_pi(e[IDX_PSI])

        # Error attenuation: clamp position error magnitude to keep the
        # LQR operating within its valid linearization regime.  Large
        # position errors produce oversized feedback that can overshoot
        # and destabilize the loop — especially at corners where the
        # gains are more aggressive.
        pos_err_mag = np.sqrt(e[IDX_X]**2 + e[IDX_Y]**2)
        if pos_err_mag > MAX_POS_ERR_FEEDBACK:
            scale = MAX_POS_ERR_FEEDBACK / pos_err_mag
            e[IDX_X] *= scale
            e[IDX_Y] *= scale

        u_raw = u_ff - K @ e

        # Saturation-aware scaling: scale only the feedback term,
        # preserve feedforward direction under saturation
        u_fb = K @ e
        s = 1.0
        a_lim  = U_BOUNDS['a'][1]
        al_lim = U_BOUNDS['alpha'][1]
        denom_a  = abs(u_ff[0] - u_raw[0]) + 1e-12
        denom_al = abs(u_ff[1] - u_raw[1]) + 1e-12
        s = min(s, a_lim  / denom_a)
        s = min(s, al_lim / denom_al)
        u = u_ff - s * u_fb

        # Hard clip + count saturations
        pre_a, pre_al = u[0], u[1]
        u[0] = np.clip(u[0], *U_BOUNDS['a'])
        u[1] = np.clip(u[1], *U_BOUNDS['alpha'])
        if abs(u[0] - pre_a)  > 1e-12: n_clip_a  += 1
        if abs(u[1] - pre_al) > 1e-12: n_clip_al += 1

        # ── Plant propagation ─────────────────────────────────────────────
        Q_true = _calc_Q(DT, q_a, q_alpha, float(x_true[IDX_PSI]))
        w = nprng.multivariate_normal(np.zeros(N_STATES), Q_true)
        x_true = predict_state_ctrv(x_true, DT, u) + w

        # ── Measurement ───────────────────────────────────────────────────
        v_meas = nprng.multivariate_normal(np.zeros(_n_meas), _R_meas)
        z_k    = _H_meas @ x_true + v_meas
        # Wrap heading component if present
        if _n_meas == 3:
            z_k[2] = wrap_to_pi(z_k[2])

        # ── EKF / UKF step ────────────────────────────────────────────────
        x_est, P_est, info = _step_fn(
            x_est, P_est, z_k, DT, _R_meas, q_a, q_alpha,
            u_k=u,
            predict_fn=predict_state_ctrv,
            innovation_gate=gate_threshold,
            H_meas=_H_meas,
        )

        # ── Logging ───────────────────────────────────────────────────────
        true_states.append(x_true.copy())
        est_states.append(x_est.copy())
        controls.append(u.copy())

        pe = np.linalg.norm(x_true[:2] - x_ref[:2])
        pos_errs.append(pe)
        cov_traces.append(np.trace(P_est))

        if not info['gated']:
            nis_log.append(info['nis'])
        gate_log.append(info['gated'])

        if debug and (k in [0, 1] or k % 100 == 0 or k == T - 2):
            print(f"  k={k:4d}  k_use={k_use:4d}  "
                  f"|err|={pe:6.2f}m  "
                  f"NIS={info['nis']:6.2f}  "
                  f"tr(P)={np.trace(P_est):7.3f}  "
                  f"gated={info['gated']}")

    # ── Saturation report ─────────────────────────────────────────────────
    ctrl_arr = np.array(controls)
    sat_report = {
        '% accel clipped':     100.0 * n_clip_a  / max(1, T - 1),
        '% yaw-accel clipped': 100.0 * n_clip_al / max(1, T - 1),
        'max |a|':             float(np.max(np.abs(ctrl_arr[:, 0]))),
        'max |alpha|':         float(np.max(np.abs(ctrl_arr[:, 1]))),
    }

    if debug:
        print("\n=== Saturation Report ===")
        for k, v in sat_report.items():
            print(f"  {k}: {v:.2f}")

    # ── NIS report ────────────────────────────────────────────────────────
    nis_summary = compute_nis_summary(nis_log, n_meas=_n_meas)
    if debug:
        print()
        print_nis_report(nis_summary)
        print(f"\nMean position error: {np.mean(pos_errs):.3f} m")
        print(f"% steps gated: {100*np.mean(gate_log):.1f}%")

    return {
        'true_states':  np.array(true_states),
        'est_states':   np.array(est_states),
        'ref_states':   Xref,
        'times':        ref_traj['times'],
        'controls':     ctrl_arr,
        'pos_errs':     np.array(pos_errs),
        'cov_traces':   np.array(cov_traces),
        'nis_log':      nis_log,
        'nis_summary':  nis_summary,
        'gate_log':     gate_log,
        'K_seq':        K_seq,
        'sat_report':   sat_report,
    }


def plot_results(results: dict, save_path: str = None) -> None:
    """6-panel analysis dashboard."""
    t      = results['times']
    t_u    = t[:-1]
    Xref   = results['ref_states']
    Xtrue  = results['true_states']
    Xest   = results['est_states']
    pos_err = results['pos_errs']
    trP    = results['cov_traces']
    K_seq  = results['K_seq']
    nis    = results['nis_log']

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(Xref[:, IDX_X], Xref[:, IDX_Y], 'b--', lw=2, label='Reference', alpha=0.6)
    ax.plot(Xtrue[:, IDX_X], Xtrue[:, IDX_Y], 'r-', lw=1.5, label='True', alpha=0.9)
    estimator_label = results.get('estimator', 'UKF').upper()
    ax.plot(Xest[:, IDX_X],  Xest[:, IDX_Y],  'g:', lw=1.0, label=f'{estimator_label} estimate', alpha=0.7)
    ax.scatter(*Xref[0, :2],  c='green', s=80, zorder=5, label='Start')
    ax.scatter(*Xref[-1, :2], c='red',   s=80, marker='s', zorder=5, label='End')
    ax.set_title('1. Trajectory Tracking')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 2. Position error
    ax = axes[0, 1]
    ax.plot(t_u, pos_err, 'b-', lw=1.2)
    ax.axhline(np.mean(pos_err), color='r', ls='--',
               label=f'Mean = {np.mean(pos_err):.2f} m')
    ax.set_title('2. Position Error')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [m]')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. LQR stiffness map
    ax = axes[1, 0]
    n_k = min(len(K_seq), len(Xref))
    K_norms = [np.linalg.norm(K) for K in K_seq[:n_k]]
    sc = ax.scatter(Xref[:n_k, IDX_X], Xref[:n_k, IDX_Y],
                    c=K_norms, cmap='viridis', s=12)
    plt.colorbar(sc, ax=ax, label='||K|| (gain magnitude)')
    ax.set_title('3. LQR Stiffness Map')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.axis('equal'); ax.grid(True, alpha=0.3)

    # 4. NIS over time
    ax = axes[1, 1]
    nis_t = t_u[:len(nis)]
    ax.plot(nis_t, nis, 'purple', lw=0.8, alpha=0.7, label='NIS')
    ax.axhline(2.0,  color='g', ls='--', label='Target (2.0)')
    ax.axhline(9.21, color='r', ls='--', label='Gate (9.21)')
    ax.axhline(np.mean(nis), color='orange', ls='-',
               label=f'Mean = {np.mean(nis):.2f}')
    ax.set_title('4. Normalized Innovation Squared (NIS)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('NIS')
    ax.set_ylim(0, min(50, np.percentile(nis, 99) * 1.5))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. Estimation uncertainty
    ax = axes[2, 0]
    ax.plot(t_u, trP, color='brown', lw=1.2)
    ax.set_title('5. Estimation Uncertainty (trace P)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('tr(P)')
    ax.grid(True, alpha=0.3)

    # 6. Velocity tracking
    ax = axes[2, 1]
    ax.plot(t, Xtrue[:, IDX_V], 'r-',  lw=1.5, label='True',      alpha=0.8)
    ax.plot(t, Xest[:, IDX_V],  'g:',  lw=1.5, label='Estimated', alpha=0.8)
    ax.plot(t, Xref[:, IDX_V],  'b--', lw=2.0, label='Reference', alpha=0.5)
    ax.set_title('6. Velocity Tracking')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed [m/s]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTRV EKF/UKF + LTV-LQR simulation')
    parser.add_argument('--estimator', choices=['ukf', 'ekf'], default='ukf')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--save',      type=str,   default='simulation_results.png',
                        help='Output plot path (default: simulation_results.png)')
    parser.add_argument('--no-heading', action='store_true',
                        help='Disable IMU heading measurement (GPS only)')
    parser.add_argument('--soft-track', action='store_true', default=True,
                        help='Use softened corner geometry (default: True)')
    args = parser.parse_args()

    # Use non-interactive backend only when saving to file
    if args.save:
        matplotlib.use('Agg')

    results = run_simulation(
        estimator=args.estimator,
        seed=args.seed,
        heading_sigma_deg=None if args.no_heading else 5.0,
        waypoints=WAYPOINTS_SOFT   if args.soft_track else WAYPOINTS,
        segment_commands=SEGMENT_COMMANDS_SOFT if args.soft_track else SEGMENT_COMMANDS,
        debug=True,
    )
    results['estimator'] = args.estimator
    plot_results(results, save_path=args.save)
