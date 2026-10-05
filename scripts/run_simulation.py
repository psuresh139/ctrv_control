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
from pathlib import Path
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctrv.constants import (IDX_X, IDX_Y, IDX_V, WAYPOINTS, SEGMENT_COMMANDS,
                            WAYPOINTS_SOFT, SEGMENT_COMMANDS_SOFT)
from ctrv.simulation import run_simulation, SCENARIOS


def plot_results(results: dict, save_path: str = None) -> None:
    """6-panel analysis dashboard."""
    t      = results['times']
    t_u    = t[1:]
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
    ax.set_title('2. Cross-track Distance to Reference Path')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [m]')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. LQR stiffness map
    ax = axes[1, 0]
    n_k = min(len(K_seq), len(Xref))
    K_norms = [np.linalg.norm(K) for K in K_seq[:n_k]]
    sc = ax.scatter(Xref[:n_k, IDX_X], Xref[:n_k, IDX_Y],
                    c=K_norms, cmap='viridis', s=12)
    plt.colorbar(sc, ax=ax, label='||K|| (gain magnitude)')
    ax.set_title('3. LQR Baseline Gain Map')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.axis('equal'); ax.grid(True, alpha=0.3)

    # 4. NIS over time
    ax = axes[1, 1]
    nis = np.asarray(nis, float)
    dof = np.asarray(results['nis_dof'])
    observed = np.isfinite(nis) & (dof > 0)
    normalized = nis[observed]/dof[observed]
    ax.plot(t_u[observed], normalized, '.', color='purple', markersize=2, label='NIS / DOF (all observations)')
    ax.axhline(1.0, color='g', ls='--', label='Target (1.0)')
    if np.any(~results['gps_available']):
        ax.fill_between(t_u, 0, 1, where=~results['gps_available'],
                        transform=ax.get_xaxis_transform(), alpha=.15, color='red', label='GPS unavailable')
    ax.set_title('4. Normalized Innovation Squared / DOF')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('NIS / DOF')
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
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTRV EKF/UKF + LTV-LQR simulation')
    parser.add_argument('--controller', choices=['lqr','mpc'], default='lqr')
    parser.add_argument('--estimator', choices=['ukf', 'ekf'], default='ukf')
    parser.add_argument('--scenario', choices=list(SCENARIOS), default='nominal')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--save',      type=str,   default='simulation_results.png',
                        help='Output plot path (default: simulation_results.png)')
    parser.add_argument('--no-heading', action='store_true',
                        help='Disable IMU heading measurement (GPS only)')
    parser.add_argument('--track', choices=['soft', 'original'], default='soft')
    args = parser.parse_args()

    # Use non-interactive backend only when saving to file
    if args.save:
        matplotlib.use('Agg')

    results = run_simulation(
        estimator=args.estimator, controller=args.controller,
        seed=args.seed,
        scenario=SCENARIOS[args.scenario],
        heading_sigma_deg=None if args.no_heading else 5.0,
        waypoints=WAYPOINTS_SOFT   if args.track == 'soft' else WAYPOINTS,
        segment_commands=SEGMENT_COMMANDS_SOFT if args.track == 'soft' else SEGMENT_COMMANDS,
        debug=True,
    )
    results['estimator'] = args.estimator
    plot_results(results, save_path=args.save)
