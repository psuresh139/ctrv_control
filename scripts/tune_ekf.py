"""
scripts/tune_ekf.py

Head-to-head comparison: EKF vs UKF on the COTA track.
Also sweeps q_a / q_alpha for the UKF to find well-tuned values.

Usage
-----
    python scripts/tune_ekf.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.run_simulation import run_simulation
from ctrv import print_nis_report, compute_nis_summary


def compare_ekf_vs_ukf(q_a=0.5, q_alpha=0.1, seed=42):
    print("=" * 60)
    print("EKF vs UKF HEAD-TO-HEAD")
    print("=" * 60)

    print(f"\n--- EKF (q_a={q_a}, q_alpha={q_alpha}) ---")
    r_ekf = run_simulation(q_a=q_a, q_alpha=q_alpha,
                           seed=seed, debug=False, estimator='ekf')
    print_nis_report(r_ekf['nis_summary'], label='EKF')
    print(f"  Mean position error: {np.mean(r_ekf['pos_errs']):.3f} m")

    print(f"\n--- UKF (q_a={q_a}, q_alpha={q_alpha}) ---")
    r_ukf = run_simulation(q_a=q_a, q_alpha=q_alpha,
                           seed=seed, debug=False, estimator='ukf')
    print_nis_report(r_ukf['nis_summary'], label='UKF')
    print(f"  Mean position error: {np.mean(r_ukf['pos_errs']):.3f} m")

    improvement = np.mean(r_ekf['pos_errs']) - np.mean(r_ukf['pos_errs'])
    pct = 100 * improvement / np.mean(r_ekf['pos_errs'])
    print(f"\n  Position error improvement: {improvement:.2f} m ({pct:.1f}%)")

    return r_ekf, r_ukf


def nis_sweep_ukf(q_a_values, q_alpha_values, seed=42):
    """Sweep (q_a, q_alpha) for UKF, return NIS and error grids."""
    nis_grid = np.zeros((len(q_a_values), len(q_alpha_values)))
    err_grid = np.zeros_like(nis_grid)

    for i, q_a in enumerate(q_a_values):
        for j, q_alpha in enumerate(q_alpha_values):
            print(f"  UKF  q_a={q_a:.2f}  q_alpha={q_alpha:.2f} ...", end=' ', flush=True)
            try:
                r = run_simulation(q_a=q_a, q_alpha=q_alpha,
                                   seed=seed, debug=False, estimator='ukf')
                nis_grid[i, j] = r['nis_summary']['mean']
                err_grid[i, j] = float(np.mean(r['pos_errs']))
                print(f"NIS={nis_grid[i,j]:.2f}  err={err_grid[i,j]:.2f}m")
            except Exception as e:
                print(f"FAILED: {e}")
                nis_grid[i, j] = err_grid[i, j] = np.nan

    return nis_grid, err_grid


def plot_comparison(r_ekf, r_ukf, save_path=None):
    """Side-by-side trajectory and NIS comparison."""
    from ctrv.constants import IDX_X, IDX_Y

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Trajectory comparison
    ax = axes[0]
    Xref = r_ekf['ref_states']
    ax.plot(Xref[:, IDX_X], Xref[:, IDX_Y], 'b--', lw=2, label='Reference', alpha=0.5)
    ax.plot(r_ekf['true_states'][:, IDX_X], r_ekf['true_states'][:, IDX_Y],
            'r-', lw=1.2, label=f"EKF  (err={np.mean(r_ekf['pos_errs']):.1f}m)", alpha=0.8)
    ax.plot(r_ukf['true_states'][:, IDX_X], r_ukf['true_states'][:, IDX_Y],
            'g-', lw=1.2, label=f"UKF  (err={np.mean(r_ukf['pos_errs']):.1f}m)", alpha=0.8)
    ax.set_title('Trajectory: EKF vs UKF')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # Position error over time
    ax = axes[1]
    t = r_ekf['times'][:-1]
    ax.plot(t, r_ekf['pos_errs'], 'r-', lw=1.0, label='EKF', alpha=0.8)
    ax.plot(t[:len(r_ukf['pos_errs'])], r_ukf['pos_errs'], 'g-', lw=1.0, label='UKF', alpha=0.8)
    ax.axhline(np.mean(r_ekf['pos_errs']), color='r', ls='--', alpha=0.5,
               label=f"EKF mean={np.mean(r_ekf['pos_errs']):.1f}m")
    ax.axhline(np.mean(r_ukf['pos_errs']), color='g', ls='--', alpha=0.5,
               label=f"UKF mean={np.mean(r_ukf['pos_errs']):.1f}m")
    ax.set_title('Position Error Over Time')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [m]')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # NIS comparison
    ax = axes[2]
    ax.plot(r_ekf['nis_log'], 'r-', lw=0.6, alpha=0.6, label='EKF NIS')
    ax.plot(r_ukf['nis_log'], 'g-', lw=0.6, alpha=0.6, label='UKF NIS')
    ax.axhline(2.0,  color='k',      ls='--', lw=1.5, label='Target (2.0)')
    ax.axhline(9.21, color='orange', ls='--', lw=1.5, label='Gate (9.21)')
    ax.set_title('NIS: EKF vs UKF')
    ax.set_xlabel('Step'); ax.set_ylabel('NIS')
    ax.set_ylim(0, 20); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # Head-to-head with original params
    r_ekf, r_ukf = compare_ekf_vs_ukf(q_a=0.5, q_alpha=0.1)
    plot_comparison(r_ekf, r_ukf,
                    save_path='/mnt/user-data/outputs/ekf_vs_ukf.png')

    # UKF sweep
    print("\n" + "=" * 60)
    print("UKF NIS SWEEP")
    print("=" * 60)
    q_a_values     = [0.5, 1.0, 2.0, 3.0]
    q_alpha_values = [0.1, 0.3, 0.5, 1.0]
    nis_grid, err_grid = nis_sweep_ukf(q_a_values, q_alpha_values)

    print("\nUKF Mean NIS grid:")
    print("q_a \\ q_alpha", q_alpha_values)
    for i, q_a in enumerate(q_a_values):
        row = [f"{nis_grid[i,j]:.2f}" for j in range(len(q_alpha_values))]
        print(f"  q_a={q_a:.1f}:  {row}")

    print("\nUKF Mean position error grid (m):")
    for i, q_a in enumerate(q_a_values):
        row = [f"{err_grid[i,j]:.1f}" for j in range(len(q_alpha_values))]
        print(f"  q_a={q_a:.1f}:  {row}")

    # Find best UKF params
    best_idx = np.unravel_index(np.nanargmin(err_grid), err_grid.shape)
    print(f"\nBest UKF params: q_a={q_a_values[best_idx[0]]}, "
          f"q_alpha={q_alpha_values[best_idx[1]]} "
          f"(err={err_grid[best_idx]:.2f}m, NIS={nis_grid[best_idx]:.2f})")
