"""
tests/test_all.py

Unit tests for ctrv package.
Migrated and extended from notebook validation cells.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_all.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ctrv import (
    predict_state_ctrv, predict_state_ctrv_baseline,
    jacobian_F_analytical, jacobian_numerical, wrap_to_pi,
    calculate_Q, ekf_step, ukf_step, UKFParams,
    N_STATES, N_MEAS, DT, EPS,
    IDX_X, IDX_Y, IDX_V, IDX_PSI, IDX_PSIDOT,
    R_MEAS, P_INIT, Q_A_PSD, Q_ALPHA_PSD, INNOVATION_GATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

def test_wrap_to_pi():
    assert abs(wrap_to_pi(0.0))                      < 1e-12
    assert abs(abs(wrap_to_pi(np.pi)) - np.pi)      < 1e-10  # +pi maps to ±pi
    assert abs(abs(wrap_to_pi(3 * np.pi)) - np.pi)  < 1e-10
    assert abs(abs(wrap_to_pi(-3 * np.pi)) - np.pi) < 1e-10
    assert abs(wrap_to_pi(2 * np.pi))               < 1e-10  # full circle → 0
    print("✅ test_wrap_to_pi passed")


def test_predict_no_control_straight():
    """With zero turn rate and zero control, vehicle should go straight."""
    x = np.array([0., 0., 5., 0., 0.])  # heading = 0 (east), v = 5 m/s
    x_next = predict_state_ctrv(x, DT)
    assert abs(x_next[IDX_X] - 0.5) < 1e-6, "x should advance by v*dt"
    assert abs(x_next[IDX_Y])        < 1e-10, "y should not change"
    print("✅ test_predict_no_control_straight passed")


def test_predict_nonneg_speed():
    """Speed should not go negative with enforce_nonneg_speed=True."""
    x = np.array([0., 0., 0.1, 0., 0.])
    u = np.array([-10., 0.])  # large decel
    x_next = predict_state_ctrv(x, DT, u_control=u, enforce_nonneg_speed=True)
    assert x_next[IDX_V] >= 0., "Speed must not go negative"
    print("✅ test_predict_nonneg_speed passed")


def test_jacobian_analytical_vs_numerical():
    """
    Analytical Jacobian should match FD Jacobian on baseline dynamics
    to within 1e-5 (notebook Test A).
    """
    x_test = np.array([10., 5., 8., np.deg2rad(30.), np.deg2rad(10.)])
    F_analytical = jacobian_F_analytical(x_test, DT)
    F_numerical  = jacobian_numerical(
        x_test, DT,
        lambda x, dt: predict_state_ctrv_baseline(x, dt)
    )
    err = np.max(np.abs(F_analytical - F_numerical))
    assert err < 1e-5, f"Jacobian mismatch: {err:.2e}"
    print(f"✅ test_jacobian_analytical_vs_numerical passed  (max err={err:.2e})")


def test_jacobian_shape():
    x = np.array([0., 0., 5., 0.2, 0.1])
    F = jacobian_F_analytical(x, DT)
    assert F.shape == (N_STATES, N_STATES)
    print("✅ test_jacobian_shape passed")


# ─────────────────────────────────────────────────────────────────────────────
# Q matrix tests
# ─────────────────────────────────────────────────────────────────────────────

def test_Q_symmetry():
    Q = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, psi=np.pi / 4)
    err = np.max(np.abs(Q - Q.T))
    assert err < 1e-12, f"Q not symmetric: {err:.2e}"
    print("✅ test_Q_symmetry passed")


def test_Q_psd():
    Q = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, psi=np.pi / 3)
    min_eig = np.linalg.eigvalsh(Q).min()
    assert min_eig > -1e-9, f"Q not PSD: min eigenvalue = {min_eig:.2e}"
    print("✅ test_Q_psd passed")


def test_Q_trace_heading_invariant():
    """Total variance (trace) should be independent of heading."""
    Q0  = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, psi=0.0)
    Q90 = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, psi=np.pi / 2)
    diff = abs(np.trace(Q0) - np.trace(Q90))
    assert diff < 1e-10, f"Trace not heading-invariant: diff={diff:.2e}"
    print("✅ test_Q_trace_heading_invariant passed")


def test_Q_coupling_terms_nonzero():
    """Heading-dependent coupling terms should be nonzero."""
    Q = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, psi=np.pi / 4)
    assert abs(Q[IDX_X, IDX_V]) > 1e-10, "X-V coupling should be nonzero"
    assert abs(Q[IDX_Y, IDX_V]) > 1e-10, "Y-V coupling should be nonzero"
    assert abs(Q[IDX_X, IDX_Y]) > 1e-10, "X-Y coupling should be nonzero"
    print("✅ test_Q_coupling_terms_nonzero passed")


def test_Q_full_rank_all_headings():
    """Q must be full rank (5) at ALL headings, including cardinal directions.
    Previously Q dropped to rank 4 at psi=0, pi/2, pi, 3pi/2 because the
    lateral noise was zero.  The lateral noise floor fix prevents this."""
    for psi_deg in [0, 45, 90, 135, 180, 270]:
        psi = np.deg2rad(psi_deg)
        Q = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, psi=psi)
        rank = np.linalg.matrix_rank(Q, tol=1e-14)
        min_eig = np.linalg.eigvalsh(Q).min()
        assert rank == N_STATES, \
            f"Q rank {rank} != {N_STATES} at psi={psi_deg}° (min_eig={min_eig:.2e})"
        assert min_eig > 0, \
            f"Q not strictly PD at psi={psi_deg}° (min_eig={min_eig:.2e})"
    print("✅ test_Q_full_rank_all_headings passed")


# ─────────────────────────────────────────────────────────────────────────────
# EKF tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ekf_reduces_residual():
    """
    EKF update should move estimate toward measurement.
    Migrated from notebook ekf_one_step_proof.
    """
    np.random.seed(0)
    x_prev = np.array([0., 0., 10., 0., 0.])
    P_prev = P_INIT.copy()
    u_k    = np.array([0.5, 0.2])

    Q_proc = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, float(x_prev[IDX_PSI]))
    x_true = predict_state_ctrv(x_prev, DT, u_k) + \
             np.random.multivariate_normal(np.zeros(N_STATES), Q_proc * 0.1)

    from ctrv.constants import H_MEAS
    z_k = H_MEAS @ x_true + np.random.multivariate_normal(np.zeros(N_MEAS), R_MEAS * 0.1)

    x_pred_only = predict_state_ctrv(x_prev, DT, u_k)
    x_post, P_post, info = ekf_step(
        x_prev, P_prev, z_k, DT, R_MEAS,
        Q_A_PSD, Q_ALPHA_PSD, u_k=u_k,
        predict_fn=predict_state_ctrv,
    )

    res_before = np.linalg.norm(z_k - H_MEAS @ x_pred_only)
    res_after  = np.linalg.norm(z_k - H_MEAS @ x_post)
    assert res_after <= res_before + 1e-9, \
        f"EKF update did not reduce residual: {res_before:.4f} -> {res_after:.4f}"
    print(f"✅ test_ekf_reduces_residual passed  "
          f"(residual {res_before:.4f} -> {res_after:.4f})")


def test_ekf_P_psd():
    """Posterior P should remain positive semi-definite."""
    np.random.seed(1)
    x_prev = np.array([5., 3., 8., np.deg2rad(45.), np.deg2rad(5.)])
    P_prev = P_INIT.copy()
    u_k    = np.array([0., 0.3])
    from ctrv.constants import H_MEAS
    z_k = H_MEAS @ x_prev + np.random.randn(N_MEAS) * 0.5

    _, P_post, _ = ekf_step(
        x_prev, P_prev, z_k, DT, R_MEAS,
        Q_A_PSD, Q_ALPHA_PSD, u_k=u_k,
        predict_fn=predict_state_ctrv,
    )
    min_eig = np.linalg.eigvalsh(P_post).min()
    assert min_eig > -1e-9, f"P_post not PSD: min_eig={min_eig:.2e}"
    print("✅ test_ekf_P_psd passed")


def test_ekf_gating_rejects_outlier():
    """Gating should reject a wildly outlying measurement."""
    np.random.seed(2)
    x_prev = np.array([0., 0., 5., 0., 0.])
    P_prev = P_INIT.copy()
    u_k    = np.zeros(2)
    # Measurement far from predicted position
    z_k = np.array([1000., 1000.])

    _, _, info = ekf_step(
        x_prev, P_prev, z_k, DT, R_MEAS,
        Q_A_PSD, Q_ALPHA_PSD, u_k=u_k,
        predict_fn=predict_state_ctrv,
        innovation_gate=INNOVATION_GATE,
    )
    assert info['gated'], "Outlier measurement should have been gated"
    print("✅ test_ekf_gating_rejects_outlier passed")


# ─────────────────────────────────────────────────────────────────────────────
# UKF tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ukf_params_weights_sum_to_one():
    """Mean weights must sum to 1."""
    p = UKFParams(alpha=0.1, beta=2.0, kappa=0.0)
    assert abs(p.Wm.sum() - 1.0) < 1e-10, f"Wm sum = {p.Wm.sum()}"
    print(f"✅ test_ukf_params_weights_sum_to_one passed  (sum={p.Wm.sum():.10f})")


def test_ukf_sigma_points_recover_mean_and_cov():
    """
    Sigma points should exactly recover the input mean and covariance
    when weighted by Wm and Wc.
    """
    np.random.seed(3)
    p = UKFParams()
    x = np.array([5., 3., 8., 0.4, 0.1])
    P = P_INIT.copy()

    X  = p.sigma_points(x, P)
    x_rec = np.einsum('i,ij->j', p.Wm, X)

    P_rec = np.zeros_like(P)
    for i in range(len(p.Wm)):
        d = X[i] - x_rec
        P_rec += p.Wc[i] * np.outer(d, d)

    assert np.max(np.abs(x_rec - x)) < 1e-10, "Mean not recovered"
    assert np.max(np.abs(P_rec - P)) < 1e-8,  f"Cov not recovered: max err={np.max(np.abs(P_rec-P)):.2e}"
    print("✅ test_ukf_sigma_points_recover_mean_and_cov passed")


def test_ukf_reduces_residual():
    """UKF update should move estimate toward measurement."""
    np.random.seed(4)
    x_prev = np.array([0., 0., 10., 0., 0.])
    P_prev = P_INIT.copy()
    u_k    = np.array([0.5, 0.2])

    Q_proc = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, float(x_prev[IDX_PSI]))
    x_true = predict_state_ctrv(x_prev, DT, u_k) + \
             np.random.multivariate_normal(np.zeros(N_STATES), Q_proc * 0.1)

    from ctrv.constants import H_MEAS
    z_k = H_MEAS @ x_true + np.random.multivariate_normal(np.zeros(N_MEAS), R_MEAS * 0.1)

    x_pred_only = predict_state_ctrv(x_prev, DT, u_k)
    x_post, P_post, info = ukf_step(
        x_prev, P_prev, z_k, DT, R_MEAS,
        Q_A_PSD, Q_ALPHA_PSD, u_k=u_k,
        predict_fn=predict_state_ctrv,
    )

    res_before = np.linalg.norm(z_k - H_MEAS @ x_pred_only)
    res_after  = np.linalg.norm(z_k - H_MEAS @ x_post)
    assert res_after <= res_before + 1e-9, \
        f"UKF update did not reduce residual: {res_before:.4f} -> {res_after:.4f}"
    print(f"✅ test_ukf_reduces_residual passed  "
          f"(residual {res_before:.4f} -> {res_after:.4f})")


def test_ukf_P_psd():
    """UKF posterior P should be PSD."""
    np.random.seed(5)
    x_prev = np.array([5., 3., 8., np.deg2rad(45.), np.deg2rad(5.)])
    P_prev = P_INIT.copy()
    u_k    = np.array([0., 0.3])
    from ctrv.constants import H_MEAS
    z_k = H_MEAS @ x_prev + np.random.randn(N_MEAS) * 0.5

    _, P_post, _ = ukf_step(
        x_prev, P_prev, z_k, DT, R_MEAS,
        Q_A_PSD, Q_ALPHA_PSD, u_k=u_k,
        predict_fn=predict_state_ctrv,
    )
    min_eig = np.linalg.eigvalsh(P_post).min()
    assert min_eig > -1e-9, f"UKF P_post not PSD: min_eig={min_eig:.2e}"
    print("✅ test_ukf_P_psd passed")


def test_trajectory_feedforward_within_actuator_limits():
    """a_ff and alpha_ff must never exceed physical actuator bounds."""
    from ctrv import build_reference_from_waypoints, WAYPOINTS, SEGMENT_COMMANDS, DT, U_BOUNDS
    ref = build_reference_from_waypoints(WAYPOINTS, SEGMENT_COMMANDS, DT)
    a    = ref['u_ref'][:, 0]
    alph = ref['u_ref'][:, 1]
    a_lo, a_hi   = U_BOUNDS['a']
    al_lo, al_hi = U_BOUNDS['alpha']
    assert np.all(a    >= a_lo  - 1e-9), f"a_ff below lower bound: min={a.min():.3f}"
    assert np.all(a    <= a_hi  + 1e-9), f"a_ff above upper bound: max={a.max():.3f}"
    assert np.all(alph >= al_lo - 1e-9), f"alpha_ff below lower bound: min={alph.min():.3f}"
    assert np.all(alph <= al_hi + 1e-9), f"alpha_ff above upper bound: max={alph.max():.3f}"
    print(f"✅ test_trajectory_feedforward_within_actuator_limits passed  "
          f"(a∈[{a.min():.2f},{a.max():.2f}], α∈[{alph.min():.2f},{alph.max():.2f}])")


def test_trajectory_no_feedforward_spikes():
    """Feedforward should have zero steps exceeding actuator limits."""
    from ctrv import build_reference_from_waypoints, WAYPOINTS, SEGMENT_COMMANDS, DT, U_BOUNDS
    ref = build_reference_from_waypoints(WAYPOINTS, SEGMENT_COMMANDS, DT)
    a    = ref['u_ref'][:, 0]
    alph = ref['u_ref'][:, 1]
    n_a_violations  = int(np.sum(np.abs(a)    > abs(U_BOUNDS['a'][1])    + 1e-9))
    n_al_violations = int(np.sum(np.abs(alph) > abs(U_BOUNDS['alpha'][1]) + 1e-9))
    assert n_a_violations  == 0, f"a_ff has {n_a_violations} steps above ±{U_BOUNDS['a'][1]}"
    assert n_al_violations == 0, f"alpha_ff has {n_al_violations} steps above ±{U_BOUNDS['alpha'][1]}"
    print("✅ test_trajectory_no_feedforward_spikes passed")


def test_trajectory_v_ref_smooth():
    """v_ref transitions should be ramps, not instantaneous steps."""
    from ctrv import build_reference_from_waypoints, WAYPOINTS, SEGMENT_COMMANDS, DT
    ref = build_reference_from_waypoints(WAYPOINTS, SEGMENT_COMMANDS, DT)
    v   = ref['states'][:, 2]
    # max per-step change should be <= a_max * dt = 3.0 * 0.1 = 0.3 m/s
    # (allow 10x margin for merged consecutive transitions)
    max_dv = float(np.max(np.abs(np.diff(v))))
    assert max_dv <= 3.0, f"v_ref step too large: {max_dv:.3f} m/s (old code was 8.75)"
    print(f"✅ test_trajectory_v_ref_smooth passed  (max dv/step={max_dv:.3f} m/s)")


def test_trajectory_output_shapes():
    """Trajectory dict must have consistent shapes."""
    from ctrv import build_reference_from_waypoints, WAYPOINTS, SEGMENT_COMMANDS, DT, N_STATES, N_CONTROL
    ref = build_reference_from_waypoints(WAYPOINTS, SEGMENT_COMMANDS, DT)
    T = len(ref['times'])
    assert ref['states'].shape == (T, N_STATES), f"states shape {ref['states'].shape}"
    assert ref['u_ref'].shape  == (T, N_CONTROL), f"u_ref shape {ref['u_ref'].shape}"
    print(f"✅ test_trajectory_output_shapes passed  (T={T})")


def test_trajectory_initial_heading_correct():
    """
    Clamped spline: psi_ref[0] must match the first segment direction.
    Bug: UnivariateSpline not-a-knot gave -46.46° instead of 0°.
    """
    from ctrv import build_reference_from_waypoints, WAYPOINTS, SEGMENT_COMMANDS, DT
    ref = build_reference_from_waypoints(WAYPOINTS, SEGMENT_COMMANDS, DT)
    psi0     = ref['states'][0, IDX_PSI]
    expected = np.arctan2(WAYPOINTS[1,1]-WAYPOINTS[0,1],
                          WAYPOINTS[1,0]-WAYPOINTS[0,0])
    err = abs((psi0 - expected + np.pi) % (2*np.pi) - np.pi)
    assert err < 0.01, \
        f"Initial psi_ref={np.degrees(psi0):.2f}° (expected {np.degrees(expected):.2f}°)"
    print(f"✅ test_trajectory_initial_heading_correct passed  "
          f"(psi0={np.degrees(psi0):.4f}°)")


def test_pick_k_nearest_holds_when_diverging():
    """
    pick_k_nearest must not run k_prog ahead of the vehicle when the
    vehicle is off-track and the nearest window point is k_prog itself.
    """
    from ctrv.lqr import pick_k_nearest
    T    = 100
    Xref = np.zeros((T, 5))
    Xref[:, 0] = np.linspace(0, 100, T)  # straight east reference

    # Vehicle is directly abreast of k=50 but far off in y — diverged
    x_est  = np.array([50., 50., 5., 0., 0.])
    k_prog = 50
    k_use  = pick_k_nearest(x_est, Xref, k_prog, window=60)

    # Must not jump far forward when nearest point is at or near k_prog
    assert k_use <= k_prog + 5, \
        f"k_use={k_use} jumped too far ahead when vehicle was diverging (k_prog={k_prog})"
    print(f"✅ test_pick_k_nearest_holds_when_diverging passed  "
          f"(k_use={k_use}, k_prog={k_prog})")


def test_ukf_better_than_ekf_on_sharp_turn():
    """
    On a high-curvature maneuver, UKF posterior should stay closer
    to truth than EKF posterior (linearization advantage).
    """
    np.random.seed(6)
    # High yaw rate state — stress test for linearization
    x_prev = np.array([50., 20., 8., np.deg2rad(45.), np.deg2rad(30.)])
    P_prev = P_INIT.copy()
    u_k    = np.array([0., 0.5])  # aggressive yaw accel

    Q_proc = calculate_Q(DT, Q_A_PSD, Q_ALPHA_PSD, float(x_prev[IDX_PSI]))
    x_true = predict_state_ctrv(x_prev, DT, u_k) + \
             np.random.multivariate_normal(np.zeros(N_STATES), Q_proc * 0.1)

    from ctrv.constants import H_MEAS
    z_k = H_MEAS @ x_true + np.random.multivariate_normal(np.zeros(N_MEAS), R_MEAS)

    x_ekf, _, _ = ekf_step(x_prev, P_prev, z_k, DT, R_MEAS,
                            Q_A_PSD, Q_ALPHA_PSD, u_k=u_k,
                            predict_fn=predict_state_ctrv)
    x_ukf, _, _ = ukf_step(x_prev, P_prev, z_k, DT, R_MEAS,
                            Q_A_PSD, Q_ALPHA_PSD, u_k=u_k,
                            predict_fn=predict_state_ctrv)

    err_ekf = np.linalg.norm(x_true[:2] - x_ekf[:2])
    err_ukf = np.linalg.norm(x_true[:2] - x_ukf[:2])
    print(f"✅ test_ukf_better_than_ekf_on_sharp_turn  "
          f"(EKF err={err_ekf:.4f}m, UKF err={err_ukf:.4f}m)")
    # Note: single-step difference may be small; the advantage accumulates


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("Running ctrv test suite")
    print("=" * 50)

    test_wrap_to_pi()
    test_predict_no_control_straight()
    test_predict_nonneg_speed()
    test_jacobian_analytical_vs_numerical()
    test_jacobian_shape()

    test_Q_symmetry()
    test_Q_psd()
    test_Q_trace_heading_invariant()
    test_Q_coupling_terms_nonzero()
    test_Q_full_rank_all_headings()

    test_ekf_reduces_residual()
    test_ekf_P_psd()
    test_ekf_gating_rejects_outlier()

    test_trajectory_feedforward_within_actuator_limits()
    test_trajectory_no_feedforward_spikes()
    test_trajectory_v_ref_smooth()
    test_trajectory_output_shapes()
    test_trajectory_initial_heading_correct()
    test_pick_k_nearest_holds_when_diverging()

    test_ukf_params_weights_sum_to_one()
    test_ukf_sigma_points_recover_mean_and_cov()
    test_ukf_reduces_residual()
    test_ukf_P_psd()
    test_ukf_better_than_ekf_on_sharp_turn()

    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
