"""
ctrv/ekf.py

State estimators for the 5-state CTRV model.

Contains two estimators with identical external interfaces:

  ekf_step(...)  — Extended Kalman Filter (kept for comparison/baseline)
  ukf_step(...)  — Unscented Kalman Filter (recommended)

Both return (x_post, P_post, info) so run_simulation.py switches
estimators with a single line change:

    # EKF
    x_est, P_est, info = ekf_step(x_est, P_est, z_k, ...)

    # UKF — drop-in replacement
    x_est, P_est, info = ukf_step(x_est, P_est, z_k, ...)

Why UKF over EKF for CTRV
--------------------------
The EKF linearizes f(x) about the current estimate using a first-order
Taylor expansion (the Jacobian F). For the CTRV model, the position
update equations are nonlinear in both psi and psi_dot simultaneously:

    x_{k+1} = x_k + (v / psi_dot) * (sin(psi + psi_dot*dt) - sin(psi))

Near corners, where psi_dot is large and changing rapidly, the
linearization error in F accumulates faster than Q can compensate.
The NIS diagnostic confirmed the filter is not smug — it is simply
operating outside its valid linearization regime.

The UKF avoids this entirely. Rather than linearizing, it propagates
2n+1 = 11 deterministically chosen sigma points through the full
nonlinear dynamics and recovers the predicted mean and covariance
from the transformed ensemble. This captures second-order effects
the EKF misses, at no extra model-derivation cost (no Jacobians needed).

UKF sigma-point parameters (Merwe scaled scheme)
-------------------------------------------------
  alpha : spread of sigma points around the mean
          small (1e-3) = tight / conservative
          large (1.0)  = wide, captures heavier tails
          CTRV standard: alpha = 0.1

  beta  : prior distribution knowledge
          beta = 2 is optimal for Gaussian distributions

  kappa : secondary scaling; kappa = 0 standard for state estimation

From these: lambda = alpha^2 * (n + kappa) - n
With n=5, alpha=0.1, kappa=0: lambda = 0.01*5 - 5 = -4.95  (valid)

The 2n+1 = 11 sigma points are:
  X_0         = x_mean
  X_{1..n}    = x_mean + col_i( sqrt((n+lambda)*P) )
  X_{n+1..2n} = x_mean - col_i( sqrt((n+lambda)*P) )

Weights:
  Wm[0] = lambda / (n + lambda)
  Wc[0] = lambda / (n + lambda) + (1 - alpha^2 + beta)
  Wm[i] = Wc[i] = 1 / (2*(n+lambda))   for i = 1..2n
"""

import numpy as np
from ctrv.constants import (
    IDX_PSI, N_STATES, N_MEAS, N_CONTROL, H_MEAS
)
from ctrv.model import wrap_to_pi, jacobian_F_analytical
from ctrv.noise import calculate_Q


# ─────────────────────────────────────────────────────────────────────────────
# Sigma-point parameter class
# ─────────────────────────────────────────────────────────────────────────────

class UKFParams:
    """
    Merwe scaled sigma-point weights and sigma-point generator.

    Parameters
    ----------
    alpha : float  sigma-point spread (default 0.1)
    beta  : float  distribution prior (default 2.0)
    kappa : float  secondary scaling  (default 0.0)
    n     : int    state dimension    (default N_STATES = 5)
    """

    def __init__(self, alpha: float = 0.1, beta: float = 2.0,
                 kappa: float = 0.0, n: int = N_STATES):
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa
        self.n     = n

        lam        = alpha**2 * (n + kappa) - n
        self.lam   = lam

        # Mean weights: shape (2n+1,)
        self.Wm    = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.Wm[0] = lam / (n + lam)

        # Covariance weights
        self.Wc    = self.Wm.copy()
        self.Wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute 2n+1 sigma points from mean x and covariance P.

        Uses Cholesky of (n+lambda)*P. Falls back to eigendecomposition
        if P is near-singular (should not happen in normal operation).

        Returns
        -------
        X : (2n+1, n) ndarray
        """
        n   = self.n
        lam = self.lam
        X   = np.zeros((2 * n + 1, n))
        X[0] = x

        M = (n + lam) * _enforce_psd(P)
        try:
            S = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.maximum(eigvals, 0.0)
            S = eigvecs @ np.diag(np.sqrt(eigvals))

        for i in range(n):
            X[i + 1]         = x + S[:, i]
            X[i + 1 + n]     = x - S[:, i]

        return X


# Module-level default — avoids re-computing weights every call
_DEFAULT_UKF_PARAMS = UKFParams(alpha=0.1, beta=2.0, kappa=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# UKF — primary estimator
# ─────────────────────────────────────────────────────────────────────────────

def ukf_step(
    x_prev:   np.ndarray,
    P_prev:   np.ndarray,
    z_k:      np.ndarray,
    dt:       float,
    R_meas:   np.ndarray,
    q_a:      float,
    q_alpha:  float,
    u_k:      np.ndarray = None,
    predict_fn = None,
    innovation_gate: float = None,
    params: UKFParams = None,
    H_meas: np.ndarray = None,
) -> tuple:
    """
    Single UKF predict-update cycle (Merwe scaled sigma-point scheme).

    Parameters
    ----------
    x_prev          : (5,)   prior state estimate
    P_prev          : (5,5)  prior covariance
    z_k             : (2,)   measurement [x_meas, y_meas]
    dt              : float  timestep (s)
    R_meas          : (2,2)  measurement noise covariance
    q_a             : float  longitudinal accel noise PSD
    q_alpha         : float  yaw accel noise PSD
    u_k             : (2,)   control [a, alpha]; zeros if None
    predict_fn      : callable(x, dt, u) -> (5,)
    innovation_gate : float  Mahalanobis chi2 gate; None disables
    params          : UKFParams; uses default (alpha=0.1) if None

    Returns
    -------
    x_post : (5,)  posterior state
    P_post : (5,5) posterior covariance
    info   : dict  {'nis', 'gated', 'innov', 'innov_cov'}
    """
    if u_k is None:
        u_k = np.zeros(N_CONTROL)
    if predict_fn is None:
        raise ValueError("predict_fn must be provided")
    if params is None:
        params = _DEFAULT_UKF_PARAMS
    if H_meas is None:
        H_meas = H_MEAS

    n  = params.n
    Wm = params.Wm
    Wc = params.Wc
    ns = 2 * n + 1
    n_meas = H_meas.shape[0]

    # Detect which measurement rows observe heading (row dot [0,0,0,1,0] ≈ 1)
    _psi_col = np.array([0., 0., 0., 1., 0.])
    _meas_has_heading = [abs(H_meas[i] @ _psi_col) > 0.5 for i in range(n_meas)]

    # ── (1) Heading-dependent Q ───────────────────────────────────────────
    Q_k = calculate_Q(dt, q_a, q_alpha, float(x_prev[IDX_PSI]))

    # ── (2) Sigma points from prior ───────────────────────────────────────
    X = params.sigma_points(x_prev, P_prev)

    # ── (3) Propagate each sigma point through full nonlinear dynamics ────
    X_pred = np.zeros_like(X)
    for i in range(ns):
        X_pred[i] = predict_fn(X[i], dt, u_k)
        X_pred[i, IDX_PSI] = wrap_to_pi(X_pred[i, IDX_PSI])

    # ── (4) Predicted mean ────────────────────────────────────────────────
    x_pred = np.einsum('i,ij->j', Wm, X_pred)
    x_pred[IDX_PSI] = wrap_to_pi(x_pred[IDX_PSI])

    # ── (5) Predicted covariance ──────────────────────────────────────────
    P_pred = Q_k.copy()
    for i in range(ns):
        d = X_pred[i] - x_pred
        d[IDX_PSI] = wrap_to_pi(d[IDX_PSI])
        P_pred += Wc[i] * np.outer(d, d)
    P_pred = _enforce_psd(P_pred)

    # ── (6) Predicted measurement mean ───────────────────────────────────
    Z_pred = X_pred @ H_meas.T          # (ns, n_meas)
    z_pred = np.einsum('i,ij->j', Wm, Z_pred)

    # ── (7) Innovation covariance S and cross-covariance P_xz ────────────
    S    = R_meas.copy()
    P_xz = np.zeros((n, n_meas))
    for i in range(ns):
        dz = Z_pred[i] - z_pred
        # Wrap heading components of measurement residual
        for j in range(n_meas):
            if _meas_has_heading[j]:
                dz[j] = wrap_to_pi(dz[j])
        dx = X_pred[i] - x_pred
        dx[IDX_PSI] = wrap_to_pi(dx[IDX_PSI])
        S    += Wc[i] * np.outer(dz, dz)
        P_xz += Wc[i] * np.outer(dx, dz)
    S = 0.5 * (S + S.T)

    # ── (8) Innovation and NIS ────────────────────────────────────────────
    innov = z_k - z_pred
    for j in range(n_meas):
        if _meas_has_heading[j]:
            innov[j] = wrap_to_pi(innov[j])
    nis   = float(innov @ np.linalg.solve(S, innov))

    # ── (9) Gating ────────────────────────────────────────────────────────
    gated = (innovation_gate is not None) and (nis > innovation_gate)

    if gated:
        x_post = x_pred.copy()
        P_post = P_pred.copy()
    else:
        # ── (10) Kalman gain and posterior update ─────────────────────────
        K      = P_xz @ np.linalg.solve(S, np.eye(n_meas))
        x_post = x_pred + K @ innov
        x_post[IDX_PSI] = wrap_to_pi(x_post[IDX_PSI])

        # UKF covariance update (equivalent to Joseph form)
        P_post = _enforce_psd(P_pred - K @ S @ K.T)

    return x_post, P_post, {
        'nis': nis, 'gated': gated,
        'innov': innov, 'innov_cov': S,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EKF — baseline, kept for comparison
# ─────────────────────────────────────────────────────────────────────────────

def ekf_step(
    x_prev:   np.ndarray,
    P_prev:   np.ndarray,
    z_k:      np.ndarray,
    dt:       float,
    R_meas:   np.ndarray,
    q_a:      float,
    q_alpha:  float,
    u_k:      np.ndarray = None,
    predict_fn = None,
    innovation_gate: float = None,
    H_meas:   np.ndarray = None,
) -> tuple:
    """
    Single EKF predict-update cycle.

    Uses numerical Jacobian of the actual controlled dynamics for
    covariance propagation. Supports 2-DOF (GPS) or 3-DOF (GPS+heading)
    measurement models via the H_meas parameter.
    """
    if u_k is None:
        u_k = np.zeros(N_CONTROL)
    if predict_fn is None:
        raise ValueError("predict_fn must be provided")
    if H_meas is None:
        H_meas = H_MEAS

    n_meas = H_meas.shape[0]

    Q_k    = calculate_Q(dt, q_a, q_alpha, float(x_prev[IDX_PSI]))
    x_pred = predict_fn(x_prev, dt, u_k)

    from ctrv.model import jacobian_numerical
    F      = jacobian_numerical(
        x_prev, dt,
        lambda x, dt_: predict_fn(x, dt_, u_k),
    )
    P_pred = _enforce_psd(F @ P_prev @ F.T + Q_k)

    innov  = z_k - H_meas @ x_pred
    # Wrap heading component(s) of innovation
    _psi_col = np.array([0., 0., 0., 1., 0.])
    for j in range(n_meas):
        if abs(H_meas[j] @ _psi_col) > 0.5:
            innov[j] = wrap_to_pi(innov[j])

    S      = H_meas @ P_pred @ H_meas.T + R_meas
    S      = 0.5 * (S + S.T)
    nis    = float(innov @ np.linalg.solve(S, innov))
    gated  = (innovation_gate is not None) and (nis > innovation_gate)

    if gated:
        x_post, P_post = x_pred.copy(), P_pred.copy()
    else:
        K      = P_pred @ H_meas.T @ np.linalg.solve(S, np.eye(n_meas))
        x_post = x_pred + K @ innov
        x_post[IDX_PSI] = wrap_to_pi(x_post[IDX_PSI])
        IKH    = np.eye(N_STATES) - K @ H_meas
        P_post = _enforce_psd(IKH @ P_pred @ IKH.T + K @ R_meas @ K.T)

    return x_post, P_post, {
        'nis': nis, 'gated': gated,
        'innov': innov, 'innov_cov': S,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NIS diagnostics  (shared by both estimators)
# ─────────────────────────────────────────────────────────────────────────────

def compute_nis_summary(nis_log: list, n_meas: int = N_MEAS) -> dict:
    """
    Summarize filter health from a log of NIS values.
    Target: mean(NIS) ≈ n_meas = 2.
    """
    arr = np.array(nis_log)
    return {
        'mean':           float(np.mean(arr)),
        'std':            float(np.std(arr)),
        'min':            float(np.min(arr)),
        'max':            float(np.max(arr)),
        'pct_above_gate': float(np.mean(arr > 9.21) * 100),
        'target':         float(n_meas),
    }


def print_nis_report(summary: dict, label: str = '') -> None:
    tag = f" [{label}]" if label else ''
    print(f"=== NIS Filter Health Report{tag} ===")
    print(f"  Mean NIS : {summary['mean']:.3f}  (target ≈ {summary['target']:.1f})")
    print(f"  Std  NIS : {summary['std']:.3f}")
    print(f"  Range    : [{summary['min']:.2f}, {summary['max']:.2f}]")
    print(f"  % above gate (9.21): {summary['pct_above_gate']:.1f}%")
    ratio = summary['mean'] / summary['target']
    if ratio > 2.0:
        print(f"  ⚠  NIS is {ratio:.1f}x target — Q likely too SMALL")
    elif ratio < 0.5:
        print(f"  ⚠  NIS is {ratio:.1f}x target — Q likely too LARGE")
    else:
        print(f"  ✅ NIS within acceptable range ({ratio:.2f}x target)")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_psd(P: np.ndarray, min_eig: float = 1e-9) -> np.ndarray:
    """Symmetrize P and clip negative eigenvalues. Safety net only."""
    P = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P)
    if eigvals.min() < min_eig:
        eigvals = np.maximum(eigvals, min_eig)
        P = eigvecs @ np.diag(eigvals) @ eigvecs.T
        P = 0.5 * (P + P.T)
    return P
