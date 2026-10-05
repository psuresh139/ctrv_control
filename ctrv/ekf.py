"""EKF and scaled UKF for controlled CTRV with local angle-chart statistics.

Both support GPS, heading-only observations, and prediction-only operation.
The measurement matrix is a selector for Euclidean states or unit heading;
this is not a general nonlinear measurement API. The UKF handles additive
process covariance in its linear measurement update using Joseph form.
Neither estimator is assumed superior without paired-seed benchmarks.
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
        if n <= 0 or alpha <= 0 or n + kappa <= 0:
            raise ValueError('Invalid sigma-point scaling parameters')
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
    anchor = X_pred[0, IDX_PSI]
    x_pred[IDX_PSI] = wrap_to_pi(anchor + Wm @ wrap_to_pi(X_pred[:, IDX_PSI] - anchor))

    # ── (5) Predicted covariance ──────────────────────────────────────────
    P_pred = Q_k.copy()
    for i in range(ns):
        d = X_pred[i] - x_pred
        d[IDX_PSI] = wrap_to_pi(d[IDX_PSI])
        P_pred += Wc[i] * np.outer(d, d)
    P_pred = _enforce_psd(P_pred)

    # H is linear in the local angle chart. Include additive process noise
    # in BOTH the innovation covariance and state-measurement covariance.
    z_pred = H_meas @ x_pred
    S = H_meas @ P_pred @ H_meas.T + R_meas
    S = 0.5 * (S + S.T)
    P_xz = P_pred @ H_meas.T
    if z_k is None or n_meas == 0:
        return x_pred, P_pred, {'nis': np.nan, 'gated': False,
                                'innov': np.empty(0), 'innov_cov': S}

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

        IKH = np.eye(n) - K @ H_meas
        P_post = _enforce_psd(IKH @ P_pred @ IKH.T + K @ R_meas @ K.T)

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

    if z_k is None or n_meas == 0:
        return x_pred, P_pred, {'nis': np.nan, 'gated': False,
                                'innov': np.empty(0), 'innov_cov': np.empty((0, 0))}

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

def compute_nis_summary(nis_log: list, n_meas: int = N_MEAS,
                        degrees_of_freedom=None, gate_probability=0.99) -> dict:
    """Summarize ALL innovations; missing observations are NaN, not zero.

    Mixed measurement dimensions are normalized by their degrees of freedom.
    This is descriptive health information, not a formal consistency test.
    """
    from scipy.stats import chi2
    arr = np.asarray(nis_log, dtype=float)
    dof = np.full(arr.shape, n_meas) if degrees_of_freedom is None else np.asarray(degrees_of_freedom)
    mask = np.isfinite(arr) & (dof > 0)
    arr, dof = arr[mask], dof[mask]
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)) if arr.size else None,
        'std': float(np.std(arr)) if arr.size else None,
        'min': float(np.min(arr)) if arr.size else None,
        'max': float(np.max(arr)) if arr.size else None,
        'pct_above_gate': float(100 * np.mean(arr > chi2.ppf(gate_probability, dof))) if arr.size else None,
        'target': float(np.mean(dof)) if arr.size else None,
        'normalized_mean': float(np.mean(arr / dof)) if arr.size else None,
    }


def print_nis_report(summary: dict, label: str = '') -> None:
    print(f"NIS report {label}: {summary['count']} observed innovations")
    if summary['count']:
        print(f"  Mean: {summary['mean']:.3f}; mean DOF: {summary['target']:.2f}")
        print(f"  Mean NIS/DOF: {summary['normalized_mean']:.3f}")
        print(f"  Above dimension-aware 99% gate: {summary['pct_above_gate']:.1f}%")


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
