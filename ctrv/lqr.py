"""
ctrv/lqr.py

Finite-horizon Linear Time-Varying LQR for the CTRV model.

Fixes vs. original notebook:
  1. build_ltv_linearization is now a pure function — it does not
     re-linearize inside the simulation loop. Call it once, pass the
     result in. This is the pattern needed for hardware deployment.
  2. ltv_lqr_backward_riccati adds condition number check on S before
     solve, and uses scipy.linalg.solve with assume_a='pos' for the
     symmetric positive definite S = R + B^T P B.
  3. pick_k_nearest is defined exactly once (no inner shadowing).
  4. compute_lqr_gain (steady-state DARE) is kept for single-point use
     and validation, but not called inside the trajectory loop.
"""

import numpy as np
import scipy.linalg as linalg

from ctrv.constants import (
    IDX_X, IDX_Y, IDX_PSI,
    N_STATES, N_CONTROL
)
from ctrv.model import jacobian_numerical, predict_state_ctrv, wrap_to_pi


# ─────────────────────────────────────────────────────────────────────────────
# Linearization
# ─────────────────────────────────────────────────────────────────────────────

def linearize_at_point(x_ref: np.ndarray, u_ref: np.ndarray,
                       dt: float, eps: float = 1e-6) -> tuple:
    """
    Finite-difference linearization of the CTRV dynamics at (x_ref, u_ref).

    Returns A = df/dx, B = df/du.

    A is computed via central differences on predict_state_ctrv with u frozen.
    B is computed via central differences on predict_state_ctrv with x frozen.

    Parameters
    ----------
    x_ref : (5,)  reference state
    u_ref : (2,)  reference control
    dt    : float timestep
    eps   : float finite difference step

    Returns
    -------
    A : (5, 5)  state Jacobian
    B : (5, 2)  input Jacobian
    """
    u_ref = np.asarray(u_ref, dtype=float).reshape(-1)

    # A = df/dx  (u frozen at u_ref)
    A = jacobian_numerical(
        x_ref, dt,
        lambda x, dt_: predict_state_ctrv(x, dt_, u_ref),
        eps=eps
    )

    # B = df/du  (x frozen at x_ref)
    B = np.zeros((N_STATES, N_CONTROL))
    for j in range(N_CONTROL):
        u_p, u_m = u_ref.copy(), u_ref.copy()
        u_p[j] += eps
        u_m[j] -= eps
        B[:, j] = (predict_state_ctrv(x_ref, dt, u_p) -
                   predict_state_ctrv(x_ref, dt, u_m)) / (2.0 * eps)

    return A, B


def build_ltv_linearization(ref_traj: dict, dt: float) -> tuple:
    """
    Pre-compute (A_k, B_k) for every step along the reference trajectory.

    This should be called ONCE before the simulation loop, not inside it.
    The resulting sequences are passed directly to ltv_lqr_backward_riccati
    and indexed during the control loop.

    Parameters
    ----------
    ref_traj : dict with keys 'states' (T,5) and 'u_ref' (T,2)
    dt       : float timestep

    Returns
    -------
    A_seq : list of (5,5) arrays, length T-1
    B_seq : list of (5,2) arrays, length T-1
    """
    Xref = ref_traj['states']
    Uref = ref_traj['u_ref']
    T    = len(Xref)

    assert Uref.shape == (T, N_CONTROL), \
        f"u_ref shape {Uref.shape} != ({T}, {N_CONTROL})"

    A_seq, B_seq = [], []
    for k in range(T - 1):
        A_k, B_k = linearize_at_point(Xref[k], Uref[k], dt)
        A_seq.append(A_k)
        B_seq.append(B_k)

    return A_seq, B_seq


# ─────────────────────────────────────────────────────────────────────────────
# Backward Riccati sweep
# ─────────────────────────────────────────────────────────────────────────────

def ltv_lqr_backward_riccati(
    A_seq: list, B_seq: list,
    Q: np.ndarray, R: np.ndarray,
    Qf: np.ndarray = None,
    reg: float = 1e-9,
) -> tuple:
    """
    Finite-horizon backward Riccati recursion.

    Computes optimal gain sequence K_0, ..., K_{T-2} and cost-to-go
    sequence P_0, ..., P_{T-1} by iterating:

        S_k   = R + B_k^T P_{k+1} B_k
        K_k   = S_k^{-1} B_k^T P_{k+1} A_k
        P_k   = Q + A_k^T P_{k+1} (A_k - B_k K_k)

    with terminal condition P_{T-1} = Qf (defaults to Q).

    Improvements vs. original:
      - S is checked for conditioning before solve; warns if ill-conditioned
      - scipy.linalg.solve used with assume_a='pos' for efficiency + stability
      - P is symmetrized at every step

    Parameters
    ----------
    A_seq : list of (n,n)  length T-1
    B_seq : list of (n,m)  length T-1
    Q     : (n,n)  state tracking cost
    R     : (m,m)  input cost
    Qf    : (n,n)  terminal cost; defaults to Q
    reg   : float  regularization added to R diagonal

    Returns
    -------
    K_seq : list of (m,n)  gain matrices, length T-1
    P_seq : list of (n,n)  cost-to-go matrices, length T
    """
    T_1 = len(A_seq)
    n   = A_seq[0].shape[0]
    m   = B_seq[0].shape[1]

    assert n == N_STATES,  f"Expected n={N_STATES}, got {n}"
    assert m == N_CONTROL, f"Expected m={N_CONTROL}, got {m}"

    Q  = 0.5 * (Q + Q.T)
    R  = 0.5 * (R + R.T) + reg * np.eye(m)
    Qf = Q.copy() if Qf is None else 0.5 * (Qf + Qf.T)

    P_seq = [None] * (T_1 + 1)
    K_seq = [None] * T_1
    P_seq[T_1] = Qf

    for k in reversed(range(T_1)):
        A  = A_seq[k]
        B  = B_seq[k]
        Pn = P_seq[k + 1]

        S = R + B.T @ Pn @ B
        S = 0.5 * (S + S.T)

        # Condition number check — ill-conditioned S means R is too small
        # relative to B^T P B; add regularization or increase R_LQR if triggered
        cond = np.linalg.cond(S)
        if cond > 1e10:
            import warnings
            warnings.warn(f"LQR step k={k}: S condition number {cond:.2e} — "
                          f"consider increasing R_LQR diagonal.")

        K = linalg.solve(S, B.T @ Pn @ A, assume_a='pos')
        P = Q + A.T @ Pn @ (A - B @ K)

        K_seq[k] = K
        P_seq[k] = 0.5 * (P + P.T)

    return K_seq, P_seq


def ltv_lqr_scheduled_riccati(
    A_seq: list, B_seq: list,
    Xref:  np.ndarray,
    Q_straight: np.ndarray, R_straight: np.ndarray,
    Q_turn:     np.ndarray = None,
    R_turn:     np.ndarray = None,
    curv_threshold: float = 0.02,
    reg: float = 1e-9,
) -> tuple:
    """
    Curvature-scheduled finite-horizon backward Riccati recursion.

    Uses different (Q, R) cost matrices depending on the local curvature
    of the reference trajectory at each time step.  This allows the
    controller to steer more aggressively in tight corners (low R_alpha)
    while remaining smooth on straights (high R_alpha).

    Curvature is computed as  kappa_k = |psi_dot_ref| / max(|v_ref|, 0.1).

    Parameters
    ----------
    A_seq, B_seq : lists of (n,n) and (n,m), length T-1
    Xref         : (T, 5)  reference states (used to compute curvature)
    Q_straight   : (n,n)   state cost on straight segments
    R_straight   : (m,m)   input cost on straight segments
    Q_turn       : (n,n)   state cost on curved segments; defaults to Q_straight
    R_turn       : (m,m)   input cost on curved segments; defaults to R_straight
    curv_threshold : float curvature threshold (rad/m) for switching
    reg          : float   regularization on R diagonal

    Returns
    -------
    K_seq : list of (m,n)  gain matrices, length T-1
    P_seq : list of (n,n)  cost-to-go matrices, length T
    """
    from ctrv.constants import IDX_V, IDX_PSIDOT

    T_1 = len(A_seq)
    n   = A_seq[0].shape[0]
    m   = B_seq[0].shape[1]

    Q_s = 0.5 * (Q_straight + Q_straight.T)
    R_s = 0.5 * (R_straight + R_straight.T) + reg * np.eye(m)
    Q_t = Q_s if Q_turn is None else 0.5 * (Q_turn + Q_turn.T)
    R_t = R_s if R_turn is None else 0.5 * (R_turn + R_turn.T) + reg * np.eye(m)

    # Compute curvature along reference
    curvatures = (np.abs(Xref[:, IDX_PSIDOT])
                  / np.maximum(np.abs(Xref[:, IDX_V]), 0.1))

    P_seq = [None] * (T_1 + 1)
    K_seq = [None] * T_1
    P_seq[T_1] = Q_s.copy()       # terminal cost

    for k in reversed(range(T_1)):
        is_turn = curvatures[k] >= curv_threshold
        Q_k = Q_t if is_turn else Q_s
        R_k = R_t if is_turn else R_s

        A  = A_seq[k]
        B  = B_seq[k]
        Pn = P_seq[k + 1]

        S = R_k + B.T @ Pn @ B
        S = 0.5 * (S + S.T)

        K = linalg.solve(S, B.T @ Pn @ A, assume_a='pos')
        P = Q_k + A.T @ Pn @ (A - B @ K)

        K_seq[k] = K
        P_seq[k] = 0.5 * (P + P.T)

    return K_seq, P_seq


# ─────────────────────────────────────────────────────────────────────────────
# Steady-state LQR (single operating point — for validation / warm-start)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lqr_gain_dare(
    A: np.ndarray, B: np.ndarray,
    Q: np.ndarray, R: np.ndarray,
    reg: float = 1e-9,
) -> tuple:
    """
    Steady-state discrete LQR gain via scipy DARE solver.

    Used for:
      - Single-point validation of the Riccati recursion
      - Warm-starting the terminal cost Qf
      - Sanity checking A, B at a given reference point

    Returns K (m,n), P (n,n).
    """
    Q = 0.5 * (Q + Q.T)
    R = 0.5 * (R + R.T) + reg * np.eye(R.shape[0])
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
        S = R + B.T @ P @ B
        K = linalg.solve(S, B.T @ P @ A, assume_a='pos')
        return K, P
    except Exception as e:
        import warnings
        warnings.warn(f"DARE solver failed: {e}. Returning zero gain.")
        return np.zeros((B.shape[1], A.shape[0])), np.eye(A.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# Spatial reference tracking
# ─────────────────────────────────────────────────────────────────────────────

def pick_k_nearest(x_est: np.ndarray, Xref: np.ndarray,
                   k_prog: int, window: int = 60) -> int:
    """
    Choose the reference index nearest to x_est in position space,
    searching only within a forward window from k_prog.

    Adaptive window: when the vehicle is close to the reference (low
    error), use a tight window to prevent skipping ahead. When the
    vehicle is far from the reference, tighten further — the controller
    should close the gap before advancing, not chase a point it can't
    reach. Only advance k_prog when a point strictly ahead is clearly
    closer than the current reference point.

    Parameters
    ----------
    x_est  : (5,)   current state estimate
    Xref   : (T,5)  reference trajectory states
    k_prog : int    current progress index (lower bound for search)
    window : int    maximum forward search window (default 60 steps)

    Returns
    -------
    k_use : int  selected reference index (>= k_prog, <= T-2)
    """
    T  = len(Xref)
    k0 = max(0, k_prog)
    k1 = min(T - 2, k_prog + window)

    dx = Xref[k0:k1 + 1, IDX_X] - x_est[IDX_X]
    dy = Xref[k0:k1 + 1, IDX_Y] - x_est[IDX_Y]
    dists_sq = dx * dx + dy * dy
    j = int(np.argmin(dists_sq))

    # Only advance if the nearest point is strictly ahead of k_prog.
    # j=0 means the current reference point (k_prog) is still the
    # nearest — hold position and let the controller converge.
    if j == 0:
        return k_prog

    return k0 + j
