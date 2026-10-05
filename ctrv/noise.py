"""Approximate local-heading process covariance for CTRV.

Longitudinal and angular double-integrator blocks use CWNA coefficients
(dt^3/3, dt^2/2, dt). An explicit lateral position disturbance supplies a
heuristic floor of 10% longitudinal intensity. This is not a full nonlinear
van Loan discretization: heading-speed coupling during a turn is omitted.
Rank-deficient process covariance is mathematically valid; the lateral floor
represents omitted disturbances, rather than a Kalman-filter requirement.
"""

import numpy as np
from ctrv.constants import IDX_X, IDX_Y, IDX_V, IDX_PSI, IDX_PSIDOT, N_STATES


def calculate_Q(dt: float, q_a: float, q_alpha: float, psi: float) -> np.ndarray:
    """
    Heading-dependent CWNA process noise covariance matrix.

    Parameters
    ----------
    dt      : float  discretization timestep (s)
    q_a     : float  longitudinal acceleration noise PSD (m^2/s^3)
    q_alpha : float  yaw acceleration noise PSD          (rad^2/s^3)
    psi     : float  current heading estimate (rad)

    Returns
    -------
    Q : (5, 5) ndarray, symmetric positive semi-definite

    Notes
    -----
    Previous (incorrect) version applied q_a * dt^3/3 isotropically to
    Q[x,x] and Q[y,y], and omitted the X-V, Y-V, and X-Y coupling terms.
    This underestimated position uncertainty during turns and contributed
    to EKF divergence (filter smugness) at high-curvature track sections.
    """
    if not np.isfinite([dt, q_a, q_alpha, psi]).all() or dt <= 0 or min(q_a, q_alpha) < 0:
        raise ValueError("dt must be positive; noise intensities nonnegative and finite")
    Q = np.zeros((N_STATES, N_STATES))
    dt2 = dt ** 2
    dt3 = dt ** 3
    cp  = np.cos(psi)
    sp  = np.sin(psi)

    # ── Longitudinal block  Q^(a) ─────────────────────────────────────────
    # Noise input: L_a = [cos(psi), sin(psi), 1, 0, 0]^T
    #
    # Position-position:
    Q[IDX_X, IDX_X] = q_a * dt3 / 3.0 * cp * cp
    Q[IDX_Y, IDX_Y] = q_a * dt3 / 3.0 * sp * sp
    Q[IDX_X, IDX_Y] = q_a * dt3 / 3.0 * cp * sp   # heading-coupled cross-term
    Q[IDX_Y, IDX_X] = Q[IDX_X, IDX_Y]

    # Position-velocity coupling:
    Q[IDX_X, IDX_V] = q_a * dt2 / 2.0 * cp
    Q[IDX_V, IDX_X] = Q[IDX_X, IDX_V]
    Q[IDX_Y, IDX_V] = q_a * dt2 / 2.0 * sp
    Q[IDX_V, IDX_Y] = Q[IDX_Y, IDX_V]

    # Velocity variance:
    Q[IDX_V, IDX_V] = q_a * dt

    # ── Lateral disturbance block  Q^(lat) ────────────────────────────────
    # Longitudinal-only position disturbance has no perpendicular variance
    # at ANY heading. Rank deficiency is valid, but omits lateral disturbances.
    #
    # Real vehicles experience lateral disturbances (wind, road camber,
    # tire slip) that are perpendicular to the heading.  We model this as
    # a fraction of the longitudinal noise PSD acting through the lateral
    # input vector L_lat = [-sin(psi), cos(psi), 0, 0, 0]^T.
    #
    # With positive q_a, q_alpha and dt this supplies a full-rank floor.
    q_lat = 0.1 * q_a                          # 10 % of longitudinal PSD
    Q[IDX_X, IDX_X] += q_lat * dt3 / 3.0 * sp * sp
    Q[IDX_Y, IDX_Y] += q_lat * dt3 / 3.0 * cp * cp
    Q[IDX_X, IDX_Y] -= q_lat * dt3 / 3.0 * cp * sp   # note: negative sign
    Q[IDX_Y, IDX_X]  = Q[IDX_X, IDX_Y]

    # ── Angular block  Q^(alpha) ──────────────────────────────────────────
    # Noise input: L_alpha = [0, 0, 0, 1, 1]^T (scalar integration chain)
    # This block was correct in the original implementation.
    Q[IDX_PSI,    IDX_PSI]    = q_alpha * dt3 / 3.0
    Q[IDX_PSI,    IDX_PSIDOT] = q_alpha * dt2 / 2.0
    Q[IDX_PSIDOT, IDX_PSI]    = q_alpha * dt2 / 2.0
    Q[IDX_PSIDOT, IDX_PSIDOT] = q_alpha * dt

    return 0.5 * (Q + Q.T)  # enforce exact symmetry
