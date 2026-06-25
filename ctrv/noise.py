"""
ctrv/noise.py

Process noise covariance matrix Q for the 5-state CTRV model.

Derivation summary
------------------
Two independent white noise acceleration sources drive the system:

  Longitudinal:  w_a     ~ N(0, q_a)      drives v -> x, y
  Yaw:           w_alpha ~ N(0, q_alpha)  drives psi_dot -> psi

Q is the sum of two independent contributions:

    Q(psi) = Q_longitudinal(psi) + Q_angular

Q_angular is heading-independent (psi/psi_dot block only).
Q_longitudinal is heading-DEPENDENT: velocity uncertainty projects into
world-frame x/y via the current heading angle psi.

The discretization follows the Continuous White Noise Acceleration (CWNA)
model via the van Loan method. For a 1D double-integrator [p, p_dot]^T
driven by acceleration noise sigma^2, the discrete Q block is:

    sigma^2 * [[dt^3/3,  dt^2/2],
               [dt^2/2,  dt   ]]

For the 5-state CTRV, the noise input vectors are:

    L_a     = [cos(psi), sin(psi), 1, 0, 0]^T   (longitudinal)
    L_alpha = [0,        0,        0, 1, 1]^T   (angular)

Q is constructed as the outer products of these vectors scaled by their
respective PSD values and the CWNA polynomial coefficients.
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
    # The longitudinal-only noise model produces a rank-deficient Q at
    # cardinal headings (psi = 0, pi/2, ...) because the noise projection
    # collapses onto one axis (e.g. Q[y,y] = 0 at psi = 0).
    #
    # Real vehicles experience lateral disturbances (wind, road camber,
    # tire slip) that are perpendicular to the heading.  We model this as
    # a fraction of the longitudinal noise PSD acting through the lateral
    # input vector L_lat = [-sin(psi), cos(psi), 0, 0, 0]^T.
    #
    # This guarantees Q is always full-rank (rank 5) at every heading.
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
