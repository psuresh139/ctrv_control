"""Controlled CTRV dynamics and angle-aware finite-difference Jacobians.

State: [x, y, speed, heading, yaw_rate]. Inputs: acceleration, yaw acceleration.
Position integrates the continuous controlled dynamics with Gaussian quadrature;
heading, speed and yaw rate use their exact polynomial updates. The analytical
Jacobian is retained only for the uncontrolled baseline and validation.
"""

import numpy as np
from ctrv.constants import (
    IDX_X, IDX_Y, IDX_V, IDX_PSI, IDX_PSIDOT,
    N_STATES, N_CONTROL, EPS
)


_QUAD_NODES, _QUAD_WEIGHTS = np.polynomial.legendre.leggauss(8)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def wrap_to_pi(a: float) -> float:
    """Map angles (rad) into [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics
# ─────────────────────────────────────────────────────────────────────────────

def predict_state_ctrv(x_prev: np.ndarray, dt: float,
                       u_control=None,
                       enforce_nonneg_speed: bool = True) -> np.ndarray:
    """
    Controlled discrete CTRV state transition: x_{k+1} = f(x_k, u_k, dt).

    State:   x = [x, y, v, psi, psi_dot]
    Control: u = [a, alpha]
               a     = longitudinal acceleration (m/s^2)
               alpha = yaw acceleration          (rad/s^2)

    Position integrates the controlled continuous dynamics with eight-point
    Gauss-Legendre quadrature. Heading integrates yaw acceleration exactly.
    Braking stops translational motion when speed reaches zero.

    Parameters
    ----------
    x_prev              : (5,) prior state
    dt                  : float timestep (s)
    u_control           : (2,) control input; zeros if None
    enforce_nonneg_speed: bool, clamp speed to >= 0

    Returns
    -------
    x_next : (5,) predicted state
    """
    if u_control is None:
        u = np.zeros(2)
    else:
        u = np.asarray(u_control, dtype=float).reshape(-1)
        if u.size == 1:
            u = np.array([0.0, float(u[0])])
        elif u.size != 2:
            raise ValueError("u_control must be shape (2,) for [a, alpha].")

    a, alpha = float(u[0]), float(u[1])
    x, y, v, psi, psi_dot = x_prev

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    # Integrate the same continuous equations for position and heading:
    # p_dot = v(t) [cos(psi(t)), sin(psi(t))], v_dot=a, omega_dot=alpha.
    # Gauss-Legendre quadrature avoids division by yaw rate near straight motion.
    duration = dt
    if enforce_nonneg_speed and a < 0 and v + a * dt < 0:
        duration = max(0.0, min(dt, -v / a))
    tau = 0.5 * duration * (_QUAD_NODES + 1.0)
    speed = v + a * tau
    if enforce_nonneg_speed:
        speed = np.maximum(speed, 0.0)
    heading = psi + psi_dot * tau + 0.5 * alpha * tau**2
    x_next = np.array([
        x + 0.5 * duration * np.dot(_QUAD_WEIGHTS, speed * np.cos(heading)),
        y + 0.5 * duration * np.dot(_QUAD_WEIGHTS, speed * np.sin(heading)),
        max(0.0, v + a * dt) if enforce_nonneg_speed else v + a * dt,
        wrap_to_pi(psi + psi_dot * dt + 0.5 * alpha * dt**2),
        psi_dot + alpha * dt,
    ])

    return x_next


def predict_state_ctrv_baseline(x_prev: np.ndarray, dt: float) -> np.ndarray:
    """
    Uncontrolled baseline CTRV model: constant psi_dot over the interval,
    no control input, no higher-order heading terms.

    This is the model the analytical Jacobian (jacobian_F_analytical) is
    derived from. Used ONLY for Jacobian validation — not the live plant.
    """
    return predict_state_ctrv(x_prev, dt, np.zeros(2), enforce_nonneg_speed=False)


# ─────────────────────────────────────────────────────────────────────────────
# Jacobians
# ─────────────────────────────────────────────────────────────────────────────

def jacobian_F_analytical(x_prev: np.ndarray, dt: float) -> np.ndarray:
    """
    Closed-form Jacobian F_k = df/dx evaluated at x_prev for the baseline
    CTRV model (predict_state_ctrv_baseline).

    Used in the EKF covariance propagation step:
        P_{k+1|k} = F_k P_{k|k} F_k^T + Q_k

    Structure (non-identity elements only):

        Turning case (|psi_dot| > EPS):

          F[x,  v]       = (1/psi_dot) * (sin(psi + psi_dot*dt) - sin(psi))
          F[x,  psi]     = (v/psi_dot) * (cos(psi + psi_dot*dt) - cos(psi))
          F[x,  psi_dot] = -(v/psi_dot^2)*(sin_sum - sin_psi)
                           + (v/psi_dot)*(cos_sum * dt)

          F[y,  v]       = (1/psi_dot) * (-cos(psi + psi_dot*dt) + cos(psi))
          F[y,  psi]     = (v/psi_dot) * (sin(psi + psi_dot*dt) - sin(psi))
          F[y,  psi_dot] = -(v/psi_dot^2)*(-cos_sum + cos_psi)
                           + (v/psi_dot)*(sin_sum * dt)

          F[psi, psi_dot] = dt

        Straight case (|psi_dot| <= EPS):

          F[x,  v]   = cos(psi)*dt
          F[x,  psi] = -v*sin(psi)*dt
          F[y,  v]   = sin(psi)*dt
          F[y,  psi] = v*cos(psi)*dt
          F[psi, psi_dot] = dt

    Returns
    -------
    F : (5, 5) ndarray
    """
    _, _, v, psi, psi_dot = (x_prev[IDX_X], x_prev[IDX_Y],
                              x_prev[IDX_V], x_prev[IDX_PSI],
                              x_prev[IDX_PSIDOT])
    F = np.eye(N_STATES)

    s_psi = np.sin(psi)
    c_psi = np.cos(psi)
    psi_sum = psi + psi_dot * dt
    s_sum = np.sin(psi_sum)
    c_sum = np.cos(psi_sum)

    if abs(psi_dot) > EPS:
        # --- x row ---
        F[IDX_X, IDX_V]      = (s_sum - s_psi) / psi_dot
        F[IDX_X, IDX_PSI]    = v / psi_dot * (c_sum - c_psi)
        F[IDX_X, IDX_PSIDOT] = (-v / psi_dot**2 * (s_sum - s_psi)
                                 + v / psi_dot * c_sum * dt)
        # --- y row ---
        F[IDX_Y, IDX_V]      = (-c_sum + c_psi) / psi_dot
        F[IDX_Y, IDX_PSI]    = v / psi_dot * (s_sum - s_psi)
        F[IDX_Y, IDX_PSIDOT] = (-v / psi_dot**2 * (-c_sum + c_psi)
                                 + v / psi_dot * s_sum * dt)
        # --- psi row ---
        F[IDX_PSI, IDX_PSIDOT] = dt
    else:
        F[IDX_X, IDX_V]        = c_psi * dt
        F[IDX_X, IDX_PSI]      = -v * s_psi * dt
        F[IDX_Y, IDX_V]        = s_psi * dt
        F[IDX_Y, IDX_PSI]      = v * c_psi * dt
        F[IDX_X, IDX_PSIDOT] = -0.5 * v * s_psi * dt**2
        F[IDX_Y, IDX_PSIDOT] = 0.5 * v * c_psi * dt**2
        F[IDX_PSI, IDX_PSIDOT] = dt

    return F


def jacobian_numerical(x: np.ndarray, dt: float, f, eps: float = 1e-6) -> np.ndarray:
    """
    Central-difference Jacobian df/dx for any discrete dynamics f(x, dt).

    Used for:
      (a) Validating jacobian_F_analytical against the baseline model
      (b) Computing A_k = df/dx in the LTV-LQR linearization sweep

    Parameters
    ----------
    x   : (n,) state
    dt  : float
    f   : callable with signature f(x, dt) -> (n,)
    eps : finite difference step size

    Returns
    -------
    J : (n, n) Jacobian
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    J = np.zeros((n, n))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        delta = np.asarray(f(x + dx, dt)) - np.asarray(f(x - dx, dt))
        delta[IDX_PSI] = wrap_to_pi(delta[IDX_PSI])
        J[:, i] = delta / (2.0 * eps)

    return J
