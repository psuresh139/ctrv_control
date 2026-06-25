"""
ctrv/model.py

CTRV (Constant Turn Rate and Velocity) vehicle dynamics.

Provides:
  - wrap_to_pi              : angle normalization
  - predict_state_ctrv      : nonlinear discrete state transition f(x, u, dt)
  - predict_state_ctrv_baseline : uncontrolled CTRV (used for Jacobian validation)
  - jacobian_F_analytical   : closed-form Jacobian df/dx (used in EKF)
  - jacobian_numerical      : central-difference Jacobian (used for validation
                              and LQR linearization)

Design note on two predict functions:
  predict_state_ctrv         includes control inputs [a, alpha] and integrates
                             higher-order terms (0.5*alpha*dt^2 in heading).
                             This is the true simulated plant.

  predict_state_ctrv_baseline is the pure CTRV model with no control and
                             constant psi_dot over the interval. This is the
                             model the analytical Jacobian is derived from,
                             and is used only for validation.

  The mismatch between the two (Test B in the notebook, ~1.93e-3) is expected
  and confirms the EKF correctly bridges the linearization gap at runtime.
"""

import numpy as np
from ctrv.constants import (
    IDX_X, IDX_Y, IDX_V, IDX_PSI, IDX_PSIDOT,
    N_STATES, N_CONTROL, EPS
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def wrap_to_pi(a: float) -> float:
    """Map angle a (rad) into (-pi, pi]."""
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

    Position update uses the exact circular arc formula when |psi_dot| > EPS,
    falling back to straight-line integration otherwise. Heading integrates
    the yaw acceleration to second order (0.5 * alpha * dt^2).

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

    # Velocity and yaw rate update (Euler forward on first-order dynamics)
    v_new = v + a * dt
    if enforce_nonneg_speed:
        v_new = max(0.0, v_new)
    psi_dot_new = psi_dot + alpha * dt

    # Heading: second-order integration captures yaw acceleration effect
    psi_new = wrap_to_pi(psi + psi_dot * dt + 0.5 * alpha * dt ** 2)

    x_next = np.zeros(N_STATES)

    # Position: exact arc formula when turning, straight-line fallback
    if abs(psi_dot_new) > EPS:
        dpsi = psi_dot_new * dt
        x_next[IDX_X] = x + (v_new / psi_dot_new) * (np.sin(psi + dpsi) - np.sin(psi))
        x_next[IDX_Y] = y + (v_new / psi_dot_new) * (-np.cos(psi + dpsi) + np.cos(psi))
    else:
        x_next[IDX_X] = x + v_new * np.cos(psi) * dt
        x_next[IDX_Y] = y + v_new * np.sin(psi) * dt

    x_next[IDX_V]      = v_new
    x_next[IDX_PSI]    = psi_new
    x_next[IDX_PSIDOT] = psi_dot_new

    return x_next


def predict_state_ctrv_baseline(x_prev: np.ndarray, dt: float) -> np.ndarray:
    """
    Uncontrolled baseline CTRV model: constant psi_dot over the interval,
    no control input, no higher-order heading terms.

    This is the model the analytical Jacobian (jacobian_F_analytical) is
    derived from. Used ONLY for Jacobian validation — not the live plant.
    """
    x, y, v, psi, psi_dot = x_prev
    x_next = np.zeros(N_STATES)

    if abs(psi_dot) > EPS:
        dpsi = psi_dot * dt
        x_next[IDX_X] = x + (v / psi_dot) * (np.sin(psi + dpsi) - np.sin(psi))
        x_next[IDX_Y] = y + (v / psi_dot) * (-np.cos(psi + dpsi) + np.cos(psi))
    else:
        x_next[IDX_X] = x + v * np.cos(psi) * dt
        x_next[IDX_Y] = y + v * np.sin(psi) * dt

    x_next[IDX_V]      = v
    x_next[IDX_PSI]    = wrap_to_pi(psi + psi_dot * dt)
    x_next[IDX_PSIDOT] = psi_dot

    return x_next


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
        J[:, i] = (np.asarray(f(x + dx, dt)) - np.asarray(f(x - dx, dt))) / (2.0 * eps)

    return J
