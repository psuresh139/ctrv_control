"""
ctrv/trajectory.py

Reference trajectory generation from sparse waypoints.

Design decisions (final version)
----------------------------------

1. Path geometry — clamped cubic spline (make_interp_spline, bc_type)
   -----------------------------------------------------------------------
   The not-a-knot (NAK) UnivariateSpline uses a global polynomial boundary
   condition that, on this U-shaped track, forces the tangent at s=0 toward
   the last waypoint (-46°) and creates a 17m y-dip on the first segment.
   The reported heading and the path coordinates were both wrong.

   A clamped spline pins the endpoint derivatives to the true first and last
   segment directions. Both x(s) and y(s) are correct: psi[0]=0°, psi[-1]=180°,
   and the path passes through all waypoints without spurious excursions.

   The path geometry between interior waypoints is slightly different from NAK
   (max deviation ~5m near the first corner), but it is geometrically correct
   and physically traversable — the vehicle is never asked to go to y=-17m.

2. Feedforward — smooth ramps, not np.gradient
   -----------------------------------------------
   np.gradient on piecewise-constant v/psi_dot produces ±50 m/s² impulse
   spikes at every segment boundary (Δv=10 m/s, dt=0.1s → spike=100/2=50).
   Replaced with linear ramp transitions of duration t_ramp=0.5s, with
   feedforward derived analytically as delta/t_ramp. Clipped to actuator
   bounds as a safety guard.
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from ctrv.constants import N_CONTROL, DT, U_BOUNDS


def build_reference_from_waypoints(
    waypoints:    np.ndarray,
    commands:     np.ndarray,
    dt:           float = DT,
    sampling_res: float = 0.1,
    t_ramp:       float = 0.5,
) -> dict:
    """
    Build a smooth, physically consistent reference trajectory.

    Parameters
    ----------
    waypoints    : (N, 2)    [x, y] waypoint coordinates
    commands     : (N-1, 2)  per-segment [v_cmd (m/s), psi_dot_cmd (rad/s)]
    dt           : float     simulation timestep (s)
    sampling_res : float     spatial sampling resolution (m)
    t_ramp       : float     velocity/yaw-rate transition ramp duration (s)

    Returns
    -------
    dict with keys:
      'states' : (T, 5)  reference states [x, y, v, psi, psi_dot]
      'times'  : (T,)    time stamps (s)
      'u_ref'  : (T, 2)  feedforward controls [a_ff, alpha_ff]
    """
    wx, wy   = waypoints[:, 0], waypoints[:, 1]
    dists    = np.sqrt(np.diff(wx)**2 + np.diff(wy)**2)
    s_sparse = np.concatenate(([0], np.cumsum(dists)))

    # ── Clamped cubic spline ──────────────────────────────────────────────
    # Endpoint tangents = unit direction of adjacent segment
    t0 = np.array([wx[1]-wx[0],  wy[1]-wy[0]])  / dists[0]
    t1 = np.array([wx[-1]-wx[-2], wy[-1]-wy[-2]]) / dists[-1]
    bc = ([(1, t0)], [(1, t1)])
    spline = make_interp_spline(
        s_sparse, np.column_stack([wx, wy]), k=3, bc_type=bc
    )

    s_dense = np.arange(0, s_sparse[-1], sampling_res)
    pts     = spline(s_dense)           # (M, 2)  — x, y
    dpts    = spline.derivative()(s_dense)  # (M, 2)  — dx/ds, dy/ds

    x_dense   = pts[:, 0]
    y_dense   = pts[:, 1]
    psi_dense = np.unwrap(np.arctan2(dpts[:, 1], dpts[:, 0]))

    # ── Piecewise-constant v / psi_dot ────────────────────────────────────
    v_step  = np.zeros_like(s_dense)
    pd_step = np.zeros_like(s_dense)
    for i in range(len(dists)):
        mask           = (s_dense >= s_sparse[i]) & (s_dense < s_sparse[i + 1])
        v_step[mask]   = commands[i, 0]
        pd_step[mask]  = commands[i, 1]
    v_step[-1]  = commands[-1, 0]
    pd_step[-1] = commands[-1, 1]

    # ── Arc-length to uniform time grid ──────────────────────────────────
    dt_step = np.diff(s_dense) / np.maximum(v_step[:-1], 1.0)
    t_dense = np.concatenate(([0], np.cumsum(dt_step)))
    t_fixed = np.arange(0, t_dense[-1], dt)

    x_ref   = np.interp(t_fixed, t_dense, x_dense)
    y_ref   = np.interp(t_fixed, t_dense, y_dense)
    psi_ref = np.interp(t_fixed, t_dense, psi_dense)
    v_raw   = np.interp(t_fixed, t_dense, v_step)
    pd_raw  = np.interp(t_fixed, t_dense, pd_step)

    # ── Smooth velocity / yaw-rate; derive feedforward analytically ───────
    v_ref,  a_ff     = _smooth_with_ramps(v_raw,  dt, t_ramp, clip=U_BOUNDS['a'])
    pd_ref, alpha_ff = _smooth_with_ramps(pd_raw, dt, t_ramp, clip=U_BOUNDS['alpha'])

    return {
        'states': np.column_stack([x_ref, y_ref, v_ref, psi_ref, pd_ref]),
        'times':  t_fixed,
        'u_ref':  np.column_stack([a_ff, alpha_ff]),
    }


def _smooth_with_ramps(
    signal: np.ndarray,
    dt:     float,
    t_ramp: float,
    clip:   tuple = None,
) -> tuple:
    """
    Replace step discontinuities with linear ramps of duration t_ramp.
    Returns (smoothed_signal, analytical_derivative).
    Consecutive transitions within n_ramp steps are merged into one event.
    """
    T      = len(signal)
    n_ramp = max(2, round(t_ramp / dt))

    raw_trans = []
    for k in range(T - 1):
        d = float(signal[k + 1]) - float(signal[k])
        if abs(d) > 1e-9:
            raw_trans.append((k, d))

    if not raw_trans:
        return signal.copy().astype(float), np.zeros(T)

    # Merge consecutive transitions within n_ramp steps
    merged, i = [], 0
    while i < len(raw_trans):
        k0, total = raw_trans[i][0], raw_trans[i][1]
        j = i + 1
        while j < len(raw_trans) and raw_trans[j][0] - k0 <= n_ramp:
            total += raw_trans[j][1]
            j += 1
        merged.append((k0, total))
        i = j

    smoothed   = signal.copy().astype(float)
    derivative = np.zeros(T)

    for idx, (k_trig, delta) in enumerate(merged):
        k_start = k_trig
        k_end   = min(T, k_start + n_ramp)
        if idx < len(merged) - 1:
            k_end = min(k_end, merged[idx + 1][0])
        k_end = max(k_end, k_start + 2)

        n = k_end - k_start
        v0 = smoothed[k_start]
        smoothed[k_start:k_end] = np.linspace(v0, v0 + delta, n)
        if k_end < T:
            smoothed[k_end:] = v0 + delta
        derivative[k_start:k_end] = delta / (n * dt)

    if clip is not None:
        derivative = np.clip(derivative, clip[0], clip[1])

    return smoothed, derivative
