"""
ctrv/constants.py

Central registry for all global constants, state indices, and default
simulation parameters. Every other module imports from here — nothing
is redefined elsewhere.
"""

import numpy as np

# ── Dimensions ────────────────────────────────────────────────────────────────
N_STATES  = 5   # [x, y, v, psi, psi_dot]
N_MEAS    = 2   # [x_meas, y_meas]
N_CONTROL = 2   # [a (longitudinal accel), alpha (yaw accel)]

# ── State indices ─────────────────────────────────────────────────────────────
IDX_X      = 0  # planar x position  (m)
IDX_Y      = 1  # planar y position  (m)
IDX_V      = 2  # longitudinal speed (m/s)
IDX_PSI    = 3  # heading / yaw      (rad)
IDX_PSIDOT = 4  # yaw rate           (rad/s)

# ── Numerics ──────────────────────────────────────────────────────────────────
EPS = 1e-6      # near-zero threshold (used for psi_dot singularity check)

# ── Simulation ────────────────────────────────────────────────────────────────
DT = 0.1        # discretization timestep (s)

# ── Default noise parameters ──────────────────────────────────────────────────
# Process noise PSDs — see ctrv/noise.py for derivation
Q_A_PSD      = 2.0    # longitudinal accel noise PSD  (m^2/s^3)
Q_ALPHA_PSD  = 0.05   # yaw accel noise PSD           (rad^2/s^3)

# Measurement noise covariance (GPS position, 1-sigma = 1.0 m)
R_MEAS = np.diag([1.0**2, 1.0**2])

# Initial state covariance
P_INIT = np.diag([
    5.0**2,                  # x uncertainty (m^2)
    5.0**2,                  # y uncertainty (m^2)
    1.0**2,                  # v uncertainty (m/s)^2
    np.deg2rad(5.0)**2,      # psi uncertainty (rad^2)
    np.deg2rad(1.0)**2,      # psi_dot uncertainty (rad/s)^2
])

# ── Measurement matrix ────────────────────────────────────────────────────────
# 2-DOF: observes x and y position only (GPS-only baseline)
H_MEAS_2DOF = np.array([
    [1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
])

# 3-DOF: also observes heading (GPS + compass/IMU)
H_MEAS_3DOF = np.array([
    [1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0., 1., 0.],
])

# Default: 2-DOF (set to 3-DOF in run_simulation if compass available)
H_MEAS = H_MEAS_2DOF
N_MEAS_2DOF = 2
N_MEAS_3DOF = 3

# Measurement noise for 3-DOF (compass heading: 5° 1-sigma)
R_MEAS_3DOF = np.diag([1.0**2, 1.0**2, np.deg2rad(5.0)**2])

# ── Track definition (COTA-inspired) ─────────────────────────────────────────
WAYPOINTS = np.array([
    [0.0,   0.0],
    [50.0,  0.0],
    [70.0,  20.0],
    [90.0,  20.0],
    [110.0, 40.0],
    [110.0, 150.0],
    [120.0, 160.0],
    [100.0, 180.0],
    [60.0,  190.0],
    [0.0,   190.0],
])

# Per-segment [v_cmd (m/s), psi_dot_cmd (rad/s)]
SEGMENT_COMMANDS = np.array([
    [10.0,  0.0],
    [8.0,   0.4],
    [8.0,  -0.4],
    [10.0,  0.0],
    [15.0,  0.0],
    [5.0,   0.0],
    [4.0,  -0.5],
    [10.0,  0.0],
    [5.0,   0.0],
])

# ── Control bounds ────────────────────────────────────────────────────────────
# Widened from original (±3.0, ±1.5) after parameter sweep showed that the
# tight corners on the COTA track are geometrically infeasible with narrow
# actuator limits.  The wider yaw-accel bound (±2.5) lets the controller
# execute the 8 m minimum-radius turns without saturating, while the wider
# longitudinal bound (±5.0) allows faster speed corrections entering/exiting
# corners.
U_BOUNDS = {
    'a':     (-5.0, 5.0),   # longitudinal accel (m/s^2)
    'alpha': (-2.5, 2.5),   # yaw accel          (rad/s^2)
}

# ── LQR cost matrices ─────────────────────────────────────────────────────────
# Tuned via parameter sweep (see scripts/tune_ekf.py).
# Key insight: with wider actuator bounds, the controller can afford higher
# position weight without saturating.  R_alpha reduced from 200→50 because
# the tight corners REQUIRE aggressive steering (κ=0.125 rad/m).
#
# Previous R_alpha=200 made the controller too timid at corners, producing
# ~6 m mean tracking error even in the noiseless oracle.
Q_LQR = np.diag([25.0, 25.0, 2.0, 25.0, 12.0])
R_LQR = np.diag([5.0, 50.0])

# ── Innovation gating ─────────────────────────────────────────────────────────
# Chi-squared 99th percentile with N_MEAS=2 degrees of freedom
# Rejects measurements whose Mahalanobis distance exceeds this threshold
INNOVATION_GATE = 9.21

# ── Softened track (transition-curve geometry) ────────────────────────────────
# Intermediate waypoints at corner entry/exit create gentler arcs that
# mimic real track transition curves (clothoids).  Max curvature drops
# from κ=0.125 rad/m (R=8m) to κ=0.060 rad/m (R≈17m).
WAYPOINTS_SOFT = np.array([
    [0.0,   0.0],
    [45.0,  0.0],
    [55.0,  5.0],
    [65.0,  15.0],
    [78.0,  22.0],
    [90.0,  22.0],
    [100.0, 28.0],
    [108.0, 45.0],
    [110.0, 70.0],
    [110.0, 130.0],
    [112.0, 150.0],
    [115.0, 162.0],
    [110.0, 175.0],
    [100.0, 183.0],
    [80.0,  188.0],
    [40.0,  190.0],
    [0.0,   190.0],
])

SEGMENT_COMMANDS_SOFT = np.array([
    [10.0,  0.0],
    [9.0,   0.15],
    [8.0,   0.20],
    [9.0,   0.10],
    [10.0, -0.05],
    [9.0,   0.15],
    [8.0,   0.20],
    [12.0,  0.0],
    [15.0,  0.0],
    [8.0,   0.0],
    [6.0,  -0.15],
    [5.0,  -0.30],
    [6.0,  -0.20],
    [8.0,  -0.05],
    [10.0,  0.0],
    [5.0,   0.0],
])
