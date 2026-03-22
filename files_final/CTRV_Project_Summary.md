# CTRV EKF-LQR Project: Final Summary & Architecture

## Project Overview

**Extended Kalman Filtering and Linear Quadratic Regulators for Optimal Trajectories of Constant Turn Rate Velocity (CTRV) Vehicles in Noisy Signal Input Environments**

This project implements a closed-loop autonomous trajectory tracking system for a 5-state Unmanned Ground Vehicle on a COTA-inspired racetrack. The system couples nonlinear state estimation (EKF/UKF) with optimal feedback control (LTV-LQR) to track a reference path under process noise and noisy sensor measurements.

---

## Performance Progression

| Stage | Mean Error | Max Error | Stable Seeds | Key Change |
|---|---|---|---|---|
| Original notebook | Diverged at t≈30s | — | — | Initial implementation |
| Bug fixes (Q, Jacobian, k_prog) | 6.74 m | ~18 m | ~8/10 | Q rank fix, numerical Jacobian, drift guard |
| Wider actuator bounds + LQR retune | 3.17 m | 7.7 m | ~6/10 | a: ±3→±5, α: ±1.5→±2.5, R_α: 200→50 |
| + Gain scheduling + error attenuation | 3.04 m | 8.6 m | **10/10** | Curvature-dependent R, position error clamp |
| + IMU heading sensor (5°) | 2.27 m | 2.6 m | 10/10 | 3-DOF measurement model |
| + Softened track geometry | **1.78 m** | **2.0 m** | **10/10** | Transition-curve waypoints, κ: 0.125→0.060 |

Final configuration: **1.78 m ± 0.14 m** across 10 random seeds. The standard deviation dropped from ±0.76 m (GPS-only, original track) to ±0.14 m, indicating the system is now robust to noise realizations.

---

## Mathematical Architecture

### State Vector

$$\mathbf{x} = \begin{bmatrix} x \\ y \\ v \\ \psi \\ \dot{\psi} \end{bmatrix} \in \mathbb{R}^5$$

where (x, y) is planar position, v is longitudinal speed, ψ is heading, and ψ̇ is yaw rate.

### Control Input

$$\mathbf{u} = \begin{bmatrix} a \\ \alpha \end{bmatrix} \in \mathbb{R}^2$$

where a is longitudinal acceleration (±5.0 m/s²) and α is yaw acceleration (±2.5 rad/s²).

---

### Stage 1: CTRV Dynamics Model (`model.py`)

The nonlinear discrete-time state transition handles two regimes:

**Turning (|ψ̇| > ε):**
$$x_{k+1} = x_k + \frac{v_{new}}{\dot{\psi}_{new}} \left[\sin(\psi_k + \dot{\psi}_{new} \Delta t) - \sin(\psi_k)\right]$$
$$y_{k+1} = y_k + \frac{v_{new}}{\dot{\psi}_{new}} \left[-\cos(\psi_k + \dot{\psi}_{new} \Delta t) + \cos(\psi_k)\right]$$

**Straight (|ψ̇| ≤ ε):**
$$x_{k+1} = x_k + v_{new} \cos(\psi_k) \Delta t, \quad y_{k+1} = y_k + v_{new} \sin(\psi_k) \Delta t$$

where v_new = v + a·dt and ψ̇_new = ψ̇ + α·dt incorporate the control input.

Heading integrates to second order: ψ_{k+1} = ψ_k + ψ̇·dt + ½·α·dt².

---

### Stage 2: Process Noise Model (`noise.py`)

The CWNA (Continuous White Noise Acceleration) process noise covariance Q(ψ) is heading-dependent and composed of two independent contributions:

**Longitudinal noise** (acceleration noise q_a projects through heading):
$$\mathbf{L}_a = [\cos\psi, \sin\psi, 1, 0, 0]^T$$

**Lateral noise** (disturbance noise q_lat projects perpendicular to heading):
$$\mathbf{L}_{lat} = [-\sin\psi, \cos\psi, 0, 0, 0]^T$$

**Angular noise** (yaw acceleration noise q_α):
$$\mathbf{L}_\alpha = [0, 0, 0, 1, 1]^T$$

The lateral noise term (q_lat = 0.1·q_a) was added to prevent the Q matrix from becoming rank-deficient at cardinal headings. Without it, Q drops to rank 4 at ψ = 0°, 90°, etc., causing the filter to become overconfident (smug) in the lateral direction.

**Verified properties:** Q is rank 5 at all headings; trace(Q) is heading-invariant; Q is symmetric positive definite everywhere.

---

### Stage 3: State Estimation — EKF and UKF (`ekf.py`)

Both estimators share the same interface and support 2-DOF (GPS: x, y) or 3-DOF (GPS+IMU: x, y, ψ) measurement models.

**EKF** uses the numerical Jacobian of the actual controlled dynamics (not the analytical baseline Jacobian, which was derived for the uncontrolled model and introduced ~0.8% per-step error in the position-heading partial derivatives).

**UKF** uses the Merwe scaled sigma-point scheme with α=0.1, β=2.0, κ=0.0, generating 2n+1 = 11 sigma points. Each sigma point is propagated through the full nonlinear dynamics, capturing second-order effects without requiring Jacobians.

Both filters use the Joseph form for covariance updates, heading wrapping in innovations, and PSD enforcement via eigenvalue clamping. Innovation gating (χ² threshold = 9.21 for 2-DOF) rejects outlier measurements.

**Key finding:** EKF and UKF produce nearly identical results (3.17 m vs 3.33 m on the original track) because the CTRV nonlinearity is mild at dt=0.1s with moderate yaw rates. The dominant error source was never the linearization — it was the control loop.

---

### Stage 4: Optimal Control — Curvature-Scheduled LTV-LQR (`lqr.py`)

The controller solves a finite-horizon LQR problem by:

1. **Linearizing** the CTRV dynamics at each reference point via central finite differences, producing time-varying (A_k, B_k) matrices.

2. **Solving the backward Riccati recursion** with curvature-scheduled costs:

$$S_k = R_k + B_k^T P_{k+1} B_k$$
$$K_k = S_k^{-1} B_k^T P_{k+1} A_k$$
$$P_k = Q_k + A_k^T P_{k+1}(A_k - B_k K_k)$$

where R_k switches between R_straight (conservative, R_α = 50) on straight segments and R_turn (aggressive, R_α = 50 with Q_pos boosted to 40) on curved segments (κ ≥ 0.02 rad/m). This allows the cost-to-go to propagate corner awareness backwards in time, creating anticipatory gains before the vehicle reaches a turn.

3. **Applying the control law** at each step:

$$\mathbf{u}_k = \mathbf{u}_{ff,k} - K_k \cdot \text{clamp}(\hat{\mathbf{x}}_k - \mathbf{x}_{ref,k})$$

where the feedforward u_ff comes from the reference trajectory's smoothed acceleration/yaw-acceleration profile, and the position error is clamped to 10m magnitude to prevent divergence when the vehicle deviates far from the linearization regime.

---

### Stage 5: Reference Trajectory Generation (`trajectory.py`)

The reference path is constructed by:

1. Fitting a clamped cubic spline through waypoints (arc-length parameterized).
2. Computing heading from spline tangent vectors via atan2.
3. Mapping per-segment velocity and yaw-rate commands onto the dense path.
4. Resampling onto a uniform dt time grid.
5. Smoothing velocity/yaw-rate profiles with linear ramps (t_ramp = 0.5s) to eliminate feedforward discontinuities.

The softened track geometry uses 17 waypoints (vs 10 original) with intermediate points at corner entry and exit to create transition curves, reducing maximum curvature from κ = 0.125 rad/m (R_min = 8m) to κ = 0.060 rad/m (R_min = 17m).

---

## Bugs Identified and Fixed

### Bug 1: Q Matrix Rank Deficiency
**Root cause:** Heading-dependent CWNA noise with only longitudinal projection produced Q[y,y] = 0 at ψ = 0° (and Q[x,x] = 0 at ψ = 90°).
**Fix:** Added lateral disturbance noise (10% of longitudinal PSD) through perpendicular input vector.
**Impact:** Q guaranteed rank 5 at all headings; eliminated filter smugness at cardinal directions.

### Bug 2: Jacobian–Dynamics Mismatch
**Root cause:** Analytical Jacobian derived from uncontrolled baseline model but applied to controlled dynamics. Position-heading derivatives differed by ~0.8% per step.
**Fix:** Replaced analytical Jacobian with numerical Jacobian of actual controlled dynamics in EKF.
**Impact:** Consistent covariance propagation; also fixed a factor-of-2 error in S symmetrization.

### Bug 3: Spatial Index Drift
**Root cause:** The reference index k_prog was purely spatial (nearest point), allowing it to stall when the vehicle was slower than the reference, causing stale LQR gains to be applied.
**Fix:** Added temporal floor: k_use = max(k_spatial, k_temporal − 50).
**Impact:** Gains always correspond to a relevant track section.

### Bug 4: Feedforward Discontinuities
**Root cause:** np.gradient on piecewise-constant velocity/yaw-rate profiles created full-range actuator jumps at segment boundaries.
**Fix:** Linear ramp smoothing with t_ramp = 0.5s and analytical derivative computation.
**Impact:** Eliminated transient control spikes at segment transitions.

---

## Codebase Structure

```
ctrv_control/
├── ctrv/
│   ├── __init__.py        (46 lines)   Public API exports
│   ├── constants.py      (164 lines)   State indices, dimensions, noise/LQR defaults,
│   │                                    track definitions (original + softened),
│   │                                    2-DOF and 3-DOF measurement matrices
│   ├── model.py          (241 lines)   CTRV dynamics, analytical + numerical Jacobians
│   ├── noise.py          (112 lines)   Heading-dependent CWNA process noise Q(ψ)
│   ├── ekf.py            (391 lines)   EKF + UKF (both 2-DOF and 3-DOF),
│   │                                    sigma-point generation, NIS diagnostics
│   ├── lqr.py            (340 lines)   FD linearization, backward Riccati (uniform +
│   │                                    curvature-scheduled), DARE, spatial tracking
│   └── trajectory.py     (170 lines)   Spline path generation, ramp-smoothed feedforward
├── tests/
│   └── test_all.py       (450 lines)   25 unit tests covering all modules
├── scripts/
│   ├── run_simulation.py (388 lines)   Closed-loop EKF/UKF + LTV-LQR simulation
│   └── tune_ekf.py       (146 lines)   NIS-based Q/R parameter sweep
├── requirements.txt
└── .gitignore

Total: 2,462 lines across 12 files
25/25 tests passing
```

---

## Key Design Decisions

1. **Hybrid linearization strategy:** Analytical Jacobians retained for validation (test_all.py confirms max error = 1.83×10⁻⁹ vs finite differences on baseline dynamics); numerical Jacobians used operationally in both EKF and LQR to ensure consistency with the actual controlled plant.

2. **Curvature-scheduled gains over uniform LQR:** A single set of Q/R costs cannot optimally handle both straights (where smooth control matters) and tight corners (where aggressive steering is necessary). The scheduled Riccati naturally propagates corner awareness backwards through the cost-to-go.

3. **Error attenuation over anti-windup:** Clamping the position error fed to the LQR (rather than adding integrator anti-windup) keeps the controller within its linearization validity regime. This is simpler and more robust than full anti-windup for a trajectory-tracking application.

4. **3-DOF measurement as modular upgrade:** The heading sensor is implemented as a parameter (`heading_sigma_deg`) rather than a code branch, making the system backward-compatible with GPS-only operation. The UKF automatically detects which measurement components observe heading via column inspection.

---

## Future Directions

- **MPC (Model Predictive Control):** Replace the precomputed LTV-LQR with online MPC that can handle state constraints (speed limits, heading bounds) explicitly. This would also naturally solve the reference index synchronization problem.
- **Adaptive Q tuning:** Use the NIS diagnostic in real-time to adjust Q_A_PSD and Q_ALPHA_PSD online, implementing a simple adaptation law that prevents both filter smugness and excess noise.
- **Dual-rate estimation:** Run the UKF at a higher rate than the control loop to reduce estimation lag, using IMU gyroscope at 100Hz and GPS at 10Hz with asynchronous measurement fusion.
- **Path optimization:** Compute a minimum-time racing line through the waypoints using the vehicle dynamics constraints, rather than prescribing speed and yaw-rate per segment.
