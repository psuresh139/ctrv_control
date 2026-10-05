# LQR versus MPC under GPS failure

A reproducible vehicle-control benchmark investigating how estimation uncertainty
and actuator constraints affect trajectory tracking when GPS disappears.

**Current milestone:** cleaned CTRV mathematics, feasible reference generation,
and an EKF/UKF + LQR/MPC comparison. Both controllers use the same track,
sensors, limits, underlying random draws, and success criteria. MPC constrains
its linearized predictions; actual noisy-plant violations are measured separately.

## Run it

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
source .venv/bin/activate
python scripts/run_simulation.py --controller mpc --scenario gps_loss_5s --save gps_loss.png
python scripts/run_simulation.py --scenario gps_loss_10s --no-heading --save gps_only_loss.png
python -m pytest tests/ -q

# Full comparison: six scenarios, two estimators/controllers (2,400 runs)
python scripts/run_benchmarks.py --seeds 100 --controllers lqr mpc --workers 4 --output benchmark_results

# Prediction-only outages: no independent heading observations
python scripts/run_benchmarks.py --seeds 100 --no-heading \
  --scenarios nominal gps_loss_5s gps_loss_10s --output gps_only_results

# Short smoke run; optionally preserve state/covariance/sensor traces
python scripts/run_benchmarks.py --seeds 3 --save-traces --output smoke_results
```

The isolated environment uses requirements.lock.txt, including pinned OSQP and pytest.
The core dependencies are NumPy, SciPy, Matplotlib, and OSQP.
Use the recorded package versions in each manifest to recreate that environment.
Seeded runs are deterministic within an environment; cross-platform bitwise
identity is not promised.

## Clean mathematics

The state is `[x, y, speed, heading, yaw_rate]`; inputs are longitudinal and yaw
acceleration. The controlled model integrates

```
x_dot = speed * cos(heading)
y_dot = speed * sin(heading)
speed_dot = acceleration
heading_dot = yaw_rate
yaw_rate_dot = yaw_acceleration
```

Position uses eight-point Gaussian quadrature of the same speed/heading
polynomials used for the other states. Braking integrates translational motion
only until speed reaches zero. Numerical state/input Jacobians wrap heading
*differences*, avoiding branch-cut artifacts at ±π. The baseline analytical
Jacobian includes the correct zero-yaw-rate limit.

The UKF computes its heading mean in a local chart anchored at the central sigma
point. Both filters wrap heading innovations and support GPS+heading,
heading-only, or prediction-only updates. The linear measurement update includes
additive process covariance in innovation and cross-covariance, and uses Joseph
form. The heading chart assumes a locally concentrated distribution; multimodal
or broadly ambiguous headings require a different estimator.

Process covariance uses local-heading CWNA longitudinal/angular blocks and an
explicit heuristic lateral disturbance floor. It is an approximation, not an
exact discretization of all nonlinear uncertainty coupling. Rank-deficient Q is
valid; full rank is not a requirement for a functioning Kalman filter.

## Feasible references and explicit limits

A clamped spline specifies geometry. Numerical **actual spline arc length** sets
travel time. Heading and curvature come from spline derivatives; yaw rate is
`curvature * speed`. Legacy segment commands supply speed ceilings; their yaw
rate column is ignored because geometry determines yaw rate.

Forward/backward speed passes enforce acceleration/braking. Global time dilation
then enforces sampled yaw acceleration, steering rate, and lateral limits.
Feedforward comes from finite differences of the final speed/yaw-rate states;
commands are never clipped independently of the reference. Endpoint samples land
on the final waypoint at a whole timestep. Generation fails if geometry exceeds
the steering-curvature limit.

Default illustrative limits in `ctrv/limits.py`:

| Quantity | Limit |
|---|---:|
| Speed | 15 m/s |
| Acceleration / braking magnitude | 3 / 5 m/s² |
| Lateral acceleration | 3 m/s² |
| Yaw rate / yaw acceleration | 0.6 rad/s / 1.5 rad/s² |
| Bicycle-equivalent wheelbase | 2.7 m |
| Steering / steering rate | 35° / 45° per second |

Reference verification checks one-step position defects below 2 cm and heading
defects below 0.002 rad at dt=0.1 s. These are discretization tolerances, not exact
continuous-time feasibility guarantees between samples.

**Plant limitation:** this remains a kinematic CTRV plant with acceleration and
yaw-acceleration actuators. Steering is inferred as `atan(wheelbase*yaw_rate/speed)`
(using a 0.1 m/s denominator floor near rest). There is no steering actuator,
tire-slip, friction-circle, or dynamic bicycle model yet. Speed, lateral, yaw,
and equivalent steering violations are measured rather than hidden by state
projection. Process noise can move states beyond the envelope. These limits are
plausible configuration values, not specifications validated for a particular car.

## Reproducible scenarios

- Nominal GPS and an independent 5° heading sensor.
- GPS loss for 5 or 10 seconds, starting at t=15 s; heading remains available.
- GPS outliers on 5% of steps (20 m Gaussian injection).
- Constant GPS bias (+3 m x, −2 m y).
- Fourfold true process covariance with unchanged filter tuning.

With `--no-heading`, GPS outages become prediction-only periods. Gating uses the
99th chi-squared percentile for the **currently available measurement dimension**.
Missing innovations are NaN, not artificial zero measurements; rejected
innovations stay in the health summary.

Independent local RNG streams keep the underlying process/sensor draws paired
across estimators and scenarios. Changing filter Q does not change true noise
parameters. The true process covariance is state-dependent, so realized noise
vectors can differ after trajectories diverge. The simulator does not reset the
global NumPy random generator.

LQR uses fixed-time reference indexing, feedforward plus feedback, and explicit
asymmetric acceleration/yaw-acceleration clipping. The earlier error clamp and
spatial/temporal index heuristics are removed so the baseline's objective and
failure modes are directly reviewable.

## What the benchmark reports

`runs.csv` / `runs.json` contain each seed's cross-track RMSE/p95/max, heading
error, estimation RMSE, outage error, endpoint progress, completion, saturation,
physical-limit violations, all-innovation NIS/DOF, and NEES. Cross-track distance
uses projection onto the reference polyline, rather than distance to a stale
reference sample. For this non-self-intersecting track, a local segment search
around the nearest reference vertex supplies the projection.

Completion requires ≥98% final progress and an endpoint error ≤5 m. Staying on
track requires all cross-track distances ≤5 m. **Success additionally requires
zero sampled physical-envelope violations.** A completed lap can fail that
criterion. Control energy/rate diagnostics combine unlike input units and must
not be interpreted as physical energy in joules.

`summary.json` contains seed-level means and bootstrap 95% intervals (2,000
resamples, fixed bootstrap seed). Intervals are descriptive; five-seed smoke
results are not strong statistical evidence. Missing metrics are null, failed
runs remain in completion/success denominators, and the CLI returns nonzero on
execution failures. NEES has a five-state target; NIS/DOF has target 1, but means
alone do not establish formal filter consistency.

`manifest.json` records source SHA-256, Git revision/status, Python and dependency
versions, seed lists, effective sensor/plant/filter/controller configurations,
reference duration, and limits. Optional `.npz` files preserve full traces.
Historical README accuracy/stability claims are superseded; compare new results
only on identical benchmark configurations.

## Code map

- `ctrv/model.py`: consistent controlled dynamics and angle-aware Jacobians.
- `ctrv/ekf.py`: EKF/UKF prediction/update and dimension-aware NIS summaries.
- `ctrv/noise.py`: documented approximate process covariance.
- `ctrv/limits.py`: explicit vehicle-envelope configuration.
- `ctrv/trajectory.py`: arc-length geometry, speed planning, retiming, defect checks.
- `ctrv/lqr.py`: linearization and finite-horizon Riccati baseline.
- `ctrv/simulation.py`: sensor failures, independent RNGs, control loop, metrics.
- `scripts/run_benchmarks.py`: paired-seed runs, exports, provenance, summaries.
- `ctrv/mpc.py`: sparse constrained MPC and solver diagnostics.
- `scripts/validate.py`: local smoke/full test and benchmark pipeline.
- `tests/test_regressions.py`: integration, branch cuts, process covariance,
  feasible references, outages, rejection logging, and reproducibility.

## Constrained MPC

`MPCConfig` defaults to a 20-step / 2 s horizon, terminal weight 10×Q, input-change
weights (0.5, 5), and normalized slack penalty 10,000×s² + 1,000×s. Sparse LTV
dynamics include the reference's one-step affine defect. Angle deviations use a
local chart; terminal horizons shorten instead of inventing post-track motion.

Actuator bounds are hard. Predicted speed, yaw rate, lateral acceleration,
bicycle-equivalent steering/rate, and the ±5 m local track corridor use nonnegative
normalized slack. Lateral and steering quantities are linearized about the
reference; this is a local model, not a nonlinear feasibility guarantee.

OSQP uses eps_abs=eps_rel=1e-5, max_iter=4,000, fixed rho=3, adaptive_rho=False,
polishing, and shifted warm starts. Only finite `solved` outputs with actuator
violations ≤1e-4 are accepted; tiny numerical excesses are clipped to the bounds.
Unaccepted results and solver exceptions use bounded LQR fallback and remain in
solver-status/fallback reports. Missing OSQP raises an installation error.

Each simulation owns its warm-start state. Reference QP templates are cached;
computation timing includes first-use template construction, update, and solve.
Timings and 100 ms deadline misses are descriptive and machine-dependent.

## Durable validation

```bash
python scripts/validate.py --profile smoke --workers 4 --output validation_results/smoke
python scripts/validate.py --profile full --workers 4 --output validation_results/full
# Continue an interrupted matching run:
python scripts/validate.py --profile full --workers 4 --output validation_results/full --resume
```

Smoke runs tests (including both CLIs, deterministic recovery, solver failure and
checkpoint resume), then three seeds across seven scenarios, two estimators, and
two controllers. Full runs tests, 2,400 heading-enabled comparisons, 240 GPS-only
comparisons, and 28 representative trace runs including deterministic recovery.

`validation.json`, JUnit XML, stage logs, complete configurations, source hashes,
exact installed dependency versions, and benchmark artifacts are preserved.
Per-run JSON checkpoints and NPZ traces use atomic replacement. Resume requires
matching source, configuration, dependencies, Python/platform, and seed list.
The worker count can change on resume; timing results then describe the original
execution of each checkpoint. Failed checkpoints remain failures, not automatic
silent retries: use a new output directory after fixing their cause.

`paired_comparisons.json` reports MPC-minus-LQR differences and seed-bootstrap
intervals; failed runs count as incomplete/unsuccessful. Numeric differences use
only pairs with two successful executions and report their count. CSV records
include full configurations and solver-status counts. Raw NPZ diagnostics include
solver status, iterations, residuals, slack, fallback, and controller timing.

GitHub Actions runs smoke validation on Ubuntu/Python 3.11 and 3.12 and
macOS/Python 3.12 for pushes and pull requests. A manual Ubuntu/Python 3.12 workflow
runs full benchmarks. Diagnostic upload runs even on failure and retains artifacts
for 90 days; local full outputs remain independently available. A written workflow
is not evidence of hosted execution; report that separately after a hosted run.

## Next milestone: bicycle dynamics

Replace equivalent steering diagnostics with a steering/acceleration bicycle
plant and adapt both controllers and estimators together. Keep the CTRV benchmark
as a historical comparison. Hardware/tire dynamics are outside this milestone.

## Uncertainty and synthetic vehicle add-on

The historical CTRV/LQR/MPC comparison remains unchanged by default. The new
vehicle experiments use eight states: position, longitudinal speed, body
heading, yaw rate, lateral speed, acceleration actuator and steering actuator.
The dynamic bicycle includes first-order actuator response, bounded acceleration
jerk and steering rate, aerodynamic drag, fixed axle normal loads, and smooth
combined longitudinal/lateral tire-force saturation inside each axle's friction
circle. Parameters are synthetic; suspension, wheel rotation, load transfer,
friction identification and hardware validation are outside this model.

The vehicle EKF and UKF predict the same discrete RK4 model as the plant. They
use commands and GPS/heading observations, never the true hidden state or the
realized disturbance. Vehicle MPC optimizes acceleration and steering demands
with hard command bounds, affine reference defects, a 20-step horizon, warm
starts, normalized soft envelopes and bounded model-consistent LQR fallback.
It predicts physical lateral, longitudinal and yaw acceleration, rather than
using CTRV acceleration proxies. Local Gaussian uncertainty margins use
`Phi^-1(confidence) * sqrt(g P g.T)` on each constraint side; steering-rate
margins retain adjacent-state covariance. These are open-loop, marginal risk
surrogates. Soft slack, nonlinear tire saturation, correlated constraints and
miscalibrated covariance prevent a formal chance-feasibility or safety claim.

Four paired policies isolate the additions: `mpc`, `tightening`, `speed`, and
`uncertainty` (both). The speed governor reduces target speed with lateral
position and heading uncertainty and reserves 20% of the declared friction
budget for cornering. The reference follows estimated path progress and is
locally retimed; slowing does not cause a fixed-time goal to run away. The time
budget is twice the historical reference duration, and completion time is
reported explicitly. The same geometry and vehicle limits apply to every
policy. Friction is declared available equally to all policies; this is not an
experiment in estimating road friction.

Vehicle command disturbance PSDs are explicitly in acceleration-demand and
steering-demand units (`2.0` and `0.00005`, respectively). They are synthetic,
not copied numerically from CTRV yaw-acceleration units. Commands are disturbed
before bounded actuator integration, rather than adding instantaneous Gaussian
jumps to the vehicle state. Sensor settings and random draws are shared across
policies. Vehicle results are a new comparison, not directly interchangeable
with historical CTRV scores.

```bash
.venv/bin/python scripts/run_vehicle_benchmarks.py --seeds 3 --workers 4 \
  --save-traces --output vehicle_results
.venv/bin/python scripts/validate.py --profile smoke
.venv/bin/python scripts/validate.py --profile full --suite vehicle
```

Default smoke validation runs existing controller tests and benchmarks, plus
three vehicle seeds across ten scenarios, four policies and both estimators
at a 30-second horizon, and a GPS-only vehicle check. Full vehicle validation
uses 20 seeds across ten scenarios (1,600 runs) and 80 representative traces.
The default full suite also runs the original 2,640-run CTRV comparison and its
28 traces. Checkpoints are atomic and resumable only with matching source,
configuration, dependencies, platform, thread settings and seeds. Execution
failures stay visible and fail validation. Reports include paired bootstrap
intervals, per-component violations, solver fallback, slack, covariance
consistency, friction utilization, completion time and descriptive controller
timing. Preserve local result directories independently of CI retention.

Vehicle NEES has eight state dimensions. Reported posterior position and heading
interval coverage uses a one-sided 99% quantile on both sides (a nominal 98%
marginal interval); it is descriptive calibration evidence, not joint safety
coverage. The friction perturbation is scheduled in simulation time, rather
than located on a spatial road map. A slow-actuator case doubles both response
time constants without changing sensors or disturbances.
