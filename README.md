# Vehicle tracking under GPS failure

A reproducible investigation of constrained control and estimation uncertainty during GPS outages. The project contains a historical CTRV LQR–MPC comparison and a separate synthetic bicycle experiment with actuator lag, friction limits, uncertainty-based speed control and bounded disturbance prediction.

## What the results support

Historical MPC improves completion under difficult disturbances but does not consistently improve tracking RMSE. In the synthetic vehicle, unbounded Gaussian acceleration margins caused the combined policy to stall. The bounded-prediction revision restored completion on six nominal/outage development routes while trading tracking accuracy for fewer physical violations.

The frozen revision `457d628` passed 94 tests, a 24-run prediction/backoff development ablation, and 160 fresh-seed confirmation runs with zero execution failures. The confirmation used seeds 1000–1004 across nominal, ten-second GPS outage, wet and friction-patch cases, both EKF/UKF, and four vehicle policies. All saved trajectories passed the finite-state, covariance and hard actuator audit. This is five seeds per group on the existing synthetic model and path, not hardware or unseen-road validation.

Across the balanced confirmation, baseline MPC completed 37/40 routes, speed-only 40/40, tightening-only 38/40 and the combined uncertainty policy 36/40. These counts do not establish overall superiority. The combined policy substantially reduced physical violation rates on nominal/outage cases but increased tracking RMSE and failed three UKF nominal/outage/wet routes and one EKF friction-patch route. Speed-only is the stronger completion candidate in this small comparison; it still has physical violations and solver fallbacks.

## Install and run

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt

# Fast numerical/regression check; restrict collection to canonical tests.
.venv/bin/python -m pytest tests -q

# Historical CTRV single run
.venv/bin/python scripts/run_simulation.py --controller mpc --scenario gps_loss_10s --save outage.png

# Focused vehicle development comparison: 24 full-route runs
.venv/bin/python scripts/run_vehicle_benchmarks.py --prediction-model bounded \
  --seeds 3 --workers 4 --scenarios nominal gps_loss_10s --estimators ukf \
  --save-traces --output vehicle_development_results

# Select the historical uncertainty predictor explicitly for an ablation.
# Use a separate output directory with --prediction-model linearized.

# Broader integration smoke; this includes 348 simulations.
.venv/bin/python scripts/validate.py --profile smoke --workers 4 --output validation_results/smoke
```

Vehicle policies are `mpc`, `speed`, `tightening`, and `uncertainty` (both). Vehicle benchmark and simulation defaults use bounded command prediction. Low-level filter/controller defaults retain the linearized predictor for compatibility; experiment manifests record the effective selection.

## Models and constraints

CTRV uses five states and acceleration/yaw-acceleration inputs. The vehicle uses eight states and acceleration/steering demands, with first-order actuators, bounded jerk/steering rate, RK4 integration, fixed axle loads and smooth tire-force saturation within each axle friction circle. Vehicle parameters and sensor noise are synthetic.

Both use dt=0.1 s and a 20-step MPC horizon. Commands are hard bounded; predicted physical envelopes use normalized penalized slack. Completion, strict success, actual violations, solver fallback and execution failure are separate metrics. Sparse OSQP solutions use warm starts, 1e−5 tolerances and a 4000-iteration limit. Historical CTRV uses fixed rho; vehicle MPC uses fixed-interval rho adaptation.

Bounded command prediction retains clipping-induced mean shifts and covariance using 81 quadrature points. Vehicle longitudinal acceleration backoffs use 512 projected Gaussian state samples and nominal-confidence quantiles. Other physical backoffs remain local Gaussian approximations. The speed governor is a covariance/friction heuristic. Neither softened constraints nor nominal-confidence labels establish nonlinear safety.

## Fresh seeds and controlled ablations

Vehicle runs accept `--start-seed` or `--seed-list`. Use unused seeds for confirmation; `--seeds 5` alone starts at zero. `--acceleration-constraints gaussian|quantile` separates acceleration backoffs from `--prediction-model linearized|bounded`, which selects process moments in both estimator and controller. With no explicit backoff selection, bounded prediction uses quantiles and linearized prediction uses Gaussian margins. Fingerprints record the resolved settings.

## Reproducibility and evidence

Results include manifests, exact configuration/dependency/source fingerprints, atomic per-run checkpoints, paired seed-bootstrap summaries and optional pickle-free NPZ traces. Resume only with matching fingerprints. Vehicle worker count is fingerprinted. Failed runs remain visible; incompatible archived results cannot validate a new source revision.

The completed historical full comparison has 2668 simulations. The completed earlier vehicle comparison has 320 runs. The bounded revision has 24 earlier development runs, a separate 24-run factorial development ablation and 160 fresh-seed confirmation runs; a stopped earlier vehicle suite remains partial. Full current-source validation is not yet established.

Full all-suite validation plans 4348 simulations and should be run deliberately after smaller development gates:

```bash
.venv/bin/python scripts/validate.py --profile full --workers 4 --output validation_results/full
```

GitHub workflows define smoke validation on Ubuntu/Python 3.11 and 3.12 plus macOS/Python 3.12, and a manual full run on Ubuntu/Python 3.12. Diagnostics upload on failure with 90-day retention. Hosted success must be verified on the published commit separately.

## Limitations and next work

Investigate the four incomplete combined-policy routes and compare their estimator consistency, progress, acceleration margins and solver fallback against matched speed-only runs before changing parameters. Quantify violation severity/distance exposure and runtime; investigate full joint state/disturbance prediction. Add standalone actuator-aware vehicle LQR before making bicycle-plant LQR–MPC comparisons.

Known friction, fixed loads, forward-only dynamics, local angle charts, reference-local gradients and soft slack limit realism. No vehicle-specific parameter fitting, friction identification, hardware deployment or real-time guarantee is claimed.

## Completed confirmation evidence

The development factorial used three existing seeds, two cases, UKF and the combined policy: linearized/Gaussian completed 0/6; bounded/Gaussian 6/6; linearized/quantile 5/6; bounded/quantile 6/6. The predictor switch affects both estimator and MPC moments; this does not isolate their contributions. No parameters were changed after this development gate.

Selected fresh-seed results below show mean cross-track RMSE and mean percentage of simulated steps with any physical violation. Each row has five runs; completion uses all five. Baseline is vehicle MPC, not LQR.

| Case/filter | Baseline RMSE m | Combined RMSE m | Baseline violations % | Combined violations % | Baseline/combined completion |
|---|---:|---:|---:|---:|---:|
| Nominal EKF | 0.355 | 1.067 | 8.766 | 0.279 | 5/5, 5/5 |
| Nominal UKF | 0.375 | 1.221 | 9.368 | 0.200 | 5/5, 4/5 |
| GPS outage EKF | 0.838 | 1.189 | 10.452 | 0.278 | 5/5, 5/5 |
| GPS outage UKF | 0.855 | 1.223 | 9.980 | 0.234 | 5/5, 4/5 |
| Wet EKF | 0.653 | 0.996 | 9.053 | 0.422 | 5/5, 5/5 |
| Wet UKF | 1.076 | 1.229 | 9.192 | 0.528 | 4/5, 4/5 |
| Friction patch EKF | 5.327 | 6.351 | 21.458 | 14.442 | 4/5, 4/5 |
| Friction patch UKF | 5.398 | 1.180 | 26.452 | 2.136 | 4/5, 5/5 |

Paired seed-bootstrap intervals support the nominal/outage violation reduction in this sample. Friction-patch EKF combined-minus-baseline RMSE is +1.025 m (95% interval −13.670 to +16.234); violation difference is −7.015 percentage points (−43.813 to +31.128). This case is inconclusive. Five seeds give weak precision; incomplete routes also have different exposure. Low violation percentages do not imply low violation severity or safety.

Conditional calibration used 18 fixed-true-state command-noise snapshots with 4096 samples each. Mean empirical coverage with 512-point quantiles was 99.62% against nominal 98%; the maximum upper-quantile range across 256/512/2048 samples was 0.0653 m/s². This indicates conservative conditional behavior, not full posterior or closed-loop calibration. The moment-order and sample-count choices remained unchanged for confirmation.

PSD covariance checks passed, but statistical consistency remained uneven: combined UKF nominal/wet mean normalized NEES was approximately 4.93/5.66, and friction-patch EKF approximately 17.91. Combined per-group mean run-level controller p95 ranged approximately 97–376 ms; many commands missed the 100 ms deadline during four-worker execution. Timing is descriptive, not a real-time guarantee.

Complete local manifests, checkpoint rows, paired intervals, traces and the audit are retained separately from the Git repository. Current-source full validation and hosted CI remain unverified; historical evidence does not substitute for those milestones.
