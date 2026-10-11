# Vehicle tracking under GPS failure

A reproducible investigation of constrained control and estimation uncertainty during GPS outages. The project contains a historical CTRV LQR–MPC comparison and a separate synthetic bicycle experiment with actuator lag, friction limits, uncertainty-based speed control and bounded disturbance prediction.

## What the results support

Historical MPC improves completion under difficult disturbances but does not consistently improve tracking RMSE. In the synthetic vehicle, unbounded Gaussian acceleration margins caused the combined policy to stall. The bounded-prediction revision restored completion on six nominal/outage development routes while trading tracking accuracy for fewer physical violations.

The latest revision passed 91 tests and 24 development simulations. Combined policy mean RMSE was approximately 0.90 m, with physical violations on approximately 0.28% of steps. These are three-seed UKF development results, not an independent confirmation, a calibrated safety guarantee, or hardware validation.

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

The completed historical full comparison has 2668 simulations. The completed earlier vehicle comparison has 320 runs. The latest bounded revision has 24 final development runs; a stopped earlier vehicle suite remains partial. Full current-source validation is not yet established.

Full all-suite validation plans 4348 simulations and should be run deliberately after smaller development gates:

```bash
.venv/bin/python scripts/validate.py --profile full --workers 4 --output validation_results/full
```

GitHub workflows define smoke validation on Ubuntu/Python 3.11 and 3.12 plus macOS/Python 3.12, and a manual full run on Ubuntu/Python 3.12. Diagnostics upload on failure with 90-day retention. Hosted success must be verified on the published commit separately.

## Limitations and next work

Calibrate uncertainty tails and quantify sampling sensitivity; add fresh seed selection and evaluate both filters on held-out nominal/outage/wet/friction-change cases. Investigate full joint state/disturbance prediction, violation severity/distance exposure, and runtime. Add standalone actuator-aware vehicle LQR before making bicycle-plant LQR–MPC comparisons.

Known friction, fixed loads, forward-only dynamics, local angle charts, reference-local gradients and soft slack limit realism. No vehicle-specific parameter fitting, friction identification, hardware deployment or real-time guarantee is claimed.

## Current confirmation status

The latest published-in-this-repository evidence remains the 24-run development comparison described above. A new factorial prediction/backoff ablation, sampling calibration, and a 160-run fresh-seed confirmation are being prepared; they are not completed evidence yet. Historical results are not current-source full validation.
