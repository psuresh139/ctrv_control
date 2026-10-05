"""Eight-state EKF/UKF for the synthetic actuator and tire model.

Only commands, declared friction and GPS/heading observations enter prediction.
No plant state or realized disturbance enters either filter. Process covariance
maps random command perturbations through the actuator dynamics; optional
model diffusion represents unmodelled acceleration, lateral and yaw dynamics.
"""
from dataclasses import dataclass
import numpy as np
from ctrv.ekf import UKFParams
from ctrv.model import wrap_to_pi
from ctrv.vehicle import VehicleConfig, step_vehicle
from ctrv.limits import DEFAULT_LIMITS


@dataclass(frozen=True)
class VehicleNoise:
    acceleration_command_psd: float = 2.
    steering_command_psd: float = .00005
    longitudinal_model_psd: float = .02
    lateral_model_psd: float = .02
    yaw_model_psd: float = .001

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in self.__dict__.values()):
            raise ValueError('Vehicle noise intensities must be finite and nonnegative')

    def command_covariance(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite timestep required')
        return np.diag([self.acceleration_command_psd, self.steering_command_psd])/dt


def validate_vehicle_covariance(P):
    P = np.asarray(P, float)
    if (P.shape != (8, 8) or not np.all(np.isfinite(P))
            or not np.allclose(P, P.T, atol=1e-9)
            or np.linalg.eigvalsh(P).min() < -1e-8):
        raise ValueError('Finite symmetric PSD eight-state covariance required')
    return P


def stabilize_covariance(P):
    P = (P+P.T)/2
    values, vectors = np.linalg.eigh(P)
    return (vectors*np.maximum(values, 1e-12))@vectors.T


def vehicle_prediction(state, command, dt, mu, config=VehicleConfig(), limits=DEFAULT_LIMITS):
    return step_vehicle(state, command, dt, mu, config, limits)[0]


def vehicle_linearization(state, command, dt, mu, config=VehicleConfig(), limits=DEFAULT_LIMITS):
    """Central differences of the same RK4 discrete map used by the plant.

    Angle differences use a local chart, including at the +/-pi boundary.
    Bounds and friction saturation make this a local derivative, not a global
    smooth model or a guarantee for large disturbances.
    """
    state = np.asarray(state, float); command = np.asarray(command, float)
    if state.shape[-1:]!=(8,) or command.shape[-1:]!=(2,) or state.shape[:-1]!=command.shape[:-1]:
        raise ValueError('Matching state and command batch dimensions required')
    states=np.repeat(state[...,None,:],21,axis=-2)
    commands=np.repeat(command[...,None,:],21,axis=-2)
    epsilons=1e-5*np.maximum(1.,np.abs(np.concatenate([state,command],axis=-1)))
    for j in range(10):
        target=states if j<8 else commands
        col=j if j<8 else j-8
        target[...,1+2*j,col]+=epsilons[...,j]
        target[...,2+2*j,col]-=epsilons[...,j]
    mapped=vehicle_prediction(states,commands,dt,mu,config,limits)
    differences=mapped[...,1::2,:]-mapped[...,2::2,:]
    differences[...,3]=wrap_to_pi(differences[...,3])
    J=np.swapaxes(differences/(2*epsilons[...,None]),-1,-2)
    return mapped[...,0,:],J[...,:8],J[...,8:]


def vehicle_process_covariance(B, dt, noise=VehicleNoise()):
    diffusion = np.zeros(8)
    diffusion[[2, 5, 4]] = [noise.longitudinal_model_psd,
                            noise.lateral_model_psd, noise.yaw_model_psd]
    return B@noise.command_covariance(dt)@B.T+np.diag(diffusion)*dt


def vehicle_filter_step(state, covariance, command, dt, mu, observation=None,
                        H=None, R=None, estimator='ekf', noise=VehicleNoise(),
                        config=VehicleConfig(), limits=DEFAULT_LIMITS, gate=None):
    """GPS/heading selector observation update using Joseph covariance form."""
    state = np.asarray(state, float)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError('Finite eight-state estimate required')
    covariance = validate_vehicle_covariance(covariance)
    if estimator not in ('ekf', 'ukf'):
        raise ValueError('Estimator must be ekf or ukf')
    predicted, A, B = vehicle_linearization(state, command, dt, mu, config, limits)
    Q = vehicle_process_covariance(B, dt, noise)
    if estimator == 'ekf':
        P = A@covariance@A.T+Q
    else:
        params = UKFParams(n=8, alpha=.2)
        points = params.sigma_points(state, covariance)
        points = vehicle_prediction(points, command, dt, mu, config, limits)
        predicted = params.Wm@points
        anchor = points[0, 3]
        predicted[3] = wrap_to_pi(anchor+params.Wm@wrap_to_pi(points[:, 3]-anchor))
        errors = points-predicted; errors[:, 3] = wrap_to_pi(errors[:, 3])
        P = np.einsum('i,ij,ik->jk', params.Wc, errors, errors)+Q
    P = stabilize_covariance(P)
    info = {'nis': None, 'gated': False, 'dof': 0}
    if observation is None:
        return predicted, P, info
    H = np.asarray(H, float); R = np.asarray(R, float); observation = np.asarray(observation, float)
    m = len(observation)
    # Deliberately limit this API to individual state selectors.
    if (H.shape != (m, 8) or R.shape != (m, m) or m == 0
            or not np.all(np.isfinite(H)) or not np.all(np.isfinite(R))
            or not np.all(np.isfinite(observation))
            or not np.all((H == 0) | (H == 1)) or not np.all(H.sum(axis=1) == 1)
            or not np.allclose(R, R.T) or np.linalg.eigvalsh(R).min() <= 0):
        raise ValueError('Finite selector observations and positive definite R required')
    innovation = observation-H@predicted
    innovation[H[:, 3] == 1] = wrap_to_pi(innovation[H[:, 3] == 1])
    S = H@P@H.T+R
    nis = float(innovation@np.linalg.solve(S, innovation))
    info.update(nis=nis, gated=bool(gate is not None and nis > gate), dof=m)
    if info['gated']:
        return predicted, P, info
    gain = np.linalg.solve(S, H@P).T
    posterior = predicted+gain@innovation; posterior[3] = wrap_to_pi(posterior[3])
    residual = np.eye(8)-gain@H
    return posterior, stabilize_covariance(residual@P@residual.T+gain@R@gain.T), info
