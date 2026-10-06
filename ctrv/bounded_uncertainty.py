"""Deterministic approximations for bounded command disturbances.

Quadrature retains clipping-induced mean shifts. Snapshot acceleration
quantiles use projected Gaussian state samples, not a safety guarantee or a
replacement for a calibrated joint state distribution.
"""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtri
from scipy.stats import qmc
from ctrv.vehicle import step_vehicle, derivative
from ctrv.model import wrap_to_pi


def command_moments(state, command, dt, mu, config, limits, noise, order=9):
    nodes, weights = hermgauss(order)
    a, b = np.meshgrid(nodes, nodes, indexing='ij')
    offsets = np.stack([a.ravel(), b.ravel()], axis=-1)*np.sqrt(2*np.diag(noise.command_covariance(dt)))
    weights = np.outer(weights, weights).ravel()/np.pi
    state = np.asarray(state); command = np.asarray(command)
    commands = command[...,None,:]+offsets
    states = np.broadcast_to(state[...,None,:],commands.shape[:-1]+(8,))
    points, _ = step_vehicle(states, commands, dt, mu, config, limits)
    nominal, _ = step_vehicle(state, command, dt, mu, config, limits)
    delta = points-nominal[...,None,:]; delta[...,3] = wrap_to_pi(delta[...,3])
    shift = np.einsum('i,...ij->...j',weights,delta)
    centered = delta-shift[...,None,:]
    covariance = np.einsum('i,...ij,...ik->...jk',weights,centered,centered)
    diffusion = np.zeros(8)
    diffusion[[2,5,4]] = [noise.longitudinal_model_psd,noise.lateral_model_psd,noise.yaw_model_psd]
    return shift, covariance+np.diag(diffusion)*dt


_NORMAL_POINTS = ndtri(qmc.Sobol(8,scramble=True,seed=2026).random_base2(9))


def acceleration_quantiles(mean, covariance, confidence, mu, config, limits):
    """512 fixed quasi-Monte Carlo samples; includes actuator/state support.

    Return physical body acceleration quantiles, not actuator demand quantiles.
    Projection is an approximation; confidence labels are nominal and must be
    checked empirically. Negative eigenvalues beyond roundoff are rejected.
    """
    if not .5<confidence<1:raise ValueError('Confidence must be between .5 and 1')
    values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
    if values.min() < -1e-8 or not np.isfinite(values).all():
        raise ValueError('Finite PSD covariance required for acceleration quantiles')
    points = mean+_NORMAL_POINTS@((vectors*np.sqrt(np.maximum(values,0.))).T)
    points[:,2] = np.maximum(points[:,2],0.)
    points[:,6] = np.clip(points[:,6],-limits.max_braking,limits.max_accel)
    points[:,7] = np.clip(points[:,7],-limits.max_steering,limits.max_steering)
    rates = derivative(points,points[:,[6,7]],mu,config,limits)
    acceleration = rates[:,2]-points[:,4]*points[:,5]
    return np.quantile(acceleration,[1-confidence,confidence])
