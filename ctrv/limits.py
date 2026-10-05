"""Explicit simulation limits; defaults are illustrative, not a validated car."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VehicleLimits:
    max_speed: float = 15.0
    max_accel: float = 3.0
    max_braking: float = 5.0
    max_lateral_accel: float = 3.0
    max_yaw_rate: float = 0.6
    max_yaw_accel: float = 1.5
    wheelbase: float = 2.7
    max_steering: float = np.deg2rad(35.0)
    max_steering_rate: float = np.deg2rad(45.0)

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('Vehicle limits must be finite and positive')
        if self.max_steering >= np.pi / 2:
            raise ValueError('max_steering must be below pi/2')

    @property
    def max_curvature(self):
        return np.tan(self.max_steering) / self.wheelbase


DEFAULT_LIMITS = VehicleLimits()
