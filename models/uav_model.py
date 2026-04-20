"""
UAV Model - Quadrotor 5-state dynamic model

State: [x, y, h, theta, phi]
  x     - x position (m)
  y     - y position (m)
  h     - altitude (m)
  theta - yaw angle (rad)
  phi   - pitch angle (rad)

Control: [v, omega, zeta]
  v     - linear velocity (m/s)
  omega - yaw rate (rad/s)
  zeta  - pitch rate (rad/s)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class UAVState:
    """Quadrotor state representation."""
    x: float
    y: float
    h: float
    theta: float  # yaw
    phi: float     # pitch

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.h, self.theta, self.phi])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'UAVState':
        return cls(x=arr[0], y=arr[1], h=arr[2], theta=arr[3], phi=arr[4])

    def __repr__(self):
        return f"UAVState(x={self.x:.2f}, y={self.y:.2f}, h={self.h:.2f}, theta={self.theta:.3f}, phi={self.phi:.3f})"


@dataclass
class UAVControl:
    """Quadrotor control input."""
    v: float      # linear velocity (m/s)
    omega: float  # yaw rate (rad/s)
    zeta: float   # pitch rate (rad/s)

    def to_array(self) -> np.ndarray:
        return np.array([self.v, self.omega, self.zeta])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'UAVControl':
        return cls(v=arr[0], omega=arr[1], zeta=arr[2])


class UAVModel:
    """
    Quadrotor discrete-time dynamics model.

    Discrete model (Euler integration, sampling time Ts):
        x_next  = x  + v * cos(theta) * cos(phi)
        y_next  = y  + v * sin(theta) * cos(phi)
        h_next  = h  + v * sin(phi)
        phi_next = phi + zeta
        theta_next = theta + omega
    """

    def __init__(self, Ts: float = 1.0):
        self.Ts = Ts  # sampling time

    def step(self, state: UAVState, control: UAVControl) -> UAVState:
        """Single-step forward dynamics."""
        v, omega, zeta = control.v, control.omega, control.zeta
        x, y, h, theta, phi = state.x, state.y, state.h, state.theta, state.phi

        x_next = x + v * np.cos(theta) * np.cos(phi)
        y_next = y + v * np.sin(theta) * np.cos(phi)
        h_next = h + v * np.sin(phi)
        phi_next = phi + zeta
        theta_next = theta + omega

        return UAVState(x=x_next, y=y_next, h=h_next, theta=theta_next, phi=phi_next)

    def step_array(self, state_arr: np.ndarray, control_arr: np.ndarray) -> np.ndarray:
        """Vectorized single-step forward dynamics from arrays."""
        v, omega, zeta = control_arr[0], control_arr[1], control_arr[2]
        x, y, h, theta, phi = state_arr[0], state_arr[1], state_arr[2], state_arr[3], state_arr[4]

        x_next = x + v * np.cos(theta) * np.cos(phi)
        y_next = y + v * np.sin(theta) * np.cos(phi)
        h_next = h + v * np.sin(phi)
        phi_next = phi + zeta
        theta_next = theta + omega

        return np.array([x_next, y_next, h_next, theta_next, phi_next])

    def predict(self, state: UAVState, controls: np.ndarray, N: int) -> np.ndarray:
        """
        Roll out prediction over N steps given a (N, 3) control sequence.
        Returns (N, 5) state trajectory.
        """
        states = np.zeros((N, 5))
        current = state.to_array()
        for i in range(N):
            current = self.step_array(current, controls[i])
            states[i] = current
        return states

    @staticmethod
    def rotation_matrix_3d(theta: float, phi: float) -> np.ndarray:
        """Compute 3x3 rotation matrix R(theta, phi) for position error transformation."""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        return np.array([
            [cos_theta * cos_phi,  cos_phi * sin_theta, -sin_phi],
            [-sin_theta,            cos_theta,            0],
            [cos_theta * sin_phi,  sin_theta * sin_phi,  cos_phi]
        ])
