"""
Formation geometry and utilities for multi-UAV formation control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FormationConfig:
    """Formation configuration for N UAVs."""
    # Formation vectors (relative positions from leader)
    # dr_ij = position_of_UAV_j - position_of_UAV_i
    dr01: np.ndarray  # [dx, dy, dh] leader to UAV1 (leader itself, usually zero)
    dr12: np.ndarray  # UAV1 to UAV2
    dr13: np.ndarray  # UAV1 to UAV3
    dr14: np.ndarray  # UAV1 to UAV4

    # Derived formation vectors
    @property
    def dr23(self) -> np.ndarray:
        return self.dr13 - self.dr12

    @property
    def dr24(self) -> np.ndarray:
        return self.dr14 - self.dr12

    @property
    def dr34(self) -> np.ndarray:
        return self.dr14 - self.dr13

    @property
    def all_vectors(self) -> Dict[str, np.ndarray]:
        return {
            'dr01': self.dr01,
            'dr12': self.dr12,
            'dr13': self.dr13,
            'dr14': self.dr14,
            'dr23': self.dr23,
            'dr24': self.dr24,
            'dr34': self.dr34,
        }

    @classmethod
    def default_3d(cls) -> 'FormationConfig':
        """Default 3D formation used in the paper."""
        dr01 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # 5D for full state
        dr12 = np.array([10.0, 27.0, 0.0, 0.0, 0.0])
        dr13 = np.array([20.0, -22.0, 0.0, 0.0, 0.0])
        dr14 = np.array([30.0, 10.0, 0.0, 0.0, 0.0])
        return cls(dr01=dr01, dr12=dr12, dr13=dr13, dr14=dr14)

    @classmethod
    def compact_3d(cls) -> 'FormationConfig':
        """Compact 3D formation."""
        dr01 = np.array([5.0, 0.0, 0.0])
        dr12 = np.array([10.0, 20.0, 0.0])
        dr13 = np.array([10.0, -5.0, 0.0])
        dr14 = np.array([5.0, -8.0, 0.0])
        return cls(
            dr01=np.append(dr01, [0.0, 0.0]),
            dr12=np.append(dr12, [0.0, 0.0]),
            dr13=np.append(dr13, [0.0, 0.0]),
            dr14=np.append(dr14, [0.0, 0.0])
        )


def sigma_cal(xp1: np.ndarray, yp1: np.ndarray, hp1: np.ndarray,
              xp2: np.ndarray, yp2: np.ndarray, hp2: np.ndarray,
              R: float) -> float:
    """
    Compute sigma - safety margin for collision avoidance.

    sigma = min(dist / 2 - R) over prediction horizon

    Args:
        xp1, yp1, hp1: Predicted states of UAV 1
        xp2, yp2, hp2: Predicted states of UAV 2
        R: Minimum safe distance

    Returns:
        sigma: Minimum safety margin
    """
    distances = np.sqrt((xp1 - xp2)**2 + (yp1 - yp2)**2 + (hp1 - hp2)**2)
    sigma = np.min(distances / 2 - R)
    return sigma


def eta_cal(xp1: np.ndarray, yp1: np.ndarray, hp1: np.ndarray,
            xp2: np.ndarray, yp2: np.ndarray, hp2: np.ndarray,
            xp3: np.ndarray, yp3: np.ndarray, hp3: np.ndarray,
            d12: np.ndarray, d23: np.ndarray,
            sigma12: float, sigma23: float) -> float:
    """
    Compute eta - coupling term for formation constraints.

    eta = phi * (alpha * xi12 + beta * xi23) / ((T - delta) * beta * (...))
    """
    phi = 0.99
    beta = 0.1
    alpha = 10.0
    T = 3.0
    delta = 0.5

    xi12 = np.max(np.sqrt(
        (xp1 - xp2 - d12[0])**2 +
        (yp1 - yp2 - d12[1])**2 +
        (hp1 - hp2 - d12[2])**2
    ))

    xi23 = np.max(np.sqrt(
        (xp2 - xp3 - d23[0])**2 +
        (yp2 - yp3 - d23[1])**2 +
        (hp2 - hp3 - d23[2])**2
    ))

    term12 = alpha * np.array([
        xp1[0] - xp2[0] - d12[0],
        yp1[0] - yp2[0] - d12[1],
        hp1[0] - hp2[0] - d12[2]
    ]).dot(np.array([
        xp1[0] - xp2[0] - d12[0],
        yp1[0] - yp2[0] - d12[1],
        hp1[0] - hp2[0] - d12[2]
    ]))

    term23 = np.array([
        xp2[0] - xp3[0] - d23[0],
        yp2[0] - yp3[0] - d23[1],
        hp2[0] - hp3[0] - d23[2]
    ]).dot(np.array([
        xp2[0] - xp3[0] - d23[0],
        yp2[0] - yp3[0] - d23[1],
        hp2[0] - hp3[0] - d23[2]
    ]))

    denom = (T - delta) * beta * ((2 * xi12 + 3 * sigma12) + (2 * xi23 + 3 * sigma23))

    eta = phi * (term12 + beta * term23) / denom

    return eta


def compute_formation_errors(
    states: Dict[int, np.ndarray],  # uav_id -> (N, 5) state trajectory
    config: FormationConfig
) -> Dict[str, np.ndarray]:
    """
    Compute formation errors for all UAV pairs.

    Returns dict of error arrays for each formation pair.
    """
    errors = {}

    # Extract positions
    x1, y1, h1 = states[1][:, 0], states[1][:, 1], states[1][:, 2]
    x2, y2, h2 = states[2][:, 0], states[2][:, 1], states[2][:, 2]
    x3, y3, h3 = states[3][:, 0], states[3][:, 1], states[3][:, 2]
    x4, y4, h4 = states[4][:, 0], states[4][:, 1], states[4][:, 2]

    # Formation errors: (actual relative position) - (desired formation vector)
    errors['err_12'] = np.sqrt((x1 - x2 - config.dr12[0])**2 +
                               (y1 - y2 - config.dr12[1])**2 +
                               (h1 - h2 - config.dr12[2])**2)

    errors['err_13'] = np.sqrt((x1 - x3 - config.dr13[0])**2 +
                               (y1 - y3 - config.dr13[1])**2 +
                               (h1 - h3 - config.dr13[2])**2)

    errors['err_14'] = np.sqrt((x1 - x4 - config.dr14[0])**2 +
                               (y1 - y4 - config.dr14[1])**2 +
                               (h1 - h4 - config.dr14[2])**2)

    errors['err_23'] = np.sqrt((x2 - x3 - config.dr23[0])**2 +
                               (y2 - y3 - config.dr23[1])**2 +
                               (h2 - h3 - config.dr23[2])**2)

    errors['err_24'] = np.sqrt((x2 - x4 - config.dr24[0])**2 +
                               (y2 - y4 - config.dr24[1])**2 +
                               (h2 - h4 - config.dr24[2])**2)

    errors['err_34'] = np.sqrt((x3 - x4 - config.dr34[0])**2 +
                               (y3 - y4 - config.dr34[1])**2 +
                               (h3 - h4 - config.dr34[2])**2)

    return errors
