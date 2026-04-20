"""
Constraint functions for DMPC optimization - mirrors MATLAB test_con*.m

Each constraint function returns (c, ceq) where:
    c <= 0  : inequality constraints
    ceq = 0 : equality constraints (model dynamics)
"""

import numpy as np
from typing import Tuple, List

# Control input limits (from MATLAB)
V_MAX = 20.0
V_MIN = 0.0  # 0 allows full speed range including deceleration
OMEGA_MAX = 45.0 * np.pi / 180.0  # rad/s
OMEGA_MIN = -45.0 * np.pi / 180.0
ZETA_MAX = 45.0 * np.pi / 180.0
ZETA_MIN = -45.0 * np.pi / 180.0

# Collision avoidance radius
R_SAFE = 2.0


def constraint_uav1(
    h: np.ndarray,
    x1now: float, y1now: float, h1now: float,
    thelt1now: float, phi1now: float,
    Nc: int, Np: int,
    x2p: np.ndarray, y2p: np.ndarray, h2p: np.ndarray,
    x3p: np.ndarray, y3p: np.ndarray, h3p: np.ndarray,
    R: float = R_SAFE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    UAV1 (Leader) constraints.

    Decision vector h layout (same as cost):
        h[0:Nc]           -> v1
        h[Nc:Nc+Nc]       -> omega1
        h[Nc+Nc:Nc+Nc+Np] -> zeta1
        h[Nc+Nc+Np:...]   -> x1, y1, h1, thelt1, phi1 (states)
    """
    v1 = h[0:Nc]
    omega1 = h[Nc:Nc+Nc]
    zeta1 = h[Nc+Nc:Nc+Nc+Np]

    x1 = h[Nc+Nc+Np:Nc+Nc+Np+Np]
    y1 = h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
    h1 = h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
    thelt1 = h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
    phi1 = h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

    # Equality constraints: model dynamics
    ceq = np.zeros(5 * Np)

    for i in range(Np):
        if i == 0:
            # First step uses current state
            ceq[i] = x1[i] - (x1now + v1[i] * np.cos(thelt1now))
            ceq[i + Np] = y1[i] - (y1now + v1[i] * np.sin(thelt1now))
            ceq[i + 2*Np] = h1[i] - (h1now + v1[i] * np.sin(phi1now))
            ceq[i + 3*Np] = thelt1[i] - (thelt1now + omega1[i])
            ceq[i + 4*Np] = phi1[i] - (phi1now + zeta1[i])
        else:
            # Subsequent steps use predicted state
            ceq[i] = x1[i] - (x1[i-1] + v1[i] * np.cos(thelt1[i-1]))
            ceq[i + Np] = y1[i] - (y1[i-1] + v1[i] * np.sin(thelt1[i-1]))
            ceq[i + 2*Np] = h1[i] - (h1[i-1] + v1[i] * np.sin(phi1[i-1]))
            ceq[i + 3*Np] = thelt1[i] - (thelt1[i-1] + omega1[i])
            ceq[i + 4*Np] = phi1[i] - (phi1[i-1] + zeta1[i])

    # Inequality constraints: control limits
    c = np.zeros(6 * Nc)

    for i in range(Nc):
        c[i] = v1[i] - V_MAX
        c[i + Nc] = V_MIN - v1[i]
        c[i + 2*Nc] = omega1[i] - OMEGA_MAX
        c[i + 3*Nc] = OMEGA_MIN - omega1[i]
        c[i + 4*Nc] = zeta1[i] - ZETA_MAX
        c[i + 5*Nc] = ZETA_MIN - zeta1[i]

    return c, ceq


def constraint_uav2(
    h: np.ndarray,
    x2now: float, y2now: float, h2now: float,
    thelt2now: float, phi2now: float,
    dr12: np.ndarray, dr23: np.ndarray, dr24: np.ndarray,
    Nc: int, Np: int,
    x1p: np.ndarray, y1p: np.ndarray, h1p: np.ndarray,
    thelt1p: np.ndarray, phi1p: np.ndarray,
    x2p_pred: np.ndarray, y2p_pred: np.ndarray, h2p_pred: np.ndarray,
    x3p: np.ndarray, y3p: np.ndarray, h3p: np.ndarray,
    x4p: np.ndarray, y4p: np.ndarray, h4p: np.ndarray,
    R: float = R_SAFE,
    k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    UAV2 (Follower) constraints including collision avoidance.
    """
    v2 = h[0:Nc]
    omega2 = h[Nc:Nc+Nc]
    zeta2 = h[Nc+Nc:Nc+Nc+Np]

    x2 = h[Nc+Nc+Np:Nc+Nc+Np+Np]
    y2 = h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
    h2 = h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
    thelt2 = h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
    phi2 = h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

    # Model dynamics equality constraints
    ceq = np.zeros(5 * Np)

    for i in range(Np):
        if i == 0:
            ceq[i] = x2[i] - (x2now + v2[i] * np.cos(thelt2now))
            ceq[i + Np] = y2[i] - (y2now + v2[i] * np.sin(thelt2now))
            ceq[i + 2*Np] = h2[i] - (h2now + v2[i] * np.sin(phi2now))
            ceq[i + 3*Np] = thelt2[i] - (thelt2now + omega2[i])
            ceq[i + 4*Np] = phi2[i] - (phi2now + zeta2[i])
        else:
            ceq[i] = x2[i] - (x2[i-1] + v2[i] * np.cos(thelt2[i-1]))
            ceq[i + Np] = y2[i] - (y2[i-1] + v2[i] * np.sin(thelt2[i-1]))
            ceq[i + 2*Np] = h2[i] - (h2[i-1] + v2[i] * np.sin(phi2[i-1]))
            ceq[i + 3*Np] = thelt2[i] - (thelt2[i-1] + omega2[i])
            ceq[i + 4*Np] = phi2[i] - (phi2[i-1] + zeta2[i])

    # Collision avoidance parameters (sigma, eta)
    from quadrotor_dmpc.models.formation import sigma_cal, eta_cal

    sigma12 = sigma_cal(x1p, y1p, h1p, x2p_pred, y2p_pred, h2p_pred, R)
    sigma23 = sigma_cal(x2p_pred, y2p_pred, h2p_pred, x3p, y3p, h3p, R)
    sigma24 = sigma_cal(x2p_pred, y2p_pred, h2p_pred, x4p, y4p, h4p, R)
    sigma2 = min(sigma12, sigma23)

    eta2 = eta_cal(x1p, y1p, h1p, x2p_pred, y2p_pred, h2p_pred,
                    x3p, y3p, h3p, dr12, dr23, sigma12, sigma23)

    # Inequality constraints
    c = np.zeros(8 * Nc)

    for i in range(Nc):
        c[i] = v2[i] - V_MAX
        c[i + Nc] = V_MIN - v2[i]
        c[i + 2*Nc] = omega2[i] - OMEGA_MAX
        c[i + 3*Nc] = OMEGA_MIN - omega2[i]
        c[i + 4*Nc] = zeta2[i] - ZETA_MAX
        c[i + 5*Nc] = ZETA_MIN - zeta2[i]

        # Collision avoidance: tracking error bounded by safety margin
        track_err = np.sqrt(
            (x2p_pred[i] - x2[i])**2 +
            (y2p_pred[i] - y2[i])**2 +
            (h2p_pred[i] - h2[i])**2
        )
        c[i + 6*Nc] = track_err - min(sigma2, eta2)

        # Safe distance constraints from neighbors
        dist_12 = np.sqrt(
            (x1p[i] - x2[i])**2 +
            (y1p[i] - y2[i])**2 +
            (h1p[i] - h2[i])**2
        )
        c[i + 7*Nc] = 2 * R + sigma12 - dist_12

    return c, ceq


def constraint_uav3(
    h: np.ndarray,
    x3now: float, y3now: float, h3now: float,
    thelt3now: float, phi3now: float,
    dr13: np.ndarray, dr23: np.ndarray,
    Nc: int, Np: int,
    x1p: np.ndarray, y1p: np.ndarray, h1p: np.ndarray,
    thelt1p: np.ndarray, phi1p: np.ndarray,
    x2p: np.ndarray, y2p: np.ndarray, h2p: np.ndarray,
    x3p_pred: np.ndarray, y3p_pred: np.ndarray, h3p_pred: np.ndarray,
    x4p: np.ndarray, y4p: np.ndarray, h4p: np.ndarray,
    R: float = R_SAFE,
    k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """UAV3 (Follower) constraints."""
    v3 = h[0:Nc]
    omega3 = h[Nc:Nc+Nc]
    zeta3 = h[Nc+Nc:Nc+Nc+Np]

    x3 = h[Nc+Nc+Np:Nc+Nc+Np+Np]
    y3 = h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
    h3 = h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
    thelt3 = h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
    phi3 = h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

    # Model dynamics
    ceq = np.zeros(5 * Np)
    for i in range(Np):
        if i == 0:
            ceq[i] = x3[i] - (x3now + v3[i] * np.cos(thelt3now))
            ceq[i + Np] = y3[i] - (y3now + v3[i] * np.sin(thelt3now))
            ceq[i + 2*Np] = h3[i] - (h3now + v3[i] * np.sin(phi3now))
            ceq[i + 3*Np] = thelt3[i] - (thelt3now + omega3[i])
            ceq[i + 4*Np] = phi3[i] - (phi3now + zeta3[i])
        else:
            ceq[i] = x3[i] - (x3[i-1] + v3[i] * np.cos(thelt3[i-1]))
            ceq[i + Np] = y3[i] - (y3[i-1] + v3[i] * np.sin(thelt3[i-1]))
            ceq[i + 2*Np] = h3[i] - (h3[i-1] + v3[i] * np.sin(phi3[i-1]))
            ceq[i + 3*Np] = thelt3[i] - (thelt3[i-1] + omega3[i])
            ceq[i + 4*Np] = phi3[i] - (phi3[i-1] + zeta3[i])

    from quadrotor_dmpc.models.formation import sigma_cal, eta_cal

    sigma13 = sigma_cal(x1p, y1p, h1p, x2p, y2p, h2p, R)
    sigma23 = sigma_cal(x2p, y2p, h2p, x3p_pred, y3p_pred, h3p_pred, R)
    sigma43 = sigma_cal(x4p, y4p, h4p, x3p_pred, y3p_pred, h3p_pred, R)
    sigma3 = min(sigma13, sigma23)
    eta3 = eta_cal(x1p, y1p, h1p, x3p_pred, y3p_pred, h3p_pred,
                    x2p, y2p, h2p, dr13, dr23, sigma13, sigma23)

    c = np.zeros(8 * Nc)
    for i in range(Nc):
        c[i] = v3[i] - V_MAX
        c[i + Nc] = V_MIN - v3[i]
        c[i + 2*Nc] = omega3[i] - OMEGA_MAX
        c[i + 3*Nc] = OMEGA_MIN - omega3[i]
        c[i + 4*Nc] = zeta3[i] - ZETA_MAX
        c[i + 5*Nc] = ZETA_MIN - zeta3[i]

        track_err = np.sqrt(
            (x3p_pred[i] - x3[i])**2 +
            (y3p_pred[i] - y3[i])**2 +
            (h3p_pred[i] - h3[i])**2
        )
        c[i + 6*Nc] = track_err - min(sigma3, eta3)

        dist_13 = np.sqrt((x1p[i]-x3[i])**2 + (y1p[i]-y3[i])**2 + (h1p[i]-h3[i])**2)
        c[i + 7*Nc] = 2 * R + sigma13 - dist_13

    return c, ceq


def constraint_uav4(
    h: np.ndarray,
    x4now: float, y4now: float, h4now: float,
    thelt4now: float, phi4now: float,
    dr14: np.ndarray, dr34: np.ndarray,
    Nc: int, Np: int,
    x1p: np.ndarray, y1p: np.ndarray, h1p: np.ndarray,
    thelt1p: np.ndarray, phi1p: np.ndarray,
    x4p_pred: np.ndarray, y4p_pred: np.ndarray, h4p_pred: np.ndarray,
    x3p: np.ndarray, y3p: np.ndarray, h3p: np.ndarray,
    x2p: np.ndarray, y2p: np.ndarray, h2p: np.ndarray,
    R: float = R_SAFE,
    k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """UAV4 (Follower) constraints."""
    v4 = h[0:Nc]
    omega4 = h[Nc:Nc+Nc]
    zeta4 = h[Nc+Nc:Nc+Nc+Np]

    x4 = h[Nc+Nc+Np:Nc+Nc+Np+Np]
    y4 = h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
    h4 = h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
    thelt4 = h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
    phi4 = h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

    ceq = np.zeros(5 * Np)
    for i in range(Np):
        if i == 0:
            ceq[i] = x4[i] - (x4now + v4[i] * np.cos(thelt4now))
            ceq[i + Np] = y4[i] - (y4now + v4[i] * np.sin(thelt4now))
            ceq[i + 2*Np] = h4[i] - (h4now + v4[i] * np.sin(phi4now))
            ceq[i + 3*Np] = thelt4[i] - (thelt4now + omega4[i])
            ceq[i + 4*Np] = phi4[i] - (phi4now + zeta4[i])
        else:
            ceq[i] = x4[i] - (x4[i-1] + v4[i] * np.cos(thelt4[i-1]))
            ceq[i + Np] = y4[i] - (y4[i-1] + v4[i] * np.sin(thelt4[i-1]))
            ceq[i + 2*Np] = h4[i] - (h4[i-1] + v4[i] * np.sin(phi4[i-1]))
            ceq[i + 3*Np] = thelt4[i] - (thelt4[i-1] + omega4[i])
            ceq[i + 4*Np] = phi4[i] - (phi4[i-1] + zeta4[i])

    from quadrotor_dmpc.models.formation import sigma_cal, eta_cal

    sigma14 = sigma_cal(x1p, y1p, h1p, x4p_pred, y4p_pred, h4p_pred, R)
    sigma43 = sigma_cal(x4p_pred, y4p_pred, h4p_pred, x3p, y3p, h3p, R)
    sigma24 = sigma_cal(x2p, y2p, h2p, x4p_pred, y4p_pred, h4p_pred, R)
    sigma4 = min(sigma14, sigma43)
    eta4 = eta_cal(x1p, y1p, h1p, x4p_pred, y4p_pred, h4p_pred,
                    x3p, y3p, h3p, dr14, dr34, sigma14, sigma43)

    c = np.zeros(8 * Nc)
    for i in range(Nc):
        c[i] = v4[i] - V_MAX
        c[i + Nc] = V_MIN - v4[i]
        c[i + 2*Nc] = omega4[i] - OMEGA_MAX
        c[i + 3*Nc] = OMEGA_MIN - omega4[i]
        c[i + 4*Nc] = zeta4[i] - ZETA_MAX
        c[i + 5*Nc] = ZETA_MIN - zeta4[i]

        track_err = np.sqrt(
            (x4p_pred[i] - x4[i])**2 +
            (y4p_pred[i] - y4[i])**2 +
            (h4p_pred[i] - h4[i])**2
        )
        c[i + 6*Nc] = track_err - min(sigma4, eta4)

        dist_14 = np.sqrt((x1p[i]-x4[i])**2 + (y1p[i]-y4[i])**2 + (h1p[i]-h4[i])**2)
        c[i + 7*Nc] = 2 * R + sigma14 - dist_14

    return c, ceq
