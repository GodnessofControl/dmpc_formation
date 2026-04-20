"""
Terminal state calculation with delay and dropout compensation.
Mirrors MATLAB terminalCal_DL.m

Computes terminal state using auxiliary feedback control law
accounting for communication delays and data dropout.
"""

import numpy as np
from typing import Tuple
from quadrotor_dmpc.models.uav_model import UAVModel
from quadrotor_dmpc.lmi.gain_compute import LMISolver, DEFAULT_K1


def terminal_cal_dl(
    last_x: np.ndarray,  # (Np,) array of predicted x
    last_y: np.ndarray,
    last_h: np.ndarray,
    last_thelt: np.ndarray,
    last_phi: np.ndarray,
    xref: float, yref: float, href: float,
    theltref: float, phiref: float,
    d: np.ndarray,  # formation vector [dx, dy, dh, dtheta, dphi]
    vr: float, wr: float, zetar: float,
    count_D: int, isdelay: int,
    count_L: int, isdropout: int,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute terminal state prediction accounting for delays and dropouts.

    This function implements the terminal auxiliary feedback control law:
        Kx = K * [prtilde; theltrtilde; phirtilde]
        vk = vr + Kx(1)
        wk = wr - Kx(2)
        zetak = zetar - Kx(3)

    Args:
        last_x ... last_phi: Last predicted state trajectories (length Np)
        xref, ... phiref: Reference values at terminal time
        d: Formation offset vector
        vr, wr, zetar: Reference control inputs
        count_D: Consecutive delay counter
        isdelay: Delay flag (0 or 1)
        count_L: Consecutive dropout counter
        isdropout: Dropout flag (0 or 1)
        K: Terminal feedback gain matrix (3, 5)

    Returns:
        state_prediction: (Np+1, 5) predicted state trajectory
        state_tilde: (5,) terminal error state
    """
    Np = len(last_x)
    model = UAVModel()

    # Shift arrays (drop oldest, append placeholder)
    new_x = np.concatenate([last_x[1:], np.array([0.0])])
    new_y = np.concatenate([last_y[1:], np.array([0.0])])
    new_h = np.concatenate([last_h[1:], np.array([0.0])])
    new_thelt = np.concatenate([last_thelt[1:], np.array([0.0])])
    new_phi = np.concatenate([last_phi[1:], np.array([0.0])])

    # Formation offsets
    xd, yd, hd = d[0], d[1], d[2]

    # Select index based on dropout/delay conditions
    if isdropout == 0 and isdelay == 0:
        idx = Np - 1  # Last element (index 7 for Np=8)
    elif isdropout == 1 and isdelay == 0:
        idx = Np - 1 - count_L
    elif isdropout == 0 and isdelay == 1:
        idx = Np - 1 - count_D
    else:  # isdropout == 1 and isdelay == 1
        idx = Np - 1 - count_D - count_L

    idx = max(0, min(idx, Np - 1))  # Clamp to valid range

    # Compute position error in body frame
    cos_theta = np.cos(new_thelt[idx])
    sin_theta = np.sin(new_thelt[idx])
    cos_phi = np.cos(new_phi[idx])
    sin_phi = np.sin(new_phi[idx])

    R = np.array([
        [cos_theta * cos_phi,  cos_phi * sin_theta, -sin_phi],
        [-sin_theta,            cos_theta,            0],
        [cos_theta * sin_phi,  sin_theta * sin_phi,  cos_phi],
    ])

    # Position error with formation offset
    pos_err = np.array([
        new_x[idx] - xref + xd,
        new_y[idx] - yref + yd,
        new_h[idx] - href + hd,
    ])

    prtilde = R @ pos_err
    xrtilde, yrtilde, hrtilde = prtilde[0], prtilde[1], prtilde[2]
    theltrtilde = new_thelt[idx] - theltref
    phirtilde = new_phi[idx] - phiref

    # Apply terminal feedback control
    state_tilde = np.array([xrtilde, yrtilde, hrtilde, theltrtilde, phirtilde])
    Kx = K @ state_tilde

    vk = vr + Kx[0]
    wk = wr - Kx[1]
    zetak = zetar - Kx[2]

    # Build prediction using selected index
    if isdropout == 0 and isdelay == 0:
        # Normal case: predict from last state
        base_idx = Np - 1
        new_x[base_idx] = new_x[base_idx - 1] + vk * np.cos(new_thelt[base_idx - 1]) * np.cos(new_phi[base_idx - 1])
        new_y[base_idx] = new_y[base_idx - 1] + vk * np.sin(new_thelt[base_idx - 1]) * np.cos(new_phi[base_idx - 1])
        new_h[base_idx] = new_h[base_idx - 1] + vk * np.sin(new_phi[base_idx - 1])
        new_thelt[base_idx] = new_thelt[base_idx - 1] + wk
        new_phi[base_idx] = new_phi[base_idx - 1] + zetak

    elif isdropout == 1 and isdelay == 0:
        # Dropout only: iterate missing steps
        count = count_L
        while count >= 0:
            idx_i = Np - 1 - count
            if idx_i > 0:
                new_x[idx_i] = new_x[idx_i - 1] + vk * np.cos(new_thelt[idx_i - 1]) * np.cos(new_phi[idx_i - 1])
                new_y[idx_i] = new_y[idx_i - 1] + vk * np.sin(new_thelt[idx_i - 1]) * np.cos(new_phi[idx_i - 1])
                new_h[idx_i] = new_h[idx_i - 1] + vk * np.sin(new_phi[idx_i - 1])
                new_thelt[idx_i] = new_thelt[idx_i - 1] + wk
                new_phi[idx_i] = new_phi[idx_i - 1] + zetak
            count -= 1

    elif isdropout == 0 and isdelay == 1:
        # Delay only
        count = count_D
        while count >= 0:
            idx_i = Np - 1 - count
            if idx_i > 0:
                new_x[idx_i] = new_x[idx_i - 1] + vk * np.cos(new_thelt[idx_i - 1]) * np.cos(new_phi[idx_i - 1])
                new_y[idx_i] = new_y[idx_i - 1] + vk * np.sin(new_thelt[idx_i - 1]) * np.cos(new_phi[idx_i - 1])
                new_h[idx_i] = new_h[idx_i - 1] + vk * np.sin(new_phi[idx_i - 1])
                new_thelt[idx_i] = new_thelt[idx_i - 1] + wk
                new_phi[idx_i] = new_phi[idx_i - 1] + zetak
            count -= 1

    else:  # Both dropout and delay
        count_DL = count_L + count_D
        while count_DL >= 0:
            idx_i = Np - 1 - count_DL
            if idx_i > 0 and idx_i < Np:
                new_x[idx_i] = new_x[idx_i - 1] + vk * np.cos(new_thelt[idx_i - 1]) * np.cos(new_phi[idx_i - 1])
                new_y[idx_i] = new_y[idx_i - 1] + vk * np.sin(new_thelt[idx_i - 1]) * np.cos(new_phi[idx_i - 1])
                new_h[idx_i] = new_h[idx_i - 1] + vk * np.sin(new_phi[idx_i - 1])
                new_thelt[idx_i] = new_thelt[idx_i - 1] + wk
                new_phi[idx_i] = new_phi[idx_i - 1] + zetak
            count_DL -= 1

    state_prediction = np.column_stack([new_x, new_y, new_h, new_thelt, new_phi])

    return state_prediction, state_tilde
