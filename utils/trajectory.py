"""
Reference trajectory generation for UAV leader.
"""

import numpy as np
from typing import Tuple


def generate_sinusoidal_3d(
    t: np.ndarray,
    x0: float = 0.0, y0: float = 0.0, h0: float = 0.0,
    Ax: float = 50.0, Ay: float = 30.0, Ah: float = 10.0,
    wx: float = 0.1, wy: float = 0.15, wh: float = 0.05,
    theta0: float = 0.0, phi0: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D sinusoidal reference trajectory.

    x(t) = x0 + Ax * sin(wx * t)
    y(t) = y0 + Ay * sin(wy * t)
    h(t) = h0 + Ah * sin(wh * t)

    Attitude angles are derived from the velocity direction to match the dynamics model:
      dx/dt = v*cos(theta)*cos(phi),  dy/dt = v*sin(theta)*cos(phi),  dh/dt = v*sin(phi)

    Returns:
        xr, yr, hr, thelr, phir: position and attitude references
    """
    xr = x0 + Ax * np.sin(wx * t)
    yr = y0 + Ay * np.sin(wy * t)
    hr = h0 + Ah * np.sin(wh * t)

    # Analytical derivatives of position
    dx_dt = Ax * wx * np.cos(wx * t)   # m/s
    dy_dt = Ay * wy * np.cos(wy * t)   # m/s
    dh_dt = Ah * wh * np.cos(wh * t)   # m/s

    # Horizontal speed (without vertical component)
    v_horizontal = np.sqrt(dx_dt**2 + dy_dt**2)

    # Yaw angle: theta = atan2(dy/dt, dx/dt) — direction of horizontal motion
    # Guard against divide-by-zero when stationary
    thelr = np.arctan2(
        np.where(v_horizontal > 1e-9, dy_dt, 0.0),
        np.where(v_horizontal > 1e-9, dx_dt, 0.0)
    )

    # Pitch angle: phi = atan2(dh/dt, v_horizontal)
    # This ensures dh/dt = v*sin(phi) and dx/dt = v*cos(theta)*cos(phi)
    phi_raw = np.arctan2(dh_dt, v_horizontal)
    # Clamp to physically plausible range [-45°, 45°]
    phi_max = 45.0 * np.pi / 180.0
    phir = np.clip(phi_raw, -phi_max, phi_max)

    return xr, yr, hr, thelr, phir


def generate_circular_3d(
    t: np.ndarray,
    x0: float = 0.0, y0: float = 0.0, h0: float = 10.0,
    R: float = 50.0, w: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D circular reference trajectory.

    x(t) = x0 + R * cos(w * t)
    y(t) = y0 + R * sin(w * t)
    h(t) = h0 (constant)

    Attitude angles derived from velocity direction (consistent with dynamics):
      dx/dt = -R*w*sin(w*t),  dy/dt = R*w*cos(w*t),  dh/dt = 0
      theta = atan2(dy/dt, dx/dt) = atan2(cos(w*t), -sin(w*t)) = w*t + pi/2
    """
    xr = x0 + R * np.cos(w * t)
    yr = y0 + R * np.sin(w * t)
    hr = np.zeros_like(t) + h0

    # Analytical velocity derivatives
    dx_dt = -R * w * np.sin(w * t)   # m/s
    dy_dt =  R * w * np.cos(w * t)   # m/s

    # Yaw angle: theta = atan2(dy/dt, dx/dt) — direction of motion
    v_horizontal = np.sqrt(dx_dt**2 + dy_dt**2)
    thelr = np.arctan2(
        np.where(v_horizontal > 1e-9, dy_dt, 0.0),
        np.where(v_horizontal > 1e-9, dx_dt, 0.0)
    )
    phir = np.zeros_like(t)  # level flight

    return xr, yr, hr, thelr, phir


def generate_trace_ref(
    trace_xr: np.ndarray, trace_yr: np.ndarray,
    trace_hr: np.ndarray, trace_theltr: np.ndarray, trace_phir: np.ndarray,
    Np: int, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract prediction horizon reference from pre-computed trajectory.

    Args:
        trace_xr, ...: Full reference trajectory arrays
        Np: Prediction horizon length
        k: Current time step

    Returns:
        xref, yref, href, theltref, phiref: Reference for prediction horizon
    """
    if k < len(trace_xr) - Np:
        xref = trace_xr[k:k+Np]
        yref = trace_yr[k:k+Np]
        href = trace_hr[k:k+Np]
        theltref = trace_theltr[k:k+Np]
        phiref = trace_phir[k:k+Np]
    else:
        # Pad with last value if near end
        remaining = len(trace_xr) - k
        xref = np.zeros(Np)
        yref = np.zeros(Np)
        href = np.zeros(Np)
        theltref = np.zeros(Np)
        phiref = np.zeros(Np)

        xref[:remaining] = trace_xr[k:]
        yref[:remaining] = trace_yr[k:]
        href[:remaining] = trace_hr[k:]
        theltref[:remaining] = trace_theltr[k:]
        phiref[:remaining] = trace_phir[k:]

        # Pad remainder with last known value
        xref[remaining:] = trace_xr[-1]
        yref[remaining:] = trace_yr[-1]
        href[remaining:] = trace_hr[-1]
        theltref[remaining:] = trace_theltr[-1]
        phiref[remaining:] = trace_phir[-1]

    return xref, yref, href, theltref, phiref


def generate_reference_velocities(
    xr: np.ndarray, yr: np.ndarray, hr: np.ndarray,
    dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reference velocities from position trajectory (finite differences).

    Returns:
        vr: linear velocity magnitude
        wr: yaw rate
        zetar: pitch rate
    """
    # Position differences
    dx = np.diff(xr) / dt
    dy = np.diff(yr) / dt
    dh = np.diff(hr) / dt

    # Velocity magnitude
    vr = np.sqrt(dx**2 + dy**2 + dh**2)

    # Yaw rate (approximate from heading change)
    dtheta = np.diff(np.arctan2(dy, dx))
    wr = dtheta / dt

    # Pitch rate (approximate from altitude change)
    zetar = np.diff(np.arcsin(np.clip(dh / (vr + 1e-6), -1, 1))) / dt

    # Pad last values
    vr = np.append(vr, vr[-1])
    wr = np.append(wr, wr[-1])
    zetar = np.append(zetar, zetar[-1])

    return vr, wr, zetar
