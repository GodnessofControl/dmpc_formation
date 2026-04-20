"""
Cost functions for DMPC optimization - mirrors MATLAB test_cost*.m

Each cost function computes the MPC objective:
    J = (X - X_ref)' P (X - X_ref) + U' R U + formation_terms
"""

import numpy as np


def cost_uav1(
    h: np.ndarray,
    Nc: int, Np: int,
    xref: np.ndarray, yref: np.ndarray,
    href: np.ndarray, theltref: np.ndarray, phiref: np.ndarray,
    dr01: np.ndarray,
    p1rtilde: np.ndarray, thelt1rtilde: float, phi1rtilde: float,
    vr1: np.ndarray, wr1: np.ndarray, zetar1: np.ndarray
) -> float:
    """
    UAV1 (Leader) cost function.

    Decision vector h layout:
        h[0:Nc]           -> v1 (velocity)
        h[Nc:Nc+Nc]       -> omega1 (yaw rate)
        h[Nc+Nc:Nc+Nc+Np] -> zeta1 (pitch rate)
        h[Nc+Nc+Np:Nc+Nc+Np+Np]         -> x1
        h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]   -> y1
        h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np] -> h1
        h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np] -> thelt1
        h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np] -> phi1
    """
    v1 = h[0:Nc]
    omega1 = h[Nc:Nc+Nc]
    zeta1 = h[Nc+Nc:Nc+Nc+Np]

    x1 = h[Nc+Nc+Np:Nc+Nc+Np+Np]
    y1 = h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
    h1 = h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
    thelt1 = h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
    phi1 = h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

    # State weighting matrix — penalizes deviation from reference trajectory
    P = np.eye(5 * Np)

    # Control effort weighting — penalizes excessive control inputs
    # Note: do NOT penalize v1 - vr1 since vr1 (reference speed) conflicts with
    # the position reference xref at the initial step (x1now = xref[0] = 0 but vr1[0] != 0).
    R_omega = 0.1 * np.eye(Nc)
    R_zeta = 0.1 * np.eye(Nc)

    # Expanded formation vector (zero for leader)
    ddr01 = np.concatenate([
        np.full(Np, dr01[0]),
        np.full(Np, dr01[1]),
        np.full(Np, dr01[2]),
        np.full(Np, dr01[3]),
        np.full(Np, dr01[4]),
    ])

    # State error from reference (formation-relative)
    # Note: theta and phi are implicitly determined by yaw/pitch RATES.
    # Penalizing theta - theltref is wrong because theltref is a reference ANGLE
    # not a control. Instead, penalize the RATE error (omega - wr1, zeta - zetar1).
    state_err_pos = np.concatenate([x1 - xref, y1 - yref, h1 - href]) - ddr01[:3*Np]
    # Omit theta/phi from state_err since they are not directly controllable.
    # They emerge from omega/zeta rates and should not be penalized against a reference angle.

    # Control effort: penalize all controls, especially rate deviations from reference
    omega_err = omega1 - wr1
    zeta_err = zeta1 - zetar1
    omega_cost = np.sum(omega_err @ R_omega @ omega_err)
    zeta_cost = np.sum(zeta_err @ R_zeta @ zeta_err)
    v_cost = np.sum(v1 @ (0.01 * np.eye(Nc)) @ v1)  # small v regularization

    J = state_err_pos @ state_err_pos + omega_cost + zeta_cost + v_cost

    return float(J)


def cost_uav2(
    h: np.ndarray,
    Nc: int, Np: int,
    dr12: np.ndarray,
    x1pby1: np.ndarray, y1pby1: np.ndarray, h1pby1: np.ndarray,
    thelt1pby1: np.ndarray, phi1pby1: np.ndarray,
    x3p: np.ndarray, y3p: np.ndarray, h3p: np.ndarray,
    dr23: np.ndarray,
    x4p: np.ndarray, y4p: np.ndarray, h4p: np.ndarray,
    dr24: np.ndarray,
    p21tilde: np.ndarray, thelt21tilde: float, phi21tilde: float,
    vr1: np.ndarray, wr1: np.ndarray, zetar1: np.ndarray,
    k: int
) -> float:
    """
    UAV2 (Follower) cost function.

    Tracks leader UAV1 and maintains formation with UAV3, UAV4.
    """
    v2 = h[0:Nc]
    omega2 = h[Nc:Nc+Nc]
    zeta2 = h[Nc+Nc:Nc+Nc+Np]

    x2 = h[Nc+Nc+Np:Nc+Nc+Np+Np]
    y2 = h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
    h2 = h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
    thelt2 = h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
    phi2 = h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

    # Weighting matrices
    Q = np.eye(5 * Np) * 500
    R = np.eye(3 * Np) * 200
    R2 = np.eye(3 * Np) * 10

    # Formation vectors expanded
    ddr12 = np.concatenate([
        np.full(Np, dr12[0]),
        np.full(Np, dr12[1]),
        np.full(Np, dr12[2]),
        np.full(Np, dr12[3]),
        np.full(Np, dr12[4]),
    ])
    ddr23 = np.concatenate([np.full(Np, dr23[0]), np.full(Np, dr23[1]), np.full(Np, dr23[2])])
    ddr24 = np.concatenate([np.full(Np, dr24[0]), np.full(Np, dr24[1]), np.full(Np, dr24[2])])

    # Formation error from leader
    form_err_12 = np.concatenate([
        x1pby1 - x2,
        y1pby1 - y2,
        h1pby1 - h2,
        thelt1pby1 - thelt2,
        phi1pby1 - phi2,
    ]) - ddr12

    # Formation error from UAV3
    form_err_23 = np.concatenate([x2 - x3p, y2 - y3p, h2 - h3p]) - ddr23

    # Formation error from UAV4
    form_err_24 = np.concatenate([x2 - x4p, y2 - y4p, h2 - h4p]) - ddr24

    # Control deviation from reference
    ctrl_err = np.concatenate([v2 - vr1, omega2 - wr1, zeta2 - zetar1])

    # Terminal error penalty
    terminal_penalty = 34 * 0.5 * (
        p21tilde @ p21tilde +
        thelt21tilde**2 +
        phi21tilde**2
    )

    J = (
        form_err_12 @ Q @ form_err_12 +
        ctrl_err @ R2 @ ctrl_err +
        form_err_24 @ R @ form_err_24 +
        form_err_23 @ R @ form_err_23 +
        terminal_penalty
    )

    return float(J)


def cost_uav3(
    h: np.ndarray,
    Nc: int, Np: int,
    dr13: np.ndarray,
    x1pby1: np.ndarray, y1pby1: np.ndarray, h1pby1: np.ndarray,
    thelt1pby1: np.ndarray, phi1pby1: np.ndarray,
    x2p: np.ndarray, y2p: np.ndarray, h2p: np.ndarray,
    dr23: np.ndarray,
    x4p: np.ndarray, y4p: np.ndarray, h4p: np.ndarray,
    dr34: np.ndarray,
    p31tilde: np.ndarray, thelt31tilde: float, phi31tilde: float,
    vr1: np.ndarray, wr1: np.ndarray, zetar1: np.ndarray,
    k: int
) -> float:
    """UAV3 (Follower) cost function - similar structure to UAV2."""
    v3 = h[0:Nc]
    omega3 = h[Nc:Nc+Nc]
    zeta3 = h[Nc+Nc:Nc+Nc+Np]

    x3 = h[Nc+Nc+Np:Nc+Nc+Np+Np]
    y3 = h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
    h3 = h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
    thelt3 = h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
    phi3 = h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

    Q = np.eye(5 * Np) * 500
    R = np.eye(3 * Np) * 200
    R2 = np.eye(3 * Np) * 10

    ddr13 = np.concatenate([
        np.full(Np, dr13[0]), np.full(Np, dr13[1]), np.full(Np, dr13[2]),
        np.full(Np, dr13[3]), np.full(Np, dr13[4]),
    ])
    ddr23 = np.concatenate([np.full(Np, dr23[0]), np.full(Np, dr23[1]), np.full(Np, dr23[2])])
    ddr34 = np.concatenate([np.full(Np, dr34[0]), np.full(Np, dr34[1]), np.full(Np, dr34[2])])

    form_err_13 = np.concatenate([
        x1pby1 - x3, y1pby1 - y3, h1pby1 - h3,
        thelt1pby1 - thelt3, phi1pby1 - phi3,
    ]) - ddr13

    form_err_23 = np.concatenate([x3 - x2p, y3 - y2p, h3 - h2p]) - ddr23
    form_err_34 = np.concatenate([x3 - x4p, y3 - y4p, h3 - h4p]) - ddr34

    ctrl_err = np.concatenate([v3 - vr1, omega3 - wr1, zeta3 - zetar1])

    terminal_penalty = 34 * 0.5 * (
        p31tilde @ p31tilde + thelt31tilde**2 + phi31tilde**2
    )

    J = (
        form_err_13 @ Q @ form_err_13 +
        ctrl_err @ R2 @ ctrl_err +
        form_err_34 @ R @ form_err_34 +
        form_err_23 @ R @ form_err_23 +
        terminal_penalty
    )

    return float(J)


def cost_uav4(
    h: np.ndarray,
    Nc: int, Np: int,
    dr14: np.ndarray,
    x1pby1: np.ndarray, y1pby1: np.ndarray, h1pby1: np.ndarray,
    thelt1pby1: np.ndarray, phi1pby1: np.ndarray,
    x3p: np.ndarray, y3p: np.ndarray, h3p: np.ndarray,
    dr34: np.ndarray,
    x2p: np.ndarray, y2p: np.ndarray, h2p: np.ndarray,
    dr24: np.ndarray,
    p41tilde: np.ndarray, thelt41tilde: float, phi41tilde: float,
    vr1: np.ndarray, wr1: np.ndarray, zetar1: np.ndarray,
    k: int
) -> float:
    """UAV4 (Follower) cost function - similar structure to UAV2/3."""
    v4 = h[0:Nc]
    omega4 = h[Nc:Nc+Nc]
    zeta4 = h[Nc+Nc:Nc+Nc+Np]

    x4 = h[Nc+Nc+Np:Nc+Nc+Np+Np]
    y4 = h[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
    h4 = h[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
    thelt4 = h[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
    phi4 = h[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

    Q = np.eye(5 * Np) * 500
    R = np.eye(3 * Np) * 200
    R2 = np.eye(3 * Np) * 10

    ddr14 = np.concatenate([
        np.full(Np, dr14[0]), np.full(Np, dr14[1]), np.full(Np, dr14[2]),
        np.full(Np, dr14[3]), np.full(Np, dr14[4]),
    ])
    ddr34 = np.concatenate([np.full(Np, dr34[0]), np.full(Np, dr34[1]), np.full(Np, dr34[2])])
    ddr24 = np.concatenate([np.full(Np, dr24[0]), np.full(Np, dr24[1]), np.full(Np, dr24[2])])

    form_err_14 = np.concatenate([
        x1pby1 - x4, y1pby1 - y4, h1pby1 - h4,
        thelt1pby1 - thelt4, phi1pby1 - phi4,
    ]) - ddr14

    form_err_34 = np.concatenate([x4 - x3p, y4 - y3p, h4 - h3p]) - ddr34
    form_err_24 = np.concatenate([x4 - x2p, y4 - y2p, h4 - h2p]) - ddr24

    ctrl_err = np.concatenate([v4 - vr1, omega4 - wr1, zeta4 - zetar1])

    terminal_penalty = 34 * 0.5 * (
        p41tilde @ p41tilde + thelt41tilde**2 + phi41tilde**2
    )

    J = (
        form_err_14 @ Q @ form_err_14 +
        ctrl_err @ R2 @ ctrl_err +
        form_err_34 @ R @ form_err_34 +
        form_err_24 @ R @ form_err_24 +
        terminal_penalty
    )

    return float(J)
