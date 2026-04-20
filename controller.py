"""
Main DMPC Controller - Ties all components together.
Mirrors the main simulation loop in MATLAB multi_delay_loss.m / OnlineDoubleRandomDL.m

This is the core Distributed Model Predictive Controller for 4 UAV formation
with random communication delays and data dropout.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from quadrotor_dmpc.models.uav_model import UAVState, UAVControl, UAVModel
from quadrotor_dmpc.models.formation import FormationConfig
from quadrotor_dmpc.utils.trajectory import generate_trace_ref, generate_reference_velocities
from quadrotor_dmpc.utils.network import NetworkSimulator
from quadrotor_dmpc.lmi.gain_compute import LMISolver, DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_K4
from quadrotor_dmpc.lmi.terminal import terminal_cal_dl
from quadrotor_dmpc.cost.functions import cost_uav1, cost_uav2, cost_uav3, cost_uav4
from quadrotor_dmpc.constraint.functions import (
    constraint_uav1, constraint_uav2, constraint_uav3, constraint_uav4,
    V_MAX, V_MIN, OMEGA_MAX, OMEGA_MIN, ZETA_MAX, ZETA_MIN
)


@dataclass
class MPCConfig:
    """MPC configuration parameters."""
    Np: int = 8        # Prediction horizon
    Nc: int = 8        # Control horizon (same as Np in this formulation)
    N_steps: int = 64  # Total simulation steps

    # Control limits
    v_max: float = V_MAX
    v_min: float = V_MIN
    omega_max: float = OMEGA_MAX
    omega_min: float = OMEGA_MIN
    zeta_max: float = ZETA_MAX
    zeta_min: float = ZETA_MIN

    # Formation
    collision_radius: float = 2.0  # R_safe

    # Network
    sigma_L: float = 0.15   # Dropout probability
    sigma_D: float = 0.20   # Delay probability

    # Optimizer
    max_iter: int = 2000
    max_fun_evals: int = 50000


@dataclass
class UAVAgent:
    """State for a single UAV agent."""
    id: int
    state: UAVState
    predicted_trajectory: np.ndarray  # (Np, 5) - current prediction
    control_history: List[UAVControl] = field(default_factory=list)
    optimal_sequence: Optional[np.ndarray] = None  # Full optimizer solution
    terminal_error: np.ndarray = field(default_factory=lambda: np.zeros(5))
    K_gain: np.ndarray = field(default_factory=lambda: DEFAULT_K1.copy())


class QuadrotorDMPC:
    """
    Distributed Model Predictive Controller for 4-UAV Formation.

    Handles:
    - Leader trajectory tracking
    - Follower formation maintenance
    - Random communication delays and data dropout
    - Collision avoidance
    - ROS/Gazebo integration interface
    """

    def __init__(
        self,
        config: MPCConfig,
        formation: FormationConfig,
        initial_states: Dict[int, UAVState],
        reference_trajectory: Dict[str, np.ndarray],
        network_sim: Optional[NetworkSimulator] = None,
        use_lmi_gains: bool = False
    ):
        self.config = config
        self.formation = formation
        self.model = UAVModel()
        self.lmi_solver = LMISolver()
        self.use_lmi_gains = use_lmi_gains

        # Initialize UAV agents
        self.uavs: Dict[int, UAVAgent] = {}
        for uav_id, state in initial_states.items():
            self.uavs[uav_id] = UAVAgent(
                id=uav_id,
                state=state,
                predicted_trajectory=self._init_prediction(state, config.Np)
            )

        # Reference trajectory
        self.trace_xr = reference_trajectory['xr']
        self.trace_yr = reference_trajectory['yr']
        self.trace_hr = reference_trajectory['hr']
        self.trace_theltr = reference_trajectory['theltr']
        self.trace_phir = reference_trajectory['phir']

        # Reference velocities
        self.vrn, self.wrn, self.zetarn = generate_reference_velocities(
            self.trace_xr, self.trace_yr, self.trace_hr
        )

        # Network simulator
        if network_sim is None:
            network_sim = NetworkSimulator.create_random(
                config.N_steps, config.sigma_L, config.sigma_D
            )
        self.network = network_sim

        # Storage for results
        self.history: Dict[int, Dict[str, list]] = {
            uav_id: {'x': [], 'y': [], 'h': [], 'theta': [], 'phi': [],
                     'v': [], 'omega': [], 'zeta': []}
            for uav_id in initial_states.keys()
        }

        # Gain matrices for each UAV
        self.K_gain = {
            1: DEFAULT_K1.copy(),
            2: DEFAULT_K2.copy(),
            3: DEFAULT_K3.copy(),
            4: DEFAULT_K4.copy(),
        }

        # Optimization initial guesses
        self._init_optimization()

    def _init_prediction(self, state: UAVState, Np: int) -> np.ndarray:
        """Initialize prediction trajectory."""
        traj = np.zeros((Np, 5))
        traj[:, 0] = state.x
        traj[:, 1] = state.y
        traj[:, 2] = state.h
        traj[:, 3] = state.theta
        traj[:, 4] = state.phi
        return traj

    def _init_optimization(self):
        """Initialize optimization variables (h0 for each UAV)."""
        Nc, Np = self.config.Nc, self.config.Np

        # Initial control sequences (use reference velocities)
        v_init = self.vrn[:Nc] if len(self.vrn) >= Nc else np.ones(Nc) * 5.0
        w_init = self.wrn[:Nc] if len(self.wrn) >= Nc else np.zeros(Nc)
        zeta_init = self.zetarn[:Nc] if len(self.zetarn) >= Nc else np.zeros(Nc)

        # Build initial predicted states consistent with dynamics:
        # Start from the leader's current state, roll out using reference velocities.
        # This avoids severe initial constraint violation that SLSQP struggles with.
        s1_init = self.uavs[1].state
        x1_init = np.zeros(Np)
        y1_init = np.zeros(Np)
        h1_init = np.zeros(Np)
        theta1_init = np.zeros(Np)
        phi1_init = np.zeros(Np)

        x1_init[0] = s1_init.x + v_init[0] * np.cos(s1_init.theta) * np.cos(s1_init.phi)
        y1_init[0] = s1_init.y + v_init[0] * np.sin(s1_init.theta) * np.cos(s1_init.phi)
        h1_init[0] = s1_init.h + v_init[0] * np.sin(s1_init.phi)
        theta1_init[0] = s1_init.theta + w_init[0]
        phi1_init[0] = s1_init.phi + zeta_init[0]

        for i in range(1, Np):
            x1_init[i] = x1_init[i-1] + v_init[i] * np.cos(theta1_init[i-1]) * np.cos(phi1_init[i-1])
            y1_init[i] = y1_init[i-1] + v_init[i] * np.sin(theta1_init[i-1]) * np.cos(phi1_init[i-1])
            h1_init[i] = h1_init[i-1] + v_init[i] * np.sin(phi1_init[i-1])
            theta1_init[i] = theta1_init[i-1] + w_init[i]
            phi1_init[i] = phi1_init[i-1] + zeta_init[i]

        self.h01 = np.concatenate([v_init, w_init, zeta_init, x1_init, y1_init, h1_init, theta1_init, phi1_init])

        # For followers, initial states are formation-offset from leader
        # Roll out leader trajectory first
        leader_traj = np.stack([x1_init, y1_init, h1_init, theta1_init, phi1_init], axis=1)  # (Np, 5)

        # Follower 2: relative to leader by dr12
        x2_init = x1_init - np.full(Np, self.formation.dr12[0])
        y2_init = y1_init - np.full(Np, self.formation.dr12[1])
        h2_init = h1_init - np.full(Np, self.formation.dr12[2])
        th2_init = theta1_init - np.full(Np, self.formation.dr12[3])
        ph2_init = phi1_init - np.full(Np, self.formation.dr12[4])

        # Follower 3: relative to leader by dr13
        x3_init = x1_init - np.full(Np, self.formation.dr13[0])
        y3_init = y1_init - np.full(Np, self.formation.dr13[1])
        h3_init = h1_init - np.full(Np, self.formation.dr13[2])
        th3_init = theta1_init - np.full(Np, self.formation.dr13[3])
        ph3_init = phi1_init - np.full(Np, self.formation.dr13[4])

        # Follower 4: relative to leader by dr14
        x4_init = x1_init - np.full(Np, self.formation.dr14[0])
        y4_init = y1_init - np.full(Np, self.formation.dr14[1])
        h4_init = h1_init - np.full(Np, self.formation.dr14[2])
        th4_init = theta1_init - np.full(Np, self.formation.dr14[3])
        ph4_init = phi1_init - np.full(Np, self.formation.dr14[4])

        self.h02 = np.concatenate([v_init, w_init, zeta_init, x2_init, y2_init, h2_init, th2_init, ph2_init])
        self.h03 = np.concatenate([v_init, w_init, zeta_init, x3_init, y3_init, h3_init, th3_init, ph3_init])
        self.h04 = np.concatenate([v_init, w_init, zeta_init, x4_init, y4_init, h4_init, th4_init, ph4_init])

    def step(self, k: int) -> Dict[int, UAVControl]:
        """
        Execute one MPC step.

        Args:
            k: Current time step

        Returns:
            Dict of control inputs for each UAV
        """
        Np, Nc = self.config.Np, self.config.Nc

        # Update network state
        count_L, count_D, level = self.network.step(k)
        isdropout = 1 if self.network.isdropout[k] else 0
        isdelay = 1 if self.network.isdelay[k] else 0

        # Get reference trajectory for prediction horizon
        xref, yref, href, theltref, phiref = generate_trace_ref(
            self.trace_xr, self.trace_yr, self.trace_hr,
            self.trace_theltr, self.trace_phir, Np, k
        )

        # Reference velocities for prediction horizon
        vr1 = self.vrn[k:k+Np] if k+Np <= len(self.vrn) else np.ones(Np) * self.vrn[-1]
        wr1 = self.wrn[k:k+Np] if k+Np <= len(self.wrn) else np.ones(Np) * self.wrn[-1]
        zetar1 = self.zetarn[k:k+Np] if k+Np <= len(self.zetarn) else np.zeros(Np)
        vr = self.vrn[k+Np] if k+Np < len(self.vrn) else self.vrn[-1]
        wr = self.wrn[k+Np] if k+Np < len(self.wrn) else self.wrn[-1]
        zetar = self.zetarn[k+Np] if k+Np < len(self.zetarn) else self.zetarn[-1]

        # Extract current UAV states
        s1 = self.uavs[1].state
        s2 = self.uavs[2].state
        s3 = self.uavs[3].state
        s4 = self.uavs[4].state

        # --- Leader UAV1 optimization ---
        uav1_opt = self._optimize_uav1(
            xref, yref, href, theltref, phiref,
            vr1, wr1, zetar1, k
        )

        # Extract leader's predicted trajectory
        x1pby1 = uav1_opt[Nc+Nc+Np:Nc+Nc+Np+Np]
        y1pby1 = uav1_opt[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
        h1pby1 = uav1_opt[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
        thelt1pby1 = uav1_opt[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
        phi1pby1 = uav1_opt[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]

        # Terminal state update for leader
        count_D_l, count_L_l = count_D if isdelay else 0, count_L if isdropout else 0
        pred1, tilde1 = terminal_cal_dl(
            x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
            self.trace_xr[k+Np], self.trace_yr[k+Np], self.trace_hr[k+Np],
            self.trace_theltr[k+Np], self.trace_phir[k+Np],
            self.formation.dr01, vr, wr, zetar,
            count_D_l, isdelay, count_L_l, isdropout,
            self.K_gain[1]
        )
        self.uavs[1].predicted_trajectory = pred1
        self.uavs[1].terminal_error = np.array([
            tilde1[0], tilde1[1], tilde1[2], tilde1[3], tilde1[4]
        ])

        # --- Follower UAV2-4 optimization (only when no delay, or with compensation) ---
        if k <= 8 or isdelay == 0:
            # Normal optimization
            uav2_opt, uav3_opt, uav4_opt = self._optimize_followers(
                x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                vr1, wr1, zetar1, k,
                count_D, count_L, isdelay, isdropout
            )
        else:
            # Delay compensation: use delayed predictions
            x1pby1_d = self.uavs[1].optimal_sequence[Nc+Nc+Np:Nc+Nc+Np+Np] if self.uavs[1].optimal_sequence is not None else x1pby1
            y1pby1_d = self.uavs[1].optimal_sequence[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np] if self.uavs[1].optimal_sequence is not None else y1pby1
            h1pby1_d = self.uavs[1].optimal_sequence[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np] if self.uavs[1].optimal_sequence is not None else h1pby1
            thelt1pby1_d = self.uavs[1].optimal_sequence[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np] if self.uavs[1].optimal_sequence is not None else thelt1pby1
            phi1pby1_d = self.uavs[1].optimal_sequence[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np] if self.uavs[1].optimal_sequence is not None else phi1pby1

            uav2_opt, uav3_opt, uav4_opt = self._optimize_followers(
                x1pby1_d, y1pby1_d, h1pby1_d, thelt1pby1_d, phi1pby1_d,
                vr1, wr1, zetar1, k,
                count_D, count_L, isdelay, isdropout
            )

        # Store optimal solutions
        self.uavs[1].optimal_sequence = uav1_opt
        self.uavs[2].optimal_sequence = uav2_opt
        self.uavs[3].optimal_sequence = uav3_opt
        self.uavs[4].optimal_sequence = uav4_opt

        # Extract control inputs (first element of each)
        v1, omega1, zeta1 = uav1_opt[0], uav1_opt[Nc], uav1_opt[Nc+Nc]
        v2, omega2, zeta2 = uav2_opt[0], uav2_opt[Nc], uav2_opt[Nc+Nc]
        v3, omega3, zeta3 = uav3_opt[0], uav3_opt[Nc], uav3_opt[Nc+Nc]
        v4, omega4, zeta4 = uav4_opt[0], uav4_opt[Nc], uav4_opt[Nc+Nc]

        controls = {
            1: UAVControl(v=v1, omega=omega1, zeta=zeta1),
            2: UAVControl(v=v2, omega=omega2, zeta=zeta2),
            3: UAVControl(v=v3, omega=omega3, zeta=zeta3),
            4: UAVControl(v=v4, omega=omega4, zeta=zeta4),
        }

        # Update LMI gains (optional)
        if self.use_lmi_gains and k > 0:
            self.K_gain[1] = self.lmi_solver.compute_gain(v1, omega1, zeta1, s1.phi, k)
            self.K_gain[2] = self.lmi_solver.compute_gain(v2, omega2, zeta2, s2.phi, k)
            self.K_gain[3] = self.lmi_solver.compute_gain(v3, omega3, zeta3, s3.phi, k)
            self.K_gain[4] = self.lmi_solver.compute_gain(v4, omega4, zeta4, s4.phi, k)

        # Apply control and update states
        for uav_id, control in controls.items():
            uav = self.uavs[uav_id]
            uav.state = self.model.step(uav.state, control)
            uav.control_history.append(control)

            # Record history
            self.history[uav_id]['x'].append(uav.state.x)
            self.history[uav_id]['y'].append(uav.state.y)
            self.history[uav_id]['h'].append(uav.state.h)
            self.history[uav_id]['theta'].append(uav.state.theta)
            self.history[uav_id]['phi'].append(uav.state.phi)
            self.history[uav_id]['v'].append(control.v)
            self.history[uav_id]['omega'].append(control.omega)
            self.history[uav_id]['zeta'].append(control.zeta)

        return controls

    def _optimize_uav1(
        self,
        xref: np.ndarray, yref: np.ndarray, href: np.ndarray,
        theltref: np.ndarray, phiref: np.ndarray,
        vr1: np.ndarray, wr1: np.ndarray, zetar1: np.ndarray,
        k: int
    ) -> np.ndarray:
        """
        Optimize UAV1 (Leader) - trajectory tracking.
        Uses fmincon-style optimization via scipy.
        """
        from scipy.optimize import minimize

        s1 = self.uavs[1].state
        p1rtilde = self.uavs[1].terminal_error[:3]
        thelt1rtilde = self.uavs[1].terminal_error[3]
        phi1rtilde = self.uavs[1].terminal_error[4]

        # Get neighbor predictions for collision avoidance
        x2p = self.uavs[2].predicted_trajectory[:, 0]
        y2p = self.uavs[2].predicted_trajectory[:, 1]
        h2p = self.uavs[2].predicted_trajectory[:, 2]
        x3p = self.uavs[3].predicted_trajectory[:, 0]
        y3p = self.uavs[3].predicted_trajectory[:, 1]
        h3p = self.uavs[3].predicted_trajectory[:, 2]

        def cost(h):
            return cost_uav1(
                h, self.config.Nc, self.config.Np,
                xref, yref, href, theltref, phiref,
                self.formation.dr01, p1rtilde, thelt1rtilde, phi1rtilde,
                vr1, wr1, zetar1
            )

        def constraint_ineq(h):
            c, _ = constraint_uav1(
                h,
                s1.x, s1.y, s1.h, s1.theta, s1.phi,
                self.config.Nc, self.config.Np,
                x2p, y2p, h2p, x3p, y3p, h3p,
                self.config.collision_radius
            )
            return c

        def constraint_eq(h):
            _, ceq = constraint_uav1(
                h,
                s1.x, s1.y, s1.h, s1.theta, s1.phi,
                self.config.Nc, self.config.Np,
                x2p, y2p, h2p, x3p, y3p, h3p,
                self.config.collision_radius
            )
            return ceq

        # Initial guess
        h0 = self.h01.copy()

        # Bounds for control inputs
        bounds = (
            [(self.config.v_min, self.config.v_max)] * self.config.Nc +
            [(self.config.omega_min, self.config.omega_max)] * self.config.Nc +
            [(self.config.zeta_min, self.config.zeta_max)] * self.config.Np +
            [(None, None)] * (5 * self.config.Np)  # State variables unconstrained
        )

        result = minimize(
            cost, h0, method='SLSQP', bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': constraint_ineq},
                {'type': 'eq', 'fun': constraint_eq}
            ],
            options={'maxiter': self.config.max_iter, 'disp': False}
        )

        self.h01 = result.x
        return result.x

    def _optimize_followers(
        self,
        x1pby1: np.ndarray, y1pby1: np.ndarray, h1pby1: np.ndarray,
        thelt1pby1: np.ndarray, phi1pby1: np.ndarray,
        vr1: np.ndarray, wr1: np.ndarray, zetar1: np.ndarray,
        k: int,
        count_D: int, count_L: int,
        isdelay: int, isdropout: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimize UAV2, UAV3, UAV4 (Followers)."""
        from scipy.optimize import minimize

        def optimize_uav2(h0, h_init_ref):
            s2 = self.uavs[2].state
            p21tilde = self.uavs[2].terminal_error[:3]
            thelt21tilde = self.uavs[2].terminal_error[3]
            phi21tilde = self.uavs[2].terminal_error[4]

            x2p = self.uavs[2].predicted_trajectory[:, 0]
            y2p = self.uavs[2].predicted_trajectory[:, 1]
            h2p = self.uavs[2].predicted_trajectory[:, 2]
            x3p = self.uavs[3].predicted_trajectory[:, 0]
            y3p = self.uavs[3].predicted_trajectory[:, 1]
            h3p = self.uavs[3].predicted_trajectory[:, 2]
            x4p = self.uavs[4].predicted_trajectory[:, 0]
            y4p = self.uavs[4].predicted_trajectory[:, 1]
            h4p = self.uavs[4].predicted_trajectory[:, 2]

            def cost(h):
                return cost_uav2(
                    h, self.config.Nc, self.config.Np,
                    self.formation.dr12,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x3p, y3p, h3p, self.formation.dr23,
                    x4p, y4p, h4p, self.formation.dr24,
                    p21tilde, thelt21tilde, phi21tilde,
                    vr1, wr1, zetar1, k
                )

            def constraint_ineq(h):
                c, _ = constraint_uav2(
                    h,
                    s2.x, s2.y, s2.h, s2.theta, s2.phi,
                    self.formation.dr12, self.formation.dr23, self.formation.dr24,
                    self.config.Nc, self.config.Np,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x2p, y2p, h2p, x3p, y3p, h3p, x4p, y4p, h4p,
                    self.config.collision_radius, k
                )
                return c

            def constraint_eq(h):
                _, ceq = constraint_uav2(
                    h,
                    s2.x, s2.y, s2.h, s2.theta, s2.phi,
                    self.formation.dr12, self.formation.dr23, self.formation.dr24,
                    self.config.Nc, self.config.Np,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x2p, y2p, h2p, x3p, y3p, h3p, x4p, y4p, h4p,
                    self.config.collision_radius, k
                )
                return ceq

            bounds = (
                [(self.config.v_min, self.config.v_max)] * self.config.Nc +
                [(self.config.omega_min, self.config.omega_max)] * self.config.Nc +
                [(self.config.zeta_min, self.config.zeta_max)] * self.config.Np +
                [(None, None)] * (5 * self.config.Np)
            )

            result = minimize(
                cost, h0, method='SLSQP', bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': constraint_ineq},
                    {'type': 'eq', 'fun': constraint_eq}
                ],
                options={'maxiter': self.config.max_iter, 'disp': False}
            )
            return result.x

        def optimize_uav3(h0, h_init_ref):
            s3 = self.uavs[3].state
            p31tilde = self.uavs[3].terminal_error[:3]
            thelt31tilde = self.uavs[3].terminal_error[3]
            phi31tilde = self.uavs[3].terminal_error[4]

            x2p = self.uavs[2].predicted_trajectory[:, 0]
            y2p = self.uavs[2].predicted_trajectory[:, 1]
            h2p = self.uavs[2].predicted_trajectory[:, 2]
            x3p = self.uavs[3].predicted_trajectory[:, 0]
            y3p = self.uavs[3].predicted_trajectory[:, 1]
            h3p = self.uavs[3].predicted_trajectory[:, 2]
            x4p = self.uavs[4].predicted_trajectory[:, 0]
            y4p = self.uavs[4].predicted_trajectory[:, 1]
            h4p = self.uavs[4].predicted_trajectory[:, 2]

            def cost(h):
                return cost_uav3(
                    h, self.config.Nc, self.config.Np,
                    self.formation.dr13,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x2p, y2p, h2p, self.formation.dr23,
                    x4p, y4p, h4p, self.formation.dr34,
                    p31tilde, thelt31tilde, phi31tilde,
                    vr1, wr1, zetar1, k
                )

            def constraint_ineq(h):
                c, _ = constraint_uav3(
                    h,
                    s3.x, s3.y, s3.h, s3.theta, s3.phi,
                    self.formation.dr13, self.formation.dr23,
                    self.config.Nc, self.config.Np,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x2p, y2p, h2p, x3p, y3p, h3p, x4p, y4p, h4p,
                    self.config.collision_radius, k
                )
                return c

            def constraint_eq(h):
                _, ceq = constraint_uav3(
                    h,
                    s3.x, s3.y, s3.h, s3.theta, s3.phi,
                    self.formation.dr13, self.formation.dr23,
                    self.config.Nc, self.config.Np,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x2p, y2p, h2p, x3p, y3p, h3p, x4p, y4p, h4p,
                    self.config.collision_radius, k
                )
                return ceq

            bounds = (
                [(self.config.v_min, self.config.v_max)] * self.config.Nc +
                [(self.config.omega_min, self.config.omega_max)] * self.config.Nc +
                [(self.config.zeta_min, self.config.zeta_max)] * self.config.Np +
                [(None, None)] * (5 * self.config.Np)
            )

            result = minimize(
                cost, h0, method='SLSQP', bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': constraint_ineq},
                    {'type': 'eq', 'fun': constraint_eq}
                ],
                options={'maxiter': self.config.max_iter, 'disp': False}
            )
            return result.x

        def optimize_uav4(h0, h_init_ref):
            s4 = self.uavs[4].state
            p41tilde = self.uavs[4].terminal_error[:3]
            thelt41tilde = self.uavs[4].terminal_error[3]
            phi41tilde = self.uavs[4].terminal_error[4]

            x2p = self.uavs[2].predicted_trajectory[:, 0]
            y2p = self.uavs[2].predicted_trajectory[:, 1]
            h2p = self.uavs[2].predicted_trajectory[:, 2]
            x3p = self.uavs[3].predicted_trajectory[:, 0]
            y3p = self.uavs[3].predicted_trajectory[:, 1]
            h3p = self.uavs[3].predicted_trajectory[:, 2]
            x4p = self.uavs[4].predicted_trajectory[:, 0]
            y4p = self.uavs[4].predicted_trajectory[:, 1]
            h4p = self.uavs[4].predicted_trajectory[:, 2]

            def cost(h):
                return cost_uav4(
                    h, self.config.Nc, self.config.Np,
                    self.formation.dr14,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x3p, y3p, h3p, self.formation.dr34,
                    x2p, y2p, h2p, self.formation.dr24,
                    p41tilde, thelt41tilde, phi41tilde,
                    vr1, wr1, zetar1, k
                )

            def constraint_ineq(h):
                c, _ = constraint_uav4(
                    h,
                    s4.x, s4.y, s4.h, s4.theta, s4.phi,
                    self.formation.dr14, self.formation.dr34,
                    self.config.Nc, self.config.Np,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x4p, y4p, h4p, x3p, y3p, h3p, x2p, y2p, h2p,
                    self.config.collision_radius, k
                )
                return c

            def constraint_eq(h):
                _, ceq = constraint_uav4(
                    h,
                    s4.x, s4.y, s4.h, s4.theta, s4.phi,
                    self.formation.dr14, self.formation.dr34,
                    self.config.Nc, self.config.Np,
                    x1pby1, y1pby1, h1pby1, thelt1pby1, phi1pby1,
                    x4p, y4p, h4p, x3p, y3p, h3p, x2p, y2p, h2p,
                    self.config.collision_radius, k
                )
                return ceq

            bounds = (
                [(self.config.v_min, self.config.v_max)] * self.config.Nc +
                [(self.config.omega_min, self.config.omega_max)] * self.config.Nc +
                [(self.config.zeta_min, self.config.zeta_max)] * self.config.Np +
                [(None, None)] * (5 * self.config.Np)
            )

            result = minimize(
                cost, h0, method='SLSQP', bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': constraint_ineq},
                    {'type': 'eq', 'fun': constraint_eq}
                ],
                options={'maxiter': self.config.max_iter, 'disp': False}
            )
            return result.x

        opt2 = optimize_uav2(self.h02, self.h02)
        opt3 = optimize_uav3(self.h03, self.h03)
        opt4 = optimize_uav4(self.h04, self.h04)

        self.h02 = opt2
        self.h03 = opt3
        self.h04 = opt4

        # Update follower predictions
        for uav_id, opt in [(2, opt2), (3, opt3), (4, opt4)]:
            Nc, Np = self.config.Nc, self.config.Np
            x_pred = opt[Nc+Nc+Np:Nc+Nc+Np+Np]
            y_pred = opt[Nc+Nc+Np+Np:Nc+Nc+Np+Np+Np]
            h_pred = opt[Nc+Nc+Np+Np+Np:Nc+Nc+Np+Np+Np+Np]
            thelt_pred = opt[Nc+Nc+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np]
            phi_pred = opt[Nc+Nc+Np+Np+Np+Np+Np:Nc+Nc+Np+Np+Np+Np+Np+Np]
            self.uavs[uav_id].predicted_trajectory = np.column_stack([
                x_pred, y_pred, h_pred, thelt_pred, phi_pred
            ])

        return opt2, opt3, opt4

    def run(self) -> Dict[int, Dict[str, list]]:
        """
        Run the full simulation.

        Returns:
            History dictionary with state/control trajectories for each UAV
        """
        print(f"Running DMPC simulation: {self.config.N_steps} steps, Np={self.config.Np}")

        for k in range(self.config.N_steps):
            if k % 10 == 0:
                print(f"  Step {k}/{self.config.N_steps}")

            self.step(k)

        print("Simulation complete.")
        return self.history

    def get_results(self) -> Dict[str, np.ndarray]:
        """Get final results as numpy arrays."""
        results = {}
        for uav_id, hist in self.history.items():
            results[f'uav{uav_id}_x'] = np.array(hist['x'])
            results[f'uav{uav_id}_y'] = np.array(hist['y'])
            results[f'uav{uav_id}_h'] = np.array(hist['h'])
            results[f'uav{uav_id}_theta'] = np.array(hist['theta'])
            results[f'uav{uav_id}_phi'] = np.array(hist['phi'])
            results[f'uav{uav_id}_v'] = np.array(hist['v'])
            results[f'uav{uav_id}_omega'] = np.array(hist['omega'])
            results[f'uav{uav_id}_zeta'] = np.array(hist['zeta'])
        return results
