#!/usr/bin/env python3
"""
Main simulation script for 4-UAV DMPC formation control.
Mirrors the MATLAB main simulation (multi_delay_loss.m / OnlineDoubleRandomDL.m)

Usage:
    python main_sim.py
    python main_sim.py --steps 100 --sigma-L 0.15 --sigma-D 0.2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from quadrotor_dmpc.controller import QuadrotorDMPC, MPCConfig, UAVAgent
from quadrotor_dmpc.models.uav_model import UAVState
from quadrotor_dmpc.models.formation import FormationConfig
from quadrotor_dmpc.utils.trajectory import generate_sinusoidal_3d, generate_circular_3d
from quadrotor_dmpc.utils.network import NetworkSimulator


def create_initial_states(formation: FormationConfig, ref_traj: dict) -> dict:
    """
    Create initial UAV states from reference trajectory and formation offsets.
    """
    x0 = ref_traj['xr'][0]
    y0 = ref_traj['yr'][0]
    h0 = ref_traj['hr'][0]
    theta0 = ref_traj['theltr'][0]
    phi0 = ref_traj['phir'][0]

    # UAV1 (Leader)
    s1 = UAVState(
        x=x0 + formation.dr01[0] - 10,
        y=y0 + formation.dr01[1],
        h=h0 + formation.dr01[2],
        theta=theta0 + formation.dr01[3],
        phi=phi0 + formation.dr01[4]
    )

    # UAV2
    s2 = UAVState(
        x=x0 - formation.dr12[0] - 20,
        y=y0 - formation.dr12[1] - 5,
        h=h0 + formation.dr12[2],
        theta=theta0 + formation.dr12[3],
        phi=phi0 + formation.dr12[4]
    )

    # UAV3
    s3 = UAVState(
        x=x0 - formation.dr13[0] - 6,
        y=y0 - formation.dr13[1] + 10,
        h=h0 + formation.dr13[2],
        theta=theta0 + formation.dr13[3],
        phi=phi0 + formation.dr13[4]
    )

    # UAV4
    s4 = UAVState(
        x=x0 - formation.dr14[0] + 15,
        y=y0 - formation.dr14[1] - 20,
        h=h0 + formation.dr14[2],
        theta=theta0 + formation.dr14[3],
        phi=phi0 + formation.dr14[4]
    )

    return {1: s1, 2: s2, 3: s3, 4: s4}


def run_simulation(args):
    """Run the DMPC formation control simulation."""

    # Generate reference trajectory
    t = np.arange(0, 300, 1.0)  # 300 seconds at 1Hz
    if args.trajectory == 'sinusoidal':
        xr, yr, hr, thelr, phir = generate_sinusoidal_3d(
            t, x0=0, y0=0, h0=10,
            Ax=50, Ay=30, Ah=10,
            wx=0.1, wy=0.15, wh=0.05
        )
    else:
        xr, yr, hr, thelr, phir = generate_circular_3d(
            t, x0=0, y0=0, h0=10,
            R=50, w=0.1
        )

    ref_traj = {
        'xr': xr, 'yr': yr, 'hr': hr,
        'theltr': thelr, 'phir': phir
    }

    # Formation configuration
    formation = FormationConfig.default_3d()

    # Initial UAV states
    initial_states = create_initial_states(formation, ref_traj)

    # Network simulator
    network = NetworkSimulator.create_random(
        args.steps, sigma_L=args.sigma_L, sigma_D=args.sigma_D,
        seed=args.seed
    )

    # MPC configuration
    config = MPCConfig(
        Np=args.np,
        Nc=args.np,
        N_steps=args.steps,
        sigma_L=args.sigma_L,
        sigma_D=args.sigma_D,
        max_iter=args.max_iter,
        max_fun_evals=args.max_fun_evals
    )

    # Create controller
    dmpc = QuadrotorDMPC(
        config=config,
        formation=formation,
        initial_states=initial_states,
        reference_trajectory=ref_traj,
        network_sim=network,
        use_lmi_gains=False
    )

    # Run simulation
    print(f"\n=== DMPC Formation Control Simulation ===")
    print(f"Steps: {args.steps}, Np: {args.np}")
    print(f"Dropout rate: {args.sigma_L:.2%}, Delay rate: {args.sigma_D:.2%}")
    print(f"Formation: dr12={formation.dr12[:2]}, dr13={formation.dr13[:2]}, dr14={formation.dr14[:2]}")
    print()

    history = dmpc.run()

    # Get results
    results = dmpc.get_results()

    # Plot results
    if args.plot:
        plot_results(results, ref_traj, args.steps)

    return results, dmpc


def plot_results(results: dict, ref_traj: dict, n_steps: int):
    """Plot 3D trajectory and formation errors."""
    fig = plt.figure(figsize=(14, 10))

    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    colors = {1: [0.50, 0.50, 0.89], 2: [0.48, 0.83, 0.39],
              3: [0.02, 0.60, 0.79], 4: [0.94, 0.34, 0.34]}

    for uav_id in [1, 2, 3, 4]:
        x = results[f'uav{uav_id}_x']
        y = results[f'uav{uav_id}_y']
        h = results[f'uav{uav_id}_h']
        k = len(x)
        ax1.plot(x[:k], y[:k], h[:k], '--', color=colors[uav_id], linewidth=1, label=f'UAV {uav_id}')
        ax1.scatter(x[0], y[0], h[0], marker='o', color=colors[uav_id], s=50)

    # Reference trajectory
    ax1.plot(ref_traj['xr'][:n_steps], ref_traj['yr'][:n_steps], ref_traj['hr'][:n_steps],
             'k:', linewidth=1, alpha=0.5, label='Reference')

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('h (m)')
    ax1.set_title('3D Formation Trajectory')
    ax1.legend()

    # XY plane
    ax2 = fig.add_subplot(2, 2, 2)
    for uav_id in [1, 2, 3, 4]:
        x = results[f'uav{uav_id}_x']
        y = results[f'uav{uav_id}_y']
        k = len(x)
        ax2.plot(x[:k], y[:k], '--', color=colors[uav_id], linewidth=1, label=f'UAV {uav_id}')

    ax2.plot(ref_traj['xr'][:n_steps], ref_traj['yr'][:n_steps], 'k:', alpha=0.5, label='Reference')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('XY Plane')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Velocity profiles
    ax3 = fig.add_subplot(2, 2, 3)
    t_range = np.arange(len(results['uav1_v']))
    for uav_id in [1, 2, 3, 4]:
        v = results[f'uav{uav_id}_v']
        ax3.plot(t_range, v, color=colors[uav_id], linewidth=1, label=f'UAV {uav_id}', alpha=0.8)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Velocity v (m/s)')
    ax3.set_title('Velocity Profiles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Altitude profile
    ax4 = fig.add_subplot(2, 2, 4)
    for uav_id in [1, 2, 3, 4]:
        h = results[f'uav{uav_id}_h']
        k = len(h)
        ax4.plot(range(k), h, '--', color=colors[uav_id], linewidth=1, label=f'UAV {uav_id}')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Altitude h (m)')
    ax4.set_title('Altitude Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/demo/mywork/data_loss_python/formation_result.png', dpi=150)
    print("\nPlot saved to /home/demo/mywork/data_loss_python/formation_result.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4-UAV DMPC Formation Control')
    parser.add_argument('--steps', type=int, default=64, help='Number of simulation steps')
    parser.add_argument('--np', type=int, default=8, help='Prediction horizon Np')
    parser.add_argument('--sigma-L', type=float, default=0.15, help='Packet dropout rate')
    parser.add_argument('--sigma-D', type=float, default=0.20, help='Communication delay rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-iter', type=int, default=2000, help='Max optimizer iterations')
    parser.add_argument('--max-fun-evals', type=int, default=50000, help='Max function evaluations')
    parser.add_argument('--trajectory', choices=['sinusoidal', 'circular'], default='sinusoidal')
    parser.add_argument('--plot', action='store_true', help='Show plots')

    args = parser.parse_args()
    results, dmpc = run_simulation(args)
