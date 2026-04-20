#!/usr/bin/env python3
"""
Single UAV DMPC Test - Visualize trajectory and network conditions.

Outputs:
1. 3D flight trajectory
2. Packet dropout / delay events timeline
3. Tracking error over time

Usage:
    python single_uav_test.py --steps 64 --sigma-L 0.15 --sigma-D 0.2 --plot
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0, '/home/demo/.openclaw/workspace/dmpc_formation')

from quadrotor_dmpc.controller import QuadrotorDMPC, MPCConfig
from quadrotor_dmpc.models.uav_model import UAVState
from quadrotor_dmpc.models.formation import FormationConfig
from quadrotor_dmpc.utils.trajectory import generate_sinusoidal_3d, generate_circular_3d
from quadrotor_dmpc.utils.network import NetworkSimulator


def run_single_uav_test(args):
    """Run single UAV DMPC with detailed network statistics."""

    # Generate reference trajectory
    t = np.arange(0, args.steps, 1.0)
    if args.trajectory == 'sinusoidal':
        xr, yr, hr, thelr, phir = generate_sinusoidal_3d(
            t, x0=0, y0=0, h0=10,
            Ax=args.Ax, Ay=args.Ay, Ah=args.Ah,
            wx=args.wx, wy=args.wy, wh=args.wh
        )
    else:
        xr, yr, hr, thelr, phir = generate_circular_3d(
            t, x0=0, y0=0, h0=10,
            R=args.R, w=args.w
        )

    ref_traj = {
        'xr': xr, 'yr': yr, 'hr': hr,
        'theltr': thelr, 'phir': phir
    }

    # Single UAV formation (leader only)
    formation = FormationConfig(
        dr01=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        dr12=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        dr13=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        dr14=np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    )

    # Initial UAV state (leader starts away from origin)
    initial_states = {
        1: UAVState(
            x=xr[0] - 5,
            y=yr[0] + 3,
            h=hr[0] + 1,
            theta=thelr[0],
            phi=phir[0]
        )
    }

    # Network simulator - record detailed events
    network = NetworkSimulator.create_random(
        args.steps,
        sigma_L=args.sigma_L,
        sigma_D=args.sigma_D,
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

    # Print header
    print(f"\n{'='*60}")
    print(f"  Single UAV DMPC Test")
    print(f"{'='*60}")
    print(f"  Steps:      {args.steps}")
    print(f"  Np (horizon): {args.np}")
    print(f"  Dropout σL: {args.sigma_L:.1%}")
    print(f"  Delay σD:   {args.sigma_D:.1%}")
    print(f"  Seed:       {args.seed}")
    print(f"  Trajectory: {args.trajectory}")
    print(f"{'='*60}\n")

    # Run simulation
    history = dmpc.run()
    results = dmpc.get_results()

    # ===== Compute Statistics =====
    x = results['uav1_x']
    y = results['uav1_y']
    h = results['uav1_h']
    v = results['uav1_v']
    theta = results['uav1_theta']

    # Tracking error (distance to reference)
    error_x = x - xr[:len(x)]
    error_y = y - yr[:len(y)]
    error_h = h - hr[:len(h)]
    tracking_error = np.sqrt(error_x**2 + error_y**2 + error_h**2)

    # Network event statistics
    dropout_events = network.isdropout
    delay_events = network.isdelay

    n_dropout = np.sum(dropout_events)
    n_delay = np.sum(delay_events)
    n_both = np.sum(dropout_events & delay_events)

    dropout_rate_actual = n_dropout / args.steps
    delay_rate_actual = n_delay / args.steps

    # Find consecutive dropout/delay bursts
    dropout_bursts = []
    delay_bursts = []
    in_dropout = False
    in_delay = False
    start = 0

    for k in range(args.steps):
        if dropout_events[k] and not in_dropout:
            start = k
            in_dropout = True
        elif not dropout_events[k] and in_dropout:
            dropout_bursts.append((start, k - start))
            in_dropout = False
        if delay_events[k] and not in_delay:
            start = k
            in_delay = True
        elif not delay_events[k] and in_delay:
            delay_bursts.append((start, k - start))
            in_delay = False
    if in_dropout:
        dropout_bursts.append((start, args.steps - start))
    if in_delay:
        delay_bursts.append((start, args.steps - start))

    # Max consecutive
    max_dropout = max([b[1] for b in dropout_bursts], default=0)
    max_delay = max([b[1] for b in delay_bursts], default=0)

    # ===== Print Statistics =====
    print(f"\n{'='*60}")
    print(f"  📊 Network Statistics")
    print(f"{'='*60}")
    print(f"  Dropout events:     {n_dropout:4d} / {args.steps}  ({dropout_rate_actual:.2%})")
    print(f"  Delay events:       {n_delay:4d} / {args.steps}  ({delay_rate_actual:.2%})")
    print(f"  Both simultaneous:  {n_both:4d}")
    print(f"  Max dropout burst:   {max_dropout} steps")
    print(f"  Max delay burst:    {max_delay} steps")
    print(f"\n  Dropout bursts: {dropout_bursts[:10]}{'...' if len(dropout_bursts) > 10 else ''}")
    print(f"  Delay bursts:   {delay_bursts[:10]}{'...' if len(delay_bursts) > 10 else ''}")

    print(f"\n{'='*60}")
    print(f"  ✈️  Flight Statistics")
    print(f"{'='*60}")
    print(f"  Total distance:     {np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(h)**2)):.2f} m")
    print(f"  Avg speed:          {np.mean(v):.2f} m/s")
    print(f"  Max speed:          {np.max(v):.2f} m/s")
    print(f"  Final position:    ({x[-1]:.2f}, {y[-1]:.2f}, {h[-1]:.2f})")
    print(f"  Final ref:          ({xr[len(x)-1]:.2f}, {yr[len(x)-1]:.2f}, {hr[len(x)-1]:.2f})")

    print(f"\n{'='*60}")
    print(f"  🎯 Tracking Error (RMS)")
    print(f"{'='*60}")
    print(f"  X error RMS:        {np.sqrt(np.mean(error_x**2)):.3f} m")
    print(f"  Y error RMS:        {np.sqrt(np.mean(error_y**2)):.3f} m")
    print(f"  H error RMS:        {np.sqrt(np.mean(error_h**2)):.3f} m")
    print(f"  Total error RMS:    {np.sqrt(np.mean(tracking_error**2)):.3f} m")
    print(f"  Max error:          {np.max(tracking_error):.3f} m")
    print(f"{'='*60}\n")

    # ===== Plot =====
    if args.plot:
        plot_single_uav_results(results, ref_traj, dropout_events, delay_events, tracking_error, args.steps)

    return {
        'x': x, 'y': y, 'h': h, 'v': v,
        'tracking_error': tracking_error,
        'dropout_events': dropout_events,
        'delay_events': delay_events,
        'dropout_bursts': dropout_bursts,
        'delay_bursts': delay_bursts
    }


def plot_single_uav_results(results, ref_traj, dropout, delay, tracking_error, n_steps):
    """Plot comprehensive results for single UAV test."""

    fig = plt.figure(figsize=(16, 12))

    x = results['uav1_x']
    y = results['uav1_y']
    h = results['uav1_h']
    v = results['uav1_v']

    # 1. 3D Trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x, y, h, 'b-', linewidth=1.5, label='UAV Trajectory')
    ax1.plot(ref_traj['xr'][:n_steps], ref_traj['yr'][:n_steps], ref_traj['hr'][:n_steps],
             'r--', linewidth=1, alpha=0.7, label='Reference')
    ax1.scatter(x[0], y[0], h[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(x[-1], y[-1], h[-1], c='red', s=100, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('H (m)')
    ax1.set_title('3D Flight Trajectory')
    ax1.legend()

    # 2. XY Plane
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x, y, 'b-', linewidth=1.5, label='UAV')
    ax2.plot(ref_traj['xr'][:n_steps], ref_traj['yr'][:n_steps], 'r--', alpha=0.7, label='Reference')
    ax2.scatter(x[0], y[0], c='green', s=100, marker='o')
    ax2.scatter(x[-1], y[-1], c='red', s=100, marker='s')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. Altitude Profile
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(range(len(h)), h, 'b-', linewidth=1.5, label='UAV Altitude')
    ax3.plot(range(len(ref_traj['hr'][:n_steps])), ref_traj['hr'][:n_steps], 'r--', alpha=0.7, label='Reference')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Network Events Timeline
    ax4 = fig.add_subplot(2, 3, 4)
    t = np.arange(n_steps)
    ax4.fill_between(t, 0, dropout, alpha=0.5, label='Dropout', color='red', step='mid')
    ax4.fill_between(t, 0, delay, alpha=0.5, label='Delay', color='orange', step='mid')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Event')
    ax4.set_title('Network Events (Dropout & Delay)')
    ax4.set_ylim(0, 1.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Tracking Error
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(range(len(tracking_error)), tracking_error, 'b-', linewidth=1.5)
    ax5.axhline(y=np.mean(tracking_error), color='r', linestyle='--', label=f'Mean: {np.mean(tracking_error):.3f}m')
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Tracking Error (m)')
    ax5.set_title('Position Tracking Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Velocity Profile
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(range(len(v)), v, 'b-', linewidth=1.5)
    ax6.axhline(y=np.mean(v), color='r', linestyle='--', label=f'Mean: {np.mean(v):.2f}m/s')
    ax6.set_xlabel('Time step')
    ax6.set_ylabel('Velocity (m/s)')
    ax6.set_title('Velocity Profile')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = '/home/demo/.openclaw/workspace/dmpc_formation/single_uav_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {output_path}")

    output_path_csv = '/home/demo/.openclaw/workspace/dmpc_formation/trajectory_data.csv'
    np.savetxt(output_path_csv,
               np.column_stack([x, y, h, v, tracking_error, dropout, delay]),
               delimiter=',',
               header='x,y,h,v,tracking_error,dropout,delay',
               comments='')
    print(f"📄 Data saved to: {output_path_csv}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single UAV DMPC Test')
    parser.add_argument('--steps', type=int, default=64, help='Number of simulation steps')
    parser.add_argument('--np', type=int, default=8, help='Prediction horizon')
    parser.add_argument('--sigma-L', type=float, default=0.15, help='Packet dropout rate')
    parser.add_argument('--sigma-D', type=float, default=0.20, help='Communication delay rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-iter', type=int, default=2000, help='Max optimizer iterations')
    parser.add_argument('--max-fun-evals', type=int, default=50000, help='Max function evaluations')
    parser.add_argument('--trajectory', choices=['sinusoidal', 'circular'], default='sinusoidal')

    # Trajectory params
    parser.add_argument('--Ax', type=float, default=50.0, help='X amplitude')
    parser.add_argument('--Ay', type=float, default=30.0, help='Y amplitude')
    parser.add_argument('--Ah', type=float, default=10.0, help='Altitude amplitude')
    parser.add_argument('--wx', type=float, default=0.1, help='X frequency')
    parser.add_argument('--wy', type=float, default=0.15, help='Y frequency')
    parser.add_argument('--wh', type=float, default=0.05, help='Altitude frequency')
    parser.add_argument('--R', type=float, default=50.0, help='Circular radius')
    parser.add_argument('--w', type=float, default=0.1, help='Circular frequency')

    parser.add_argument('--plot', action='store_true', help='Show plots')

    args = parser.parse_args()
    results = run_single_uav_test(args)
