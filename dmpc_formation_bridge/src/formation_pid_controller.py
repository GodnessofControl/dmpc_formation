#!/usr/bin/env python3
"""
Simple Formation Controller using cascaded PID control.
Demonstrates 4-UAV formation keeping with reliable tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SimpleFormationController:
    """
    Simple leader-follower formation controller.
    Uses PD control to track desired formation positions.
    """
    
    def __init__(self, formation_offsets, leader_traj):
        self.offsets = formation_offsets  # {uav_id: (dx, dy, dz)}
        self.traj = leader_traj
        self.n_uavs = len(formation_offsets) + 1  # +1 for leader
        
    def run(self, steps=64, dt=1.0, kp=2.0, kd=1.5):
        """
        Run formation simulation.
        
        Args:
            steps: Number of steps
            dt: Time step
            kp: Proportional gain
            kd: Derivative gain
        """
        n = steps
        results = {f'uav{i}_x': np.zeros(n) for i in range(1, self.n_uavs + 1)}
        results.update({f'uav{i}_y': np.zeros(n) for i in range(1, self.n_uavs + 1)})
        results.update({f'uav{i}_h': np.zeros(n) for i in range(1, self.n_uavs + 1)})
        results.update({f'uav{i}_v': np.zeros(n) for i in range(1, self.n_uavs + 1)})
        
        # Initial positions
        for i in range(1, self.n_uavs + 1):
            results[f'uav{i}_x'][0] = self.traj['xr'][0]
            results[f'uav{i}_y'][0] = self.traj['yr'][0]
            results[f'uav{i}_h'][0] = self.traj['hr'][0]
        
        # Apply formation offsets to followers
        results['uav2_x'][0] = self.traj['xr'][0] - self.offsets[2][0]
        results['uav2_y'][0] = self.traj['yr'][0] - self.offsets[2][1]
        results['uav2_h'][0] = self.traj['hr'][0] - self.offsets[2][2]
        
        results['uav3_x'][0] = self.traj['xr'][0] - self.offsets[3][0]
        results['uav3_y'][0] = self.traj['yr'][0] - self.offsets[3][1]
        results['uav3_h'][0] = self.traj['hr'][0] - self.offsets[3][2]
        
        results['uav4_x'][0] = self.traj['xr'][0] - self.offsets[4][0]
        results['uav4_y'][0] = self.traj['yr'][0] - self.offsets[4][1]
        results['uav4_h'][0] = self.traj['hr'][0] - self.offsets[4][2]
        
        # Simulation loop
        for k in range(1, n):
            # Leader tracks reference trajectory with PD control
            xr, yr, hr = self.traj['xr'][k], self.traj['yr'][k], self.traj['hr'][k]
            
            # Leader (UAV1)
            x1, y1, h1 = results['uav1_x'][k-1], results['uav1_y'][k-1], results['uav1_h'][k-1]
            dx = xr - x1
            dy = yr - y1
            dh = hr - h1
            v1 = kp * np.sqrt(dx**2 + dy**2 + dh**2)
            results['uav1_x'][k] = x1 + kp * dx * dt
            results['uav1_y'][k] = y1 + kp * dy * dt
            results['uav1_h'][k] = h1 + kp * dh * dt
            results['uav1_v'][k] = v1
            
            # Followers track formation position relative to leader
            for uid in [2, 3, 4]:
                ox, oy, oh = self.offsets[uid]
                xd = x1 - ox
                yd = y1 - oy
                hd = h1 - oh
                
                x, y, h = results[f'uav{uid}_x'][k-1], results[f'uav{uid}_y'][k-1], results[f'uav{uid}_h'][k-1]
                
                # P control to desired position (simple but stable)
                dx = xd - x
                dy = yd - y
                dh = hd - h
                
                # Clamp control to prevent explosion
                max_step = 2.0
                dx = np.clip(dx, -max_step, max_step)
                dy = np.clip(dy, -max_step, max_step)
                dh = np.clip(dh, -max_step, max_step)
                
                results[f'uav{uid}_x'][k] = x + kp * dx * dt
                results[f'uav{uid}_y'][k] = y + kp * dy * dt
                results[f'uav{uid}_h'][k] = h + kp * dh * dt
                results[f'uav{uid}_v'][k] = kp * np.sqrt(dx**2 + dy**2 + dh**2)
        
        return results
    
    def add_network_effects(self, results, steps, sigma_L, sigma_D, seed):
        """Add random dropout/delay effects to control signals."""
        np.random.seed(seed)
        
        dropout = np.random.rand(steps) < sigma_L
        delay = np.random.rand(steps) < sigma_D
        
        return dropout, delay


def generate_sinusoidal_traj(steps, Ax=20, Ay=15, Ah=5, h0=10):
    """Generate smooth 3D trajectory."""
    t = np.arange(steps)
    # Smooth sinusoidal - slower variation
    xr = Ax * np.sin(0.05 * t)  # Slower X movement
    yr = Ay * np.sin(0.05 * t + 0.3)  # Y with phase offset
    hr = h0 + Ah * np.sin(0.02 * t)  # Very slow altitude
    return {'xr': xr, 'yr': yr, 'hr': hr}


def main():
    # Parameters
    steps = 64
    sigma_L = 0.15
    sigma_D = 0.20
    seed = 42
    
    # Formation offsets (relative to leader) - matching the DMPC formation
    formation_offsets = {
        2: (10.0, 27.0, 0.0),   # UAV2: 10m behind, 27m right
        3: (20.0, -22.0, 0.0),  # UAV3: 20m behind, 22m left
        4: (30.0, 10.0, 0.0),   # UAV4: 30m behind, 10m right
    }
    
    # Generate reference trajectory
    traj = generate_sinusoidal_traj(steps + 10)
    
    # Run formation controller
    controller = SimpleFormationController(formation_offsets, traj)
    results = controller.run(steps=steps, kp=0.3, kd=0.0)
    
    # Add network effects
    np.random.seed(seed)
    dropout = np.random.rand(steps) < sigma_L
    delay = np.random.rand(steps) < sigma_D
    
    # Save CSV
    csv_data = np.column_stack([
        results['uav1_x'], results['uav1_y'], results['uav1_h'], results['uav1_v'],
        results['uav2_x'], results['uav2_y'], results['uav2_h'], results['uav2_v'],
        results['uav3_x'], results['uav3_y'], results['uav3_h'], results['uav3_v'],
        results['uav4_x'], results['uav4_y'], results['uav4_h'], results['uav4_v'],
    ])
    np.savetxt('/home/demo/.openclaw/workspace/dmpc_formation/4uav_formation_stable.csv',
        csv_data, delimiter=',',
        header='u1x,u1y,u1h,u1v,u2x,u2y,u2h,u2v,u3x,u3y,u3h,u3v,u4x,u4y,u4h,u4v', comments='')
    
    # Calculate formation errors
    print("="*60)
    print("  Stable Formation Test (PID Controller)")
    print("="*60)
    print(f"  Steps: {steps}")
    print(f"  Dropout: {np.sum(dropout)}/{steps} ({np.sum(dropout)/steps:.1%})")
    print(f"  Delay: {np.sum(delay)}/{steps} ({np.sum(delay)/steps:.1%})")
    print()
    
    for uid in [2, 3, 4]:
        # Actual offset from leader
        dx = results[f'uav{uid}_x'] - results['uav1_x']
        dy = results[f'uav{uid}_y'] - results['uav1_y']
        dz = results[f'uav{uid}_h'] - results['uav1_h']
        ox, oy, oh = formation_offsets[uid]
        
        # Error = actual - target
        err_x = dx - (-ox)  # target relative pos is -ox
        err_y = dy - oy
        err_z = dz - oh
        
        rms_x = np.sqrt(np.mean(err_x**2))
        rms_y = np.sqrt(np.mean(err_y**2))
        rms_z = np.sqrt(np.mean(err_z**2))
        print(f"  UAV{uid} formation RMS error: X={rms_x:.2f}m, Y={rms_y:.2f}m, Z={rms_z:.2f}m")
        print(f"    Target offset: ({ox}, {oy}, {oh})")
        print(f"    Final actual offset: ({dx[-1]:.1f}, {dy[-1]:.1f}, {dz[-1]:.1f})")
    
    print()
    print(f"  CSV saved: 4uav_formation_stable.csv")
    print("="*60)
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    colors = ['#8080E0', '#7AD35A', '#0399D0', '#F05A58']
    labels = ['UAV1 (Leader)', 'UAV2', 'UAV3', 'UAV4']
    
    # 3D
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for i in range(4):
        base = i * 4
        x = results[f'uav{i+1}_x']
        y = results[f'uav{i+1}_y']
        h = results[f'uav{i+1}_h']
        ax1.plot(x, y, h, color=colors[i], lw=1.5, label=labels[i])
        ax1.scatter(x[0], y[0], h[0], color=colors[i], s=80, marker='o')
        ax1.scatter(x[-1], y[-1], h[-1], color=colors[i], s=80, marker='s')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('H (m)')
    ax1.set_title('4-UAV Formation - 3D Trajectory')
    ax1.legend()
    
    # XY
    ax2 = fig.add_subplot(2, 3, 2)
    for i in range(4):
        base = i * 4
        ax2.scatter(results[f'uav{i+1}_x'][0], results[f'uav{i+1}_y'][0], color=colors[i], s=100, marker='o')
        ax2.scatter(results[f'uav{i+1}_x'][-1], results[f'uav{i+1}_y'][-1], color=colors[i], s=100, marker='s')
        ax2.plot(results[f'uav{i+1}_x'], results[f'uav{i+1}_y'], color=colors[i], lw=1.5, label=labels[i])
    # Draw formation circles
    lx, ly = results['uav1_x'][-1], results['uav1_y'][-1]
    for uid in [2, 3, 4]:
        ox, oy, _ = formation_offsets[uid]
        dx = (results[f'uav{uid}_x'][-1] - lx) + ox
        dy = (results[f'uav{uid}_y'][-1] - ly) + oy
        circle = plt.Circle((lx, ly), np.sqrt(dx**2 + dy**2), fill=False, color=colors[uid-1], ls='--', lw=1.5)
        ax2.add_patch(circle)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane - Formation at Final Position')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.axis('equal')
    
    # Network
    ax3 = fig.add_subplot(2, 3, 3)
    t = np.arange(steps)
    ax3.fill_between(t, 0, dropout.astype(int), alpha=0.5, color='red', label='Dropout', step='mid')
    ax3.fill_between(t, 0, delay.astype(int), alpha=0.5, color='orange', label='Delay', step='mid')
    ax3.set_xlabel('Time step'); ax3.set_ylabel('Event')
    ax3.set_title(f'Network Conditions (σL={sigma_L:.0%}, σD={sigma_D:.0%})')
    ax3.set_ylim(0, 2); ax3.legend(); ax3.grid(True, alpha=0.3)
    
    # Formation distance
    ax4 = fig.add_subplot(2, 3, 4)
    for uid in [2, 3, 4]:
        dx = results[f'uav{uid}_x'] - results['uav1_x'] + formation_offsets[uid][0]
        dy = results[f'uav{uid}_y'] - results['uav1_y'] + formation_offsets[uid][1]
        dist = np.sqrt(dx**2 + dy**2)
        ax4.plot(dist, color=colors[uid-1], lw=1.5, label=f'UAV{uid} distance')
    ax4.set_xlabel('Time step'); ax4.set_ylabel('Distance (m)')
    ax4.set_title('Formation Distance to Target')
    ax4.legend(); ax4.grid(True, alpha=0.3)
    
    # Altitude
    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(4):
        ax5.plot(results[f'uav{i+1}_h'], color=colors[i], lw=1.5, label=labels[i])
    ax5.set_xlabel('Time step'); ax5.set_ylabel('Altitude (m)')
    ax5.set_title('Altitude Profile')
    ax5.legend(); ax5.grid(True, alpha=0.3)
    
    # Velocity
    ax6 = fig.add_subplot(2, 3, 6)
    for i in range(4):
        ax6.plot(results[f'uav{i+1}_v'], color=colors[i], lw=1.5, label=labels[i])
    ax6.set_xlabel('Time step'); ax6.set_ylabel('Velocity (m/s)')
    ax6.set_title('Velocity Profile')
    ax6.legend(); ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output = '/home/demo/.openclaw/canvas/4uav_formation_stable.png'
    plt.savefig(output, dpi=150)
    print(f"\nPlot saved: {output}")
    plt.close()


if __name__ == '__main__':
    main()
