#!/usr/bin/env python3
"""
Triangle Formation Gazebo Controller using gz command.
Moves 3 Iris UAVs in triangle formation.
"""

import subprocess
import numpy as np
import time

class TriangleFormationController:
    def __init__(self, csv_path, rate=5.0):
        self.rate = rate
        self.data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        self.n_steps = len(self.data)
        
        self.colors = ['\033[94m', '\033[92m', '\033[91m']
        self.reset = '\033[0m'
        
        print(f"\n{'='*60}")
        print(f"  Triangle Formation Controller")
        print(f"{'='*60}")
        print(f"  Steps: {self.n_steps} at {rate} Hz")
        print(f"  Formation: Triangle (3 UAVs)")
        print(f"{'='*60}\n")
        
    def set_uav_position(self, name, x, y, z, roll=0, pitch=0, yaw=0):
        cmd = [
            '/usr/bin/gz', 'model',
            '-m', name,
            '-x', str(x), '-y', str(y), '-z', str(z),
            '-R', str(roll), '-P', str(pitch), '-Y', str(yaw)
        ]
        subprocess.run(cmd, capture_output=True)
    
    def run(self):
        dt = 1.0 / self.rate
        
        for step in range(self.n_steps):
            row = self.data[step]
            
            for i in range(3):
                base = i * 4
                x_ned = row[base]
                y_ned = row[base + 1]
                z_ned = row[base + 2]
                
                # NED to ENU
                x_enu = y_ned
                y_enu = x_ned
                z_enu = -z_ned
                
                name = f'uav{i + 1}'
                color = self.colors[i]
                
                self.set_uav_position(name, x_enu, y_enu, z_enu)
                
                if step % 10 == 0:
                    print(f"{color}[Step {step:3d}] {name}: ({x_enu:.1f}, {y_enu:.1f}, {z_enu:.1f}){self.reset}")
            
            time.sleep(dt)
        
        print(f"\n{'='*60}")
        print(f"  Triangle Formation Complete!")
        print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='/home/demo/.openclaw/workspace/dmpc_formation/triangle_3uav.csv')
    parser.add_argument('--rate', type=float, default=5.0)
    args = parser.parse_args()
    
    controller = TriangleFormationController(args.csv, args.rate)
    controller.run()
