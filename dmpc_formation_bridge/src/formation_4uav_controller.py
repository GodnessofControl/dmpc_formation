#!/usr/bin/env python3
"""
4-UAV Formation Controller for Gazebo.
"""
import subprocess, numpy as np, time

class Controller:
    def __init__(self, csv_path, rate=3.0):
        self.data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        self.n = len(self.data)
        self.rate = rate
        self.colors = ['\033[94m', '\033[92m', '\033[93m', '\033[91m']
        print('4-UAV Formation, %d steps at %.1f Hz' % (self.n, rate))
        
    def set_pos(self, name, x, y, z):
        subprocess.run(['/usr/bin/gz', 'model', '-m', name, '-x', str(x), '-y', str(y), '-z', str(z), '-R', '0', '-P', '0', '-Y', '0'], capture_output=True)
    
    def run(self):
        dt = 1.0 / self.rate
        for step in range(self.n):
            row = self.data[step]
            for i in range(4):
                base = i * 4
                # NED to ENU: ENU_x = NED_y, ENU_y = NED_x, ENU_z = -NED_z
                x_enu = row[base + 1]
                y_enu = row[base]
                z_enu = -row[base + 2]
                self.set_pos('uav%d' % (i+1), x_enu, y_enu, z_enu)
                if step % 20 == 0:
                    print('%s[Step %3d] uav%d: ENU(%.1f, %.1f, %.1f)%s' % (self.colors[i], step, i+1, x_enu, y_enu, z_enu, '\033[0m'))
            time.sleep(dt)
        print('Done!')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='/home/demo/.openclaw/workspace/dmpc_formation/4uav_formation_trace.csv')
    p.add_argument('--rate', type=float, default=3.0)
    Controller(p.parse_args().csv, p.parse_args().rate).run()
