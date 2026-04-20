#!/usr/bin/env python3
"""
Formation Gazebo Visualizer
Reads trajectory CSV and moves 4 UAVs in Gazebo to visualize formation.
Uses /gazebo/set_model_state to position UAVs.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Pose, Twist, Quaternion
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
import tf.transformations as T
import os


class FormationVisualizer:
    def __init__(self, csv_path, rate=5.0):
        rospy.init_node('formation_visualizer', anonymous=True)
        
        self.rate = rate
        self.running = True
        
        # Load trajectory data
        self.data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        self.n_steps = len(self.data)
        
        # UAV colors for console output
        self.colors = ['\033[94m', '\033[92m', '\033[96m', '\033[91m']  # blue, green, cyan, red
        self.reset = '\033[0m'
        
        rospy.loginfo(f"Loaded {self.n_steps} steps from {csv_path}")
        rospy.loginfo("Waiting for Gazebo services...")
        
        # Wait for Gazebo services
        rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Model names
        self.uav_names = ['uav1', 'uav2', 'uav3', 'uav4']
        
        # Initial positions (matchgazebo spawn positions)
        self.init_positions = [
            (0, 0, 0.5),      # UAV1
            (-10, -27, 0.5),  # UAV2
            (-20, 22, 0.5),   # UAV3
            (-30, -10, 0.5),  # UAV4
        ]
        
        rospy.loginfo("Services ready. Starting visualization...")
        
    def euler_to_quat(self, roll, pitch, yaw):
        q = T.quaternion_from_euler(roll, pitch, yaw)
        return Quaternion(*q)
    
    def set_uav_position(self, name, x, y, z, roll=0, pitch=0, yaw=0):
        """Move a UAV to a specific position in Gazebo."""
        state = ModelState()
        state.model_name = name
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation = self.euler_to_quat(roll, pitch, yaw)
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0
        state.reference_frame = 'world'
        
        try:
            self.set_state(state)
            return True
        except Exception as e:
            return False
    
    def run(self):
        """Main loop - move UAVs according to trajectory."""
        rate = rospy.Rate(self.rate)
        
        print("\n" + "="*60)
        print("  4-UAV Formation Visualization in Gazebo")
        print("="*60)
        print(f"  Trajectory: {self.n_steps} steps at {self.rate} Hz")
        print(f"  UAV Colors: Blue=Leader, Green=UAV2, Cyan=UAV3, Red=UAV4")
        print("="*60 + "\n")
        
        for step in range(self.n_steps):
            if rospy.is_shutdown():
                break
                
            row = self.data[step]
            
            # Move each UAV (NED -> ENU conversion)
            # CSV: u1x,u1y,u1h,u1v, u2x,u2y,u2h,u2v, ...
            for i in range(4):
                base = i * 4
                # NED to ENU: swap x/y, negate z
                x_ned = row[base]      # NED x = North
                y_ned = row[base+1]   # NED y = East  
                z_ned = row[base+2]   # NED z = Down (so -z_ned = altitude)
                
                # Convert to ENU for Gazebo
                x_enu = y_ned   # ENU x = East = NED y
                y_enu = x_ned   # ENU y = North = NED x
                z_enu = -z_ned  # ENU z = Up = -NED z
                
                name = self.uav_names[i]
                color = self.colors[i]
                
                if step % 10 == 0:
                    print(f"{color}[Step {step:3d}] {name}: ENU({x_enu:.1f}, {y_enu:.1f}, {z_enu:.1f}){self.reset}")
                
                self.set_uav_position(name, x_enu, y_enu, z_enu)
            
            rate.sleep()
        
        print("\n" + "="*60)
        print("  Visualization Complete!")
        print("="*60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Formation Gazebo Visualizer')
    parser.add_argument('--csv', 
        default='/home/demo/.openclaw/workspace/dmpc_formation/4uav_trajectory.csv',
        help='Path to trajectory CSV')
    parser.add_argument('--rate', type=float, default=5.0,
        help='Visualization rate (Hz)')
    args = parser.parse_args()
    
    try:
        viz = FormationVisualizer(args.csv, args.rate)
        viz.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
