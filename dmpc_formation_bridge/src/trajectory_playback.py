#!/usr/bin/env python3
"""
Trajectory Playback Node
Reads 4-UAV trajectory from CSV and plays it back as ROS markers.
Visualize in RViz: add MarkerArray display pointing to /formation/markers
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import tf.transformations as T


class TrajectoryPlayback:
    def __init__(self, csv_path, rate=1.0):
        rospy.init_node('trajectory_playback', anonymous=True)

        self.rate = rate
        self.data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        self.n_steps = len(self.data)

        self.uav_names = ['UAV1', 'UAV2', 'UAV3', 'UAV4']
        self.uav_colors = [
            ColorRGBA(0.50, 0.50, 0.89, 1.0),  # blue
            ColorRGBA(0.48, 0.83, 0.39, 1.0),  # green
            ColorRGBA(0.02, 0.60, 0.79, 1.0),  # cyan
            ColorRGBA(0.94, 0.34, 0.34, 1.0),  # red
        ]

        # Publisher for markers
        self.marker_pub = rospy.Publisher('/formation/markers', MarkerArray, queue_size=1)

        # Publishers for each UAV pose (for following camera)
        self.pose_pubs = {}
        for i in range(4):
            pub = rospy.Publisher(f'/uav{i+1}/trajectory_pose', PoseStamped, queue_size=1)
            self.pose_pubs[i] = pub

        rospy.loginfo(f"Loaded {self.n_steps} steps from {csv_path}")
        rospy.loginfo("Columns: u1x,u1y,u1h,u1v, u2x,u2y,u2h,u2v, u3x,u3y,u3h,u3v, u4x,u4y,u4h,u4v")

    def quat_from_euler(self, roll, pitch, yaw):
        q = T.quaternion_from_euler(roll, pitch, yaw)
        return Quaternion(*q)

    def run(self):
        """Playback the trajectory."""
        r = rospy.Rate(self.rate)

        for step in range(self.n_steps):
            if rospy.is_shutdown():
                break

            row = self.data[step]
            markers = MarkerArray()

            for uav_idx in range(4):
                base = uav_idx * 4
                x, y, h = row[base], row[base+1], row[base+2]

                # Publish pose (convert NED -> ENU for Gazebo)
                # NED: x=North, y=East, z=Down
                # ENU: x=East, y=North, z=Up
                pose_enu = PoseStamped()
                pose_enu.header = Header(frame_id='map', stamp=rospy.Time.now())
                pose_enu.pose.position = Point(x=y, y=x, z=h)  # NED->ENU swap
                pose_enu.pose.orientation = Quaternion(0, 0, 0, 1)
                self.pose_pubs[uav_idx].publish(pose_enu)

                # Create marker (sphere for current position)
                marker = Marker()
                marker.header = Header(frame_id='map', stamp=rospy.Time.now())
                marker.ns = f'uav{uav_idx+1}'
                marker.id = step
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = Point(x=y, y=x, z=h)  # NED->ENU swap
                marker.pose.orientation = Quaternion(0, 0, 0, 1)
                marker.scale.x = 2.0
                marker.scale.y = 2.0
                marker.scale.z = 2.0
                marker.color = self.uav_colors[uav_idx]
                markers.markers.append(marker)

                # Trail marker (smaller spheres showing path so far)
                if step > 0:
                    trail_marker = Marker()
                    trail_marker.header = Header(frame_id='map', stamp=rospy.Time.now())
                    trail_marker.ns = f'uav{uav_idx+1}_trail'
                    trail_marker.id = uav_idx * 10000 + step
                    trail_marker.type = Marker.LINE_STRIP
                    trail_marker.action = Marker.ADD
                    trail_marker.scale.x = 0.3  # line width
                    trail_marker.color = self.uav_colors[uav_idx]
                    trail_marker.color.a = 0.3

                    # Build path points
                    for t in range(max(0, step-50), step+1):
                        r_row = self.data[t]
                        tx, ty, th = r_row[base], r_row[base+1], r_row[base+2]
                        p = Point(x=ty, y=tx, z=th)
                        trail_marker.points.append(p)

                    markers.markers.append(trail_marker)

            self.marker_pub.publish(markers)

            if step % 10 == 0:
                rospy.loginfo(f"Step {step}/{self.n_steps}: UAV1=({row[0]:.1f},{row[1]:.1f},{row[2]:.1f})")

            r.sleep()

        rospy.loginfo("Playback complete!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trajectory Playback for RViz')
    parser.add_argument('--csv', default='/home/demo/.openclaw/workspace/dmpc_formation/4uav_trajectory.csv',
                        help='Path to trajectory CSV')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='Playback rate (Hz)')
    args = parser.parse_args()

    try:
        pb = TrajectoryPlayback(args.csv, args.rate)
        pb.run()
    except rospy.ROSInterruptException:
        pass
