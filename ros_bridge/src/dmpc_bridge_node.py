#!/usr/bin/env python3
"""
DMPC Formation Controller Bridge for PX4/Gazebo

Connects the DMPC formation control algorithm with the PX4/Gazebo simulation.
Listens to UAV odometry and publishes velocity commands.

Usage:
    # Start Gazebo with 4 UAVs first, then:
    roslaunch dmpc_formation_bridge formation_bridge.launch
    
    # Or run node directly:
    rosrun dmpc_formation_bridge dmpc_bridge_node.py
"""

import rospy
import numpy as np
import threading
import time
from collections import deque

from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State, ExtendedState
from mavros_msgs.srv import SetMode, CommandBool
from std_msgs.msg import Header, Float64MultiArray


class UAVStateBuffer:
    """Thread-safe buffer for UAV state."""
    def __init__(self, uav_id):
        self.uav_id = uav_id
        self.lock = threading.Lock()
        self._position = np.array([0.0, 0.0, 0.0])  # x, y, z (ENU frame)
        self._velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        self._yaw = 0.0
        self._pitch = 0.0
        self._roll = 0.0
        self._has_data = False
        self._stamp = rospy.Time.now()

    def update_odom(self, pos, vel, yaw, pitch, roll):
        with self.lock:
            self._position = pos.copy()
            self._velocity = vel.copy()
            self._yaw = yaw
            self._pitch = pitch
            self._roll = roll
            self._has_data = True
            self._stamp = rospy.Time.now()

    def get_state(self):
        with self.lock:
            return {
                'pos': self._position.copy(),
                'vel': self._velocity.copy(),
                'yaw': self._yaw,
                'pitch': self._pitch,
                'roll': self._roll,
                'has_data': self._has_data,
                'stamp': self._stamp
            }


class DMPCBridgeNode:
    """
    Bridge node between DMPC controller and PX4/Gazebo simulation.
    
    Handles:
    - Subscribing to odometry from 4 UAVs
    - Converting between Gazebo (ENU) and DMPC (NED) frames
    - Running DMPC at specified rate
    - Publishing velocity commands to each UAV
    """

    def __init__(self):
        rospy.init_node('dmpc_formation_bridge', anonymous=True)

        # Parameters
        self.n_uavs = rospy.get_param('~n_uavs', 4)
        self.mpc_rate = rospy.get_param('~mpc_rate', 1.0)  # Hz
        self.uav_namespace = rospy.get_param('~uav_namespace', 'uav')
        self.use_mavros = rospy.get_param('~use_mavros', True)
        self.frame_conversion = rospy.get_param('~frame_conversion', 'ENUtoNED')

        # State buffers for each UAV
        self.uav_buffers = {
            i: UAVStateBuffer(i) for i in range(1, self.n_uavs + 1)
        }

        # DMPC controller (lazy import, only when needed)
        self.dmpc_controller = None
        self.dmpc_ready = False
        self.dmpc_lock = threading.Lock()

        # Publishers and subscribers
        self.cmd_pubs = {}
        self._setup_topics()

        # Network simulator state
        self.sigma_L = rospy.get_param('~sigma_L', 0.15)
        self.sigma_D = rospy.get_param('~sigma_D', 0.20)

        # Formation configuration
        self.formation_config = self._load_formation_config()

        # Initialize DMPC
        self._init_dmpc_controller()

        # Control timer
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.mpc_rate),
            self._mpc_callback
        )

        rospy.loginfo(f"DMPC Bridge initialized: {self.n_uavs} UAVs at {self.mpc_rate} Hz")
        rospy.loginfo(f"Frame conversion: {self.frame_conversion}")

    def _setup_topics(self):
        """Setup subscribers and publishers for each UAV."""
        for i in range(1, self.n_uavs + 1):
            ns = f"{self.uav_namespace}{i}"

            if self.use_mavros:
                # Subscribe to MAVROS state/odom
                odom_sub = rospy.Subscriber(
                    f'/{ns}/mavros/local_position/odom',
                    Odometry,
                    self._odom_callback,
                    callback_args=(i, ns)
                )

                # Publish to MAVROS setpoint
                cmd_pub = rospy.Publisher(
                    f'/{ns}/mavros/setpoint_velocity/cmd_vel_unstamped',
                    Twist,
                    queue_size=1
                )
            else:
                # Direct Gazebo odometry
                odom_sub = rospy.Subscriber(
                    f'/{ns}/ground_truth/state',
                    Odometry,
                    self._odom_callback,
                    callback_args=(i, ns)
                )

                cmd_pub = rospy.Publisher(
                    f'/{ns}/cmd_vel',
                    Twist,
                    queue_size=1
                )

            self.cmd_pubs[i] = cmd_pub

            rospy.loginfo(f"UAV {i}: subscribed to {ns} odom, cmd -> {ns}/cmd_vel")

    def _odom_callback(self, msg, args):
        """Process odometry message from a UAV."""
        uav_id, ns = args

        # Extract position (ENU frame in Gazebo/MAVROS)
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # Extract velocity
        vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

        # Extract orientation (quaternion -> Euler)
        q = msg.pose.pose.orientation
        roll, pitch, yaw = self._quaternion_to_euler(q.x, q.y, q.z, q.w)

        # Frame conversion: ENU -> NED for DMPC
        if self.frame_conversion == 'ENUtoNED':
            pos = self._enu_to_ned(pos)
            vel = self._enu_to_ned(vel)
            # Yaw: ENU 0=E, NED 0=N, so rotate by -90 deg
            yaw = yaw - np.pi/2

        self.uav_buffers[uav_id].update_odom(pos, vel, yaw, pitch, roll)

    def _enu_to_ned(self, vec):
        """Convert ENU coordinates to NED."""
        # ENU: x=East, y=North, z=Up
        # NED: x=North, y=East, z=Down
        return np.array([vec[1], vec[0], -vec[2]])

    def _ned_to_enu(self, vec):
        """Convert NED coordinates to ENU."""
        return np.array([vec[1], vec[0], -vec[2]])

    def _quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _load_formation_config(self):
        """Load formation configuration from parameter server."""
        # Default formation for 4 UAVs
        # Formation offsets relative to leader (dr01, dr12, dr13, dr14)
        return {
            'dr01': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # Leader at origin
            'dr12': np.array([
                rospy.get_param('~formation_dr12_x', 10.0),
                rospy.get_param('~formation_dr12_y', 27.0),
                rospy.get_param('~formation_dr12_z', 0.0),
                0.0, 0.0
            ]),
            'dr13': np.array([
                rospy.get_param('~formation_dr13_x', 20.0),
                rospy.get_param('~formation_dr13_y', -22.0),
                rospy.get_param('~formation_dr13_z', 0.0),
                0.0, 0.0
            ]),
            'dr14': np.array([
                rospy.get_param('~formation_dr14_x', 30.0),
                rospy.get_param('~formation_dr14_y', 10.0),
                rospy.get_param('~formation_dr14_z', 0.0),
                0.0, 0.0
            ]),
        }

    def _init_dmpc_controller(self):
        """Initialize the DMPC controller from the formation package."""
        try:
            # Add the dmpc_formation package to path
            import sys
            dmpc_path = '/home/demo/.openclaw/workspace/dmpc_formation'
            if dmpc_path not in sys.path:
                sys.path.insert(0, dmpc_path)

            from quadrotor_dmpc.controller import QuadrotorDMPC, MPCConfig
            from quadrotor_dmpc.models.formation import FormationConfig
            from quadrotor_dmpc.models.uav_model import UAVState
            from quadrotor_dmpc.utils.trajectory import generate_sinusoidal_3d
            from quadrotor_dmpc.utils.network import NetworkSimulator

            # Generate reference trajectory
            t = np.arange(0, 300, 1.0)
            xr, yr, hr, thelr, phir = generate_sinusoidal_3d(
                t, x0=0, y0=0, h0=10,
                Ax=50, Ay=30, Ah=10,
                wx=0.1, wy=0.15, wh=0.05
            )

            ref_traj = {
                'xr': xr, 'yr': yr, 'hr': hr,
                'theltr': thelr, 'phir': phir
            }

            # Create formation config matching our parameterization
            formation = FormationConfig(
                dr01=self.formation_config['dr01'],
                dr12=self.formation_config['dr12'],
                dr13=self.formation_config['dr13'],
                dr14=self.formation_config['dr14']
            )

            # Create network simulator
            network = NetworkSimulator.create_random(
                300, sigma_L=self.sigma_L, sigma_D=self.sigma_D, seed=42
            )

            # Create MPC config
            config = MPCConfig(
                Np=8,
                Nc=8,
                N_steps=300,
                sigma_L=self.sigma_L,
                sigma_D=self.sigma_D
            )

            # Initial states - use current positions from Gazebo
            initial_states = {}
            for uav_id in range(1, self.n_uavs + 1):
                buf = self.uav_buffers[uav_id]
                state = buf.get_state()
                if state['has_data']:
                    pos = state['pos']
                    initial_states[uav_id] = UAVState(
                        x=pos[0], y=pos[1], h=pos[2],
                        theta=state['yaw'], phi=state['pitch']
                    )
                else:
                    # Default initial positions if no odom yet
                    initial_states[uav_id] = UAVState(
                        x=0, y=0, h=10, theta=0, phi=0
                    )

            self.dmpc_controller = QuadrotorDMPC(
                config=config,
                formation=formation,
                initial_states=initial_states,
                reference_trajectory=ref_traj,
                network_sim=network,
                use_lmi_gains=False
            )

            self.dmpc_ready = True
            rospy.loginfo("DMPC controller initialized successfully")

        except Exception as e:
            rospy.logerr(f"Failed to initialize DMPC controller: {e}")
            import traceback
            traceback.print_exc()

    def _mpc_callback(self, event):
        """Main MPC control loop callback."""
        if not self.dmpc_ready:
            return

        try:
            with self.dmpc_lock:
                # Update states from buffers
                for uav_id in range(1, self.n_uavs + 1):
                    buf = self.uav_buffers[uav_id]
                    state = buf.get_state()
                    if state['has_data']:
                        pos = state['pos']
                        self.dmpc_controller.uavs[uav_id].state.x = pos[0]
                        self.dmpc_controller.uavs[uav_id].state.y = pos[1]
                        self.dmpc_controller.uavs[uav_id].state.h = pos[2]
                        self.dmpc_controller.uavs[uav_id].state.theta = state['yaw']
                        self.dmpc_controller.uavs[uav_id].state.phi = state['pitch']

                # Run one MPC step
                controls = self.dmpc_controller.step(self.dmpc_controller.config.N_steps - 1)

                # Publish commands to each UAV
                for uav_id in range(1, self.n_uavs + 1):
                    if uav_id in controls:
                        ctrl = controls[uav_id]
                        self._publish_cmd(uav_id, ctrl.v, ctrl.omega, ctrl.zeta)

        except Exception as e:
            rospy.logerr(f"MPC callback error: {e}")
            import traceback
            traceback.print_exc()

    def _publish_cmd(self, uav_id, v, omega, zeta):
        """Publish velocity command to a UAV.

        Args:
            uav_id: UAV identifier
            v: Forward velocity (m/s) in NED frame
            omega: Yaw rate (rad/s)
            zeta: Pitch rate (rad/s)
        """
        cmd = Twist()

        # Convert NED velocity to ENU for Gazebo/MAVROS
        # NED: x=North(+v), y=East
        # ENU: x=East, y=North, z=Up
        enu_v = v  # Forward velocity maps to x in body frame

        # Body frame velocity components
        cmd.linear.x = v  # Forward
        cmd.linear.y = 0.0
        cmd.linear.z = zeta  # Vertical velocity component

        # Angular velocities
        cmd.angular.x = 0.0  # Roll rate (not used)
        cmd.angular.y = zeta  # Pitch rate
        cmd.angular.z = omega  # Yaw rate

        self.cmd_pubs[uav_id].publish(cmd)

    def run(self):
        """Run the bridge node."""
        rospy.loginfo("DMPC Formation Bridge running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = DMPCBridgeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
