cmake_minimum_required(VERSION 3.0.2)
project(dmpc_formation_bridge)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs geometry_msgs mavros_msgs nav_msgs trajectory_msgs message_generation)

add_message_files(
  FILES
  FormationState.msg
)

generate_messages()
catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs mavros_msgs nav_msgs message_runtime
)

## Install Python scripts
catkin_install_python(PROGRAMS
  src/dmpc_bridge_node.py
  src/trajectory_playback.py
  src/formation_gazebo_visualizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(FILES
  launch/formation_bridge.launch
  launch/px4_formation_simulation.launch
  launch/gazebo_formation.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Install URDF/SDF
install(FILES
  sdf/quadrotor_basic.sdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/sdf
)
