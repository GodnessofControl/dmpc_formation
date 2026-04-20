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
