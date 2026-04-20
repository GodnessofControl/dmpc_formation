#!/bin/bash
# Start Gazebo with 4 UAVs and visualize DMPC formation trajectory

set -e
export GAZEBO_MODEL_PATH=/home/demo/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:$GAZEBO_MODEL_PATH
export PYTHONPATH=/home/demo/.openclaw/workspace:$PYTHONPATH

WORKSPACE="/home/demo/catkin_ws"
PX4="/home/demo/PX4-Autopilot"
WORLD_FILE="$PX4/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/empty.world"

echo "========================================="
echo "  4-UAV Formation Visualizer"
echo "========================================="

# Kill any existing processes
pkill -f gzserver 2>/dev/null || true
pkill -f gzclient 2>/dev/null || true
sleep 1

# Source environment
source /opt/ros/noetic/setup.bash
source $PX4/Tools/simulation/gazebo-classic/setup_gazebo.bash $PX4 $PX4/build
source $WORKSPACE/devel/setup.bash

# Start Gazebo server (headless)
echo "[1/4] Starting Gazebo server..."
rosrun gazebo_ros gzserver $WORLD_FILE --verbose &
sleep 3

# Start Gazebo client (GUI)
echo "[2/4] Starting Gazebo client..."
rosrun gazebo_ros gzclient &
sleep 2

# Spawn 4 UAVs
echo "[3/4] Spawning 4 UAVs..."
SPAWN_ARGS=(
    "0,0,0.5"      # UAV1 (Leader) - ENU
    "-10,-27,0.5"  # UAV2
    "-20,22,0.5"   # UAV3
    "-30,-10,0.5"  # UAV4
)

for i in $(seq 1 4); do
    idx=$((i-1))
    pos=(${SPAWN_ARGS[$idx]})
   IFS=',' read -r x y z <<< "$pos"
    
    echo "  Spawning UAV$i at x=$x y=$y z=$z"
    
    rosrun gazebo_ros spawn_model \
        -file "$PX4/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf" \
        -sdf \
        -model uav$i \
        -x $x -y $y -z $z \
        -R 0 -P 0 -Y 0 &
    
    sleep 1
done

echo "[4/4] Starting trajectory playback..."
echo ""
echo "========================================="
echo "  UAVs Spawned - Starting Playback"
echo "========================================="
echo "  Open another terminal and run:"
echo "    source /opt/ros/noetic/setup.bash"
echo "    source $WORKSPACE/devel/setup.bash"
echo "    rosrun dmpc_formation_bridge trajectory_playback.py"
echo ""
echo "  Or in RViz, add MarkerArray topic: /formation/markers"
echo "========================================="

# Keep running
wait
