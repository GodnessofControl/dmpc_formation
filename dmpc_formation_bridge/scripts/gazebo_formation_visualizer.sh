#!/bin/bash
# Complete Gazebo + Formation Visualization Setup
# Spawns 4 UAVs and moves them according to DMPC trajectory

set -e
export DISPLAY=:0

echo "========================================="
echo "  DMPC Formation - Gazebo Visualizer"
echo "========================================="

# Setup paths
WORKSPACE="/home/demo/catkin_ws"
SDF_PATH="$WORKSPACE/src/dmpc_formation_bridge/sdf"
CSV_PATH="/home/demo/.openclaw/workspace/dmpc_formation/4uav_trajectory.csv"

# Source environment
source /opt/ros/noetic/setup.bash
source $WORKSPACE/devel/setup.bash

# Kill existing processes
echo "[0/6] Cleaning up..."
pkill -9 -f gzserver 2>/dev/null || true
pkill -9 -f gzclient 2>/dev/null || true
pkill -9 -f spawn_model 2>/dev/null || true
sleep 1

# Start roscore
echo "[1/6] Starting roscore..."
nohup roscore > /tmp/roscore.log 2>&1 &
sleep 2

# Start Gazebo server
echo "[2/6] Starting Gazebo server..."
export GAZEBO_PLUGIN_PATH=/home/demo/PX4-Autopilot/build/px4_sitl_default/build_gazebo-classic:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=$SDF_PATH:$GAZEBO_MODEL_PATH
export LD_LIBRARY_PATH=/home/demo/PX4-Autopilot/build/px4_sitl_default/build_gazebo-classic:$LD_LIBRARY_PATH

nohup gzserver /usr/share/gazebo-11/worlds/empty.world > /tmp/gzserver.log 2>&1 &
sleep 4

# Start Gazebo client
echo "[3/6] Starting Gazebo GUI..."
nohup gzclient > /tmp/gzclient.log 2>&1 &
sleep 3

# Check if Gazebo is running
if ! pgrep -x gzserver > /dev/null; then
    echo "ERROR: gzserver failed to start!"
    cat /tmp/gzserver.log
    exit 1
fi
echo "  Gazebo running (PID: $(pgrep -x gzserver))"

# Spawn 4 UAVs at initial positions
echo "[4/6] Spawning 4 UAVs..."
declare -a SPAWN_POS=(
    "0;0;0.5"      # UAV1
    "-10;-27;0.5"  # UAV2
    "-20;22;0.5"   # UAV3
    "-30;-10;0.5"  # UAV4
)

for i in 1 2 3 4; do
    IFS=';' read -r x y z <<< "${SPAWN_POS[$i-1]}"
    echo "  Spawning UAV$i at ($x, $y, $z)"
    
    # Use simple quadrotor SDF
    rosrun gazebo_ros spawn_model \
        -file "$SDF_PATH/quadrotor_basic.sdf" \
        -sdf \
        -model uav$i \
        -x $x -y $y -z $z \
        -Y 0 &
    sleep 0.5
done

sleep 3

# Check spawned models
echo "[5/6] Checking models..."
timeout 5 rosservice call /gazebo/list_models 2>/dev/null || echo "  (service unavailable)"

# Start formation visualizer
echo "[6/6] Starting formation visualizer..."
echo ""
echo "========================================="
echo "  Starting Formation Animation!"
echo "========================================="
echo ""

nohup /usr/bin/python3 $WORKSPACE/src/dmpc_formation_bridge/src/formation_gazebo_visualizer.py \
    --csv $CSV_PATH \
    --rate 5 > /tmp/formation_viz.log 2>&1 &

echo "  Formation visualizer started (PID: $!)"
echo ""
echo "========================================="
echo "  All systems running!"
echo "========================================="
echo "  - Gazebo server: $(pgrep -x gzserver)"
echo "  - Gazebo client: $(pgrep -x gzclient)"
echo "  - 4 UAVs spawned"
echo "  - Formation visualizer running"
echo ""
echo "  Check Gazebo window to see formation!"
echo "========================================="
echo ""
echo "To watch logs:"
echo "  tail -f /tmp/formation_viz.log"
echo ""
