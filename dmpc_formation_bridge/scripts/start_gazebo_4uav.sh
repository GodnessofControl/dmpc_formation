#!/bin/bash
# Clean start of Gazebo with 4 UAVs

export DISPLAY=:0
export GAZEBO_PLUGIN_PATH=/home/demo/PX4-Autopilot/build/px4_sitl_default/build_gazebo-classic:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=/home/demo/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:$GAZEBO_MODEL_PATH
export LD_LIBRARY_PATH=/home/demo/PX4-Autopilot/build/px4_sitl_default/build_gazebo-classic:$LD_LIBRARY_PATH
export PYTHONPATH=/home/demo/.openclaw/workspace:$PYTHONPATH

PX4="/home/demo/PX4-Autopilot"
WORKSPACE="/home/demo/catkin_ws"

# Source ROS
source /opt/ros/noetic/setup.bash
source $WORKSPACE/devel/setup.bash

echo "[1/5] Starting roscore..."
nohup roscore > /tmp/roscore.log 2>&1 &
sleep 2

echo "[2/5] Starting gzserver..."
nohup bash -c "rosrun gazebo_ros gzserver $PX4/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/empty.world --verbose" > /tmp/gzserver.log 2>&1 &
sleep 4

echo "[3/5] Starting gzclient (GUI)..."
nohup bash -c "rosrun gazebo_ros gzclient" > /tmp/gzclient.log 2>&1 &
sleep 3

echo "[4/5] Spawning 4 UAVs..."
for i in 1 2 3 4; do
    case $i in
        1) x=0; y=0; z=0.5 ;;
        2) x=-10; y=-27; z=0.5 ;;
        3) x=-20; y=22; z=0.5 ;;
        4) x=-30; y=-10; z=0.5 ;;
    esac
    
    echo "  Spawning UAV$i at ($x, $y, $z)"
    rosrun gazebo_ros spawn_model \
        -file "$PX4/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf" \
        -sdf -model uav$i \
        -x $x -y $y -z $z -R 0 -P 0 -Y 0 &
    sleep 1
done

echo "[5/5] Starting trajectory playback..."
nohup /usr/bin/python3 $WORKSPACE/src/dmpc_formation_bridge/src/trajectory_playback.py --rate 2 > /tmp/playback.log 2>&1 &

echo ""
echo "========================================="
echo "  All systems started!"
echo "========================================="
echo "  gzserver: $(ps aux | grep gzserver | grep -v grep | wc -l) processes"
echo "  gzclient: $(ps aux | grep gzclient | grep -v grep | wc -l) processes"
echo "  UAVs spawned: 4"
echo "  Trajectory playback: running"
echo "========================================="
echo ""
echo "To view remotely, set up VNC or use:"
echo "  ssh -X user@$HOSTNAME"
echo ""
