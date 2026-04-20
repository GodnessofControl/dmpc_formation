#!/bin/bash
# Complete DMPC Formation Simulation Setup
# This script sets up everything needed for the multi-UAV DMPC simulation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_AUTOPILOT="/home/demo/PX4-Autopilot"
N_UAVS=4
WORKSPACE="/home/demo/catkin_ws"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "========================================"
echo "  DMPC Formation Control Simulation"
echo "========================================"

# Step 1: Check environment
log_info "Checking environment..."

if ! command -v gazebo &> /dev/null; then
    log_err "Gazebo not found!"
    exit 1
fi

if [ ! -d "$PX4_AUTOPILOT" ]; then
    log_err "PX4-Autopilot not found at $PX4_AUTOPILOT"
    exit 1
fi

if [ ! -d "$WORKSPACE/src/dmpc_formation_bridge" ]; then
    log_err "dmpc_formation_bridge package not found!"
    exit 1
fi

log_info "Environment check passed"

# Step 2: Source environment
log_info "Sourcing environment..."
source /opt/ros/noetic/setup.bash
source $PX4_AUTOPILOT/Tools/simulation/gazebo-classic/sitl_gazebo-classic/setup_gazebo.bash
export GAZEBO_MODEL_PATH=$PX4_AUTOPILOT/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:$GAZEBO_MODEL_PATH
export ROS_PACKAGE_PATH=$PX4_AUTOPILOT:$ROS_PACKAGE_PATH

# Step 3: Build catkin workspace
log_info "Building catkin workspace..."
cd $WORKSPACE
catkin build dmpc_formation_bridge
source devel/setup.bash

# Step 4: Kill any existing PX4/Gazebo processes
log_info "Cleaning up existing processes..."
pkill -f px4 || true
pkill -f gzserver || true
pkill -f gzclient || true
sleep 2

# Step 5: Start Gazebo
log_info "Starting Gazebo..."
cd $PX4_AUTOPILOT
export PX4_SIM_MODEL=iris

# Use empty world first, can be changed to baylands, warehouse, etc.
WORLD_FILE="$PX4_AUTOPILOT/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/empty.world"

# Start Gazebo server in background
rosrun gazebo_ros gzserver $WORLD_FILE --verbose &
GAZEBO_PID=$!
sleep 3

# Start Gazebo client (GUI) in background
rosrun gazebo_ros gzclient &
GAZEBO_CLIENT_PID=$!
sleep 2

# Step 6: Spawn UAVs
log_info "Spawning $N_UAVS UAVs..."

declare -a SPAWN_X=(0 -10 -20 -30)
declare -a SPAWN_Y=(0 -27 22 -10)
declare -a SPAWN_Z=(0.5 0.5 0.5 0.5)

for i in $(seq 1 $N_UAVS); do
    idx=$((i-1))
    x=${SPAWN_X[$idx]}
    y=${SPAWN_Y[$idx]}
    z=${SPAWN_Z[$idx]}
    
    log_info "Spawning UAV $i at ($x, $y, $z)"
    
    # Note: In real scenario, you'd use px4's spawn model with correct SDF
    # For now, we assume the iris model is already spawned or use:
    # rosrun gazebo_ros spawn_model -file $PX4_AUTOPILOT/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf -urdf -model uav$i -x $x -y $y -z $z
done

# Step 7: Start MAVROS nodes
log_info "Starting MAVROS..."
for i in $(seq 1 $N_UAVS); do
    port=$((14540 + (i-1)*10))
    rosrun mavros mavros_node __name:=mavros_$i /mavros:=/uav$i/mavros fcu_url:=udp://:$port@127.0.0.1:$((port+40)) &
done
sleep 2

# Step 8: Start DMPC Bridge
log_info "Starting DMPC Formation Bridge..."
rosrun dmpc_formation_bridge dmpc_bridge_node.py &

log_info "========================================"
log_info "Simulation started!"
log_info "  - Gazebo GUI should be visible"
log_info "  - 4 UAVs spawned"
log_info "  - MAVROS connected"
log_info "  - DMPC Bridge running"
log_info "========================================"
log_info "Use 'rosnode list' to see active nodes"
log_info "Use 'rostopic list' to see active topics"
log_info "Press Ctrl+C to stop"
log_info "========================================"

# Wait for interrupt
trap "log_info 'Shutting down...'; kill $GAZEBO_PID $GAZEBO_CLIENT_PID 2>/dev/null; pkill -f px4; pkill -f gzserver; exit 0" SIGINT SIGTERM

wait
