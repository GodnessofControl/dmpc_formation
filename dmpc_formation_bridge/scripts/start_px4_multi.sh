#!/bin/bash
# Start PX4 SITL with multiple UAVs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_AUTOPILOT="/home/demo/PX4-Autopilot"
N_UAVS=${N_UAVS:-4}
WORLD=${WORLD:-empty}

echo "Starting $N_UAVS PX4 SITL instances with world: $WORLD"

# Source ROS and PX4
source /opt/ros/noetic/setup.bash
source $PX4_AUTOPILOT/Tools/simulation/gazebo-classic/gazebo_sitl_multiple_run.sh

# Initial positions for 4 UAVs (x, y, z in ENU frame for Gazebo)
# These correspond to the formation offsets
declare -a POSITIONS=(
    "0,0,0.5,0,0,0"      # UAV1 (Leader)
    "-10,-27,0.5,0,0,0"  # UAV2
    "-20,22,0.5,0,0,0"  # UAV3
    "-30,-10,0.5,0,0,0" # UAV4
)

# MAVLINK ports (UDP)
declare -a PORTS=(
    "14540"  # UAV1
    "14550"  # UAV2
    "14560"  # UAV3
    "14570"  # UAV4
)

# Start PX4 instances
for i in $(seq 1 $N_UAVS); do
    instance=$((i-1))
    pos=(${POSITIONS[$instance]})
    port=${PORTS[$instance]}
    
    echo "Starting UAV $i on port $port at position ${pos[*]}"
    
    (
        cd $PX4_AUTOPILOT
        export MAV_SYS_ID=$i
        export PX4_SIM_MODEL=iris
        export PX4_GZ_MODEL_POS="x=${pos[0]},y=${pos[1]},z=${pos[2]},roll=${pos[3]},pitch=${pos[4]},yaw=${pos[5]}"
        
        # Use PX4's multi-UAV script
        ./Tools/simulation/gazebo-classic/sitl_multiple_run.sh -n $i -w $WORLD -t iris &
    )
    
    sleep 1
done

echo "All PX4 instances started"
wait
