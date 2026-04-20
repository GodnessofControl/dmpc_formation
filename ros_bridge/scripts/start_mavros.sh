#!/bin/bash
# Start MAVROS nodes for multiple UAVs

set -e

N_UAVS=${N_UAVS:-4}
BASE_PORT=${BASE_PORT:-14540}

echo "Starting MAVROS for $N_UAVS UAVs"

for i in $(seq 1 $N_UAVS); do
    uav_ns="uav$i"
    port=$((BASE_PORT + (i-1)*10))
    fcu_url="udp://:$port@127.0.0.1:$((port+40))"
    
    echo "Starting MAVROS for $uav_ns at $fcu_url"
    
    rosrun mavros mavros_node \
        /mavros:=$uav_ns/mavros \
        __name:=mavros_$i \
        fcu_url:=$fcu_url \
        gcs_url:= \
        target_system_id:=$i \
        target_component_id:=$((i*10)) \
        fcu_protocol:=v2.0 \
        &
        
    sleep 0.5
done

echo "All MAVROS nodes started"
wait
