#!/bin/bash
# Kill all ROS 2 / Gazebo processes from simulation tests

pkill -f gzserver
pkill -f gzclient
pkill -f robot_state_publisher
pkill -f joint_state_publisher
pkill -f ekf_node
pkill -f maze_navigator
pkill -f color_wall_counter
pkill -f "ros2 launch"

echo "All simulation processes killed."
