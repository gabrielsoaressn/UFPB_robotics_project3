"""Launch file for the JetAuto Maze Solver — starts navigation + color detection."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='jetauto_maze_solver',
            executable='maze_navigator',
            name='maze_navigator',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='jetauto_maze_solver',
            executable='color_wall_counter',
            name='color_wall_counter',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                # Match robotica-main color_detector_node pipeline (strict_count_gates:=false)
                'strict_count_gates': False,
                'min_color_ratio': 0.06,
                # UFPB depth_camera.urdf.xacro — robotica uses 1.3962634 (80°); set that if you use their URDF
                'camera_hfov': 1.2,
                'wall_dedupe_dist': 2.0,
                'wall_dedupe_dist_back': 4.0,
                'wall_max_proj': 1.925,
                'publish_debug_image': False,
                # If counts duplicate on same wall from both faces, raise wall_dedupe_dist_back (e.g. 5.0–5.5)
                # If false positives when passing walls, set strict_count_gates:=true and tune strict params
            }],
        ),
    ])
