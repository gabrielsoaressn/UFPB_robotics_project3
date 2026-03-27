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
                'camera_hfov': 1.2,
                'publish_debug_image': False,
                'wall_max_proj': 4.0,
                'wall_dedupe_dist': 2.4,
                'wall_dedupe_dist_back': 5.5,
                'stable_color_frames': 4,
                'color_min_area_fraction': 0.065,
                'max_wall_distance_to_count': 1.65,
                'centroid_max_abs': 0.32,
                'max_lidar_bearing_deg': 20.0,
                'max_front_range_for_count': 2.05,
                'front_window_deg': 14.0,
                'max_range_vs_front_disagree': 0.75,
            }],
        ),
    ])
