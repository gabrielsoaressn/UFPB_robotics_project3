colcon build --packages-select jetauto_maze_solver
source install/setup.bash
ros2 launch jetauto_maze_solver start_all.launch.py
