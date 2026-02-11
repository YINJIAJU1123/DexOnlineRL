from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # ==========================================
        # 1. 启动手腕相机 (D405)
        # ==========================================
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='wrist',
            name='camera',
            parameters=[{
                'serial_no': '352122271914',  # 这里的数字我已经帮你填好了
                'enable_color': True,
                'enable_depth': True,
                'publish_tf': True,
            }],
            remappings=[
                ('color/image_raw', 'image_raw'),
                ('depth/image_rect_raw', 'depth_raw'),
            ]
        ),

        # ==========================================
        # 2. 启动右侧相机 (D435)
        # ==========================================
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='right',
            name='camera',
            parameters=[{
                'serial_no': '213322070221',  # 这里的数字我已经帮你填好了
                'enable_color': True,
                'enable_depth': True,
                'publish_tf': True,
            }],
            remappings=[
                ('color/image_raw', 'image_raw'),
                ('depth/image_rect_raw', 'depth_raw'),
            ]
        ),

        # ==========================================
        # 3. 启动 RViz2
        # ==========================================
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        )
    ])