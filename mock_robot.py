import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import time
import math

class MockRobot(Node):
    def __init__(self):
        super().__init__('mock_robot_publisher')
        
        # === 1. 定义发布者 ===
        # 机械臂状态
        self.pub_arm = self.create_publisher(JointState, '/franka/joint_states', 10)
        # 灵巧手状态
        self.pub_hand = self.create_publisher(JointState, '/leap/joint_states', 10)
        # 人工干预
        self.pub_human = self.create_publisher(JointState, '/human/leap_command', 10)
        
        # === 📸 关键修改：发布 Policy 需要的所有相机 Topic ===
        self.pub_cam_wrist = self.create_publisher(Image, '/camera/wrist/image_raw', 10)
        self.pub_cam_chest = self.create_publisher(Image, '/camera/cam_chest/image_raw', 10)
        self.pub_cam_head  = self.create_publisher(Image, '/camera/cam_head/image_raw', 10)

        # Spacenav
        self.pub_spacenav = self.create_publisher(PoseStamped, '/spacenav/pose', 10)

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.033, self.timer_callback) # 30Hz
        self.start_time = time.time()
        

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        t = time.time() - self.start_time

        # === 2. 造假：机械臂 ===
        msg_arm = JointState()
        msg_arm.header.stamp = now
        msg_arm.name = [f'fr3_joint{i+1}' for i in range(7)]
        msg_arm.position = [math.sin(t + i) * 0.5 for i in range(7)] 
        self.pub_arm.publish(msg_arm)

        # === 3. 造假：灵巧手 ===
        msg_hand = JointState()
        msg_hand.header.stamp = now
        msg_hand.name = [f'joint_{i}' for i in range(19)]
        msg_hand.position = [math.cos(t) * 0.5 + 0.5] * 19 
        self.pub_hand.publish(msg_hand)
        self.pub_human.publish(msg_hand)

        # === 4. 造假：生成一张通用噪点图 ===
        # 224x224 适配你的 ResNet
        random_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        msg_img = self.bridge.cv2_to_imgmsg(random_image, encoding="bgr8")
        msg_img.header.stamp = now
        msg_img.header.frame_id = "camera_optical_frame"

        # === 📸 分发给三个相机话题 ===
        # 1. 腕部
        self.pub_cam_wrist.publish(msg_img)
        
        # 2. 胸部 (Cam Chest)
        self.pub_cam_chest.publish(msg_img)
        
        # 3. 头部 (Cam Head)
        self.pub_cam_head.publish(msg_img)

        # === 5. Spacenav ===
        msg_pose = PoseStamped()
        msg_pose.header.stamp = now
        self.pub_spacenav.publish(msg_pose)

def main(args=None):
    rclpy.init(args=args)
    node = MockRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()