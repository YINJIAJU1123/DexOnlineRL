# ros2_bridge.py
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import threading
import numpy as np
import time
import pinocchio as pin 
import os
from std_msgs.msg import Float64MultiArray, Float32
import builtins

# 伪装 Joints 类
class Joints:
    def __init__(self, values, num_of_dofs=7):
        self.joints = np.array(values)
        self.num_of_dofs = num_of_dofs
    def __repr__(self):
        return f"Joints(dofs={self.num_of_dofs}, val={self.joints})"

class Ros2Bridge(Node):
    def __init__(self, robot_type="franka"):
        # 0. 初始化检查
        if not rclpy.ok(): rclpy.init()
        super().__init__('online_rl_bridge')

        self.robot_type = robot_type 
        print(f">>> [Ros2Bridge] 初始化开始，模式: {self.robot_type}")
        
        # === 1. 配置 ===
        self.arm_joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        #self.hand_joint_names = [f"joint_{i}" for i in range(16)]
        self.hand_joint_names = [f"joint_{i}" for i in range(16)]


        # === 2. 发布者 (Publishers) ===
        # A. 机械臂
        self.arm_pub = self.create_publisher(JointTrajectory, '/fr3_arm_controller/joint_trajectory', 10)
        
        # B. 灵巧手 (Leap Hand) - 无论什么模式都初始化，方便调试
        self.hand_pub = self.create_publisher(JointState, '/cmd_leap', 10)
        
        # C. 普通夹爪
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/franka_gripper/commands', 10) 

        # === 3. 订阅者 (Subscribers) ===
        self.bridge = CvBridge()
        
        # 状态缓存初始化
        self.latest_joints = None       # 7维
        self.latest_hand_joints = None  # 16维
        self.latest_ee_pose = None      # 7维 (pos + quat)
        self.latest_images = {}
        self.human_pose = None          # Spacenav 7维
        # self.human_gripper = np.zeros(16) # 干预信号
        self.human_gripper = np.zeros(19)

        # [重要] 订阅机械臂状态
        self.create_subscription(JointState, '/franka/joint_states', self.arm_state_cb, 10)
        self.create_subscription(JointState, '/leap/joint_states', self.hand_state_cb, 10)

        # [重要] 订阅图像
        # self.create_subscription(Image, '/camera/wrist/image_raw', lambda m: self.img_cb_wrist(m, 'wrist'), 10)
        # self.create_subscription(Image, '/camera/right/image_raw', lambda m: self.img_cb_right(m, 'right'), 10)

        self.create_subscription(Image, '/camera/wrist/image_raw', 
                                 lambda m: self.image_callback(m, 'wrist'), 10)
        
        self.create_subscription(Image, '/camera/cam_chest/image_raw', 
                                 lambda m: self.image_callback(m, 'cam_chest'), 10)
        
        self.create_subscription(Image, '/camera/cam_head/image_raw', 
                                 lambda m: self.image_callback(m, 'cam_head'), 10)

        # [重要] 订阅人工干预 (Human Intervention)
        self.create_subscription(JointState, '/human/leap_command', self.human_hand_cb, 10)
        
        # [重要] 订阅 Spacenav
        self.create_subscription(PoseStamped, '/spacenav/pose', self.spacenav_cb, 10)

        # === 4. Pinocchio 模型 ===
        urdf_path = "/home/lixin/OnlineRl/fr3.urdf" 
        if os.path.exists(urdf_path):
            try:
                self.model = pin.buildModelFromUrdf(urdf_path)
                self.data = self.model.createData()
                self.has_model = True
                print(f">>> URDF loaded from {urdf_path}")
            except Exception as e:
                print(f"Error loading URDF: {e}")
                self.has_model = False
        else:
            print(f">>> URDF not found at {urdf_path}, FK/IK will fail.")
            self.has_model = False

        # === 5. Executor 管理 (全局单例模式) ===
        self.executor = None
        
        # 优先复用全局对象
        if hasattr(builtins, "GLOBAL_ROS_EXECUTOR"):
            print(">>> [Ros2Bridge]   检测到全局 Executor，直接复用！")
            self.executor = builtins.GLOBAL_ROS_EXECUTOR
        else:
            print("⚠️ [Ros2Bridge] 未找到全局 Executor，新建 SingleThreadedExecutor...")
            try:
                self.executor = SingleThreadedExecutor()
            except Exception as e:
                 raise RuntimeError(f"❌ Executor 创建失败: {e}")

        # 将自己加入 Executor
        self.executor.add_node(self)
        
        # 启动后台线程 (如果是自己新建的 Executor 或者 全局的还没跑)
        if not getattr(self.executor, "_is_spinning_thread_started", False):
             self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
             self.spin_thread.start()
             self.executor._is_spinning_thread_started = True
             print("  [Ros2Bridge] 后台 Spin 线程已启动。")
        else:
             print("  [Ros2Bridge] 全局 Executor 已经在运行中。")

    # --- 回调函数 ---
    def image_callback(self, msg, camera_name):
        import cv2
        try:
            # 1. ROS -> OpenCV (BGR)
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # 2. OpenCV (BGR) -> RGB (RL环境通常需要 RGB)
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # 3. 存入字典 (存进去的就是 Numpy 数组了！)
            self.latest_images[camera_name] = rgb_img
            
        except Exception as e:
            print(f"Image callback error ({camera_name}): {e}")
    def human_hand_cb(self, msg):
        # 接收人工干预信号 (16维)
        if len(msg.position) == 16:
            self.human_gripper = np.array(msg.position)

    def arm_state_cb(self, msg):
        # 调试：收到第一帧数据时打印
        if self.latest_joints is None:
            print(f"  [Ros2Bridge] 收到机械臂数据! {msg.name[:2]}...")

        state_map = {n: p for n, p in zip(msg.name, msg.position)}
        self.latest_joints = np.array([state_map.get(n, 0.0) for n in self.arm_joint_names])
        
        # 更新 Pinocchio FK
        if self.has_model:
            try:
                pin.forwardKinematics(self.model, self.data, self.latest_joints)
                pin.updateFramePlacements(self.model, self.data)
                # 如果报错 Frame not found，请检查 URDF
                if self.model.existFrame("fr3_link8"):
                    fid = self.model.getFrameId("fr3_link8")
                    tf = self.data.oMf[fid]
                    quat = pin.Quaternion(tf.rotation).coeffs()
                    self.latest_ee_pose = np.concatenate((tf.translation, quat))
            except Exception:
                pass # 忽略计算错误

    def hand_state_cb(self, msg):
        if self.latest_hand_joints is None:
            print(f"  [Ros2Bridge] 收到灵巧手数据! 长度: {len(msg.position)}")
        self.latest_hand_joints = np.array(msg.position)

    def img_cb_wrist(self, msg, key):
        if key not in self.latest_images:
            print(f"  [Ros2Bridge] 收到cam_chest图像数据: {key}")
        self.latest_images[key] = msg

    def img_cb_right(self, msg, key):
        if key not in self.latest_images:
            print(f"  [Ros2Bridge] 收到cam_head图像数据: {key}")
        self.latest_images[key] = msg
        
    def spacenav_cb(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.human_pose = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w])

    # --- 核心控制函数 (模拟实时控制) ---
    def send_joints(self, arm_joints, hand_val):
        # 1. 发送手臂 (通用)
        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names
        pt = JointTrajectoryPoint()
        pt.positions = arm_joints.tolist()
        pt.time_from_start.nanosec = 33333333
        traj.points = [pt]
        self.arm_pub.publish(traj)
        
        # 2. 发送手部 (分流)
        if "frankaleap" in self.robot_type:
            # === 分支 A: Leap 灵巧手 (16 关节) ===
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.hand_joint_names
            
            # 协同映射：如果 RL 只输出了 1 个浮点数 (0~1)，映射到 16 个电机
            # 假设 0=张开, 1=握拳 (范围根据 leaphand-node.py 调整，那里是 radians)
            # 你的 leaphand-node 似乎处理了 bias，这里发 0~pi 即可
            if isinstance(hand_val, (float, int, np.float32, np.float64)):
                # 简单 Synergy: 所有手指弯曲
                # 假设最大弯曲是 1.6 rad
                cmd_rad = float(hand_val) * 1.6 
                msg.position = [cmd_rad] * 16
            else:
                # 如果 RL 输出了 16 维数组
                msg.position = hand_val.tolist()
                
            self.hand_pub.publish(msg)
            
        else:
            # === 分支 B: 普通 Franka 夹爪 ===
            # 原先的逻辑，发给夹爪控制器
            msg = Float64MultiArray()
            # 假设 0 是关，1 是开，或者是宽度
            msg.data = [float(hand_val)]
            self.gripper_pub.publish(msg)

    # --- 适配 BaseEnv 的接口 ---
    def connect(self): pass
    def generate_station_handle(self): return self
    def get_robot_handle(self): return {"fr3": self}
    
    def reach_target_joint(self, goal_joints_obj):
        # Reset 时的慢速移动
        target = goal_joints_obj.joints
        # Reset 时手通常归零
        # self.send_joints(target, np.zeros(16))
        self.send_joints(target, np.zeros(19))

        time.sleep(2.0) # 简单 sleep，因为 trajectory controller 会自己规划
        return True

    def step(self, robot_target):
        # BaseEnv 传进来的 robot_target
        arm = robot_target['arm_joints']['single']
        
        # 提取手部动作
        # 注意：wrapper 可能会把手部动作放在 hand_joints 里
        hand = robot_target.get('hand_joints', {}).get('single', 0.0)
        
        self.send_joints(arm, hand)
        return self.get_obs()

    def get_obs(self):
        # === 1. 等待逻辑：给 ROS 一点时间接收第一帧数据 ===
        timeout = 1.0  # 最多等1秒
        start = time.time()
        # 只要关节数据还是空的，就稍微等一等
        while self.latest_joints is None and (time.time() - start < timeout):
            time.sleep(0.01)

        obs = {}

        # === 2. 机械臂状态兜底 (修复 ValueError 关键点) ===
        if self.latest_ee_pose is not None:
            obs['arm_pose'] = {'single': self.latest_ee_pose}
        else:
            # 🚨 修复核心：不能全为0！必须是合法的四元数 [x,y,z, qx,qy,qz,qw]
            # 我们设置 w=1，代表“无旋转”，防止 BaseEnv 计算时除以0崩溃
            dummy_pose = np.zeros(7)
            dummy_pose[6] = 1.0  # quaternion w
            obs['arm_pose'] = {'single': dummy_pose}

        if self.latest_joints is not None:
             obs['arm_joints'] = {'single': self.latest_joints}
        else:
             print("非正常得到arm_joints")
             obs['arm_joints'] = {'single': np.zeros(7)}

        if self.latest_hand_joints is not None:
             obs['hand_joints'] = {'single': self.latest_hand_joints}
        else:
             print("非正常得到hand_joints")
             #obs['hand_joints'] = {'single': np.zeros(16)}
             obs['hand_joints'] = {'single': np.zeros(19)}


        # === 3. 图像强制交货 (修复“非正常交货”) ===
        obs['images'] = {}
        # 必须列出所有需要的相机，不能依赖 self.latest_images.keys()
        target_cameras = ["cam_chest", "cam_head", "wrist"]
        
        for cam in target_cameras:
            if self.latest_images.get(cam) is not None:
                obs['images'][cam] = self.latest_images[cam]
            else:
                # 兜底：如果没有收到图，生成全黑图 (224, 224, 3)
                # 这样下游永远不会报 KeyError
                print("非正常得到图片")
                obs['images'][cam] = np.zeros((480, 640, 3), dtype=np.uint8)
                
        return obs
    
    def get_ee_pose_from_joint(self, joints):
        # 必须用 Pinocchio，因为 Controller 不支持查表
        if not self.has_model: return np.zeros(7)
        pin.forwardKinematics(self.model, self.data, joints)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.model.getFrameId("fr3_link8")
        tf = self.data.oMf[fid]
        quat = pin.Quaternion(tf.rotation).coeffs()
        return np.concatenate((tf.translation, quat))

    def get_xtele_data(self):
        return {
            'pose': self.human_pose,
            'gripper': self.human_gripper
        }
    
    def step_ee(self, robot_target):
        """
        处理 Pose 控制指令
        robot_target 结构: 
        {
            "arm_pose": {"single": [x,y,z, qx,qy,qz,qw]},
            "hand_joints": {"single": gripper_val}
        }
        """
        # 1. 提取目标 Pose
        target_pose = robot_target['arm_pose']['single']
        
        # 2. 提取夹爪
        # 注意: base_env 里逻辑是如果没传 hand_joints 就用空的
        hand_dict = robot_target.get('hand_joints', {})
        hand_val = hand_dict.get('single', 0.0)
        
        # 3. 运行 IK 算出关节角
        # 使用当前关节角作为 IK 的种子 (seed)
        target_joints = self.solve_ik(target_pose)         # todo by Jiaju
        
        # 4. 发送指令
        self.send_joints(target_joints, hand_val)
        
        return self.get_obs()
    
    def solve_ik(self, target_pose, max_iter=100, dt=1e-2, damp=1e-12):            # todo by Jiaju
        """
        输入: target_pose [x,y,z, qx,qy,qz,qw]
        输出: 7个关节角度
        """
        if not self.has_model:
            print("Error: No URDF model loaded, cannot solve IK.")
            return self.latest_joints # 失败返回当前角度

        # 转换目标 pose 为 SE3 矩阵
        t = target_pose[:3]
        q = pin.Quaternion(target_pose[6], target_pose[3], target_pose[4], target_pose[5]) # 注意 pinocchio 顺序是 x,y,z,w 还是 w,x,y,z，这里假设输入是 [x,y,z, qx,qy,qz,qw]
        R = q.matrix() 
        oMdes = pin.SE3(R, t)

        # 初始猜测 (当前关节角)
        q = self.latest_joints.copy()
        
        # 迭代求解 (CLIK 算法)
        frame_id = self.model.getFrameId("fr3_link8") # 确保名字对
        
        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[frame_id]
            
            # 计算误差 (Log map)
            dMf = oMdes.actInv(oMf)
            err = pin.log(dMf).vector
            
            if np.linalg.norm(err) < 1e-4:
                # print(f"IK Converged in {i} iters")
                break
                
            # 计算雅可比矩阵
            J = pin.computeFrameJacobian(self.model, self.data, q, frame_id)
            
            # 阻尼最小二乘法更新
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * dt)
            
        return q