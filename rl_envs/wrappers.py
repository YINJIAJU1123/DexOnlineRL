import time
import numpy as np
from gymnasium import Env, spaces
import gymnasium as gym
from scipy.spatial.transform import Rotation
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space, flatten
# from xrocs.utils.logger.logger_loader import logger
from rl_envs.shared_state import shared_state
import cv2
import traceback
import sys

class HumanIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.robot_type = env.unwrapped.robot_type
        self.env.unwrapped.init_xtele() # init xtele
        self.control_mode = env.unwrapped.control_mode
    

    def reset(self, **kwargs):
        """Reset the environment and sync robot position."""
        obs, info = self.env.reset(**kwargs)
        shared_state.human_intervention_key = False
        self.env.unwrapped.sync_xtele(timeout=2)
        info["is_intervention"] = False
        return obs, info

    def pose2matrix(self, pose):
        pose_t, pose_quat = pose[0:3], pose[3:7]
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = Rotation.from_quat(pose_quat).as_matrix()
        pose_matrix[:3, 3] = pose_t
        return pose_matrix
    

    def action(self, action: np.ndarray) -> np.ndarray:
        # intervened = True
        intervened = shared_state.human_intervention_key
        if intervened:
            try:
                obs = self.env.unwrapped.get_xtele()
                xtele_joints, xtele_pose = obs['joints'], obs['pose']
                # print("gripper_value:", len(xtele_joints), xtele_joints[-1])
                if self.control_mode == "joint":
                    expert_a = xtele_joints
                else:
                    curr_matrix = self.pose2matrix(self.env.unwrapped.currpos)
                    tar_matrix = self.pose2matrix(xtele_pose)
                    T_diff_matrix = np.dot(np.linalg.inv(curr_matrix), tar_matrix)

                    
                rel_rot = Rotation.from_matrix(T_diff_matrix[:3, :3]).as_euler("xyz")
                rel_pos = T_diff_matrix[:3, 3]
                expert_a = np.zeros(7, dtype=np.float32)
                expert_a[:3] = rel_pos / self.env.unwrapped.action_scale[0]
                expert_a[3:6] = rel_rot / self.env.unwrapped.action_scale[1]
                expert_a[6:] = xtele_joints[-1] / self.env.unwrapped.action_scale[2]
                
                # expert_a = np.clip(expert_a, [-1]*7, [1]*7)
                """
                intervention action 边缘裁剪
                """
                epsilon = 1e-6
                expert_a[0:6]= expert_a[0:6].clip(-1+epsilon, 1-epsilon)

                return expert_a, xtele_joints, True
            except Exception as e:
                print(f"Error in action: {e}")
                print(f"[{type(e).__name__}] {e!r}")
                traceback.print_exc()          # full stacktrace
                sys.exit(1)
        return action, None, False

    def step(self, action):
        action, xtele_joints,replaced = self.action(action)
        if replaced:
            obs, rew, terminated, truncated, info = self.env.step(action)
            info["intervene_action"] = action
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)        
            self.env.unwrapped.sync_xtele(timeout=0.1)
        

        info["is_intervention"] = replaced
        return obs, rew, terminated, truncated, info




class AugmentedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env

    # def observation(self, obs):
    #     images = obs['images']
    #     env = self.env.unwrapped
    #     for key, img in images.items():
    #         if hasattr(env, 'image_crop'):
    #             cropped_rgb = env.image_crop[key](img) if key in env.image_crop else img
    #         else:
    #             cropped_rgb = img
    #         cropped_rgb = cv2.resize(
    #             cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
    #         )
    #         images[key] = cropped_rgb

    #     return obs
    
    def observation(self, obs):
        # [侦查点 1]：确认数据进入了 Wrapper
        print(f"🔍 [Wrapper] 收到 Obs! Keys: {list(obs['images'].keys())}") 
        
        images = obs['images']
        env = self.env.unwrapped
        for key, img in images.items():
            # [侦查点 2]：打印输入尺寸
            print(f"   -> {key} 输入尺寸: {img.shape}")

            if hasattr(env, 'image_crop'):
                cropped_rgb = env.image_crop[key](img) if key in env.image_crop else img
            else:
                cropped_rgb = img
            
            # [侦查点 3]：打印目标尺寸 (这步最关键！看看 observation_space 到底是多少)
            target_size = self.observation_space["images"][key].shape[:2][::-1]
            print(f"   -> {key} 计划 Resize 到: {target_size}")

            cropped_rgb = cv2.resize(cropped_rgb, target_size)
            
            # [侦查点 4]：打印实际输出尺寸
            print(f"   -> {key} 实际输出尺寸: {cropped_rgb.shape}")
            
            images[key] = cropped_rgb

        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], Rotation.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )


        return observation


from collections import OrderedDict


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None, use_force=False):
        super().__init__(env)
        if use_force:
            self.proprio_keys = proprio_keys
        else:
            self.proprio_keys = proprio_keys[:2]

        print("proprio_keys:", self.proprio_keys)    

        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            OrderedDict((key, self.env.observation_space["state"][key]) for key in self.proprio_keys)
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        from collections import OrderedDict
        obs = {
            "state": flatten(
                self.proprio_space,
                OrderedDict((key, obs["state"][key]) for key in self.proprio_keys),
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

  
def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs