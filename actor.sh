# export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
# export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
# #export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889
# #export http_proxy=http://127.0.0.1:2080 && export https_proxy=http://127.0.0.1:2080
# export no_proxy=localhost,127.0.0.1,0.0.0.0,::1



# #task_name=close_trashbin_franka_1028
# task_name=revoarm_bottle 

# mkdir -p experiments/${task_name}
# cd experiments/${task_name}

# # python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true use_human_intervention=true ego_mode=true policy_type=silri

# # [debug]
# # python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true freeze_actor=true use_human_intervention=false ego_mode=false policy_type=silri




# ########################                  conda                   ###########################
# #PYTHONPATH="" AMENT_PREFIX_PATH="" python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true freeze_actor=true use_human_intervention=false ego_mode=false policy_type=silri




# # export PYTHONPATH=$PYTHONPATH:/opt/ros/humble/local/lib/python3.10/dist-packages
# # AMENT_PREFIX_PATH="" python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true freeze_actor=true use_human_intervention=false ego_mode=false policy_type=silri

# # source /opt/ros/humble/setup.bash

# # echo ">>> 启动 Actor (System Python Mode)..."
# # echo ">>> ROS_DISTRO: $ROS_DISTRO"
# # echo ">>> Python Path: $(which python3)"

# # # 2. 启动命令
# # # 注意：这里直接运行，不要加任何前缀
# # /usr/bin/python3 ../../actor.py \
# #     robot_type@_global_=franka \
# #     task@_global_=${task_name} \
# #     classifier_cfg.require_train=true \
# #     freeze_actor=true \
# #     use_human_intervention=false \
# #     ego_mode=false \
# #     policy_type=silri

# SYSTEM_LIBSTDCPP=$(find /usr/lib -name "libstdc++.so.6" | grep "x86_64" | head -n 1)

# if [ -z "$SYSTEM_LIBSTDCPP" ]; then
#     echo "❌ 警告: 未找到系统 libstdc++.so.6，可能会发生 ABI 冲突！"
# else
#     echo ">>> 🛡️  已启用 ABI 保护模式"
#     echo ">>> 预加载系统库: $SYSTEM_LIBSTDCPP"
# fi

# # 2. 加载 ROS 2 环境
# source /opt/ros/humble/setup.bash

# # 3. 打印调试信息
# echo ">>> ROS_DISTRO: $ROS_DISTRO"
# echo ">>> Python: $(which python3)"

# # 4. 启动 Actor (带上核武器 LD_PRELOAD)
# #    注意：PYTHONPATH="" 确保不被 Conda 干扰 (如果你有 Conda 残留)
# #    LD_PRELOAD=... 确保优先使用系统 C++ 库，完美模拟 "ros_first" 的效果

# LD_PRELOAD=$SYSTEM_LIBSTDCPP /usr/bin/python3 ../../actor.py \
#     robot_type@_global_=franka \
#     task@_global_=${task_name} \
#     classifier_cfg.require_train=true \
#     freeze_actor=true \
#     use_human_intervention=false \
#     ego_mode=false \
#     policy_type=silri


#!/bin/bash

# ================= 原始配置 =================

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
#export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1

task_name=revoarm_bottle 

mkdir -p experiments/${task_name}
cd experiments/${task_name}

# 加载 ROS
source /opt/ros/humble/setup.bash

echo ">>> [Actor.sh] 启动 Python Actor..."

/usr/bin/python3 ../../actor.py \
    robot_type@_global_=frankaleap \
    task@_global_=${task_name} \
    classifier_cfg.require_train=true \
    freeze_actor=true \
    use_human_intervention=false \
    ego_mode=false \
    policy_type=sac