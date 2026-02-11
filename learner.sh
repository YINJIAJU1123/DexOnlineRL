
export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
# export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
#export http_proxy=http://127.0.0.1:7890 && export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7897 && export https_proxy=http://127.0.0.1:7897


#task_name=close_trashbin_franka_1028
task_name=revoarm_bottle 

mkdir -p experiments/${task_name}
cd experiments/${task_name}

#python3 ../../learner.py robot_type@_global_=franka task@_global_=${task_name} policy_type=silri dataset.repo_id=${task_name}_success dataset.root=/home/yin/Online_RL/HIL-RL/experiments/${task_name}/offline_dataset
python3 ../../learner.py robot_type@_global_=franka task@_global_=${task_name}
