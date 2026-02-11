<div align="center">

# DexOnlineRL: Real-World Reinforcement Learning for Dexterous Manipulation

<sup>1</sup> The Hong Kong University of Science and Technology

[\[📖 Documents\]](docs/) [\[🚀 Installation\]](#-installation) [\[📖 Training Recipe\]](#-training-recipe)  [\[🙋 FAQs\]](#-faqs)

</div>

## 📖 Introduction
**DexOnlineRL** is a framework for real-world reinforcement learning, extending [HIL-SERL](https://github.com/rail-berkeley/hil-serl) to support **dexterous manipulation** tasks. It integrates **Franka Emika FR3** arms with **LEAP Hands**, enabling efficient online learning from human demonstrations and interventions.

## Repository Structure

* `actor.py`: Actor script that queries actions from policy and executes them on the robot via `rl_envs`.
* `learner.py`: Learner script that receives transitions from the actor and updates model parameters.
* `rl_envs/`: Robot environments and wrappers (Supports Franka FR3 + Leap Hand).
* `lerobot/`: Open-source RL baseline library. Extended with `SilRI` and `HG-Dagger`.
* `collect_data.py`: Collects offline demonstrations for reward classifier training.
* `train_reward_classifier.py`: Trains the task-specific reward classifier.
* `cfg/`: Configuration files for robot (hardware) and task setup.
* `docker/`: Docker environment setup scripts.

## ⚙️ Hardware Setup
* **Robot Arm**: Franka Emika FR3
* **End Effector**: LEAP Hand (16 DoF)
* **Cameras**: Intel RealSense D435 / D405
* **Compute**: NVIDIA GPU (RTX 4090 recommended)

## 🚀 Installation

We provide a Docker environment to ensure consistent dependencies.

### 1. Build the Docker Image
```bash
cd docker
bash build.sh
```

### 2. Run the Container
```bash
bash run.sh
```
*Make sure you have installed the NVIDIA Container Toolkit and drivers properly.*

## 📖 Training Recipe

The overall training pipeline follows the **HIL-SERL** paradigm. Real-world RL training consists of three stages:

### 📑 Stage 1: Offline Data Collection

Collect 20 demonstration trajectories to initialize the buffer:

```bash
bash collect_data.sh
```

In `collect_data.sh`, set `robot_type` (e.g., `frankaleap`) and `task_name`. Key parameters in `cfg/task/`:

```yaml
# Task Configuration (e.g., cfg/task/revoarm_bottle.yaml)
abs_pose_limit_high: Upper bound of pose limits (safety).
abs_pose_limit_low: Lower bound of pose limits.
reset_joint: Joint configuration for reset.
max_episode_length: Maximum steps per episode.
```

After collection, split the dataset into success/failure subsets:
```bash
bash split_data.sh
```

### 📑 Stage 2: Reward Classifier Training

Train a task-specific reward classifier to distinguish between successful and failed states:

```bash
bash train_reward_classifier.sh
```
*Ensure you set the correct `task_name` and `dataset.root` in the script.*

### 📑 Stage 3: Online RL Training

Train the robust policy using Online RL/IL. Configure `actor.sh` and `learner.sh`:

```yaml
# actor.sh / learner.sh settings:
task_name: Your task name (e.g., revoarm_bottle)
robot_type: frankaleap
classifier_cfg.require_train: True (continue training classifier online)
use_human_intervention: True (enable human corrections)
```

**Execution Order:**
1.  **Server**: Start the learner first.
    ```bash
    bash learner.sh
    ```
2.  **Robot PC**: Start the actor.
    ```bash
    bash actor.sh
    ```

## License

This project is released under the [Apache License](LICENSE).

## Citation

If you find this project useful in your research, please consider citing:


## 🙋 FAQs

