# 1. 基础镜像：Ubuntu 22.04 + CUDA 12.1
FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04

# 避免交互式提问
ENV DEBIAN_FRONTEND=noninteractive

# 2. 传入构建参数（用于创建用户）
ARG USERNAME=user
ARG UID=1000
ARG GID=1000

# 3. 更换 apt 源为清华源（加快构建速度，可选）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 4. 安装系统基础依赖
# LeRobot 和常见 RL 环境通常需要 git, python3.10, ffmpeg, 以及一些 GL 库用于渲染
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    build-essential \
    cmake \
    python3-pip \
    python3-dev \
    python3-venv \
    ffmpeg \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists*

RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8

# 添加 ROS 2 GPG Key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 添加 ROS 2 仓库 (使用清华源加速 ROS 下载)
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu jammy main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装 ROS 2 Humble Desktop (包含 Rviz 等工具) 和 基础构建工具 (colcon)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists*
# =========================================================

# 建立 python3 到 python 的软链接
RUN ln -s /usr/bin/python3 /usr/bin/python

# 5. 创建与宿主机一致的用户（解决权限问题）
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 切换到非 root 用户
USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# 6. 配置 pip 清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 7. 安装 Python 核心包
RUN pip install --upgrade pip setuptools wheel

# 8. 安装 PyTorch 2.1.1 (CUDA 12.1)
# 注意：这里直接指定 index-url 为官方源下载 gpu 版本
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 9. 复制 requirements.txt 并安装其余依赖
RUN pip install datasets==3.5.0 safetensors==0.5.3 protobuf==6.32.0 \
    hydra-core==1.3.2 draccus==0.10.0 numpy==1.26.4 huggingface-hub==0.35.3 \
    opencv-python diffusers gymnasium cmake

#########################################################################
RUN pip install \
    einops \
    av \
    "imageio[ffmpeg]" \
    jsonlines \
    deepdiff \
    pyserial \
    termcolor \
    grpcio \
    grpcio-tools \
    wandb \
    transformers==4.44.2 \
    pillow \
    scipy \
    pin \
    "numpy==1.26.4"

#########################################################################
# 10. (可选) 预先安装 LeRobot 的其他依赖，
# 但 LeRobot 本身通过挂载后 pip install -e . 安装
# 这里设置 git lfs 
RUN git lfs install

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]