#!/usr/bin/env bash
set -e

PROJECT_ROOT="/home/lixin/OnlineRl"

DATASET_PATH="${PROJECT_ROOT}/docker/data" 

IMAGE_NAME="onlinerl:v1"
CONTAINER_NAME="onlinerl_dev"
# ===========================================
echo "Allowing X11 access..."
xhost +local:root

echo "Starting Container: $CONTAINER_NAME with Image: $IMAGE_NAME"

docker run -d \
    --name "onlinerl_dev" \
    --gpus all \
    --network host \
    --ipc=host \
    --shm-size=16g \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "/home/lixin/OnlineRl:/home/$(whoami)/OnlineRl" \
    -v "/home/lixin/OnlineRl/docker/data:/home/$(whoami)/data" \
    -w "/home/lixin/OnlineRl" \
    "${IMAGE_NAME}" \
    tail -f /dev/null