#!/usr/bin/env bash
set -e

IMAGE_NAME="onlinerl"
TAG="v1"

CONTEXT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Building Docker Image: ${IMAGE_NAME}:${TAG}..."

DOCKER_BUILDKIT=1 docker build \
    --network host \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg USERNAME="$(whoami)" \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${CONTEXT_DIR}/Dockerfile" \
    "${CONTEXT_DIR}"

echo "Build Success!"