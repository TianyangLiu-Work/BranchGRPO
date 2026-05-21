#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-branchgrpo:sglang-verl}
HF_CACHE=${HF_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}
mkdir -p "${HF_CACHE}"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --net=host \
  --shm-size=16g \
  -v "$(pwd)":/workspace \
  -v "${HF_CACHE}":/root/.cache/huggingface \
  -w /workspace \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src:/workspace \
  "${IMAGE}" "$@"

