#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-branchgrpo:sglang-verl}
HF_CACHE=${HF_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}
RAY_TMPDIR=${RAY_TMPDIR:-/workspace/outputs/ray_tmp}
mkdir -p "${HF_CACHE}"
mkdir -p outputs/ray_tmp

docker_tty_args=()
if [[ -t 0 && -t 1 ]]; then
  docker_tty_args=(-it)
elif [[ ! -t 0 ]]; then
  docker_tty_args=(-i)
fi

docker_extra_args=()
WANDB_NETRC=${WANDB_NETRC:-${HOME}/.netrc}
if [[ -f "${WANDB_NETRC}" ]]; then
  docker_extra_args+=(-v "${WANDB_NETRC}":/root/.netrc:ro)
fi
for env_name in WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_MODE WANDB_NAME WANDB_PROJECT; do
  if [[ -n "${!env_name:-}" ]]; then
    docker_extra_args+=(-e "${env_name}=${!env_name}")
  fi
done

docker run --rm "${docker_tty_args[@]}" \
  --gpus all \
  --ipc=host \
  --net=host \
  --shm-size=16g \
  -v "$(pwd)":/workspace \
  -v "${HF_CACHE}":/root/.cache/huggingface \
  -w /workspace \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src:/workspace \
  -e RAY_TMPDIR="${RAY_TMPDIR}" \
  "${docker_extra_args[@]}" \
  "${IMAGE}" "$@"
