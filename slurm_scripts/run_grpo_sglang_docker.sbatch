#!/bin/bash
#SBATCH -p gpu
#SBATCH -J grpo_sglang
#SBATCH --gres=gpushare:4
#SBATCH --time=72:00:00
#SBATCH -o ./outputs/%x-%j.out
#SBATCH -e ./outputs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "START $(date) JobID=${SLURM_JOB_ID}"
mkdir -p ./outputs

IMAGE=${IMAGE:-branchgrpo:sglang-verl}
HF_CACHE=${HF_CACHE:-/data/shared/cache}
BUILD_IMAGE=${BUILD_IMAGE:-1}

if [[ "${BUILD_IMAGE}" == "1" ]]; then
  docker build -t "${IMAGE}" .
fi

docker run --rm --gpus all --ipc=host --net=host --shm-size=16g \
  -v "${HF_CACHE}":/root/.cache/huggingface \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src:/workspace \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${IMAGE}" \
  bash -lc "python scripts/prepare_math_data.py --output-dir data/math && ./scripts/train_grpo_sglang.sh"

echo "DONE $(date)"

