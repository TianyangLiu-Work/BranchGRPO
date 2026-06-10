#!/usr/bin/env bash
#SBATCH --job-name=verl-aime-psmc
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-/home/tyliu/ghworkspace/vllm-ps/verl}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/home/tyliu/ghworkspace/vllm-ps}
IMAGE=${IMAGE:-vllm-ps-verl-power-smc-vllm:latest}
cd "${REPO_ROOT}"

mkdir -p outputs/slurm

DOCKER_GPU_ARGS=(--gpus all)
if docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
    DOCKER_GPU_ARGS+=(--runtime=nvidia)
fi

exec docker run --rm \
    "${DOCKER_GPU_ARGS[@]}" \
    --net=host \
    --ipc=host \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add=SYS_ADMIN \
    -v "${WORKSPACE_ROOT}:/workspace/vllm-ps" \
    -v /data:/data \
    -w /workspace/vllm-ps/verl \
    -e VLLM_PS_ROOT=/workspace/vllm-ps \
    -e HF_HOME="${HF_HOME:-/data/shared/huggingface}" \
    -e HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data/shared/huggingface/hub}" \
    -e TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/shared/huggingface/hub}" \
    -e VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" \
    -e VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
    -e MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}" \
    -e TRAIN_FILE="${TRAIN_FILE:-/workspace/vllm-ps/data/aime2026/train.parquet}" \
    -e VAL_FILE="${VAL_FILE:-/workspace/vllm-ps/data/aime2026/test.parquet}" \
    -e NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}" \
    -e ROLLOUT_TP="${ROLLOUT_TP:-1}" \
    -e ROLLOUT_N="${ROLLOUT_N:-8}" \
    -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-30}" \
    -e PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-30}" \
    -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}" \
    -e MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}" \
    -e PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}" \
    -e POWER_SMC_ALPHA="${POWER_SMC_ALPHA:-1.4}" \
    -e POWER_SMC_PARTICLES="${POWER_SMC_PARTICLES:-4}" \
    -e TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}" \
    -e SAVE_FREQ="${SAVE_FREQ:--1}" \
    -e TEST_FREQ="${TEST_FREQ:-1}" \
    -e WANDB_MODE="${WANDB_MODE:-offline}" \
    "${IMAGE}" \
    bash -lc 'examples/power_smc_aime/run_qwen3_8b_aime2026_power_smc.sh "$@"' \
    bash "$@"
