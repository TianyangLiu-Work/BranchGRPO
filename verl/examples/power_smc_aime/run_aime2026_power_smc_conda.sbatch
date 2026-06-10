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

REPO_ROOT="/home/tyliu/ghworkspace/vllm-ps/verl"
WORKSPACE_ROOT="/home/tyliu/ghworkspace/vllm-ps"
CONDA_ENV="${POWER_SMC_CONDA_ENV:-/fast/conda_envs/power-smc-vllm}"

cd "${REPO_ROOT}"
mkdir -p outputs/slurm

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PS_ROOT="${WORKSPACE_ROOT}"
export PYTHONPATH="${WORKSPACE_ROOT}/vllm:${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_SKIP_OVERLAY=1
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${CONDA_ENV}/targets/x86_64-linux/lib:${CONDA_ENV}/targets/x86_64-linux/lib/stubs:${CONDA_ENV}/lib64:${CONDA_ENV}/lib64/stubs:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CONDA_ENV}/lib:${CONDA_ENV}/targets/x86_64-linux/lib:${CONDA_ENV}/targets/x86_64-linux/lib/stubs:${CONDA_ENV}/lib64:${CONDA_ENV}/lib64/stubs:${LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME:-${CONDA_ENV}}"

export HF_HOME="${HF_HOME:-/data/shared/cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data/shared/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/shared/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/shared/cache/tyliu/hf_datasets}"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/shared/cache/tyliu/ray}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/shared/cache/tyliu/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/data/shared/cache/tyliu/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/data/shared/cache/tyliu/cuda}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/data/shared/cache/tyliu}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-DEBUG}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export VERL_VLLM_FORCE_SHM_WEIGHT_SYNC="${VERL_VLLM_FORCE_SHM_WEIGHT_SYNC:-1}"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export TRAIN_FILE="${TRAIN_FILE:-${WORKSPACE_ROOT}/data/aime2026/train.parquet}"
export VAL_FILE="${VAL_FILE:-${WORKSPACE_ROOT}/data/aime2026/test.parquet}"

export NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.27}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-22048}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
export UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-4096}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20000}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"

export POWER_SMC_ALPHA="${POWER_SMC_ALPHA:-1.4}"
export POWER_SMC_PARTICLES="${POWER_SMC_PARTICLES:-4}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
export ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-False}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p \
    "${HF_DATASETS_CACHE}" \
    "${RAY_TMPDIR}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer"

python - <<'PY'
import torch
import vllm
import vllm._C
import vllm.v1.power_smc
print("env smoke ok", torch.__version__, vllm.__version__)
PY

exec bash examples/power_smc_aime/run_qwen3_8b_aime2026_power_smc.sh
