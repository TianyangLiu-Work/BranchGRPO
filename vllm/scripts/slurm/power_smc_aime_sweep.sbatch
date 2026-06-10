#!/usr/bin/env bash
#SBATCH --job-name=power-smc-aime-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/power-smc-aime-sweep-%j.out
#SBATCH --error=outputs/slurm/power-smc-aime-sweep-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tyliu/ghworkspace/vllm-ps/vllm}"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm outputs/power_smc/aime2025

export POWER_SMC_CONDA_PREFIX="${POWER_SMC_CONDA_PREFIX:-/data/conda_envs/power-smc-vllm}"
export POWER_SMC_PYTHON="${POWER_SMC_PYTHON:-${POWER_SMC_CONDA_PREFIX}/bin/python}"
export POWER_SMC_USE_SOURCE_VLLM=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HOME="${HF_HOME:-/data/shared/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

export CUDA_HOME="${CUDA_HOME:-${POWER_SMC_CONDA_PREFIX}}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export CC="${CC:-${CUDA_HOME}/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-${CUDA_HOME}/bin/x86_64-conda-linux-gnu-g++}"
export PATH="${CUDA_HOME}/bin:${PATH}"
cuda_stub_dir="${CUDA_HOME}/targets/x86_64-linux/lib/stubs"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${cuda_stub_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${cuda_stub_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LDFLAGS="-L${cuda_stub_dir}${LDFLAGS:+ ${LDFLAGS}}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PROMPT_FILE="${PROMPT_FILE:-aime2025.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/power_smc/aime2025}"
ALPHAS="${ALPHAS:-1.2,1.4}"
PARTICLES="${PARTICLES:-4,8,16}"
BATCH_SIZES="${BATCH_SIZES:-4,8,16}"
ATTEMPTS_PER_PROBLEM="${ATTEMPTS_PER_PROBLEM:-8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
ALPHA_RAMP_TOKENS="${ALPHA_RAMP_TOKENS:-400}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
LIMIT="${LIMIT:-}"
PROBLEM_INDICES="${PROBLEM_INDICES:-7,18,20}"

echo "[aime-sweep] python=${POWER_SMC_PYTHON}"
echo "[aime-sweep] model=${MODEL}"
echo "[aime-sweep] alphas=${ALPHAS}"
echo "[aime-sweep] particles=${PARTICLES}"
echo "[aime-sweep] batch_sizes=${BATCH_SIZES}"
echo "[aime-sweep] attempts/problem=${ATTEMPTS_PER_PROBLEM}"
if [[ -n "${LIMIT}" ]]; then
  echo "[aime-sweep] limit=${LIMIT}"
fi
if [[ -n "${PROBLEM_INDICES}" ]]; then
  echo "[aime-sweep] problem_indices=${PROBLEM_INDICES}"
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  echo "[aime-sweep] enforce_eager=1"
fi

cmd=(
  "${POWER_SMC_PYTHON}" examples/generate/benchmark_power_smc_aime.py
  --model "${MODEL}" \
  --prompt-file "${PROMPT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --alphas "${ALPHAS}" \
  --particles "${PARTICLES}" \
  --batch-sizes "${BATCH_SIZES}" \
  --attempts-per-problem "${ATTEMPTS_PER_PROBLEM}" \
  --max-tokens "${MAX_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  --ess-threshold "${ESS_THRESHOLD}" \
  --alpha-ramp-tokens "${ALPHA_RAMP_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --attention-backend "${ATTENTION_BACKEND}"
)
if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi
if [[ -n "${PROBLEM_INDICES}" ]]; then
  cmd+=(--problem-indices "${PROBLEM_INDICES}")
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

exec "${cmd[@]}"
