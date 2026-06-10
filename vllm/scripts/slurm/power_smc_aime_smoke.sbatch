#!/usr/bin/env bash
#SBATCH --job-name=power-smc-aime
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/power-smc-aime-%j.out
#SBATCH --error=outputs/slurm/power-smc-aime-%j.err

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

echo "[aime-smoke] python=${POWER_SMC_PYTHON}"
echo "[aime-smoke] model=${MODEL:-Qwen/Qwen3-8B}"
echo "[aime-smoke] problems=${NUM_PROBLEMS:-3} samples=${SAMPLES_PER_PROBLEM:-4}"
echo "[aime-smoke] alphas=${ALPHAS:-2.0,3.0,4.0,5.0,6.0,8.0}"

exec "${POWER_SMC_PYTHON}" examples/generate/power_smc_aime_smoke.py
