#!/bin/bash
#SBATCH -p gpu
#SBATCH -J std_grpo
#SBATCH --gres=gpushare:4
#SBATCH --time=24:00:00
#SBATCH -o ./outputs/%x-%j.out
#SBATCH -e ./outputs/%x-%j.err
#SBATCH --export=ALL,CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps,CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

set -eo pipefail
echo "START $(date) JobID=$SLURM_JOB_ID"
echo "Method: standard_grpo"

eval "$(/data/conda/bin/conda shell.bash hook)"
conda activate /data/conda_envs/branch-grpo

export HF_HOME=/data/shared/cache
export TRITON_CACHE_DIR=/data/shared/cache/triton
export SGLANG_DISABLE_DEEP_GEMM=1
export TORCH_CUDA_ARCH_LIST="9.0"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_NUM_THREADS=8

mkdir -p ./outputs

python train.py --config configs/standard_grpo.yaml --no_wandb

echo "DONE $(date)"
