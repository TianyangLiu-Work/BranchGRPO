#!/bin/bash
#SBATCH -p gpu
#SBATCH -J smoke
#SBATCH --gres=gpushare:4
#SBATCH --time=01:00:00
#SBATCH -o ./outputs/smoke-%j.out
#SBATCH -e ./outputs/smoke-%j.err
#SBATCH --export=ALL

set -eo pipefail
echo "START $(date) JobID=$SLURM_JOB_ID"
mkdir -p ./outputs

eval "$(/data/conda/bin/conda shell.bash hook)"
conda activate /data/conda_envs/branch-grpo
export HF_HOME=/data/shared/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py --config configs/smoketest.yaml --no_wandb

echo "DONE $(date)"
