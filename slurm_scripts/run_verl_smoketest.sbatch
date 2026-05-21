#!/bin/bash
#SBATCH -p gpu
#SBATCH -J verl_smoke
#SBATCH --gres=gpushare:4
#SBATCH --time=01:00:00
#SBATCH -o ./outputs/verl_smoke-%j.out
#SBATCH -e ./outputs/verl_smoke-%j.err
#SBATCH --export=ALL

set -eo pipefail
echo "START $(date) JobID=$SLURM_JOB_ID"
mkdir -p ./outputs

docker run --rm --gpus all --ipc=host \
  -v /data/shared/cache:/root/.cache/huggingface \
  -v /home/tyliu/ghworkspace/verl-official:/workspace \
  -w /workspace \
  verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2 \
  bash -c "
cd /workspace
pip install -q datasets 2>&1 | tail -1
python -m recipe.branch_grpo.main_branch_grpo \
  --config-path=/workspace/recipe/branch_grpo/config \
  --config-name=single_gpu_sglang
"

echo "DONE $(date)"
