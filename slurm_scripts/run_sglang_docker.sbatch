#!/bin/bash
#SBATCH -p gpu
#SBATCH -J sgl_train
#SBATCH --gres=gpushare:4
#SBATCH --time=72:00:00
#SBATCH -o ./outputs/sgl_train-%j.out
#SBATCH -e ./outputs/sgl_train-%j.err
#SBATCH --export=ALL

set -eo pipefail
echo "START $(date) JobID=$SLURM_JOB_ID"
mkdir -p ./outputs

docker run --rm --gpus all --ipc=host \
  -v /data/shared/cache:/root/.cache/huggingface \
  -v /home/tyliu/ghworkspace/BranchGRPO:/workspace \
  -w /workspace \
  verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2 \
  bash -c "
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q datasets peft wandb verl 2>&1 | tail -1
python train.py --config configs/standard_grpo.yaml --no_wandb
"

echo "DONE $(date)"
