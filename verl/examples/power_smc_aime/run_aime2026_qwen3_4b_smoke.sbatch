#!/usr/bin/env bash
#SBATCH --job-name=verl-branchgrpo-qwen3-4b
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/home/tyliu/ghworkspace/vllm-ps/verl/outputs/slurm/verl-aime-qwen3-4b-%j.out
#SBATCH --error=/home/tyliu/ghworkspace/vllm-ps/verl/outputs/slurm/verl-aime-qwen3-4b-%j.err

set -euxo pipefail

mkdir -p /home/tyliu/ghworkspace/vllm-ps/verl/outputs/slurm
cd /home/tyliu/ghworkspace/vllm-ps

exec bash verl/examples/power_smc_aime/run_qwen3_4b_aime2026_power_smc.sh
