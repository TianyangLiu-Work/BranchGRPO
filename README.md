# BranchGRPO on VeRL + SGLang

This repo is now a thin VeRL recipe. Training runs through VeRL's Ray PPO/GRPO
entrypoint, with SGLang as the rollout backend and a local rule-based math
reward.

## Layout

- `configs/grpo_sglang.yaml`: default GRPO/SGLang training settings.
- `src/branch_grpo/reward.py`: custom VeRL reward function.
- `src/branch_grpo/data.py`: VeRL parquet schema helpers.
- `scripts/prepare_math_data.py`: converts MATH-style JSON or Hugging Face data to VeRL parquet.
- `scripts/launch_verl_grpo.py`: builds and launches `python -m verl.trainer.main_ppo`.
- `Dockerfile` and `docker-compose.yml`: GPU Docker environment based on VeRL's SGLang image.
- `slurm_scripts/run_grpo_sglang_docker.sbatch`: cluster entrypoint.

## Local smoke checks

```bash
pip install -r requirements.txt -e .
pytest -q
python scripts/prepare_math_data.py \
  --local-json data/smoketest_20.json \
  --output-dir data/smoketest \
  --train-limit 18 \
  --val-limit 2
python scripts/launch_verl_grpo.py --config configs/grpo_sglang.yaml --dry-run
```

## Docker

```bash
./scripts/docker_build.sh
./scripts/docker_run.sh bash
```

Inside the container:

```bash
python scripts/prepare_math_data.py --output-dir data/math
./scripts/train_grpo_sglang.sh
```

For a small local data smoke run inside Docker:

```bash
python scripts/prepare_math_data.py \
  --local-json data/smoketest_20.json \
  --output-dir data/smoketest \
  --train-limit 18 \
  --val-limit 2
python scripts/launch_verl_grpo.py \
  --config configs/grpo_sglang.yaml \
  data.train_files='["/workspace/data/smoketest/train.parquet"]' \
  data.val_files='["/workspace/data/smoketest/test.parquet"]' \
  trainer.total_epochs=1 \
  trainer.logger='["console"]'
```

## Main Training

Default config uses:

- VeRL GRPO: `algorithm.adv_estimator=grpo`
- rollout backend: `actor_rollout_ref.rollout.name=sglang`
- custom reward: `custom_reward_function.path=/workspace/src/branch_grpo/reward.py`

Run:

```bash
./scripts/train_grpo_sglang.sh
```

Override any VeRL Hydra key after the command:

```bash
./scripts/train_grpo_sglang.sh \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  trainer.n_gpus_per_node=8 \
  trainer.logger='["console","wandb"]'
```

