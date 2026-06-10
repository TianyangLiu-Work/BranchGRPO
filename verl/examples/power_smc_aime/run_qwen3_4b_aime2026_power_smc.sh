#!/usr/bin/env bash
set -euxo pipefail

VLLM_PS_ROOT=${VLLM_PS_ROOT:-/home/tyliu/ghworkspace/vllm-ps}
CONDA_ENV=${CONDA_ENV:-/fast/conda_envs/power-smc-vllm}

export PATH="${CONDA_ENV}/bin:${CONDA_ENV}/nvvm/bin:${PATH}"
export PYTHONPATH="${VLLM_PS_ROOT}/vllm:${VLLM_PS_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_SKIP_OVERLAY=${VLLM_SKIP_OVERLAY:-1}
export HF_HOME=${HF_HOME:-/data/shared/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data/shared/cache/tyliu/hf_datasets}
export RAY_TMPDIR=${RAY_TMPDIR:-/data/shared/cache/tyliu/ray}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/dev/shm/tyliu/triton_cache_${SLURM_JOB_ID:-local}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/tyliu/torchinductor_cache_${SLURM_JOB_ID:-local}}
export POWER_SMC_CUDA_SHIM=${POWER_SMC_CUDA_SHIM:-/dev/shm/tyliu/cuda_home_${SLURM_JOB_ID:-local}}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}

mkdir -p \
  "${HF_DATASETS_CACHE}" \
  "${RAY_TMPDIR}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${POWER_SMC_CUDA_SHIM}/bin" \
  "${POWER_SMC_CUDA_SHIM}/include" \
  "${POWER_SMC_CUDA_SHIM}/lib64/stubs"

ln -sfn "${CONDA_ENV}/bin/nvcc" "${POWER_SMC_CUDA_SHIM}/bin/nvcc"
ln -sfn "${CONDA_ENV}/nvvm/bin/cicc" "${POWER_SMC_CUDA_SHIM}/bin/cicc"
set +x
ln -sfn "${CONDA_ENV}/targets/x86_64-linux/include/"* \
  "${POWER_SMC_CUDA_SHIM}/include/"
set -x
ln -sfn "${CONDA_ENV}/lib/libcudart.so" \
  "${POWER_SMC_CUDA_SHIM}/lib64/libcudart.so"
ln -sfn "${CONDA_ENV}/lib/libcudart.so.12" \
  "${POWER_SMC_CUDA_SHIM}/lib64/libcudart.so.12"
ln -sfn "${CONDA_ENV}/targets/x86_64-linux/lib/stubs/libcuda.so" \
  "${POWER_SMC_CUDA_SHIM}/lib64/stubs/libcuda.so"

export CUDA_HOME="${POWER_SMC_CUDA_SHIM}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${POWER_SMC_CUDA_SHIM}/lib64:${LD_LIBRARY_PATH:-}"
cd "${VLLM_PS_ROOT}/verl"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_branchgrpo_math_psmc_lora_merge_a1p4_p8_20k_100step_20260609}
BRANCH_GRPO_DATA_DIR=${BRANCH_GRPO_DATA_DIR:-/data/shared/branch-grpo/data}
TRAIN_FILE=${TRAIN_FILE:-${BRANCH_GRPO_DATA_DIR}/math__combined_54.4k.parquet}
VAL_FILE=${VAL_FILE:-${BRANCH_GRPO_DATA_DIR}/math__aime_repeated_8x_240.parquet}
REWARD_FILE=${REWARD_FILE:-${VLLM_PS_ROOT}/verl/examples/power_smc_aime/branchgrpo_math_reward.py}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20000}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}

ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.27}
ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-22048}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}

POWER_SMC_ALPHA=${POWER_SMC_ALPHA:-1.4}
POWER_SMC_PARTICLES=${POWER_SMC_PARTICLES:-8}
POWER_SMC_BLOCK_SIZE=${POWER_SMC_BLOCK_SIZE:-64}
POWER_SMC_ESS_THRESHOLD=${POWER_SMC_ESS_THRESHOLD:-0.5}
POWER_SMC_ALPHA_RAMP_TOKENS=${POWER_SMC_ALPHA_RAMP_TOKENS:-400}

LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_MERGE=${LORA_MERGE:-True}

test -f "${TRAIN_FILE}"
test -f "${VAL_FILE}"
test -f "${REWARD_FILE}"

python3 - <<'PY'
import torch
import vllm

print("env smoke ok", torch.__version__, vllm.__version__)
PY

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  "data.train_files=['${TRAIN_FILE}']" \
  "data.val_files=['${VAL_FILE}']" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  reward.custom_reward_function.path="${REWARD_FILE}" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.lora.merge="${LORA_MERGE}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.power_smc.enabled=True \
  actor_rollout_ref.rollout.power_smc.alpha="${POWER_SMC_ALPHA}" \
  actor_rollout_ref.rollout.power_smc.particles="${POWER_SMC_PARTICLES}" \
  actor_rollout_ref.rollout.power_smc.block_size="${POWER_SMC_BLOCK_SIZE}" \
  actor_rollout_ref.rollout.power_smc.ess_threshold="${POWER_SMC_ESS_THRESHOLD}" \
  actor_rollout_ref.rollout.power_smc.alpha_ramp_tokens="${POWER_SMC_ALPHA_RAMP_TOKENS}" \
  actor_rollout_ref.rollout.power_smc.proposal=power_temperature \
  actor_rollout_ref.rollout.power_smc.return_diagnostics=True \
  actor_rollout_ref.rollout.power_smc.kv_cow=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.balance_batch=True \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=verl_power_smc_aime2026 \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs=1 \
  trainer.val_before_train=False \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  hydra.run.dir="outputs/hydra/${EXPERIMENT_NAME}"
