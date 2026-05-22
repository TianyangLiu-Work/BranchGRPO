#!/bin/bash
#SBATCH -p gpu
#SBATCH -J grpo_plan_pilot
#SBATCH --gres=gpushare:4
#SBATCH --time=04:00:00
#SBATCH -o ./outputs/%x-%j.out
#SBATCH -e ./outputs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "START $(date) JobID=${SLURM_JOB_ID}"

IMAGE=${IMAGE:-branchgrpo:sglang-verl}
HF_CACHE=${HF_CACHE:-/data/shared/cache}
BUILD_IMAGE=${BUILD_IMAGE:-0}
METHODS=${METHODS:-"standard_grpo_g8 low_temp_grpo_g8"}
DATA_DIR=${DATA_DIR:-data/math}
LOCAL_JSON=${LOCAL_JSON:-}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
VAL_LIMIT=${VAL_LIMIT:-}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
STANDARD_TEMP=${STANDARD_TEMP:-0.7}
STANDARD_TOP_P=${STANDARD_TOP_P:-0.9}
LOW_TEMP=${LOW_TEMP:-0.7}
LOW_TOP_P=${LOW_TOP_P:-0.9}
MH_ALPHA=${MH_ALPHA:-1.5}
MH_BRANCH_STRATEGY=${MH_BRANCH_STRATEGY:-topk_entropy}
MH_TOP_LOGPROBS=${MH_TOP_LOGPROBS:-20}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
TOKEN_LEN_PER_GPU=${TOKEN_LEN_PER_GPU:-24576}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
STANDARD_ROLLOUT_N=${STANDARD_ROLLOUT_N:-32}
SGLANG_MAX_NUM_BATCHED_TOKENS=${SGLANG_MAX_NUM_BATCHED_TOKENS:-32768}
SGLANG_MAX_NUM_SEQS=${SGLANG_MAX_NUM_SEQS:-32}
USE_LEGACY_WORKER_IMPL=${USE_LEGACY_WORKER_IMPL:-disable}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES:-}

mkdir -p ./outputs/experiment_plan ./outputs/ray_tmp

if [[ "${BUILD_IMAGE}" == "1" ]]; then
  docker build -t "${IMAGE}" .
fi

docker_base_args=(
  --rm
  --gpus all
  --ipc=host
  --net=host
  --shm-size=16g
  -v "${HF_CACHE}":/root/.cache/huggingface
  -v "$(pwd)":/workspace
  -w /workspace
  -e HF_HOME=/root/.cache/huggingface
  -e PYTHONPATH=/workspace/src:/workspace
  -e RAY_TMPDIR=/workspace/outputs/ray_tmp
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
)

WANDB_NETRC=${WANDB_NETRC:-${HOME}/.netrc}
if [[ -f "${WANDB_NETRC}" ]]; then
  docker_base_args+=(-v "${WANDB_NETRC}":/root/.netrc:ro)
fi
for env_name in WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_MODE WANDB_NAME WANDB_PROJECT; do
  if [[ -n "${!env_name:-}" ]]; then
    docker_base_args+=(-e "${env_name}=${!env_name}")
  fi
done

prepare_data_args=(python scripts/prepare_math_data.py --output-dir "${DATA_DIR}")
if [[ -n "${LOCAL_JSON}" ]]; then
  prepare_data_args+=(--local-json "${LOCAL_JSON}")
fi
if [[ -n "${TRAIN_LIMIT}" ]]; then
  prepare_data_args+=(--train-limit "${TRAIN_LIMIT}")
fi
if [[ -n "${VAL_LIMIT}" ]]; then
  prepare_data_args+=(--val-limit "${VAL_LIMIT}")
fi

docker run "${docker_base_args[@]}" "${IMAGE}" "${prepare_data_args[@]}"

read -r -a methods <<< "${METHODS}"
for method in "${methods[@]}"; do
  rollout_n="8"
  top_p="${STANDARD_TOP_P}"
  extra_hydra_overrides=""
  extra_env_args=()
  case "${method}" in
    standard_grpo_g8)
      temperature="${STANDARD_TEMP}"
      ;;
    standard_grpo_g32)
      temperature="${STANDARD_TEMP}"
      rollout_n="${STANDARD_ROLLOUT_N}"
      ;;
    low_temp_grpo_g8)
      temperature="${LOW_TEMP}"
      top_p="${LOW_TOP_P}"
      ;;
    mh_chain_only_k4)
      temperature="1.0"
      top_p="1.0"
      rollout_n="5"
      extra_hydra_overrides="+actor_rollout_ref.rollout.agent.agent_loop_manager_class=branch_grpo.mh_agent_loop.MHPowerAgentLoopManager"
      extra_env_args+=(
        -e BRANCH_GRPO_MH_VARIANT=chain_only
        -e BRANCH_GRPO_MH_STEPS=4
        -e BRANCH_GRPO_MH_ALPHA="${MH_ALPHA}"
        -e BRANCH_GRPO_MH_DEDUP_EXACT=0
        -e BRANCH_GRPO_MH_BRANCH_STRATEGY="${MH_BRANCH_STRATEGY}"
        -e BRANCH_GRPO_MH_TOP_LOGPROBS="${MH_TOP_LOGPROBS}"
      )
      ;;
    mh_all_proposals_k4)
      temperature="1.0"
      top_p="1.0"
      rollout_n="9"
      extra_hydra_overrides="+actor_rollout_ref.rollout.agent.agent_loop_manager_class=branch_grpo.mh_agent_loop.MHPowerAgentLoopManager"
      extra_env_args+=(
        -e BRANCH_GRPO_MH_VARIANT=all_proposals
        -e BRANCH_GRPO_MH_STEPS=4
        -e BRANCH_GRPO_MH_ALPHA="${MH_ALPHA}"
        -e BRANCH_GRPO_MH_DEDUP_EXACT=0
        -e BRANCH_GRPO_MH_BRANCH_STRATEGY="${MH_BRANCH_STRATEGY}"
        -e BRANCH_GRPO_MH_TOP_LOGPROBS="${MH_TOP_LOGPROBS}"
      )
      ;;
    mh_all_proposals_k4_exact_dedup)
      temperature="1.0"
      top_p="1.0"
      rollout_n="9"
      extra_hydra_overrides="+actor_rollout_ref.rollout.agent.agent_loop_manager_class=branch_grpo.mh_agent_loop.MHPowerAgentLoopManager"
      extra_env_args+=(
        -e BRANCH_GRPO_MH_VARIANT=all_proposals
        -e BRANCH_GRPO_MH_STEPS=4
        -e BRANCH_GRPO_MH_ALPHA="${MH_ALPHA}"
        -e BRANCH_GRPO_MH_DEDUP_EXACT=1
        -e BRANCH_GRPO_MH_BRANCH_STRATEGY="${MH_BRANCH_STRATEGY}"
        -e BRANCH_GRPO_MH_TOP_LOGPROBS="${MH_TOP_LOGPROBS}"
      )
      ;;
    *)
      echo "Unsupported method for current code path: ${method}" >&2
      exit 64
      ;;
  esac

  worker_overrides=(
    "trainer.use_legacy_worker_impl=${USE_LEGACY_WORKER_IMPL}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
  )

  log_path="/workspace/outputs/experiment_plan/${method}-${SLURM_JOB_ID}.log"
  echo "RUN ${method} temperature=${temperature} top_p=${top_p} rollout_n=${rollout_n} $(date)"
  docker run "${docker_base_args[@]}" "${extra_env_args[@]}" "${IMAGE}" bash -lc "
    set -euo pipefail
    ./scripts/train_grpo_sglang.sh \
      trainer.experiment_name=${method} \
      trainer.default_local_dir=/workspace/checkpoints/branch_grpo/${method} \
      trainer.n_gpus_per_node=1 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      data.train_files=[/workspace/${DATA_DIR}/train.parquet] \
      data.val_files=[/workspace/${DATA_DIR}/test.parquet] \
      data.train_batch_size=${TRAIN_BATCH_SIZE} \
      actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
      actor_rollout_ref.rollout.n=${rollout_n} \
      actor_rollout_ref.rollout.temperature=${temperature} \
      actor_rollout_ref.rollout.top_p=${top_p} \
      actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS} \
      data.max_response_length=${MAX_RESPONSE_LENGTH} \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${TOKEN_LEN_PER_GPU} \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${TOKEN_LEN_PER_GPU} \
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${TOKEN_LEN_PER_GPU} \
      actor_rollout_ref.rollout.max_num_batched_tokens=${SGLANG_MAX_NUM_BATCHED_TOKENS} \
      actor_rollout_ref.rollout.max_num_seqs=${SGLANG_MAX_NUM_SEQS} \
      trainer.total_epochs=1 \
      trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
      trainer.val_before_train=False \
      trainer.save_freq=-1 \
      trainer.test_freq=-1 \
      'trainer.logger=${TRAINER_LOGGER}' \
      ${worker_overrides[*]} \
      ${extra_hydra_overrides} \
      ${EXTRA_HYDRA_OVERRIDES} 2>&1 | tee ${log_path}
  "
done

echo "DONE $(date)"
