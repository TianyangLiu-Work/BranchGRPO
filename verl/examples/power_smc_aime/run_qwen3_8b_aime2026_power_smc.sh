#!/usr/bin/env bash
# GRPO | Qwen3-8B | AIME2026 | vLLM Power-SMC rollout

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
VLLM_PS_ROOT=${VLLM_PS_ROOT:-$(cd "${VERL_ROOT}/.." && pwd)}

prepare_vllm_overlay() {
    local local_vllm_pkg="${VLLM_PS_ROOT}/vllm/vllm"
    local overlay_root="${VLLM_OVERLAY_ROOT:-/tmp/vllm_ps_overlay}"
    local overlay_pkg="${overlay_root}/vllm"
    local image_vllm_pkg

    if [ "${VLLM_SKIP_OVERLAY:-0}" = "1" ] || [ ! -d "${local_vllm_pkg}" ]; then
        return 0
    fi

    image_vllm_pkg=$(PYTHONPATH= python3 -c \
        'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)' \
        2>/dev/null || true)
    if [ -z "${image_vllm_pkg}" ] || [ ! -d "${image_vllm_pkg}" ]; then
        return 0
    fi

    rm -rf "${overlay_root}"
    mkdir -p "${overlay_root}"
    cp -a "${local_vllm_pkg}" "${overlay_pkg}"

    # The mounted workspace may contain vLLM extensions built against a
    # different torch/CUDA ABI. Keep the local Python changes, but use the
    # binary extensions from the Docker image.
    find "${overlay_pkg}" -maxdepth 1 -type f -name '*.so' -delete
    find "${image_vllm_pkg}" -maxdepth 1 -type f -name '*.so' \
        -exec cp -a {} "${overlay_pkg}/" \;

    echo "${overlay_root}"
}

VLLM_PYTHON_ROOT=$(prepare_vllm_overlay)
if [ -n "${VLLM_PYTHON_ROOT}" ]; then
    export PYTHONPATH="${VLLM_PYTHON_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${VLLM_PS_ROOT}/vllm:${VERL_ROOT}:${PYTHONPATH:-}"
fi

expand_path() {
    local path="$1"
    echo "${path/#\~/${HOME}}"
}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
TRAIN_FILE=$(expand_path "${TRAIN_FILE:-${VLLM_PS_ROOT}/data/aime2026/train.parquet}")
VAL_FILE=$(expand_path "${VAL_FILE:-${VLLM_PS_ROOT}/data/aime2026/test.parquet}")
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_MERGE=${LORA_MERGE:-}
if [ -z "${LORA_MERGE}" ]; then
    if [ "${LORA_RANK}" != "0" ]; then
        LORA_MERGE=True
    else
        LORA_MERGE=False
    fi
fi

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Missing TRAIN_FILE=${TRAIN_FILE}" >&2
    echo "Create it with examples/data_preprocess/aime2026.py or set TRAIN_FILE." >&2
    exit 2
fi
if [ ! -f "${VAL_FILE}" ]; then
    echo "Missing VAL_FILE=${VAL_FILE}" >&2
    echo "Create it with examples/data_preprocess/aime2026.py or set VAL_FILE." >&2
    exit 2
fi

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-2048}

POWER_SMC_ENABLED=${POWER_SMC_ENABLED:-True}
POWER_SMC_ALPHA=${POWER_SMC_ALPHA:-1.4}
POWER_SMC_PARTICLES=${POWER_SMC_PARTICLES:-4}
POWER_SMC_BLOCK_SIZE=${POWER_SMC_BLOCK_SIZE:-64}
POWER_SMC_ESS_THRESHOLD=${POWER_SMC_ESS_THRESHOLD:-0.5}
POWER_SMC_ALPHA_RAMP_TOKENS=${POWER_SMC_ALPHA_RAMP_TOKENS:-400}

ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-}
ENABLE_THINKING=${ENABLE_THINKING:-}

PROJECT_NAME=${PROJECT_NAME:-verl_power_smc_aime2026}
if [ -z "${EXPERIMENT_NAME:-}" ]; then
    if [ "${POWER_SMC_ENABLED}" = "True" ] || [ "${POWER_SMC_ENABLED}" = "true" ] || [ "${POWER_SMC_ENABLED}" = "1" ]; then
        EXPERIMENT_NAME=qwen3_8b_aime2026_psmc_a${POWER_SMC_ALPHA}_p${POWER_SMC_PARTICLES}_$(date +%Y%m%d_%H%M)
    else
        EXPERIMENT_NAME=qwen3_8b_aime2026_plain_vllm_$(date +%Y%m%d_%H%M)
    fi
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

if [ -n "${TRAIN_MAX_SAMPLES}" ]; then
    DATA+=(data.train_max_samples=${TRAIN_MAX_SAMPLES})
fi
if [ -n "${VAL_MAX_SAMPLES}" ]; then
    DATA+=(data.val_max_samples=${VAL_MAX_SAMPLES})
fi
if [ -n "${ENABLE_THINKING}" ]; then
    DATA+=(+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING})
fi

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    actor_rollout_ref.model.lora.merge=${LORA_MERGE}
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=${ROLLOUT_ATTENTION_BACKEND}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.power_smc.enabled=${POWER_SMC_ENABLED}
    actor_rollout_ref.rollout.power_smc.alpha=${POWER_SMC_ALPHA}
    actor_rollout_ref.rollout.power_smc.particles=${POWER_SMC_PARTICLES}
    actor_rollout_ref.rollout.power_smc.block_size=${POWER_SMC_BLOCK_SIZE}
    actor_rollout_ref.rollout.power_smc.ess_threshold=${POWER_SMC_ESS_THRESHOLD}
    actor_rollout_ref.rollout.power_smc.alpha_ramp_tokens=${POWER_SMC_ALPHA_RAMP_TOKENS}
    actor_rollout_ref.rollout.power_smc.proposal=power_temperature
    actor_rollout_ref.rollout.power_smc.return_diagnostics=True
    actor_rollout_ref.rollout.power_smc.kv_cow=True
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    hydra.run.dir=outputs/hydra/${EXPERIMENT_NAME}
)

if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    TRAINER+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then
    TRAINER+=(trainer.validation_data_dir="${VALIDATION_DATA_DIR}")
fi
if [ -n "${ROLLOUT_DATA_DIR}" ]; then
    TRAINER+=(trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}")
fi

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
