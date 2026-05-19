#!/bin/bash
# Run all experiment variants for the MH Power Sampling GRPO experiment
# Usage: bash scripts/run_experiments.sh

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-7B"}
TRAIN_PROMPTS=${TRAIN_PROMPTS:-500}
VAL_SAMPLES=${VAL_SAMPLES:-500}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-6}
MAX_LEN=${MAX_LEN:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}

echo "=== BranchGRPO Experiment Runner ==="
echo "Model: $MODEL"
echo "Train prompts: $TRAIN_PROMPTS"
echo "Epochs: $EPOCHS"
echo "Output dir: $OUTPUT_DIR"
echo ""

run_exp() {
    local method=$1
    local desc=$2
    local extra_args=$3

    echo "--- Running: $desc ($method) ---"
    python train.py \
        --method "$method" \
        --model "$MODEL" \
        --train_prompts "$TRAIN_PROMPTS" \
        --num_epochs "$EPOCHS" \
        --learning_rate "$LR" \
        --max_response_length "$MAX_LEN" \
        --output_dir "$OUTPUT_DIR" \
        --val_samples "$VAL_SAMPLES" \
        $extra_args
    echo "--- Complete: $desc ---"
    echo ""
}

# 1. Standard GRPO baseline
run_exp "standard_grpo" "Standard GRPO (8 rollouts)" \
    "--num_rollouts 8 --temperature 1.0 --loss_mask full_response"

# 2. Low-temperature GRPO
run_exp "low_temp_grpo" "Low-temp GRPO (8 rollouts, temp=0.5)" \
    "--num_rollouts 8 --temperature 0.5 --loss_mask full_response"

# 3. MH Final Only
run_exp "mh_final_only" "MH Final Only (K=4)" \
    "--num_rollouts 1 --mh_steps 4 --alpha 1.5 --span_len 16 --loss_mask branch_after_only"

# 4. MH Chain Only
run_exp "mh_chain_only" "MH Chain Only (K=4)" \
    "--num_rollouts 1 --mh_steps 4 --alpha 1.5 --span_len 16 --loss_mask branch_after_only"

# 5. MH All Proposals
run_exp "mh_all_proposals" "MH All Proposals (K=4)" \
    "--num_rollouts 1 --mh_steps 4 --alpha 1.5 --span_len 16 --loss_mask branch_after_only"

# 6. MH All Proposals + Dedup
run_exp "mh_all_proposals_dedup" "MH All Proposals + Dedup (K=4)" \
    "--num_rollouts 1 --mh_steps 4 --alpha 1.5 --span_len 16 --loss_mask branch_after_only"

echo "=== All experiments complete ==="
