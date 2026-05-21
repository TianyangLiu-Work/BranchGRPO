#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/grpo_sglang.yaml}
python scripts/launch_verl_grpo.py --config "${CONFIG}" "$@"

