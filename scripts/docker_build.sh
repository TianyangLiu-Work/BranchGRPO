#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-branchgrpo:sglang-verl}
docker build -t "${IMAGE}" .

