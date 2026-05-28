#!/usr/bin/env bash
# Source after activating the conda environment on uclaa100.
#
#   conda activate /data2/fyyang/conda_envs/branch-grpo-verl
#   source scripts/cluster_runtime_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this file instead of executing it" >&2
  exit 2
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "activate the BranchGRPO conda environment before sourcing this file" >&2
  return 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT="${BRANCH_GRPO_CACHE_ROOT:-/data2/fyyang}"

PY_MM="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

CUDA_RUNTIME="${CONDA_PREFIX}/lib/python${PY_MM}/site-packages/nvidia/cuda_runtime/lib"
CUDA_NVRTC="${CONDA_PREFIX}/lib/python${PY_MM}/site-packages/nvidia/cuda_nvrtc/lib"

if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda-12.4 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.4
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export HF_HOME="${CACHE_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${CACHE_ROOT}/.cache"
export TORCH_HOME="${CACHE_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/.triton/cache"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/.cache/torchinductor"
export CUDA_CACHE_PATH="${CACHE_ROOT}/.cache/nv"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}"
export FLASHINFER_CUBIN_DIR="${CACHE_ROOT}/.cache/flashinfer/cubins"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-80}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r46}"
export TMPDIR="${TMPDIR:-/tmp/fyyang}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
  "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}" \
  "${FLASHINFER_CUBIN_DIR}" "${RAY_TMPDIR}" "${TMPDIR}"

LD_PARTS=()
[[ -d "${CUDA_RUNTIME}" ]] && LD_PARTS+=("${CUDA_RUNTIME}")
[[ -d "${CUDA_NVRTC}" ]] && LD_PARTS+=("${CUDA_NVRTC}")
LD_PARTS+=("${CONDA_PREFIX}/lib")
[[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]] && LD_PARTS+=("${CUDA_HOME}/lib64")
[[ -n "${LD_LIBRARY_PATH:-}" ]] && LD_PARTS+=("${LD_LIBRARY_PATH}")
export LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PARTS[*]}")"

if [[ -f "${CUDA_RUNTIME}/libcudart.so.12" ]]; then
  export LD_PRELOAD="${CUDA_RUNTIME}/libcudart.so.12${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
