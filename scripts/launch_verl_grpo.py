#!/usr/bin/env python3
"""Launch VeRL GRPO with SGLang rollout from a compact project config."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _bool(value: Any) -> str:
    return "True" if bool(value) else "False"


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return _bool(value)
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return str(value)


def _override(key: str, value: Any) -> str:
    return f"{key}={_value(value)}"


def _path_list(values: list[str], base_dir: Path) -> list[str]:
    resolved = []
    for value in values:
        expanded = os.path.expandvars(os.path.expanduser(str(value)))
        if "://" in expanded:
            resolved.append(expanded)
            continue
        path = Path(expanded)
        if not path.is_absolute():
            path = base_dir / path
        resolved.append(str(path.resolve()))
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if _get(config, "rollout.backend", "sglang") != "sglang":
        raise ValueError("This launcher is intentionally scoped to rollout.backend=sglang")
    return config


def build_overrides(config: dict[str, Any], repo_root: Path = REPO_ROOT) -> list[str]:
    train_files = _path_list(_get(config, "data.train_files", []), repo_root)
    val_files = _path_list(_get(config, "data.val_files", []), repo_root)
    reward_path = _path_list([_get(config, "reward.path", "src/branch_grpo/reward.py")], repo_root)[0]

    overrides = [
        _override("algorithm.adv_estimator", _get(config, "algorithm.adv_estimator", "grpo")),
        _override("algorithm.use_kl_in_reward", _get(config, "algorithm.use_kl_in_reward", False)),
        _override(
            "algorithm.norm_adv_by_std_in_grpo",
            _get(config, "algorithm.norm_adv_by_std_in_grpo", True),
        ),
        _override("data.train_files", train_files),
        _override("data.val_files", val_files),
        _override("data.train_batch_size", _get(config, "data.train_batch_size", 64)),
        _override("data.max_prompt_length", _get(config, "data.max_prompt_length", 1024)),
        _override("data.max_response_length", _get(config, "data.max_response_length", 2048)),
        _override("data.filter_overlong_prompts", _get(config, "data.filter_overlong_prompts", True)),
        _override("data.truncation", _get(config, "data.truncation", "error")),
        _override("data.prompt_key", _get(config, "data.prompt_key", "prompt")),
        _override("data.reward_fn_key", _get(config, "data.reward_fn_key", "data_source")),
        _override("data.return_raw_chat", _get(config, "data.return_raw_chat", True)),
        _override("data.trust_remote_code", _get(config, "data.trust_remote_code", True)),
        _override("actor_rollout_ref.model.path", _get(config, "model.path", "Qwen/Qwen2.5-Math-7B")),
        _override("actor_rollout_ref.model.trust_remote_code", _get(config, "model.trust_remote_code", True)),
        _override("actor_rollout_ref.model.use_remove_padding", _get(config, "model.use_remove_padding", True)),
        _override(
            "actor_rollout_ref.model.enable_gradient_checkpointing",
            _get(config, "model.enable_gradient_checkpointing", True),
        ),
        _override("actor_rollout_ref.actor.optim.lr", _get(config, "actor.lr", 1e-6)),
        _override("actor_rollout_ref.actor.ppo_mini_batch_size", _get(config, "actor.ppo_mini_batch_size", 64)),
        _override("actor_rollout_ref.actor.use_dynamic_bsz", _get(config, "actor.use_dynamic_bsz", True)),
        _override(
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
            _get(config, "actor.ppo_max_token_len_per_gpu", 24576),
        ),
        _override("actor_rollout_ref.actor.use_kl_loss", _get(config, "actor.use_kl_loss", True)),
        _override("actor_rollout_ref.actor.kl_loss_coef", _get(config, "actor.kl_loss_coef", 0.001)),
        _override("actor_rollout_ref.actor.kl_loss_type", _get(config, "actor.kl_loss_type", "low_var_kl")),
        _override("actor_rollout_ref.actor.entropy_coeff", _get(config, "actor.entropy_coeff", 0)),
        _override("actor_rollout_ref.actor.grad_clip", _get(config, "actor.grad_clip", 1.0)),
        _override("actor_rollout_ref.actor.fsdp_config.param_offload", _get(config, "actor.param_offload", False)),
        _override(
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
            _get(config, "actor.optimizer_offload", False),
        ),
        _override("actor_rollout_ref.rollout.name", "sglang"),
        _override("actor_rollout_ref.rollout.n", _get(config, "rollout.n", 8)),
        _override("actor_rollout_ref.rollout.temperature", _get(config, "rollout.temperature", 1.0)),
        _override("actor_rollout_ref.rollout.top_p", _get(config, "rollout.top_p", 1.0)),
        _override(
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            _get(config, "rollout.tensor_model_parallel_size", 2),
        ),
        _override(
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            _get(config, "rollout.gpu_memory_utilization", 0.6),
        ),
        _override(
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
            _get(config, "rollout.log_prob_use_dynamic_bsz", True),
        ),
        _override(
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
            _get(config, "rollout.log_prob_max_token_len_per_gpu", 24576),
        ),
        _override(
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            _get(config, "rollout.max_num_batched_tokens", 8192),
        ),
        _override("actor_rollout_ref.rollout.max_num_seqs", _get(config, "rollout.max_num_seqs", 1024)),
        _override(
            "actor_rollout_ref.rollout.multi_stage_wake_up",
            _get(config, "rollout.multi_stage_wake_up", False),
        ),
        _override("actor_rollout_ref.rollout.free_cache_engine", _get(config, "rollout.free_cache_engine", True)),
        _override(
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz",
            _get(config, "ref.log_prob_use_dynamic_bsz", True),
        ),
        _override(
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
            _get(config, "ref.log_prob_max_token_len_per_gpu", 24576),
        ),
        _override("actor_rollout_ref.ref.fsdp_config.param_offload", _get(config, "ref.param_offload", True)),
        _override("custom_reward_function.path", reward_path),
        _override("custom_reward_function.name", _get(config, "reward.name", "compute_score")),
        _override("trainer.balance_batch", True),
        _override("trainer.project_name", _get(config, "trainer.project_name", "branch_grpo")),
        _override("trainer.experiment_name", _get(config, "trainer.experiment_name", "grpo_sglang")),
        _override("trainer.logger", _get(config, "trainer.logger", ["console"])),
        _override("trainer.nnodes", _get(config, "trainer.nnodes", 1)),
        _override("trainer.n_gpus_per_node", _get(config, "trainer.n_gpus_per_node", 4)),
        _override("trainer.total_epochs", _get(config, "trainer.total_epochs", 3)),
        _override("trainer.save_freq", _get(config, "trainer.save_freq", 20)),
        _override("trainer.test_freq", _get(config, "trainer.test_freq", 5)),
        _override("trainer.val_before_train", _get(config, "trainer.val_before_train", True)),
        _override(
            "trainer.default_local_dir",
            str((repo_root / _get(config, "trainer.default_local_dir", "checkpoints/branch_grpo")).resolve()),
        ),
    ]

    val_batch_size = _get(config, "data.val_batch_size", None)
    if val_batch_size is not None:
        overrides.append(_override("data.val_batch_size", val_batch_size))

    lora_rank = int(_get(config, "model.lora.rank", 0) or 0)
    if lora_rank > 0:
        overrides.extend(
            [
                _override("actor_rollout_ref.model.lora_rank", lora_rank),
                _override("actor_rollout_ref.model.lora_alpha", _get(config, "model.lora.alpha", 32)),
                _override("actor_rollout_ref.model.target_modules", _get(config, "model.lora.target_modules", "all-linear")),
            ]
        )

    attention_backend = _get(config, "rollout.attention_backend", None)
    if attention_backend:
        overrides.append(
            _override("+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend", attention_backend)
        )

    agent_num_workers = _get(config, "rollout.agent.num_workers", None)
    if agent_num_workers is not None:
        overrides.append(_override("actor_rollout_ref.rollout.agent.num_workers", agent_num_workers))

    agent_loop_manager_class = _get(config, "rollout.agent.agent_loop_manager_class", None)
    if agent_loop_manager_class:
        overrides.append(
            _override("+actor_rollout_ref.rollout.agent.agent_loop_manager_class", agent_loop_manager_class)
        )

    return overrides


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def build_env(config: dict[str, Any]) -> dict[str, str]:
    mh_config = config.get("mh") or {}
    if not isinstance(mh_config, dict):
        return {}

    mapping = {
        "variant": "BRANCH_GRPO_MH_VARIANT",
        "alpha": "BRANCH_GRPO_MH_ALPHA",
        "steps": "BRANCH_GRPO_MH_STEPS",
        "min_prefix_tokens": "BRANCH_GRPO_MH_MIN_PREFIX_TOKENS",
        "dedup_exact": "BRANCH_GRPO_MH_DEDUP_EXACT",
        "seed": "BRANCH_GRPO_MH_SEED",
    }
    env = {}
    for key, env_key in mapping.items():
        value = mh_config.get(key)
        if value is not None:
            env[env_key] = _env_value(value)
    return env


def build_command(config: dict[str, Any], extra_overrides: list[str]) -> list[str]:
    module = _get(config, "verl.entrypoint", "verl.trainer.main_ppo")
    return [sys.executable, "-m", module, *build_overrides(config), *extra_overrides]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_sglang.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the VeRL command without executing it")
    return parser.parse_known_args()


def main() -> None:
    args, extra_overrides = parse_args()
    config = load_config(args.config)
    command = build_command(config, extra_overrides)

    if args.dry_run:
        print(shlex.join(command))
        return

    env = os.environ.copy()
    env.update(build_env(config))
    pythonpath = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
