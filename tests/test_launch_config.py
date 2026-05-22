from pathlib import Path

from scripts.launch_verl_grpo import build_command, build_env, build_overrides


def test_build_overrides_selects_sglang_and_custom_reward(tmp_path: Path):
    config = {
        "model": {
            "path": "Qwen/Qwen2.5-Math-7B",
            "lora": {"rank": 8, "alpha": 16, "merge": True},
        },
        "data": {
            "train_files": ["data/train.parquet"],
            "val_files": ["data/test.parquet"],
        },
        "actor": {"param_offload": True},
        "rollout": {
            "backend": "sglang",
            "attention_backend": "flashinfer",
            "load_format": "auto",
            "skip_tokenizer_init": False,
        },
        "trainer": {"use_legacy_worker_impl": "disable"},
        "reward": {"path": "src/branch_grpo/reward.py", "name": "compute_score"},
    }
    overrides = build_overrides(config, repo_root=tmp_path)
    assert "actor_rollout_ref.rollout.name=sglang" in overrides
    assert "algorithm.adv_estimator=grpo" in overrides
    assert "custom_reward_function.name=compute_score" in overrides
    assert "actor_rollout_ref.rollout.load_format=auto" in overrides
    assert "actor_rollout_ref.rollout.skip_tokenizer_init=False" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in overrides
    assert "+actor_rollout_ref.model.lora.merge=True" in overrides
    assert "trainer.use_legacy_worker_impl=disable" in overrides
    assert any(item.startswith("custom_reward_function.path=") for item in overrides)
    assert (
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer"
        in overrides
    )


def test_build_command_allows_extra_hydra_overrides(tmp_path: Path):
    config = {
        "verl": {"entrypoint": "verl.trainer.main_ppo"},
        "data": {"train_files": ["train.parquet"], "val_files": ["test.parquet"]},
        "rollout": {"backend": "sglang"},
    }
    command = build_command(config, ["trainer.total_epochs=1"])
    assert command[:3]
    assert "verl.trainer.main_ppo" in command
    assert "trainer.total_epochs=1" in command


def test_build_overrides_and_env_support_mh_agent_loop(tmp_path: Path):
    config = {
        "data": {"train_files": ["train.parquet"], "val_files": ["test.parquet"]},
        "rollout": {
            "backend": "sglang",
            "n": 9,
            "agent": {
                "num_workers": 8,
                "agent_loop_manager_class": "branch_grpo.mh_agent_loop.MHPowerAgentLoopManager",
            },
        },
        "mh": {
            "variant": "all_proposals",
            "alpha": 1.5,
            "steps": 4,
            "dedup_exact": False,
            "branch_strategy": "topk_entropy",
            "top_logprobs": 20,
        },
    }
    overrides = build_overrides(config, repo_root=tmp_path)
    assert "actor_rollout_ref.rollout.n=9" in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=8" in overrides
    assert (
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class="
        "branch_grpo.mh_agent_loop.MHPowerAgentLoopManager"
    ) in overrides

    env = build_env(config)
    assert env["BRANCH_GRPO_MH_VARIANT"] == "all_proposals"
    assert env["BRANCH_GRPO_MH_ALPHA"] == "1.5"
    assert env["BRANCH_GRPO_MH_STEPS"] == "4"
    assert env["BRANCH_GRPO_MH_DEDUP_EXACT"] == "0"
    assert env["BRANCH_GRPO_MH_BRANCH_STRATEGY"] == "topk_entropy"
    assert env["BRANCH_GRPO_MH_TOP_LOGPROBS"] == "20"
