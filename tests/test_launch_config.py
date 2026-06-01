from pathlib import Path

from scripts.launch_verl_grpo import build_command, build_env, build_overrides, load_config


def test_build_overrides_selects_sglang_and_custom_reward(tmp_path: Path):
    config = {
        "model": {
            "path": "Qwen/Qwen2.5-3B-Instruct",
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
            "chains": 32,
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
    assert env["BRANCH_GRPO_MH_CHAINS"] == "32"
    assert env["BRANCH_GRPO_MH_DEDUP_EXACT"] == "0"
    assert env["BRANCH_GRPO_MH_BRANCH_STRATEGY"] == "topk_entropy"
    assert env["BRANCH_GRPO_MH_TOP_LOGPROBS"] == "20"


def test_qwen3_ps_config_maps_old_runtime_knobs():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs/qwen3_8b_ps.yaml")

    overrides = build_overrides(config, repo_root=repo_root)
    env = build_env(config)
    command = build_command(config, [])

    assert config["data"]["apply_chat_template_kwargs"] == {"enable_thinking": True}
    assert config["model"]["override_config"] == {
        "attn_implementation": "flash_attention_2"
    }
    assert config["rollout"]["val_kwargs"]["do_sample"] is True
    assert config["reward"]["kwargs"]["overlong_buffer_cfg"]["enable"] is True

    assert "verl.trainer.main_ppo" in command
    assert "actor_rollout_ref.model.path=Qwen/Qwen3-8B" in overrides
    assert "++data.apply_chat_template_kwargs={enable_thinking:True}" in overrides
    assert not any("lora" in item for item in overrides)
    assert (
        'data.train_files=["/data2/fyyang/branch-grpo-old/data/math__combined_54.4k.parquet"]'
        in overrides
    )
    assert (
        'data.val_files=["/data2/fyyang/branch-grpo-old/data/math__aime_repeated_8x_240.parquet"]'
        in overrides
    )
    assert "data.max_response_length=10000" in overrides
    assert "actor_rollout_ref.rollout.response_length=10000" in overrides
    assert "actor_rollout_ref.rollout.n=9" in overrides
    assert "actor_rollout_ref.rollout.top_k=-1" in overrides
    assert "++actor_rollout_ref.rollout.repetition_penalty=1.0" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=11024" in overrides
    assert "data.train_batch_size=4" in overrides
    assert "++data.gen_batch_size=4" in overrides
    assert "data.seed=1" in overrides
    assert "trainer.n_gpus_per_node=2" in overrides
    assert "trainer.total_training_steps=200" in overrides
    assert "trainer.save_freq=50" in overrides
    assert "trainer.test_freq=10" in overrides
    assert "trainer.resume_mode=disable" in overrides
    assert "trainer.experiment_name=qwen3_8b_ps_2gpu_fresh" in overrides
    assert any(
        item.startswith("trainer.default_local_dir=")
        and item.endswith("outputs/ckpts/qwen3_8b_ps_2gpu_fresh")
        for item in overrides
    )
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=4" in overrides
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_low=0.2" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_high=0.28" in overrides
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps=10" in overrides
    assert "actor_rollout_ref.actor.optim.weight_decay=0.1" in overrides
    assert "++actor_rollout_ref.model.override_config={attn_implementation:flash_attention_2}" in overrides
    assert (
        "++actor_rollout_ref.rollout.val_kwargs="
        "{temperature:1.0,top_p:0.7,top_k:-1,do_sample:True,n:1}"
    ) in overrides
    assert (
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class="
        "branch_grpo.mh_agent_loop.MHPowerAgentLoopManager"
    ) in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in overrides
    assert any(
        item.startswith("reward.reward_manager.module.path=")
        and item.endswith("src/branch_grpo/reward_overlong.py")
        for item in overrides
    )
    assert "reward.reward_manager.name=OverlongRewardManager" in overrides
    assert "reward.reward_manager.source=importlib" in overrides
    assert not any(item.startswith("reward.reward_manager.module.name=") for item in overrides)
    assert "++reward.reward_kwargs.max_resp_len=10000" in overrides
    assert any(
        item.startswith("++reward.reward_kwargs.overlong_buffer_cfg=")
        and "len:1024" in item
        for item in overrides
    )

    assert env["BRANCH_GRPO_MH_VARIANT"] == "all_proposals"
    assert env["BRANCH_GRPO_MH_ALPHA"] == "2"
    assert env["BRANCH_GRPO_MH_STEPS"] == "4"
    assert env["BRANCH_GRPO_MH_DEDUP_EXACT"] == "0"
    assert env["BRANCH_GRPO_MH_BRANCH_STRATEGY"] == "topk_entropy"
    assert env["BRANCH_GRPO_MH_TOP_LOGPROBS"] == "20"
    assert "BRANCH_GRPO_DIAGNOSTICS_JSONL" not in env
    assert "BRANCH_GRPO_MAX_RESPONSE_LENGTH" not in env
    assert "BRANCH_GRPO_OVERLONG_ENABLE" not in env
    assert "BRANCH_GRPO_OVERLONG_BUFFER_LEN" not in env
    assert "BRANCH_GRPO_OVERLONG_PENALTY_FACTOR" not in env


def test_qwen3_grpo_config_disables_power_sampling():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs/qwen3_8b_grpo.yaml")

    overrides = build_overrides(config, repo_root=repo_root)
    env = build_env(config)

    assert config["data"]["apply_chat_template_kwargs"] == {"enable_thinking": True}
    assert config["model"]["path"] == "Qwen/Qwen3-8B"
    assert "actor_rollout_ref.model.path=Qwen/Qwen3-8B" in overrides
    assert "++data.apply_chat_template_kwargs={enable_thinking:True}" in overrides
    assert not any("lora" in item for item in overrides)
    assert "trainer.n_gpus_per_node=2" in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in overrides
    assert "trainer.experiment_name=qwen3_8b_grpo" in overrides
    assert "trainer.resume_mode=disable" in overrides
    assert any(
        item.startswith("trainer.default_local_dir=")
        and item.endswith("outputs/ckpts/qwen3_8b_grpo")
        for item in overrides
    )
    assert not any("agent_loop_manager_class" in item for item in overrides)
    assert not any(key.startswith("BRANCH_GRPO_MH_") for key in env)
