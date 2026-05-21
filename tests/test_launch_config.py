from pathlib import Path

from scripts.launch_verl_grpo import build_command, build_overrides


def test_build_overrides_selects_sglang_and_custom_reward(tmp_path: Path):
    config = {
        "model": {"path": "Qwen/Qwen2.5-Math-7B", "lora": {"rank": 8, "alpha": 16}},
        "data": {"train_files": ["data/train.parquet"], "val_files": ["data/test.parquet"]},
        "rollout": {"backend": "sglang", "attention_backend": "flashinfer"},
        "reward": {"path": "src/branch_grpo/reward.py", "name": "compute_score"},
    }
    overrides = build_overrides(config, repo_root=tmp_path)
    assert "actor_rollout_ref.rollout.name=sglang" in overrides
    assert "algorithm.adv_estimator=grpo" in overrides
    assert "custom_reward_function.name=compute_score" in overrides
    assert any(item.startswith("custom_reward_function.path=") for item in overrides)
    assert "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer" in overrides


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

