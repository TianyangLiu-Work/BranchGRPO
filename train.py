"""BranchGRPO Training Entry Point — VeRL-powered with PyTorch generation."""

import argparse
import os
import random
import numpy as np
import torch

try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False
    wandb = None

from src.config import ExperimentConfig, MethodConfig, MHConfig
from src.data import load_train_data, load_val_data
from src.model_utils import load_model_and_tokenizer
from src.trainer import BranchGRPOTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="BranchGRPO Training (VeRL+PyTorch)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--method", type=str, default="mh_all_proposals",
                        choices=["standard_grpo", "low_temp_grpo", "mh_final_only",
                                 "mh_chain_only", "mh_all_proposals", "mh_all_proposals_dedup"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--train_prompts", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_response_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--mh_steps", type=int, default=4)
    parser.add_argument("--span_len", type=int, default=16)
    parser.add_argument("--loss_mask", type=str, default="branch_after_only",
                        choices=["full_response", "branch_after_only"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--val_samples", type=int, default=500)
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        mh_cfg = MHConfig(
            alpha=args.alpha,
            mh_steps=args.mh_steps,
            span_len=args.span_len,
        )
        method_cfg = MethodConfig(
            method=args.method,
            num_rollouts_per_prompt=args.num_rollouts,
            temperature=args.temperature,
            mh=mh_cfg,
            exact_dedup=(args.method == "mh_all_proposals_dedup"),
            include_chain_states=(args.method == "mh_chain_only"),
            include_rejected=(args.method in ("mh_all_proposals", "mh_all_proposals_dedup")),
        )
        config = ExperimentConfig(method=method_cfg)
        config.training.model_name = args.model
        config.training.train_prompts = args.train_prompts
        config.training.num_epochs = args.num_epochs
        config.training.learning_rate = args.learning_rate
        config.training.max_response_length = args.max_response_length
        config.logging.output_dir = args.output_dir
        config.loss_mask = args.loss_mask
        config.seed = args.seed

    set_seed(config.seed)
    config.logging.use_wandb = not args.no_wandb

    if args.run_name:
        config.logging.run_name = args.run_name
    elif config.logging.run_name is None:
        config.logging.run_name = config.get_method_name()

    os.makedirs(config.logging.output_dir, exist_ok=True)

    if config.logging.use_wandb and _has_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.logging.run_name,
            config={
                "method": config.method.method,
                "model": config.training.model_name,
                "lr": config.training.learning_rate,
                "epochs": config.training.num_epochs,
                "max_response_length": config.training.max_response_length,
                "loss_mask": config.loss_mask,
                "alpha": config.method.mh.alpha if config.method.mh else None,
                "mh_steps": config.method.mh.mh_steps if config.method.mh else None,
                "span_len": config.method.mh.span_len if config.method.mh else None,
                "num_rollouts": config.method.num_rollouts_per_prompt,
            },
        )

    print(f"Method: {config.get_method_name()}")
    print(f"Model: {config.training.model_name}")
    print(f"Train prompts: {config.training.train_prompts}")
    print(f"Loss mask: {config.loss_mask}")
    if config.method.mh:
        print(f"MH alpha={config.method.mh.alpha} steps={config.method.mh.mh_steps} span={config.method.mh.span_len}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(config)

    print("Loading data...")
    train_data = load_train_data(config, num_prompts=config.training.train_prompts)
    val_data = load_val_data(config, num_samples=args.val_samples)
    print(f"Train: {len(train_data)} prompts, Val: {len(val_data)} prompts")

    print("\nStarting training...")
    trainer = BranchGRPOTrainer(config, model, tokenizer, train_data, val_data)
    trainer.train()

    print("Training complete.")


if __name__ == "__main__":
    main()
