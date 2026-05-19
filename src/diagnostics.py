import torch
import math
from typing import List, Dict

try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False
    wandb = None


def log_mh_diagnostics(candidate_pools: List[List[Dict]], step: int):
    all_accepted = []
    all_rejected = []
    all_accept_logprobs = []
    step_rewards_by_source = {}
    total_proposals = 0
    total_accepted = 0

    for pool in candidate_pools:
        for c in pool:
            source = c.get("source", "unknown")
            if source not in step_rewards_by_source:
                step_rewards_by_source[source] = []
            if "reward" in c:
                step_rewards_by_source[source].append(c["reward"])

            if c.get("source") == "proposal":
                total_proposals += 1
                if c.get("accepted"):
                    total_accepted += 1
                    all_accepted.append(c.get("reward", 0))
                    all_accept_logprobs.append(c.get("accept_logprob", 0))
                else:
                    all_rejected.append(c.get("reward", 0))

    metrics = {}
    if total_proposals > 0:
        metrics["mh/acceptance_rate"] = total_accepted / total_proposals
    if all_accept_logprobs:
        metrics["mh/mean_accept_logprob"] = sum(all_accept_logprobs) / len(all_accept_logprobs)
    if all_accepted:
        metrics["mh/accepted_reward_mean"] = sum(all_accepted) / len(all_accepted)
        metrics["mh/accepted_correct_rate"] = sum(1 for r in all_accepted if r > 0.5) / len(all_accepted)
    if all_rejected:
        metrics["mh/rejected_reward_mean"] = sum(all_rejected) / len(all_rejected)
        metrics["mh/rejected_correct_rate"] = sum(1 for r in all_rejected if r > 0.5) / len(
            all_rejected
        )

    for source, rewards in step_rewards_by_source.items():
        if rewards:
            metrics[f"mh/reward_mean_{source}"] = sum(rewards) / len(rewards)
            metrics[f"mh/reward_std_{source}"] = (
                (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5
            )

    if _has_wandb and wandb.run:
        wandb.log(metrics, step=step)

    return metrics


def log_group_quality(candidate_pools: List[List[Dict]], step: int):
    group_sizes = []
    group_reward_means = []
    group_reward_stds = []
    all_correct_count = 0
    all_wrong_count = 0

    for pool in candidate_pools:
        if not pool:
            continue
        rewards = [c.get("reward", 0) for c in pool]
        group_sizes.append(len(pool))
        mu = sum(rewards) / len(rewards)
        group_reward_means.append(mu)
        if len(rewards) > 1:
            std = (sum((r - mu) ** 2 for r in rewards) / len(rewards)) ** 0.5
        else:
            std = 0.0
        group_reward_stds.append(std)

        if all(r > 0.5 for r in rewards):
            all_correct_count += 1
        if all(r < 0.5 for r in rewards):
            all_wrong_count += 1

    metrics = {}
    if group_sizes:
        metrics["group/size_mean"] = sum(group_sizes) / len(group_sizes)
    if group_reward_means:
        metrics["group/reward_mean"] = sum(group_reward_means) / len(group_reward_means)
    if group_reward_stds:
        metrics["group/reward_std"] = sum(group_reward_stds) / len(group_reward_stds)
    if candidate_pools:
        metrics["group/all_correct_rate"] = all_correct_count / len(candidate_pools)
        metrics["group/all_wrong_rate"] = all_wrong_count / len(candidate_pools)

    if _has_wandb and wandb.run:
        wandb.log(metrics, step=step)

    return metrics


def log_training_metrics(metrics: Dict, step: int):
    m = {f"train/{k}": v for k, v in metrics.items()}
    if _has_wandb and wandb.run:
        wandb.log(m, step=step)


def log_eval_metrics(metrics: Dict, step: int):
    m = {f"eval/{k}": v for k, v in metrics.items()}
    if _has_wandb and wandb.run:
        wandb.log(m, step=step)
