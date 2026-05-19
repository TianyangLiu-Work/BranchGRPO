import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple


def compute_grpo_loss(
    model,
    tokenizer,
    prompts: List[str],
    candidate_pools: List[List[Dict]],
    rewards_list: List[List[float]],
    clip_epsilon: float = 0.2,
    loss_mask: str = "branch_after_only",
) -> Tuple[torch.Tensor, Dict]:
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_kl = 0.0
    total_clip_frac = 0.0
    total_tokens = 0
    num_prompts = len(prompts)

    for prompt_idx, (prompt, pool, rewards) in enumerate(
        zip(prompts, candidate_pools, rewards_list)
    ):
        M = len(pool)
        if M == 0:
            continue
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        mu = rewards_t.mean()
        sigma = rewards_t.std() + 1e-8
        advantages = (rewards_t - mu) / sigma

        prompt_loss = torch.tensor(0.0, device=device)
        prompt_kl = 0.0
        prompt_clip_frac = 0.0
        num_tokens = 0

        for i, candidate in enumerate(pool):
            response_ids = candidate["response_ids"].to(device)
            branch_point = candidate.get("branch_point")
            old_token_logprobs = candidate.get("old_token_logprobs")
            if old_token_logprobs is None:
                continue

            prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = prompt_enc.input_ids.shape[1]

            full_ids = torch.cat([prompt_enc.input_ids[0], response_ids])
            full_attn = torch.ones_like(full_ids).unsqueeze(0)

            outputs = model(input_ids=full_ids.unsqueeze(0), attention_mask=full_attn)
            logits = outputs.logits[0]
            logprobs_new = F.log_softmax(logits, dim=-1)

            for t in range(len(response_ids)):
                if loss_mask == "branch_after_only" and branch_point is not None:
                    if t < branch_point:
                        continue

                pos = prompt_len + t - 1
                token_id = response_ids[t].item()

                logp_new = logprobs_new[pos, token_id]
                logp_old = old_token_logprobs[t]

                ratio = torch.exp(logp_new - logp_old)
                advantage = advantages[i]

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                token_loss = -torch.min(surr1, surr2)

                prompt_loss = prompt_loss + token_loss
                prompt_kl += (logp_old - logp_new.item()) if isinstance(logp_new, torch.Tensor) else (logp_old - logp_new)
                if ratio > 1.0 + clip_epsilon or ratio < 1.0 - clip_epsilon:
                    prompt_clip_frac += 1.0
                num_tokens += 1

        if num_tokens > 0:
            total_loss = total_loss + prompt_loss / num_tokens
            total_kl += prompt_kl / num_tokens
            total_clip_frac += prompt_clip_frac / num_tokens
        total_tokens += num_tokens

    total_loss = total_loss / max(num_prompts, 1)
    total_kl = total_kl / max(num_prompts, 1)
    total_clip_frac = total_clip_frac / max(num_prompts, 1)

    metrics = {
        "loss": total_loss.item(),
        "approx_kl": total_kl,
        "clip_fraction": total_clip_frac,
    }

    return total_loss, metrics


def compute_advantages(rewards: List[float]) -> List[float]:
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mu = rewards_t.mean()
    sigma = rewards_t.std() + 1e-8
    return ((rewards_t - mu) / sigma).tolist()
