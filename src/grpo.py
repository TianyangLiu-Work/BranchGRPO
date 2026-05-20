"""GRPO loss and advantage computation, powered by VeRL's core algorithms.

Uses batched forward passes for efficient logprob computation.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    compute_policy_loss_vanilla,
)


def compute_grpo_loss(
    model, tokenizer, prompts, candidate_pools, rewards_list,
    clip_epsilon=0.2, loss_mask="branch_after_only", loss_agg_mode="token-mean",
):
    device = next(model.parameters()).device

    # Flatten
    flat_cands, flat_rewards, group_idx, branch_pts = [], [], [], []
    for pi, (pool, rewards) in enumerate(zip(candidate_pools, rewards_list)):
        for c, r in zip(pool, rewards):
            flat_cands.append(c)
            flat_rewards.append(r)
            group_idx.append(pi)
            branch_pts.append(c.get("branch_point"))

    if not flat_cands:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}

    N = len(flat_cands)
    group_np = np.array(group_idx, dtype=np.int64)
    pad_id = tokenizer.pad_token_id or 0

    # -- Build new logprobs via batched forward, per prompt group --
    new_lps_list = [[] for _ in range(N)]
    old_lps_list = [[] for _ in range(N)]
    mask_list = [[] for _ in range(N)]

    # Group candidates by prompt
    prompt_groups = {}
    for i in range(N):
        pi = group_idx[i]
        prompt_groups.setdefault(pi, []).append(i)

    max_chunk = 4  # chunk size to avoid OOM from large logits tensor

    for pi, cand_indices in prompt_groups.items():
        prompt = prompts[pi]
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
        p_len = len(prompt_ids)

        # Process candidates in chunks
        ci_list = list(cand_indices)
        for c_start in range(0, len(ci_list), max_chunk):
            c_chunk = ci_list[c_start:c_start + max_chunk]
            B = len(c_chunk)
            resp_lens = [len(flat_cands[i]["response_ids"]) for i in c_chunk]
            max_r = max(resp_lens)

            input_ids = torch.full((B, p_len + max_r), pad_id, device=device)
            for b, ci in enumerate(c_chunk):
                r_ids = flat_cands[ci]["response_ids"].to(device)
                input_ids[b, :p_len] = prompt_ids
                input_ids[b, p_len:p_len + len(r_ids)] = r_ids

            attn_mask = (input_ids != pad_id).long()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logprobs = F.log_softmax(outputs.logits, dim=-1)

            for b, ci in enumerate(c_chunk):
                r_ids = flat_cands[ci]["response_ids"].to(device)
                r_len = len(r_ids)
                old_lps_i = flat_cands[ci].get("old_token_logprobs")
                if old_lps_i is None:
                    old_lps_i = [0.0] * r_len
                bp = branch_pts[ci]

                new_lp, old_lp, mask = [], [], []
                for t in range(r_len):
                    pos = p_len + t - 1
                    tok = r_ids[t].item()
                    new_lp.append(logprobs[b, pos, tok])
                    old_lp.append(float(old_lps_i[t]) if t < len(old_lps_i) else 0.0)
                    if loss_mask == "branch_after_only" and bp is not None:
                        mask.append(1.0 if t >= bp else 0.0)
                    else:
                        mask.append(1.0)
                new_lps_list[ci] = new_lp
                old_lps_list[ci] = old_lp
                mask_list[ci] = mask

    # -- Pad to max length for batch tensor construction --
    max_len = max((len(x) for x in new_lps_list), default=1)
    max_len = max(max_len, 1)

    new_rows, old_rows, mask_rows, reward_rows = [], [], [], []

    def _safe_old(lst, L):
        out = [float(x) if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else 0.0 for x in lst[:L]]
        return torch.tensor(out + [0.0] * (L - len(out)), device=device)

    for i in range(N):
        L = len(new_lps_list[i])
        L = max(L, 1)

        new_pad = torch.zeros(L, device=device)
        for j, v in enumerate(new_lps_list[i]):
            if isinstance(v, torch.Tensor):
                new_pad[j] = torch.where(torch.isfinite(v), v, torch.tensor(0.0, device=device))
            elif v is not None:
                new_pad[j] = float(v)
        if L < max_len:
            new_pad = torch.cat([new_pad, torch.zeros(max_len - L, device=device)])
        new_rows.append(new_pad)

        old_rows.append(_safe_old(old_lps_list[i], max_len))

        m = mask_list[i][:L] + [0.0] * (max_len - L)
        mask_rows.append(torch.tensor(m[:max_len], device=device))

        r = torch.zeros(max_len, device=device)
        last_pos = sum(mask_list[i]) - 1
        if last_pos >= 0 and last_pos < max_len:
            r[int(last_pos)] = flat_rewards[i]
        else:
            r[-1] = flat_rewards[i]
        reward_rows.append(r)

    new_logprobs = torch.stack(new_rows)      # [N, max_len] with grad
    old_logprobs = torch.stack(old_rows)       # [N, max_len]
    response_mask = torch.stack(mask_rows)      # [N, max_len]
    token_rewards = torch.stack(reward_rows)    # [N, max_len]

    # -- VeRL advantage --
    with torch.no_grad():
        advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=group_np,
            norm_adv_by_std_in_grpo=True,
        )

    # -- VeRL policy loss --
    fake_cfg = type("FakeActorConfig", (), {
        "clip_ratio": clip_epsilon,
        "clip_ratio_low": None, "clip_ratio_high": None,
        "loss_agg_mode": loss_agg_mode, "global_batch_info": {},
        "get": lambda s, k, d: getattr(s, k, d),
    })()

    pg_loss, pg_metrics = compute_policy_loss_vanilla(
        old_log_prob=old_logprobs, log_prob=new_logprobs,
        advantages=advantages, response_mask=response_mask,
        loss_agg_mode=loss_agg_mode, config=fake_cfg,
    )

    return pg_loss, {
        "loss": pg_loss.item(),
        "approx_kl": pg_metrics.get("actor/ppo_kl", 0.0),
        "clip_fraction": pg_metrics.get("actor/pg_clipfrac", 0.0),
    }
