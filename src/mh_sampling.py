import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple
from .proposal_kernel import (
    select_branch_point,
    propose_span_then_continue,
    exact_dedup,
)


@torch.no_grad()
def mh_power_sampling(model, tokenizer, prompt: str, config) -> List[Dict]:
    method = config.method.method
    max_response_length = config.training.max_response_length
    temperature = config.method.temperature
    top_p = config.method.top_p
    num_rollouts = config.method.num_rollouts_per_prompt

    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)

    if method in ("standard_grpo", "low_temp_grpo"):
        candidate_pool = []
        for _ in range(num_rollouts):
            ids = _generate_rollout(
                model, tokenizer, prompt_enc, temperature, top_p, max_response_length
            )
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
            candidate_pool.append({
                "response_ids": ids[0],
                "response_text": text,
                "source": "sample",
                "mh_step": 0,
                "accepted": True,
                "branch_point": None,
            })
        _compute_and_store_old_logprobs(model, tokenizer, prompt, candidate_pool)
        return candidate_pool

    mh_cfg = config.method.mh
    alpha = mh_cfg.alpha
    mh_steps = mh_cfg.mh_steps
    span_len = mh_cfg.span_len
    branch_strategy = mh_cfg.branch_selection
    include_chain_states = config.method.include_chain_states
    include_rejected = config.method.include_rejected
    exact_dedup_flag = config.method.exact_dedup

    if method == "mh_final_only":
        candidate_pool = []
        for _ in range(num_rollouts):
            init_ids = _generate_rollout(
                model, tokenizer, prompt_enc, temperature, top_p, max_response_length
            )
            current_ids = init_ids[0]
            for k in range(1, mh_steps + 1):
                branch_point = select_branch_point(
                    model, tokenizer, prompt, current_ids, branch_strategy
                )
                proposed_ids, _ = propose_span_then_continue(
                    model, tokenizer, prompt, current_ids,
                    branch_point, span_len, temperature, top_p, max_response_length,
                )
                logp_cur = _sequence_logprob(model, tokenizer, prompt, current_ids)
                logp_prop = _sequence_logprob(model, tokenizer, prompt, proposed_ids)
                accept_logp = min(0.0, alpha * (logp_prop - logp_cur))
                accept = math.log(torch.rand(1).item() + 1e-10) < accept_logp
                if accept:
                    current_ids = proposed_ids
            final_text = tokenizer.decode(current_ids, skip_special_tokens=True)
            candidate_pool.append({
                "response_ids": current_ids,
                "response_text": final_text,
                "source": "final",
                "mh_step": mh_steps,
                "accepted": True,
                "branch_point": None,
            })
        _compute_and_store_old_logprobs(model, tokenizer, prompt, candidate_pool)
        return candidate_pool

    init_ids = _generate_rollout(
        model, tokenizer, prompt_enc, temperature, top_p, max_response_length
    )
    init_text = tokenizer.decode(init_ids[0], skip_special_tokens=True)
    candidate_pool = [{
        "response_ids": init_ids[0],
        "response_text": init_text,
        "source": "initial",
        "mh_step": 0,
        "accepted": True,
        "branch_point": None,
    }]

    current_ids = init_ids[0]
    for k in range(1, mh_steps + 1):
        branch_point = select_branch_point(
            model, tokenizer, prompt, current_ids, branch_strategy
        )
        proposed_ids, proposal_info = propose_span_then_continue(
            model, tokenizer, prompt, current_ids,
            branch_point, span_len, temperature, top_p, max_response_length,
        )

        logp_cur = _sequence_logprob(model, tokenizer, prompt, current_ids)
        logp_prop = _sequence_logprob(model, tokenizer, prompt, proposed_ids)

        accept_logp = min(0.0, alpha * (logp_prop - logp_cur))
        accept = math.log(torch.rand(1).item() + 1e-10) < accept_logp

        prop_text = tokenizer.decode(proposed_ids, skip_special_tokens=True)
        candidate_pool.append({
            "response_ids": proposed_ids,
            "response_text": prop_text,
            "source": "proposal",
            "mh_step": k,
            "accepted": accept,
            "accept_logprob": accept_logp,
            "branch_point": branch_point,
            "proposal_info": proposal_info,
        })

        if accept:
            current_ids = proposed_ids

        if include_chain_states:
            chain_text = tokenizer.decode(current_ids, skip_special_tokens=True)
            candidate_pool.append({
                "response_ids": current_ids,
                "response_text": chain_text,
                "source": "chain_state",
                "mh_step": k,
                "accepted": True,
                "branch_point": branch_point,
            })

    if exact_dedup_flag or method == "mh_all_proposals_dedup":
        candidate_pool = exact_dedup(candidate_pool)

    if not include_rejected and method == "mh_chain_only":
        candidate_pool = [c for c in candidate_pool if c["accepted"]]
        seen = {}
        deduped = []
        for c in candidate_pool:
            key = tuple(c["response_ids"].tolist())
            if key not in seen:
                seen[key] = c
                deduped.append(c)
        candidate_pool = deduped

    _compute_and_store_old_logprobs(model, tokenizer, prompt, candidate_pool)
    return candidate_pool


@torch.no_grad()
def _compute_and_store_old_logprobs(model, tokenizer, prompt: str, pool: List[Dict]):
    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_enc.input_ids.shape[1]

    for candidate in pool:
        response_ids = candidate["response_ids"].to(device)
        full_ids = torch.cat([prompt_enc.input_ids[0], response_ids])
        full_attn = torch.ones_like(full_ids).unsqueeze(0)

        outputs = model(input_ids=full_ids.unsqueeze(0), attention_mask=full_attn)
        logits = outputs.logits[0]
        logprobs = F.log_softmax(logits, dim=-1)

        token_logprobs = []
        for t in range(len(response_ids)):
            pos = prompt_len + t - 1
            token_id = response_ids[t].item()
            token_logprobs.append(logprobs[pos, token_id].item())

        candidate["old_token_logprobs"] = token_logprobs


def _sequence_logprob(model, tokenizer, prompt: str, response_ids: torch.Tensor) -> float:
    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_enc.input_ids.shape[1]

    full_ids = torch.cat([prompt_enc.input_ids[0], response_ids.to(device)])
    full_attn = torch.ones_like(full_ids).unsqueeze(0)

    outputs = model(input_ids=full_ids.unsqueeze(0), attention_mask=full_attn)
    logits = outputs.logits[0]
    logprobs = F.log_softmax(logits, dim=-1)

    total = 0.0
    for t in range(len(response_ids)):
        pos = prompt_len + t - 1
        token_id = response_ids[t].item()
        total += logprobs[pos, token_id].item()
    return total


@torch.no_grad()
def _generate_rollout(
    model, tokenizer, prompt_enc, temperature: float, top_p: float, max_length: int
) -> torch.Tensor:
    input_ids = prompt_enc.input_ids.clone()
    attention_mask = prompt_enc.attention_mask.clone()
    generated = []

    for _ in range(max_length):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :] / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum_probs > top_p] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=input_ids.device)], dim=1
        )
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, device=attention_mask.device)], dim=1
        )

    return torch.tensor([generated])
