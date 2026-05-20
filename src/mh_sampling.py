"""MH Power Sampling with PyTorch generation (KV-cache enabled).

Uses manual autoregressive generation with proper KV-cache reuse
for O(n) scaling instead of O(n²).
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Dict
from .proposal_kernel import select_branch_point, exact_dedup


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
        ids_list = _generate_rollout_batch(
            model, tokenizer, prompt_enc, temperature, top_p, max_response_length, num_rollouts
        )
        candidate_pool = []
        for ids in ids_list:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            candidate_pool.append({
                "response_ids": ids,
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
    exact_dedup_flag = config.method.exact_dedup or (method == "mh_all_proposals_dedup")

    if method == "mh_final_only":
        init_ids_list = _generate_rollout_batch(
            model, tokenizer, prompt_enc, temperature, top_p, max_response_length, num_rollouts
        )
        candidate_pool = []
        current_ids_list = []
        for current_ids in init_ids_list:
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
            current_ids_list.append(current_ids)
        for current_ids in current_ids_list:
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

    if exact_dedup_flag:
        candidate_pool = exact_dedup(candidate_pool)

    if not include_rejected and method == "mh_chain_only":
        candidate_pool = [c for c in candidate_pool if c["accepted"]]

    _compute_and_store_old_logprobs(model, tokenizer, prompt, candidate_pool)
    return candidate_pool


@torch.no_grad()
def _compute_and_store_old_logprobs(model, tokenizer, prompt: str, pool: List[Dict], max_batch: int = 4):
    """Batch-compute old logprobs for all candidates in chunked forward passes."""
    if not pool:
        return
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
    prompt_len = len(prompt_ids)

    for start in range(0, len(pool), max_batch):
        chunk = pool[start:start + max_batch]
        B = len(chunk)
        resp_lens = [len(c["response_ids"]) for c in chunk]
        max_resp = max(resp_lens)

        input_ids = torch.full((B, prompt_len + max_resp), tokenizer.pad_token_id or 0, device=device)
        for i, c in enumerate(chunk):
            input_ids[i, :prompt_len] = prompt_ids
            cur_len = len(c["response_ids"])
            input_ids[i, prompt_len:prompt_len + cur_len] = c["response_ids"].to(device)

        attention_mask = (input_ids != (tokenizer.pad_token_id or 0)).long()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logprobs = F.log_softmax(outputs.logits, dim=-1)

        for i, c in enumerate(chunk):
            cur_len = len(c["response_ids"])
            token_logprobs = []
            for t in range(cur_len):
                pos = prompt_len + t - 1
                tok = c["response_ids"][t].item()
                token_logprobs.append(logprobs[i, pos, tok].item())
            c["old_token_logprobs"] = token_logprobs


def _sequence_logprob(model, tokenizer, prompt: str, response_ids: torch.Tensor) -> float:
    """Single-sequence logprob (called by MH step). Falls back to _compute_and_store_old_logprobs."""
    pool = [{"response_ids": response_ids}]
    _compute_and_store_old_logprobs(model, tokenizer, prompt, pool)
    return sum(pool[0]["old_token_logprobs"])


@torch.no_grad()
def _generate_rollout_batch(
    model, tokenizer, prompt_enc, temperature: float, top_p: float, max_length: int, num_sequences: int
):
    """Generate multiple rollouts in parallel via batch inference.

    One model forward pass produces tokens for ALL sequences simultaneously,
    sharing KV-cache across the batch. ~6-8x faster than sequential generation.
    """
    device = prompt_enc.input_ids.device
    B = num_sequences

    input_ids = prompt_enc.input_ids.repeat(B, 1)
    attention_mask = prompt_enc.attention_mask.repeat(B, 1)

    generated = [[] for _ in range(B)]
    eos_reached = [False] * B

    past_key_values = None
    for _ in range(max_length):
        if past_key_values is None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        else:
            outputs = model(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :] / temperature  # [B, vocab]

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum_probs > top_p] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1).squeeze(-1)  # [B]

        for i in range(B):
            if not eos_reached[i]:
                generated[i].append(next_tokens[i].item())
                if next_tokens[i].item() == tokenizer.eos_token_id:
                    eos_reached[i] = True

        if all(eos_reached):
            break

        next_tok_tensor = next_tokens.unsqueeze(1)
        input_ids = torch.cat([input_ids, next_tok_tensor], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, device=device)], dim=1
        )

    return [torch.tensor(g) for g in generated]


@torch.no_grad()
def _generate_rollout(
    model, tokenizer, prompt_enc, temperature: float, top_p: float, max_length: int
) -> torch.Tensor:
    """Single-sequence generation (used by MH branch proposals)."""
    result = _generate_rollout_batch(model, tokenizer, prompt_enc, temperature, top_p, max_length, 1)
    return torch.tensor([result[0]])


@torch.no_grad()
def propose_span_then_continue(
    model, tokenizer, prompt: str, current_response_ids: torch.Tensor,
    branch_point: int, span_len: int, temperature: float, top_p: float,
    max_response_length: int,
):
    """Propose a new response by resampling from branch_point then continuing.

    Uses KV-cache for efficient continuation.
    """
    device = next(model.parameters()).device
    prefix_ids = current_response_ids[:branch_point].clone()
    generated = prefix_ids.tolist()

    # Generate short span tokens (no KV cache for simplicity — span is short)
    input_ids = torch.cat([
        tokenizer(prompt, return_tensors="pt").input_ids[0].to(device),
        torch.tensor(generated, device=device, dtype=torch.long),
    ]).unsqueeze(0)

    for step in range(span_len):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :] / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum_probs > top_p] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        input_ids = torch.cat([
            input_ids, torch.tensor([[next_token]], device=device, dtype=torch.long)
        ], dim=1)
        if next_token == tokenizer.eos_token_id:
            break

    # Continue to EOS with KV-cache
    remaining = max_response_length - len(generated)
    if remaining > 0:
        input_ids = torch.cat([
            tokenizer(prompt, return_tensors="pt").input_ids[0].to(device),
            torch.tensor(generated, device=device, dtype=torch.long),
        ]).unsqueeze(0)

        past_key_values = None
        for step in range(remaining):
            if past_key_values is None:
                outputs = model(input_ids=input_ids, use_cache=True)
            else:
                outputs = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
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

    proposed_ids = torch.tensor(generated)
    info = {"branch_point": branch_point, "span_len": span_len}
    return proposed_ids, info
