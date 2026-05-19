import torch
import torch.nn.functional as F
from typing import Tuple
from .model_utils import compute_token_entropies, sequence_logprob


def select_branch_point(
    model, tokenizer, prompt: str, response_ids: torch.Tensor, strategy: str
) -> int:
    if strategy == "random_position":
        return torch.randint(1, len(response_ids), (1,)).item()

    entropies = compute_token_entropies(model, tokenizer, prompt, response_ids)

    if strategy == "entropy_top1":
        return int(entropies.argmax().item())

    if strategy == "entropy_top4_random_one":
        top4 = entropies.topk(min(4, len(entropies))).indices
        return int(top4[torch.randint(0, len(top4), (1,))].item())

    return int(entropies.argmax().item())


@torch.no_grad()
def propose_span_then_continue(
    model,
    tokenizer,
    prompt: str,
    current_response_ids: torch.Tensor,
    branch_point: int,
    span_len: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_response_length: int = 2048,
) -> Tuple[torch.Tensor, dict]:
    device = next(model.parameters()).device
    prefix_ids = current_response_ids[:branch_point].to(device)

    new_response_ids = prefix_ids.clone().tolist()

    input_ids = torch.cat(
        [
            tokenizer(prompt, return_tensors="pt").input_ids[0].to(device),
            torch.tensor(new_response_ids, device=device, dtype=torch.long),
        ]
    ).unsqueeze(0)

    for step in range(span_len):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :] / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum_probs > top_p] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        new_response_ids.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device, dtype=torch.long)], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    continuation_ids = tokenizer(
        tokenizer.decode(new_response_ids, skip_special_tokens=True),
        return_tensors="pt",
    ).input_ids[0].tolist()

    remaining_len = max_response_length - len(new_response_ids)
    if remaining_len > 0:
        input_ids = torch.cat(
            [
                tokenizer(prompt, return_tensors="pt").input_ids[0].to(device),
                torch.tensor(continuation_ids, device=device, dtype=torch.long),
            ]
        ).unsqueeze(0)

        past_key_values = None
        for step in range(remaining_len):
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
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cum_probs > top_p] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            continuation_ids.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

    proposed_ids = torch.tensor(continuation_ids)
    info = {
        "branch_point": branch_point,
        "span_len": span_len,
        "proposal_temperature": temperature,
    }
    return proposed_ids, info


def exact_dedup(candidate_pool: list) -> list:
    seen = set()
    deduped = []
    for c in candidate_pool:
        key = tuple(c["response_ids"].tolist())
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped
