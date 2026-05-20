"""Proposal kernel utilities for MH-based GRPO.

The heavy generation work is now handled by SGLang backend (src/sglang_backend.py).
This module retains only:
- select_branch_point: entropy-based branch point selection
- exact_dedup: exact token-level deduplication of candidate pools
"""

import torch
from typing import List
from .model_utils import compute_token_entropies


def select_branch_point(
    model, tokenizer, prompt: str, response_ids: torch.Tensor, strategy: str
) -> int:
    """Select a branch point in the response for MH proposal generation.

    Args:
        model: PyTorch model (for entropy computation).
        tokenizer: Tokenizer.
        prompt: Input prompt string.
        response_ids: Current response token IDs.
        strategy: Branch selection strategy ("entropy_top1", "entropy_top4_random_one", "random_position").

    Returns:
        Integer index into response_ids for the branch point.
    """
    if strategy == "random_position":
        return torch.randint(1, len(response_ids), (1,)).item()

    entropies = compute_token_entropies(model, tokenizer, prompt, response_ids)

    if strategy == "entropy_top1":
        return int(entropies.argmax().item())

    if strategy == "entropy_top4_random_one":
        top4 = entropies.topk(min(4, len(entropies))).indices
        return int(top4[torch.randint(0, len(top4), (1,))].item())

    return int(entropies.argmax().item())


def exact_dedup(candidate_pool: list) -> list:
    """Remove exact duplicate response sequences from a candidate pool.

    Keeps the first occurrence of each unique token sequence.
    """
    seen = set()
    deduped = []
    for c in candidate_pool:
        key = tuple(c["response_ids"].tolist())
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped
