"""Small helpers for MH proposal-pool rollout generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass
class MHCandidate:
    response_ids: list[int]
    logprobs: list[float]
    source: str
    mh_step: int
    accepted: bool
    accept_logprob: float | None = None
    branch_point: int | None = None

    @property
    def sequence_logprob(self) -> float:
        return sequence_logprob(self.logprobs)


def sequence_logprob(logprobs: Iterable[float | None]) -> float:
    return float(sum(float(value or 0.0) for value in logprobs))


def mh_accept_logprob(current_logprob: float, proposal_logprob: float, alpha: float) -> float:
    return min(0.0, float(alpha) * (float(proposal_logprob) - float(current_logprob)))


def should_accept(log_accept_prob: float, rng: random.Random | None = None) -> bool:
    rng = rng or random
    return math.log(max(rng.random(), 1e-12)) < log_accept_prob


def choose_low_logprob_branch(
    logprobs: list[float],
    *,
    min_prefix_tokens: int = 1,
    max_prefix_tokens: int | None = None,
) -> int:
    """Pick a branch point before the lowest-logprob sampled token.

    This is a cheap proxy for the plan's entropy branch point. SGLang returns
    sampled-token logprobs here, not full next-token entropy.
    """

    if len(logprobs) <= 1:
        return 0

    min_prefix_tokens = max(0, min_prefix_tokens)
    upper = len(logprobs) - 1
    if max_prefix_tokens is not None:
        upper = min(upper, max_prefix_tokens)
    if upper <= min_prefix_tokens:
        return min(min_prefix_tokens, len(logprobs) - 1)

    candidates = range(min_prefix_tokens, upper + 1)
    return min(candidates, key=lambda idx: float(logprobs[idx]))


def exact_dedup(candidates: list[MHCandidate]) -> list[MHCandidate]:
    seen: set[tuple[int, ...]] = set()
    output: list[MHCandidate] = []
    for candidate in candidates:
        key = tuple(candidate.response_ids)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output
