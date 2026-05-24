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
    chain_id: int = 0
    accept_logprob: float | None = None
    branch_point: int | None = None
    top_logprobs: list[list[float]] | None = None
    branch_entropy: float | None = None
    branch_strategy: str = "sample_logprob"

    @property
    def sequence_logprob(self) -> float:
        return sequence_logprob(self.logprobs)


def sequence_logprob(logprobs: Iterable[float | None]) -> float:
    return float(sum(float(value or 0.0) for value in logprobs))


def mh_accept_logprob(
    current_logprob: float, proposal_logprob: float, alpha: float
) -> float:
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


def truncated_entropy(logprobs: Iterable[float | None]) -> float:
    """Entropy of a distribution renormalized over the provided top-k logprobs."""

    values = [
        float(value)
        for value in logprobs
        if value is not None and math.isfinite(float(value))
    ]
    if not values:
        return 0.0

    max_value = max(values)
    weights = [math.exp(value - max_value) for value in values]
    total = sum(weights)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for weight in weights:
        prob = weight / total
        if prob > 0:
            entropy -= prob * math.log(prob)
    return float(entropy)


def choose_high_entropy_branch(
    top_logprobs: list[list[float]],
    *,
    min_prefix_tokens: int = 1,
    max_prefix_tokens: int | None = None,
) -> tuple[int, float | None]:
    """Pick a branch point before the highest top-k truncated-entropy sampled token."""

    if len(top_logprobs) <= 1:
        return 0, None

    entropies = [truncated_entropy(values) for values in top_logprobs]
    min_prefix_tokens = max(0, min_prefix_tokens)
    upper = len(entropies) - 1
    if max_prefix_tokens is not None:
        upper = min(upper, max_prefix_tokens)
    if upper <= min_prefix_tokens:
        branch_point = min(min_prefix_tokens, len(entropies) - 1)
        return branch_point, entropies[branch_point] if entropies else None

    candidates = range(min_prefix_tokens, upper + 1)
    branch_point = max(candidates, key=lambda idx: entropies[idx])
    return branch_point, entropies[branch_point]


def coerce_top_logprobs(raw: object) -> list[list[float]]:
    """Normalize common SGLang/OpenAI top-logprob shapes to per-token logprob lists."""

    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in (
            "output_top_logprobs",
            "top_logprobs",
            "output_top_log_probs",
            "top_log_probs",
        ):
            if key in raw:
                return coerce_top_logprobs(raw[key])
        for key in ("output_top_logprobs_val", "top_logprobs_val"):
            if key in raw:
                return coerce_top_logprobs(raw[key])
        return [_extract_logprob_values(raw)]
    if isinstance(raw, (list, tuple)):
        if raw and all(_is_number(item) for item in raw):
            return [[float(item) for item in raw]]
        if _is_logprob_tuple(raw):
            return [[float(raw[0])]]
        return [_extract_logprob_values(item) for item in raw]
    return []


def _extract_logprob_values(raw: object) -> list[float]:
    if raw is None:
        return []
    if _is_number(raw):
        return [float(raw)]
    if isinstance(raw, dict):
        for key in ("logprob", "log_prob", "log_probs", "value"):
            if key in raw and _is_number(raw[key]):
                return [float(raw[key])]
        values: list[float] = []
        for value in raw.values():
            values.extend(_extract_logprob_values(value))
        return values
    if isinstance(raw, (list, tuple)):
        if raw and all(_is_number(item) for item in raw):
            return [float(item) for item in raw]
        if _is_logprob_tuple(raw):
            return [float(raw[0])]
        values = []
        for item in raw:
            values.extend(_extract_logprob_values(item))
        return values
    return []


def _is_logprob_tuple(raw: object) -> bool:
    return (
        isinstance(raw, (list, tuple))
        and len(raw) >= 2
        and _is_number(raw[0])
        and _is_number(raw[1])
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
