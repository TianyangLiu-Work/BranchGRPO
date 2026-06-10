# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Power-SMC configuration and pure helpers for vLLM V1.

The scheduler integration is intentionally separate from this module.  Keeping
these pieces pure makes the request-side validation and correctness tests
independent of CUDA, workers, and KV-cache state.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

PowerSMCProposal = Literal["power_temperature"]
_POWER_SMC_CHILD_ID_RE = re.compile(r"^psmc(?P<particle_idx>\d+)_(?P<parent_id>.+)$")


@dataclass(frozen=True)
class PowerSMCConfig:
    """Validated Power-SMC decoding configuration."""

    enabled: bool = False
    alpha: float = 4.0
    particles: int = 32
    block_size: int = 64
    ess_threshold: float = 0.5
    alpha_ramp_tokens: int = 400
    proposal: PowerSMCProposal = "power_temperature"
    return_diagnostics: bool = False
    kv_cow: bool = True
    kv_pool_diagnostics: bool = False

    @classmethod
    def from_sampling_params(
        cls,
        sampling_params: SamplingParams | None,
    ) -> PowerSMCConfig | None:
        if sampling_params is None or sampling_params.extra_args is None:
            return None
        raw_config = sampling_params.extra_args.get("power_smc")
        if raw_config is None:
            return None
        if not isinstance(raw_config, Mapping):
            raise VLLMValidationError(
                "extra_args['power_smc'] must be a mapping.",
                parameter="extra_args.power_smc",
                value=raw_config,
            )
        enabled = bool(raw_config.get("enabled", False))
        if not enabled:
            return None

        config = cls(
            enabled=True,
            alpha=float(raw_config.get("alpha", cls.alpha)),
            particles=int(raw_config.get("particles", cls.particles)),
            block_size=int(raw_config.get("block_size", cls.block_size)),
            ess_threshold=float(
                raw_config.get("ess_threshold", cls.ess_threshold)),
            alpha_ramp_tokens=int(
                raw_config.get("alpha_ramp_tokens",
                               cls.alpha_ramp_tokens)),
            proposal=raw_config.get("proposal", cls.proposal),
            return_diagnostics=bool(
                raw_config.get("return_diagnostics",
                               cls.return_diagnostics)),
            kv_cow=bool(raw_config.get("kv_cow", cls.kv_cow)),
            kv_pool_diagnostics=bool(
                raw_config.get("kv_pool_diagnostics",
                               cls.kv_pool_diagnostics)),
        )
        config.validate_sampling_params(sampling_params)
        return config

    def validate_sampling_params(self, sampling_params: SamplingParams) -> None:
        self._validate_self()

        unsupported: list[str] = []
        if sampling_params.n != 1:
            unsupported.append("n > 1")
        if sampling_params.temperature != 1.0:
            unsupported.append("temperature != 1")
        if sampling_params.top_p != 1.0:
            unsupported.append("top_p != 1")
        if sampling_params.top_k not in (0, -1):
            unsupported.append("top_k enabled")
        if sampling_params.min_p != 0.0:
            unsupported.append("min_p enabled")
        if sampling_params.repetition_penalty != 1.0:
            unsupported.append("repetition_penalty")
        if sampling_params.frequency_penalty != 0.0:
            unsupported.append("frequency_penalty")
        if sampling_params.presence_penalty != 0.0:
            unsupported.append("presence_penalty")
        if sampling_params.structured_outputs is not None:
            unsupported.append("structured_outputs")
        if sampling_params.allowed_token_ids is not None:
            unsupported.append("allowed_token_ids")
        if sampling_params.bad_words:
            unsupported.append("bad_words")
        if sampling_params.logit_bias:
            unsupported.append("logit_bias")

        if unsupported:
            raise VLLMValidationError(
                "Power-SMC V1 currently supports only full-softmax sampling "
                "without logits processors or penalties. Unsupported: "
                + ", ".join(unsupported),
                parameter="extra_args.power_smc",
                value=sampling_params.extra_args,
            )

    def _validate_self(self) -> None:
        if self.alpha <= 0.0:
            raise VLLMValidationError(
                "Power-SMC alpha must be > 0.",
                parameter="extra_args.power_smc.alpha",
                value=self.alpha,
            )
        if self.particles < 1:
            raise VLLMValidationError(
                "Power-SMC particles must be >= 1.",
                parameter="extra_args.power_smc.particles",
                value=self.particles,
            )
        if self.block_size < 1:
            raise VLLMValidationError(
                "Power-SMC block_size must be >= 1.",
                parameter="extra_args.power_smc.block_size",
                value=self.block_size,
            )
        if not 0.0 < self.ess_threshold <= 1.0:
            raise VLLMValidationError(
                "Power-SMC ess_threshold must be in (0, 1].",
                parameter="extra_args.power_smc.ess_threshold",
                value=self.ess_threshold,
            )
        if self.alpha_ramp_tokens < 1:
            raise VLLMValidationError(
                "Power-SMC alpha_ramp_tokens must be >= 1.",
                parameter="extra_args.power_smc.alpha_ramp_tokens",
                value=self.alpha_ramp_tokens,
            )
        if self.proposal != "power_temperature":
            raise VLLMValidationError(
                "Power-SMC currently supports only proposal='power_temperature'.",
                parameter="extra_args.power_smc.proposal",
                value=self.proposal,
            )


def alpha_ramp(step: int, alpha_final: float, ramp_tokens: int) -> float:
    """Linear alpha ramp from 1.0 to ``alpha_final``."""

    if ramp_tokens <= 1:
        return float(alpha_final)
    if step < ramp_tokens:
        frac = float(step + 1) / float(ramp_tokens)
        return 1.0 + (float(alpha_final) - 1.0) * frac
    return float(alpha_final)


def proposal_temperature(alpha_t: float) -> float:
    """Power-temperature proposal T_t = 1 / alpha_t."""

    if alpha_t <= 0.0:
        raise ValueError(f"alpha_t must be positive, got {alpha_t}.")
    return 1.0 / alpha_t


def _max_log_weight_and_uniform(
    log_weights: Sequence[float],
) -> tuple[float, int, bool]:
    iterator = iter(log_weights)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("max() arg is an empty sequence") from None

    max_log_weight = first
    count = 1
    all_equal = True
    for weight in iterator:
        count += 1
        if weight != first:
            all_equal = False
        if weight > max_log_weight:
            max_log_weight = weight
    return max_log_weight, count, all_equal


def normalize_log_weights(log_weights: Sequence[float]) -> list[float]:
    max_log_weight, count, all_equal = _max_log_weight_and_uniform(log_weights)
    if all_equal:
        return [1.0 / count] * count
    unnormalized = [math.exp(w - max_log_weight) for w in log_weights]
    total = math.fsum(unnormalized)
    if total == 0.0 or not math.isfinite(total):
        return [1.0 / count] * count
    return [w / total for w in unnormalized]


def effective_sample_size(weights: Sequence[float]) -> float:
    denom = math.fsum(weight * weight for weight in weights)
    return 0.0 if denom <= 0.0 else 1.0 / denom


def _normalize_log_weights_and_ess(
    log_weights: Sequence[float],
) -> tuple[list[float], float]:
    max_log_weight, count, all_equal = _max_log_weight_and_uniform(log_weights)
    if all_equal:
        return [1.0 / count] * count, float(count)
    return _normalize_log_weights_and_ess_from_max(
        log_weights,
        max_log_weight=max_log_weight,
        count=count,
    )


def _normalize_log_weights_and_ess_from_max(
    log_weights: Sequence[float],
    *,
    max_log_weight: float,
    count: int,
) -> tuple[list[float], float]:
    unnormalized = [math.exp(w - max_log_weight) for w in log_weights]
    total = math.fsum(unnormalized)
    if total == 0.0 or not math.isfinite(total):
        return [1.0 / count] * count, float(count)

    weights: list[float] = []
    denom = 0.0
    for unnormalized_weight in unnormalized:
        weight = unnormalized_weight / total
        weights.append(weight)
        denom += weight * weight
    ess = 0.0 if denom <= 0.0 else 1.0 / denom
    return weights, ess


def update_log_weight(
    *,
    log_weight: float,
    cum_logp: float,
    prev_alpha: float,
    alpha_t: float,
    base_logp: float,
    proposal_logq: float,
) -> tuple[float, float, float]:
    """Update one particle's log weight in log space.

    Target density at step ``t`` is annealed as ``p(y_<=t)^alpha_t`` while
    proposal density contributes ``q(y_t | prefix)``.
    """

    if alpha_t != prev_alpha:
        log_weight += (alpha_t - prev_alpha) * cum_logp
    log_weight += alpha_t * base_logp - proposal_logq
    cum_logp += base_logp
    return log_weight, cum_logp, alpha_t


def systematic_resample(
    weights: Sequence[float],
    rng: random.Random,
) -> list[int]:
    n = len(weights)
    inv_n = 1.0 / n
    start = rng.random() * inv_n
    ancestors: list[int] = []
    append_ancestor = ancestors.append
    cumulative = 0.0
    idx = 0
    for i in range(n):
        pos = start + i * inv_n
        while idx < n - 1 and cumulative + weights[idx] < pos:
            cumulative += weights[idx]
            idx += 1
        append_ancestor(idx)
    return ancestors


def config_dict(config: PowerSMCConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "alpha": config.alpha,
        "particles": config.particles,
        "block_size": config.block_size,
        "ess_threshold": config.ess_threshold,
        "alpha_ramp_tokens": config.alpha_ramp_tokens,
        "proposal": config.proposal,
        "return_diagnostics": config.return_diagnostics,
        "kv_cow": config.kv_cow,
        "kv_pool_diagnostics": config.kv_pool_diagnostics,
    }


def validate_power_smc_engine_features(
    config: PowerSMCConfig | None,
    *,
    stream_input: bool = False,
    lora_request: Any | None = None,
    is_encoder_decoder: bool = False,
    speculative_config: Any | None = None,
    kv_block_size: int | None = None,
) -> None:
    """Reject engine-level features outside the first Power-SMC scope."""

    if config is None:
        return

    unsupported: list[str] = []
    if stream_input:
        unsupported.append("streaming input")
    if is_encoder_decoder:
        unsupported.append("encoder-decoder models")
    if speculative_config is not None:
        unsupported.append("speculative decoding")
    if kv_block_size is not None and config.block_size % kv_block_size != 0:
        unsupported.append(
            f"block_size not divisible by kv_block_size={kv_block_size}")

    if unsupported:
        raise VLLMValidationError(
            "Power-SMC V1 currently supports only non-streaming decoder-only "
            "requests without speculative decoding. Unsupported: "
            + ", ".join(unsupported),
            parameter="extra_args.power_smc",
            value=config_dict(config),
        )


@dataclass(frozen=True)
class PowerSMCChildRequestInfo:
    parent_id: str
    particle_idx: int


def make_power_smc_child_request_id(parent_id: str, particle_idx: int) -> str:
    return f"psmc{particle_idx}_{parent_id}"


def parse_power_smc_child_request_id(
    request_id: str,
) -> PowerSMCChildRequestInfo | None:
    match = _POWER_SMC_CHILD_ID_RE.match(request_id)
    if match is None:
        return None
    return PowerSMCChildRequestInfo(
        parent_id=match.group("parent_id"),
        particle_idx=int(match.group("particle_idx")),
    )


@dataclass
class PowerSMCParticleState:
    """Mutable scheduler-side state for one Power-SMC particle."""

    token_ids: list[int] = field(default_factory=list)
    log_weight: float = 0.0
    cum_logp: float = 0.0
    prev_alpha: float = 1.0
    done: bool = False
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    history_start_length: int = 0
    history_start_state: tuple[float, float, float] = (0.0, 0.0, 1.0)
    boundary_state_history: list[tuple[int, tuple[float, float, float]]] = (
        field(default_factory=list))

    def state_at(self, output_length: int) -> tuple[float, float, float]:
        if output_length == len(self.token_ids):
            return self.log_weight, self.cum_logp, self.prev_alpha
        if output_length <= 0:
            return 0.0, 0.0, 1.0
        if output_length == self.history_start_length:
            return self.history_start_state
        if output_length < self.history_start_length:
            raise ValueError(
                "Power-SMC particle state before compacted history was "
                "requested.")
        for length, state in reversed(self.boundary_state_history):
            if length == output_length:
                return state
        raise ValueError(
            "Power-SMC particle state is not available for this length.")


@dataclass(frozen=True)
class PowerSMCResamplePlan:
    """Concrete request reset plan produced by a block-boundary resample."""

    ancestors: list[int]
    particle_token_ids: list[list[int]]


class PowerSMCGroupManager:
    """Pure Power-SMC particle-group state machine.

    This manager owns algorithmic state only. The scheduler integration is
    responsible for mapping these logical particle slots to child requests and
    KV-cache blocks.
    """

    def __init__(
        self,
        config: PowerSMCConfig,
        rng: random.Random | None = None,
    ) -> None:
        config._validate_self()
        self.config = config
        self.rng = rng or random.Random()
        self._particle_count = config.particles
        self._block_size = config.block_size
        self._resample_ess_threshold = config.ess_threshold * config.particles
        self._alpha_final = float(config.alpha)
        self._alpha_ramp_tokens = config.alpha_ramp_tokens
        self._alpha_is_constant = self._alpha_ramp_tokens <= 1
        self._alpha_slope = (
            0.0 if self._alpha_is_constant else
            (self._alpha_final - 1.0) / float(self._alpha_ramp_tokens)
        )
        self.particles = [
            PowerSMCParticleState() for _ in range(self._particle_count)
        ]
        self.resample_count = 0
        self.ess_history: list[float] = []
        self.unique_ancestors_per_resample: list[int] = []
        self.ancestor_history: list[list[int]] = []
        self.kv_resample_events: list[dict[str, Any]] = []
        self.maybe_resample_calls = 0
        self.block_boundary_checks = 0
        self.resample_skip_reasons: dict[str, int] = {}
        self.last_boundary_check_length = 0
        self.last_resample_plan: PowerSMCResamplePlan | None = None
        self.chosen_particle: int | None = None
        self._chosen_final_ess: float | None = None

    def update_after_token(
        self,
        particle_idx: int,
        token_id: int,
        *,
        base_logp: float,
        proposal_logq: float,
        done: bool = False,
        finish_reason: str | None = None,
        stop_reason: int | str | None = None,
    ) -> bool:
        particle = self.particles[particle_idx]
        if particle.done:
            raise ValueError(
                f"Power-SMC particle {particle_idx} is already done.")

        self._chosen_final_ess = None
        token_ids = particle.token_ids
        step = len(token_ids)
        alpha_t = (
            self._alpha_final
            if self._alpha_is_constant or step >= self._alpha_ramp_tokens else
            1.0 + self._alpha_slope * float(step + 1)
        )
        if alpha_t != particle.prev_alpha:
            particle.log_weight += (
                alpha_t - particle.prev_alpha) * particle.cum_logp
        particle.log_weight += alpha_t * base_logp - proposal_logq
        particle.cum_logp += base_logp
        particle.prev_alpha = alpha_t
        token_ids.append(token_id)
        output_length = step + 1
        is_block_boundary = output_length % self._block_size == 0
        if is_block_boundary:
            particle.boundary_state_history.append((
                output_length,
                (particle.log_weight, particle.cum_logp, particle.prev_alpha),
            ))
        particle.done = done
        particle.finish_reason = finish_reason if done else None
        particle.stop_reason = stop_reason if done else None
        return is_block_boundary

    def _alpha_at_step(self, step: int) -> float:
        if step >= self._alpha_ramp_tokens:
            return self._alpha_final
        if self._alpha_is_constant:
            return self._alpha_final
        return 1.0 + self._alpha_slope * float(step + 1)

    def maybe_resample(self) -> bool:
        self.maybe_resample_calls += 1
        self.last_resample_plan = None
        if self._particle_count == 1:
            self._record_resample_skip("particles_one")
            return False
        boundary_length, has_done_particle = self._eligible_resample_boundary()
        if has_done_particle:
            self._record_resample_skip("done_particle")
            return False
        if boundary_length is None:
            self._record_resample_skip("not_block_boundary")
            return False
        if boundary_length <= self.last_boundary_check_length:
            self._record_resample_skip("stale_block_boundary")
            return False

        self.block_boundary_checks += 1
        boundary_states: list[tuple[float, float, float]] = []
        log_weights: list[float] = []
        append_boundary_state = boundary_states.append
        append_log_weight = log_weights.append
        first_log_weight: float | None = None
        max_log_weight = -math.inf
        all_log_weights_equal = True
        for particle in self.particles:
            state = particle.state_at(boundary_length)
            append_boundary_state(state)
            log_weight = state[0]
            append_log_weight(log_weight)
            if first_log_weight is None:
                first_log_weight = log_weight
                max_log_weight = log_weight
            else:
                if log_weight != first_log_weight:
                    all_log_weights_equal = False
                if log_weight > max_log_weight:
                    max_log_weight = log_weight
        if all_log_weights_equal:
            ess = float(self._particle_count)
            weights = None
        else:
            weights, ess = _normalize_log_weights_and_ess_from_max(
                log_weights,
                max_log_weight=max_log_weight,
                count=self._particle_count,
            )
        self.ess_history.append(ess)
        self.last_boundary_check_length = boundary_length
        if ess >= self._resample_ess_threshold:
            self._record_resample_skip("ess_above_threshold")
            return False

        assert weights is not None
        ancestors = systematic_resample(weights, self.rng)
        resampled_particles: list[PowerSMCParticleState] = []
        particle_token_ids: list[list[int]] = []
        ancestor_prefixes: dict[int, tuple[int, ...]] = {}
        for ancestor in ancestors:
            ancestor_particle = self.particles[ancestor]
            _, cum_logp, prev_alpha = boundary_states[ancestor]
            prefix = ancestor_prefixes.get(ancestor)
            if prefix is None:
                prefix = tuple(ancestor_particle.token_ids[:boundary_length])
                ancestor_prefixes[ancestor] = prefix
            token_ids = list(prefix)
            particle_token_ids.append(token_ids)
            resampled_particles.append(
                PowerSMCParticleState(
                    token_ids=token_ids,
                    log_weight=0.0,
                    cum_logp=cum_logp,
                    prev_alpha=prev_alpha,
                    done=False,
                    finish_reason=None,
                    stop_reason=None,
                    history_start_length=len(token_ids),
                    history_start_state=(0.0, cum_logp, prev_alpha),
                ))
        self.particles = resampled_particles
        self._chosen_final_ess = None
        self.last_resample_plan = PowerSMCResamplePlan(
            ancestors=ancestors,
            particle_token_ids=particle_token_ids,
        )
        self.resample_count += 1
        self.ancestor_history.append(ancestors)
        self.unique_ancestors_per_resample.append(len(ancestor_prefixes))
        return True

    def normalized_weights(self) -> list[float]:
        return normalize_log_weights(
            [particle.log_weight for particle in self.particles])

    def record_kv_resample_event(self, event: Mapping[str, Any]) -> None:
        self.kv_resample_events.append(dict(event))

    def _record_resample_skip(self, reason: str) -> None:
        self.resample_skip_reasons[reason] = (
            self.resample_skip_reasons.get(reason, 0) + 1)

    def final_select(self) -> int:
        log_weights = [particle.log_weight for particle in self.particles]
        weights, final_ess = _normalize_log_weights_and_ess(log_weights)
        self._chosen_final_ess = final_ess
        self.chosen_particle = sample_index(weights, self.rng)
        return self.chosen_particle

    def diagnostics(self) -> dict[str, Any]:
        particle_lengths: list[int] = []
        min_particle_length: int | None = None
        max_particle_length = 0
        done_count = 0
        log_weights: list[float] = []
        finish_reason_counts: dict[str, int] = {}
        stop_reason_counts: dict[str, int] = {}
        for particle in self.particles:
            length = len(particle.token_ids)
            particle_lengths.append(length)
            if min_particle_length is None or length < min_particle_length:
                min_particle_length = length
            if length > max_particle_length:
                max_particle_length = length
            if particle.done:
                done_count += 1
            log_weights.append(particle.log_weight)
            if particle.finish_reason is not None:
                finish_reason_counts[particle.finish_reason] = (
                    finish_reason_counts.get(particle.finish_reason, 0) + 1)
            if particle.stop_reason is not None:
                key = str(particle.stop_reason)
                stop_reason_counts[key] = stop_reason_counts.get(key, 0) + 1
        if self._chosen_final_ess is None:
            _, final_ess = _normalize_log_weights_and_ess(log_weights)
        else:
            final_ess = self._chosen_final_ess
        kv_alias_success_count = 0
        kv_alias_fallback_count = 0
        kv_aliased_blocks = 0
        kv_aliased_tokens = 0
        kv_cow_physical_blocks = 0
        kv_cow_saved_blocks = 0
        kv_cow_saved_tokens = 0
        for event in self.kv_resample_events:
            kv_alias_success_count += int(event.get("alias_success_count", 0))
            kv_alias_fallback_count += int(event.get("fallback_count", 0))
            kv_aliased_blocks += int(event.get("aliased_blocks", 0))
            kv_aliased_tokens += int(event.get("aliased_tokens", 0))
            kv_cow_physical_blocks += int(
                event.get("cow_physical_blocks", 0))
            kv_cow_saved_blocks += int(event.get("cow_saved_blocks", 0))
            kv_cow_saved_tokens += int(event.get("cow_saved_tokens", 0))
        return {
            "alpha": self.config.alpha,
            "particles": self.config.particles,
            "chosen_particle": self.chosen_particle,
            "resample_count": self.resample_count,
            "final_ess": final_ess,
            "mean_ess": math.fsum(self.ess_history) / len(self.ess_history)
            if self.ess_history else final_ess,
            "ess_history": list(self.ess_history),
            "ancestor_history": [
                list(ancestors) for ancestors in self.ancestor_history
            ],
            "maybe_resample_calls": self.maybe_resample_calls,
            "block_boundary_checks": self.block_boundary_checks,
            "last_boundary_check_length": self.last_boundary_check_length,
            "resample_skip_reasons": dict(
                sorted(self.resample_skip_reasons.items())),
            "unique_ancestors_per_resample":
                list(self.unique_ancestors_per_resample),
            "particle_lengths": particle_lengths,
            "min_particle_length": min_particle_length or 0,
            "max_particle_length": max_particle_length,
            "done_count": done_count,
            "finish_reason_counts": finish_reason_counts,
            "stop_reason_counts": stop_reason_counts,
            "avg_log_weight": math.fsum(log_weights) / self._particle_count,
            "kv_resample_events": list(self.kv_resample_events),
            "kv_alias_success_count": kv_alias_success_count,
            "kv_alias_fallback_count": kv_alias_fallback_count,
            "kv_aliased_blocks": kv_aliased_blocks,
            "kv_aliased_tokens": kv_aliased_tokens,
            "kv_cow_physical_blocks": kv_cow_physical_blocks,
            "kv_cow_saved_blocks": kv_cow_saved_blocks,
            "kv_cow_saved_tokens": kv_cow_saved_tokens,
        }

    def _eligible_resample_boundary(self) -> tuple[int | None, bool]:
        min_length: int | None = None
        for particle in self.particles:
            if particle.done:
                return None, True
            length = len(particle.token_ids)
            if length == 0:
                min_length = 0
                continue
            if min_length is None or length < min_length:
                min_length = length
        if min_length is None:
            return None, False
        boundary = min_length // self._block_size * self._block_size
        if boundary == 0:
            return None, False
        return boundary, False


def sample_index(weights: Sequence[float], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if draw <= cumulative:
            return idx
    return len(weights) - 1
