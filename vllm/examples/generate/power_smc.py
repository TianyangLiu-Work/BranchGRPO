# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Power-SMC decoding with vLLM offline inference.

This example implements an efficient, minimally invasive Power-SMC variant
using vLLM as the batched decoding backend. It targets the sequence-level
power distribution p(y | x)^alpha with a base-model proposal q=p, which keeps
the importance correction exact while requiring only sampled-token raw
logprobs instead of full-vocabulary logprobs.

The implementation uses:
  * batched particles through LLM.generate()
  * block-level ESS checks and systematic resampling
  * token-id prompts so vLLM prefix caching can reuse common prefixes after
    resampling
  * grouping identical particle prefixes into one request with n>1

Example:
    python examples/generate/power_smc.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --prompt "Solve: what is 19 * 23? Put the answer in \\boxed{}." \
        --alpha 4 --particles 64 --block-size 32 --max-tokens 512

Notes:
    The Power-SMC paper recommends the adaptive proposal temperature
    T_t = 1 / alpha_t. Correcting that proposal without bias requires either
    processed proposal logprobs and raw model logprobs for the same sampled
    tokens or full-vocabulary logprobs to compute the proposal normalizer.
    This public-API example intentionally uses q=p to avoid that extra cost.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import LLM, TokensPrompt
    from vllm.inputs import PromptType
else:
    LLM = Any
    PromptType = Any
    TokensPrompt = dict[str, list[int]]


@dataclass
class PowerSMCConfig:
    """Configuration for vLLM-backed Power-SMC decoding."""

    max_tokens: int = 512
    alpha: float = 4.0
    particles: int = 64
    ess_threshold: float = 0.5
    block_size: int = 32
    alpha_ramp_tokens: int = 128
    min_tokens: int = 0
    seed: int = 0
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0 for power sampling.")
        if self.particles < 1:
            raise ValueError("particles must be at least 1.")
        if not 0.0 < self.ess_threshold <= 1.0:
            raise ValueError("ess_threshold must be in (0, 1].")
        if self.block_size < 1:
            raise ValueError("block_size must be at least 1.")
        if self.alpha_ramp_tokens < 1:
            raise ValueError("alpha_ramp_tokens must be at least 1.")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative.")


@dataclass
class PowerSMCParticle:
    """Mutable state for one logical SMC particle."""

    token_ids: list[int] = field(default_factory=list)
    cum_logp: float = 0.0
    log_weight: float = 0.0
    prev_alpha: float = 1.0
    finished: bool = False
    finish_reason: str | None = None
    stop_reason: int | str | None = None


@dataclass
class PowerSMCOutput:
    """Final Power-SMC result."""

    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]
    selected_particle: int
    normalized_weights: list[float]
    particles: list[PowerSMCParticle]
    stats: dict[str, Any]


def alpha_ramp(step: int, alpha_final: float, ramp_tokens: int) -> float:
    """Linear alpha ramp from 1 to alpha_final."""

    if step < ramp_tokens:
        frac = float(step + 1) / float(ramp_tokens)
        return 1.0 + (float(alpha_final) - 1.0) * frac
    return float(alpha_final)


def normalize_log_weights(log_weights: Sequence[float]) -> list[float]:
    """Normalize log weights into probabilities."""

    max_log_weight = max(log_weights)
    unnormalized = [math.exp(w - max_log_weight) for w in log_weights]
    total = math.fsum(unnormalized)
    if total == 0.0 or not math.isfinite(total):
        uniform = 1.0 / len(log_weights)
        return [uniform for _ in log_weights]
    return [w / total for w in unnormalized]


def effective_sample_size(weights: Sequence[float]) -> float:
    """Compute ESS from normalized weights."""

    denom = math.fsum(w * w for w in weights)
    if denom <= 0.0:
        return 0.0
    return 1.0 / denom


def systematic_resample(weights: Sequence[float],
                        rng: random.Random) -> list[int]:
    """Draw systematic-resampling ancestors from normalized weights."""

    n = len(weights)
    start = rng.random() / n
    positions = [start + i / n for i in range(n)]

    ancestors: list[int] = []
    cumulative = 0.0
    idx = 0
    for pos in positions:
        while idx < n - 1 and cumulative + weights[idx] < pos:
            cumulative += weights[idx]
            idx += 1
        ancestors.append(idx)
    return ancestors


def sample_index(weights: Sequence[float], rng: random.Random) -> int:
    """Sample one index from normalized weights."""

    draw = rng.random()
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if draw <= cumulative:
            return idx
    return len(weights) - 1


def sampled_token_logprobs(completion: Any) -> list[float]:
    """Extract sampled-token logprobs from a vLLM completion."""

    if completion.logprobs is None:
        raise ValueError("Power-SMC requires SamplingParams(logprobs=0).")

    token_ids = list(completion.token_ids)
    if hasattr(completion.logprobs, "start_indices"):
        logprobs: list[float] = []
        if len(completion.logprobs) != len(token_ids):
            raise ValueError(
                "vLLM returned a different number of logprob positions "
                "than generated tokens.")
        for pos, token_id in enumerate(token_ids):
            start = completion.logprobs.start_indices[pos]
            end = completion.logprobs.end_indices[pos]
            for flat_idx in range(start, end):
                if completion.logprobs.token_ids[flat_idx] == token_id:
                    logprobs.append(completion.logprobs.logprobs[flat_idx])
                    break
            else:
                raise ValueError(
                    f"Missing sampled-token logprob for token id {token_id}.")
        return logprobs

    if len(completion.logprobs) != len(token_ids):
        raise ValueError(
            "vLLM returned a different number of logprob positions "
            "than generated tokens.")
    return [
        position_logprobs[token_id].logprob
        for token_id, position_logprobs in zip(token_ids, completion.logprobs)
    ]


def make_sampling_params(**kwargs: Any) -> Any:
    """Create SamplingParams while tolerating older installed vLLM wheels."""

    from vllm import SamplingParams

    signature = inspect.signature(SamplingParams)
    if "flat_logprobs" not in signature.parameters:
        kwargs.pop("flat_logprobs", None)
    return SamplingParams(**kwargs)


class VLLMPowerSMCSampler:
    """Power-SMC sampler built on the vLLM public offline API."""

    def __init__(self, llm: LLM, config: PowerSMCConfig) -> None:
        self.llm = llm
        self.config = config
        self.rng = random.Random(config.seed)
        self.tokenizer = llm.get_tokenizer()

    def generate(self, prompt: PromptType) -> PowerSMCOutput:
        prompt_token_ids = self._prompt_token_ids(prompt)
        particles = [
            PowerSMCParticle() for _ in range(self.config.particles)
        ]
        stats: dict[str, Any] = {
            "blocks": 0,
            "resample_count": 0,
            "ess_history": [],
            "active_particles_history": [],
            "prompt_groups_history": [],
            "ancestor_history": [],
        }

        while self._has_active_particles(particles):
            active_indices = [
                idx for idx, particle in enumerate(particles)
                if (not particle.finished
                    and len(particle.token_ids) < self.config.max_tokens)
            ]
            if not active_indices:
                break

            stats["blocks"] += 1
            stats["active_particles_history"].append(len(active_indices))

            prompt_groups: dict[tuple[int, ...], list[int]] = {}
            for idx in active_indices:
                key = tuple(particles[idx].token_ids)
                prompt_groups.setdefault(key, []).append(idx)
            stats["prompt_groups_history"].append(len(prompt_groups))

            prompts: list[TokensPrompt] = []
            sampling_params: list[Any] = []
            output_groups: list[list[int]] = []
            for suffix_token_ids, group_indices in prompt_groups.items():
                particle = particles[group_indices[0]]
                remaining = self.config.max_tokens - len(particle.token_ids)
                block_tokens = min(self.config.block_size, remaining)
                min_tokens = min(
                    block_tokens,
                    max(0, self.config.min_tokens - len(particle.token_ids)),
                )
                prompts.append({
                    "prompt_token_ids": prompt_token_ids + list(suffix_token_ids),
                })
                output_groups.append(group_indices)
                sampling_params.append(
                    make_sampling_params(
                        n=len(group_indices),
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                        max_tokens=block_tokens,
                        min_tokens=min_tokens,
                        stop_token_ids=self.config.stop_token_ids,
                        ignore_eos=self.config.ignore_eos,
                        logprobs=0,
                        flat_logprobs=True,
                        detokenize=False,
                        skip_special_tokens=False,
                    ))

            outputs = self.llm.generate(
                prompts,
                sampling_params,
                use_tqdm=False,
            )

            for group_indices, output in zip(output_groups, outputs):
                completions = sorted(output.outputs, key=lambda item: item.index)
                if len(completions) != len(group_indices):
                    raise ValueError(
                        "vLLM returned an unexpected number of particle "
                        "continuations.")
                for particle_idx, completion in zip(group_indices, completions):
                    self._apply_completion(particles[particle_idx], completion)

            if not self._has_active_particles(particles):
                break

            weights = normalize_log_weights(
                [particle.log_weight for particle in particles])
            ess = effective_sample_size(weights)
            stats["ess_history"].append(ess)

            if ess < self.config.ess_threshold * self.config.particles:
                ancestors = systematic_resample(weights, self.rng)
                particles = [copy.deepcopy(particles[i]) for i in ancestors]
                for particle in particles:
                    particle.log_weight = 0.0
                stats["resample_count"] += 1
                stats["ancestor_history"].append(ancestors)

        for particle in particles:
            self._finish_alpha_ramp(particle)

        final_weights = normalize_log_weights(
            [particle.log_weight for particle in particles])
        selected = sample_index(final_weights, self.rng)
        stats["final_ess"] = effective_sample_size(final_weights)
        stats["mean_ess"] = (
            math.fsum(stats["ess_history"]) / len(stats["ess_history"])
            if stats["ess_history"] else stats["final_ess"])
        stats["selected_particle"] = selected
        selected_tokens = particles[selected].token_ids
        text = self.tokenizer.decode(
            selected_tokens,
            skip_special_tokens=True,
        )
        return PowerSMCOutput(
            text=text,
            token_ids=selected_tokens,
            prompt_token_ids=prompt_token_ids,
            selected_particle=selected,
            normalized_weights=final_weights,
            particles=particles,
            stats=stats,
        )

    def _prompt_token_ids(self, prompt: PromptType) -> list[int]:
        if isinstance(prompt, str):
            return list(self.tokenizer.encode(prompt))
        if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            return list(prompt)
        if isinstance(prompt, Mapping) and "prompt_token_ids" in prompt:
            return list(prompt["prompt_token_ids"])
        raise TypeError(
            "Power-SMC example supports text prompts and token-id prompts.")

    def _update_particle_weight(self, particle: PowerSMCParticle,
                                token_logp: float) -> None:
        step = len(particle.token_ids)
        alpha_t = alpha_ramp(
            step,
            self.config.alpha,
            self.config.alpha_ramp_tokens,
        )

        delta = alpha_t - particle.prev_alpha
        if delta != 0.0:
            particle.log_weight += delta * particle.cum_logp
            particle.prev_alpha = alpha_t

        # Exact q=p proposal correction:
        # target increment contributes alpha_t * log p(token), while proposal
        # contributes log p(token), leaving (alpha_t - 1) * log p(token).
        particle.log_weight += (alpha_t - 1.0) * token_logp
        particle.cum_logp += token_logp

    def _apply_completion(self, particle: PowerSMCParticle,
                          completion: Any) -> None:
        token_ids = list(completion.token_ids)
        token_logprobs = sampled_token_logprobs(completion)

        for token_id, token_logp in zip(token_ids, token_logprobs):
            self._update_particle_weight(particle, token_logp)
            particle.token_ids.append(token_id)

        if completion.finish_reason != "length":
            particle.finished = True
            particle.finish_reason = completion.finish_reason
            particle.stop_reason = completion.stop_reason
        elif len(particle.token_ids) >= self.config.max_tokens:
            particle.finished = True
            particle.finish_reason = "length"

    def _finish_alpha_ramp(self, particle: PowerSMCParticle) -> None:
        delta = self.config.alpha - particle.prev_alpha
        if delta != 0.0:
            particle.log_weight += delta * particle.cum_logp
            particle.prev_alpha = self.config.alpha

    def _has_active_particles(self, particles: Sequence[PowerSMCParticle]) -> bool:
        return any(
            (not particle.finished
             and len(particle.token_ids) < self.config.max_tokens)
            for particle in particles)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact base-proposal Power-SMC with vLLM.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--alpha-ramp-tokens", type=int, default=128)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-token-id", action="append", type=int, default=[])
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    return parser


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    raise ValueError("Pass either --prompt or --prompt-file.")


def main() -> None:
    args = build_arg_parser().parse_args()
    prompt = load_prompt(args)

    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.disable_prefix_caching,
        logprobs_mode="raw_logprobs",
    )
    config = PowerSMCConfig(
        max_tokens=args.max_tokens,
        alpha=args.alpha,
        particles=args.particles,
        ess_threshold=args.ess_threshold,
        block_size=args.block_size,
        alpha_ramp_tokens=args.alpha_ramp_tokens,
        min_tokens=args.min_tokens,
        seed=args.seed,
        stop_token_ids=args.stop_token_id,
        ignore_eos=args.ignore_eos,
    )
    result = VLLMPowerSMCSampler(llm, config).generate(prompt)

    print(result.text)
    print()
    print("Power-SMC stats:")
    print(f"  selected_particle: {result.selected_particle}")
    print(f"  generated_tokens: {len(result.token_ids)}")
    print(f"  resample_count: {result.stats['resample_count']}")
    print(f"  blocks: {result.stats['blocks']}")
    if result.stats["prompt_groups_history"]:
        print(f"  final_prompt_groups: "
              f"{result.stats['prompt_groups_history'][-1]}")
    if result.stats["ess_history"]:
        print(f"  final_ess: {result.stats['ess_history'][-1]:.2f}")


if __name__ == "__main__":
    main()
