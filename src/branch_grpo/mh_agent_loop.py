"""Metropolis-Hastings proposal-pool rollout manager for VeRL/SGLang."""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import replace
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import ray
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
)
from verl.workers.rollout.replica import TokenOutput

from branch_grpo.mh_sampling import (
    MHCandidate,
    choose_low_logprob_branch,
    exact_dedup,
    mh_accept_logprob,
    should_accept,
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


class MHPowerAgentLoop(AgentLoopBase):
    """Generate a fixed-size candidate pool from one prompt using MH suffix resampling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.variant = os.environ.get("BRANCH_GRPO_MH_VARIANT", "all_proposals")
        self.alpha = _env_float("BRANCH_GRPO_MH_ALPHA", 1.5)
        self.steps = _env_int("BRANCH_GRPO_MH_STEPS", 4)
        self.min_prefix_tokens = _env_int("BRANCH_GRPO_MH_MIN_PREFIX_TOKENS", 1)
        self.dedup_exact = _env_bool("BRANCH_GRPO_MH_DEDUP_EXACT", False)
        seed = os.environ.get("BRANCH_GRPO_MH_SEED")
        self.rng = random.Random(int(seed)) if seed not in (None, "") else random.Random()

    async def run_pool(
        self,
        sampling_params: dict[str, Any],
        *,
        pool_size: int,
        **kwargs,
    ) -> list[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        start = perf_counter()
        num_preempted = -1
        initial_output = await self._generate(prompt_ids, sampling_params, images=images, videos=videos)
        num_preempted = max(num_preempted, initial_output.num_preempted or -1)
        current = self._candidate_from_output(initial_output, source="initial", step=0, accepted=True)
        candidates = [current]

        for step in range(1, self.steps + 1):
            if len(candidates) >= pool_size:
                break

            proposal, output_num_preempted = await self._propose(
                prompt_ids=prompt_ids,
                current=current,
                sampling_params=sampling_params,
                images=images,
                videos=videos,
                step=step,
            )
            num_preempted = max(num_preempted, output_num_preempted)
            accept_logprob = mh_accept_logprob(current.sequence_logprob, proposal.sequence_logprob, self.alpha)
            accepted = should_accept(accept_logprob, self.rng)
            proposal = replace(proposal, accepted=accepted, accept_logprob=accept_logprob)

            if self.variant in {"all_proposals", "accepted_only"}:
                if self.variant == "all_proposals" or accepted:
                    candidates.append(proposal)

            if accepted:
                current = replace(proposal, source="chain_state", accepted=True)
            else:
                current = replace(
                    current,
                    source="chain_state",
                    mh_step=step,
                    accepted=False,
                    accept_logprob=accept_logprob,
                    branch_point=proposal.branch_point,
                )

            if self.variant in {"all_proposals", "chain_only"} and len(candidates) < pool_size:
                candidates.append(current)

        if self.variant == "final_only":
            candidates = [current]

        if self.dedup_exact:
            candidates = exact_dedup(candidates)

        while len(candidates) < pool_size:
            candidates.append(replace(current, source="pool_padding", mh_step=self.steps))

        elapsed = perf_counter() - start
        return [
            self._to_agent_loop_output(
                candidate,
                prompt_ids=prompt_ids,
                multi_modal_data=multi_modal_data,
                elapsed=elapsed,
                num_preempted=num_preempted,
            )
            for candidate in candidates[:pool_size]
        ]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        return (await self.run_pool(sampling_params, pool_size=1, **kwargs))[0]

    async def _propose(
        self,
        *,
        prompt_ids: list[int],
        current: MHCandidate,
        sampling_params: dict[str, Any],
        images: Any,
        videos: Any,
        step: int,
    ) -> tuple[MHCandidate, int]:
        branch_point = choose_low_logprob_branch(
            current.logprobs,
            min_prefix_tokens=self.min_prefix_tokens,
            max_prefix_tokens=self.response_length - 1,
        )
        prefix_ids = current.response_ids[:branch_point]
        prefix_logprobs = current.logprobs[:branch_point]
        remaining = max(1, self.response_length - len(prefix_ids))
        output = await self._generate(
            prompt_ids + prefix_ids,
            sampling_params,
            max_new_tokens=remaining,
            images=images,
            videos=videos,
        )
        suffix_ids = output.token_ids[:remaining]
        suffix_logprobs = self._logprobs(output, len(suffix_ids))
        proposal_ids = (prefix_ids + suffix_ids)[: self.response_length]
        proposal_logprobs = (prefix_logprobs + suffix_logprobs)[: len(proposal_ids)]
        proposal = MHCandidate(
            response_ids=proposal_ids,
            logprobs=proposal_logprobs,
            source="proposal",
            mh_step=step,
            accepted=False,
            branch_point=branch_point,
        )
        return proposal, output.num_preempted or -1

    async def _generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        images: Any = None,
        videos: Any = None,
    ) -> TokenOutput:
        params = dict(sampling_params)
        params["logprobs"] = True
        if max_new_tokens is not None:
            params.pop("max_tokens", None)
            params["max_new_tokens"] = max_new_tokens
        output: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=params,
            image_data=images,
            video_data=videos,
        )
        return output

    def _candidate_from_output(self, output: TokenOutput, *, source: str, step: int, accepted: bool) -> MHCandidate:
        response_ids = output.token_ids[: self.response_length]
        return MHCandidate(
            response_ids=response_ids,
            logprobs=self._logprobs(output, len(response_ids)),
            source=source,
            mh_step=step,
            accepted=accepted,
        )

    def _logprobs(self, output: TokenOutput, length: int) -> list[float]:
        if output.log_probs is None:
            return [0.0] * length
        values = [float(value or 0.0) for value in output.log_probs[:length]]
        return values + [0.0] * max(0, length - len(values))

    def _to_agent_loop_output(
        self,
        candidate: MHCandidate,
        *,
        prompt_ids: list[int],
        multi_modal_data: dict[str, Any],
        elapsed: float,
        num_preempted: int,
    ) -> AgentLoopOutput:
        extra_fields = {
            "turn_scores": [],
            "tool_rewards": [],
            "mh_source": candidate.source,
            "mh_step": candidate.mh_step,
            "mh_accepted": candidate.accepted,
            "mh_accept_logprob": candidate.accept_logprob,
            "mh_branch_point": candidate.branch_point,
            "mh_variant": self.variant,
            "mh_alpha": self.alpha,
            "mh_sequence_logprob": candidate.sequence_logprob,
        }
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=candidate.response_ids[: self.response_length],
            response_mask=[1] * min(len(candidate.response_ids), self.response_length),
            response_logprobs=candidate.logprobs[: self.response_length],
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics={
                "generate_sequences": elapsed,
                "tool_calls": 0.0,
                "num_preempted": num_preempted,
            },
            extra_fields=extra_fields,
        )


class MHPowerAgentLoopWorker(AgentLoopWorker):
    """Worker that collapses repeated GRPO rows into per-prompt MH pools."""

    async def generate_sequences(self, batch):
        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=True,
        )
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        outputs = []
        expanded_non_tensor_batch: dict[str, list[Any]] = {key: [] for key in batch.non_tensor_batch}
        for indices in self._consecutive_groups(batch):
            kwargs = {key: values[indices[0]] for key, values in batch.non_tensor_batch.items()}
            pool = await self._run_mh_pool(sampling_params, pool_size=len(indices), **kwargs)
            for output, source_index in zip(pool, indices, strict=True):
                output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
                outputs.append(await self._agent_loop_postprocess(output, **kwargs))
                for key, values in batch.non_tensor_batch.items():
                    expanded_non_tensor_batch[key].append(values[source_index])

        expanded_arrays = {
            key: np.array(values, dtype=batch.non_tensor_batch[key].dtype)
            for key, values in expanded_non_tensor_batch.items()
        }
        return self._postprocess(outputs, input_non_tensor_batch=expanded_arrays)

    async def _run_mh_pool(
        self,
        sampling_params: dict[str, Any],
        *,
        pool_size: int,
        **kwargs,
    ) -> list[AgentLoopOutput]:
        agent_loop = MHPowerAgentLoop(
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )
        return await agent_loop.run_pool(sampling_params, pool_size=pool_size, **kwargs)

    def _consecutive_groups(self, batch) -> list[list[int]]:
        key = "uid" if "uid" in batch.non_tensor_batch else "index" if "index" in batch.non_tensor_batch else None
        if key is None:
            return [[idx] for idx in range(len(batch))]

        values = batch.non_tensor_batch[key]
        groups: list[list[int]] = []
        current: list[int] = []
        marker = object()
        last: Any = marker
        for idx, value in enumerate(values):
            comparable = value.item() if hasattr(value, "item") else value
            if current and comparable != last:
                groups.append(current)
                current = []
            current.append(idx)
            last = comparable
        if current:
            groups.append(current)
        return groups


class MHPowerAgentLoopManager(AgentLoopManager):
    """AgentLoopManager entry point used by actor_rollout_ref rollout config."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(MHPowerAgentLoopWorker)
        super().__init__(*args, **kwargs)
