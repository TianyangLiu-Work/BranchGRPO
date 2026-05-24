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
    choose_high_entropy_branch,
    choose_low_logprob_branch,
    coerce_top_logprobs,
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
        self.chains = max(1, _env_int("BRANCH_GRPO_MH_CHAINS", 1))
        self.min_prefix_tokens = _env_int("BRANCH_GRPO_MH_MIN_PREFIX_TOKENS", 1)
        self.dedup_exact = _env_bool("BRANCH_GRPO_MH_DEDUP_EXACT", False)
        self.branch_strategy = os.environ.get(
            "BRANCH_GRPO_MH_BRANCH_STRATEGY", "topk_entropy"
        ).lower()
        self.top_logprobs = max(0, _env_int("BRANCH_GRPO_MH_TOP_LOGPROBS", 20))
        self.http_timeout = _env_float("BRANCH_GRPO_MH_HTTP_TIMEOUT", 120.0)
        self.strict_top_logprobs = _env_bool(
            "BRANCH_GRPO_MH_STRICT_TOP_LOGPROBS", False
        )
        self._top_logprobs_disabled = False
        seed = os.environ.get("BRANCH_GRPO_MH_SEED")
        self.rng = (
            random.Random(int(seed)) if seed not in (None, "") else random.Random()
        )

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
        prompt_ids = await self.apply_chat_template(
            messages, images=images, videos=videos
        )

        start = perf_counter()
        chain_tasks = [
            self._run_chain(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                images=images,
                videos=videos,
                chain_id=chain_id,
                max_candidates=pool_size if self.chains == 1 else None,
            )
            for chain_id in range(self.chains)
        ]
        chain_results = await asyncio.gather(*chain_tasks)

        candidates: list[MHCandidate] = []
        num_preempted = -1
        padding_candidate: MHCandidate | None = None
        for chain_candidates, final_current, chain_num_preempted in chain_results:
            candidates.extend(chain_candidates)
            padding_candidate = final_current
            num_preempted = max(num_preempted, chain_num_preempted)

        if self.dedup_exact:
            candidates = exact_dedup(candidates)

        padding_candidate = padding_candidate or candidates[-1]
        while len(candidates) < pool_size:
            candidates.append(
                replace(padding_candidate, source="pool_padding", mh_step=self.steps)
            )

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

    async def _run_chain(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Any,
        videos: Any,
        chain_id: int,
        max_candidates: int | None,
    ) -> tuple[list[MHCandidate], MHCandidate, int]:
        num_preempted = -1
        initial_output = await self._generate(
            prompt_ids, sampling_params, images=images, videos=videos
        )
        num_preempted = max(num_preempted, initial_output.num_preempted or -1)
        current = self._candidate_from_output(
            initial_output,
            source="initial",
            step=0,
            accepted=True,
            chain_id=chain_id,
        )
        candidates = [] if self.variant == "proposals_only" else [current]

        for step in range(1, self.steps + 1):
            if max_candidates is not None and len(candidates) >= max_candidates:
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
            accept_logprob = mh_accept_logprob(
                current.sequence_logprob, proposal.sequence_logprob, self.alpha
            )
            accepted = should_accept(accept_logprob, self.rng)
            proposal = replace(
                proposal, accepted=accepted, accept_logprob=accept_logprob
            )

            if self.variant in {"all_proposals", "accepted_only"}:
                if self.variant == "all_proposals" or accepted:
                    candidates.append(proposal)
            elif self.variant == "proposals_only":
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

            if (
                self.variant in {"all_proposals", "chain_only"}
                and len(candidates) < pool_size
            ):
                candidates.append(current)

        if self.variant == "final_only":
            candidates = [current]

        return candidates, current, num_preempted

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
        branch_point, branch_entropy, branch_strategy = self._select_branch(current)
        prefix_ids = current.response_ids[:branch_point]
        prefix_logprobs = current.logprobs[:branch_point]
        prefix_top_logprobs = (
            current.top_logprobs[:branch_point] if current.top_logprobs else []
        )
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
        suffix_top_logprobs = self._top_logprobs(output, len(suffix_ids))
        proposal_ids = (prefix_ids + suffix_ids)[: self.response_length]
        proposal_logprobs = (prefix_logprobs + suffix_logprobs)[: len(proposal_ids)]
        proposal_top_logprobs = None
        if suffix_top_logprobs is not None:
            merged_top_logprobs = (prefix_top_logprobs + suffix_top_logprobs)[
                : len(proposal_ids)
            ]
            if len(merged_top_logprobs) == len(proposal_ids):
                proposal_top_logprobs = merged_top_logprobs
        proposal = MHCandidate(
            response_ids=proposal_ids,
            logprobs=proposal_logprobs,
            source="proposal",
            mh_step=step,
            accepted=False,
            chain_id=current.chain_id,
            branch_point=branch_point,
            top_logprobs=proposal_top_logprobs,
            branch_entropy=branch_entropy,
            branch_strategy=branch_strategy,
        )
        return proposal, output.num_preempted or -1

    def _select_branch(self, current: MHCandidate) -> tuple[int, float | None, str]:
        if (
            self.branch_strategy in {"topk_entropy", "entropy", "auto"}
            and current.top_logprobs
        ):
            branch_point, entropy = choose_high_entropy_branch(
                current.top_logprobs,
                min_prefix_tokens=self.min_prefix_tokens,
                max_prefix_tokens=self.response_length - 1,
            )
            if entropy is not None:
                return branch_point, entropy, "topk_entropy"

        branch_point = choose_low_logprob_branch(
            current.logprobs,
            min_prefix_tokens=self.min_prefix_tokens,
            max_prefix_tokens=self.response_length - 1,
        )
        strategy = (
            "sample_logprob"
            if self.branch_strategy == "sample_logprob"
            else "sample_logprob_fallback"
        )
        return branch_point, None, strategy

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
        elif "max_new_tokens" not in params and "max_tokens" not in params:
            params["max_new_tokens"] = max(
                0,
                min(
                    self.response_length,
                    self.prompt_length + self.response_length - len(prompt_ids),
                ),
            )
        if self._should_request_top_logprobs(images=images, videos=videos):
            try:
                return await self._generate_with_top_logprobs(
                    prompt_ids=prompt_ids,
                    sampling_params=params,
                    images=images,
                    request_id=uuid4().hex,
                )
            except Exception as exc:
                if self.strict_top_logprobs:
                    raise
                self._top_logprobs_disabled = True
                output = await self._generate_without_top_logprobs(
                    prompt_ids, params, images=images, videos=videos
                )
                output.extra_fields["top_logprobs_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                return output
        return await self._generate_without_top_logprobs(
            prompt_ids, params, images=images, videos=videos
        )

    async def _generate_without_top_logprobs(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        images: Any = None,
        videos: Any = None,
    ) -> TokenOutput:
        output: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images,
            video_data=videos,
        )
        return output

    def _should_request_top_logprobs(self, *, images: Any, videos: Any) -> bool:
        if self._top_logprobs_disabled or self.top_logprobs <= 0:
            return False
        if self.branch_strategy not in {"topk_entropy", "entropy", "auto"}:
            return False
        if images is not None or videos is not None:
            return False
        return hasattr(self.server_manager, "_acquire_server") and hasattr(
            self.server_manager, "_release_server"
        )

    async def _generate_with_top_logprobs(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Any,
        request_id: str,
    ) -> TokenOutput:
        import aiohttp

        params = dict(sampling_params)
        return_logprob = bool(params.pop("logprobs", False))
        server_id, _server = await self.server_manager._acquire_server(request_id)
        try:
            url = f"http://{server_id}/generate"
            payload = {
                "rid": request_id,
                "input_ids": prompt_ids,
                "sampling_params": params,
                "return_logprob": return_logprob,
                "top_logprobs_num": self.top_logprobs,
                "return_text_in_logprobs": False,
                "image_data": images,
            }
            timeout = aiohttp.ClientTimeout(total=self.http_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            f"SGLang /generate failed with HTTP {response.status}: {body[:500]}"
                        )
                    raw_output = await response.json(content_type=None)
        finally:
            self.server_manager._release_server(server_id)

        if isinstance(raw_output, list):
            raw_output = raw_output[0]
        meta_info = raw_output.get("meta_info") or {}
        token_ids = list(raw_output.get("output_ids") or [])
        log_probs = None
        if return_logprob:
            output_token_logprobs = meta_info.get("output_token_logprobs") or []
            if output_token_logprobs:
                log_probs = [float(item[0] or 0.0) for item in output_token_logprobs]
                token_ids = [int(item[1]) for item in output_token_logprobs]
            else:
                log_probs = [0.0] * len(token_ids)

        finish_reason = meta_info.get("finish_reason")
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type")

        extra_fields = {
            "meta_info": meta_info,
            "output_top_logprobs": meta_info.get("output_top_logprobs"),
        }
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=finish_reason,
            num_preempted=meta_info.get("num_preempted"),
            extra_fields=extra_fields,
        )

    def _candidate_from_output(
        self,
        output: TokenOutput,
        *,
        source: str,
        step: int,
        accepted: bool,
        chain_id: int = 0,
    ) -> MHCandidate:
        response_ids = output.token_ids[: self.response_length]
        return MHCandidate(
            response_ids=response_ids,
            logprobs=self._logprobs(output, len(response_ids)),
            source=source,
            mh_step=step,
            accepted=accepted,
            chain_id=chain_id,
            top_logprobs=self._top_logprobs(output, len(response_ids)),
            branch_strategy=self.branch_strategy,
        )

    def _logprobs(self, output: TokenOutput, length: int) -> list[float]:
        if output.log_probs is None:
            return [0.0] * length
        values = [float(value or 0.0) for value in output.log_probs[:length]]
        return values + [0.0] * max(0, length - len(values))

    def _top_logprobs(
        self, output: TokenOutput, length: int
    ) -> list[list[float]] | None:
        extra_fields = output.extra_fields or {}
        raw_values = []
        for container in (
            extra_fields,
            (
                extra_fields.get("meta_info")
                if isinstance(extra_fields.get("meta_info"), dict)
                else None
            ),
        ):
            if not container:
                continue
            for key in (
                "output_top_logprobs",
                "top_logprobs",
                "output_top_log_probs",
                "top_log_probs",
                "output_top_logprobs_val",
                "top_logprobs_val",
            ):
                if key in container:
                    raw_values = coerce_top_logprobs(container[key])
                    break
            if raw_values:
                break
        if not raw_values:
            return None

        values = raw_values[:length]
        if len(values) < length:
            return None
        return values

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
            "mh_chain_id": candidate.chain_id,
            "mh_step": candidate.mh_step,
            "mh_accepted": candidate.accepted,
            "mh_accept_logprob": candidate.accept_logprob,
            "mh_branch_point": candidate.branch_point,
            "mh_branch_strategy": candidate.branch_strategy,
            "mh_branch_entropy": candidate.branch_entropy,
            "mh_variant": self.variant,
            "mh_alpha": self.alpha,
            "mh_sequence_logprob": candidate.sequence_logprob,
            "mh_top_logprobs_k": self.top_logprobs,
            "mh_entropy_available": candidate.top_logprobs is not None,
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
        expanded_non_tensor_batch: dict[str, list[Any]] = {
            key: [] for key in batch.non_tensor_batch
        }
        for indices in self._consecutive_groups(batch):
            kwargs = {
                key: values[indices[0]]
                for key, values in batch.non_tensor_batch.items()
            }
            pool = await self._run_mh_pool(
                sampling_params, pool_size=len(indices), **kwargs
            )
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
        key = (
            "uid"
            if "uid" in batch.non_tensor_batch
            else "index" if "index" in batch.non_tensor_batch else None
        )
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
