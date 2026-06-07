"""Math reward with 0/1 correctness and DAPO-style overlong penalty."""

from __future__ import annotations

import asyncio
from typing import Any

from branch_grpo.reward import exact_match_reward, extract_answer, numeric_match_reward

_MH_DIAGNOSTIC_KEYS = (
    "mh_source",
    "mh_step",
    "mh_accepted",
    "mh_accept_logprob",
    "mh_branch_point",
    "mh_branch_entropy",
    "mh_sequence_logprob",
    "mh_response_length",
    "mh_avg_token_logprob",
)

_MH_SOURCES = ("initial", "proposal", "chain_state", "pool_padding")

_MH_NUMERIC_SENTINELS = {
    "mh_step": -1.0,
    "mh_accept_logprob": 0.0,
    "mh_branch_point": -1.0,
    "mh_branch_entropy": 0.0,
    "mh_sequence_logprob": 0.0,
    "mh_response_length": 0.0,
    "mh_avg_token_logprob": 0.0,
}


def correctness_reward(solution_str: Any, ground_truth: Any) -> float:
    """Return 1.0 for a correct final math answer, otherwise 0.0."""
    candidate = solution_str or ""
    target = extract_answer(ground_truth)

    try:
        from verl.utils.reward_score.math_reward import compute_score as verl_math_score

        score = float(verl_math_score(candidate, target))
        if score > 0:
            return 1.0
    except Exception:
        pass

    return max(exact_match_reward(candidate, target), numeric_match_reward(candidate, target))


def overlong_penalty(
    valid_response_length: int | float | None,
    max_response_length: int,
    overlong_buffer_len: int,
    penalty_factor: float,
) -> float:
    """Return the old DAPO-style length penalty, or 0.0 when not overlong."""
    if valid_response_length is None:
        return 0.0
    if overlong_buffer_len <= 0:
        raise ValueError("overlong_buffer_len must be positive")
    if max_response_length < overlong_buffer_len:
        raise ValueError("max_response_length must be >= overlong_buffer_len")

    expected_len = max_response_length - overlong_buffer_len
    exceed_len = float(valid_response_length) - expected_len
    return min(-exceed_len / overlong_buffer_len * float(penalty_factor), 0.0)


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    valid_response_length: int | None = None,
    max_response_length: int | None = None,
    overlong_buffer_len: int | None = None,
    overlong_penalty_factor: float | None = None,
    overlong_penalty_enabled: bool | None = None,
    **_: Any,
) -> float:
    """Return 0/1 correctness plus an optional overlong penalty.

    The exact token-length penalty is available when the caller explicitly passes
    ``valid_response_length`` or when called by ``OverlongRewardManager``.
    """
    del data_source, extra_info

    score = correctness_reward(solution_str or "", ground_truth)
    if overlong_penalty_enabled is False or valid_response_length is None:
        return score

    if max_response_length is None:
        raise ValueError("max_response_length is required when applying overlong penalty")
    if overlong_buffer_len is None:
        raise ValueError("overlong_buffer_len is required when applying overlong penalty")

    factor = 1.0 if overlong_penalty_factor is None else overlong_penalty_factor
    return score + overlong_penalty(
        valid_response_length,
        max_response_length,
        overlong_buffer_len,
        factor,
    )


def _get_config_value(config: Any, path: str) -> Any:
    current = config
    for key in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _plain_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _tool_extra_fields(non_tensor_batch: dict[str, Any]) -> dict[str, Any]:
    value = non_tensor_batch.get("tool_extra_fields", {})
    value = _plain_value(value)
    return value if isinstance(value, dict) else {}


def _numeric_or_zero(value: Any) -> float:
    value = _plain_value(value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _numeric_or_sentinel(value: Any, sentinel: float) -> float:
    value = _plain_value(value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return sentinel


def _bool_or_false(value: Any) -> bool:
    value = _plain_value(value)
    return value if isinstance(value, bool) else False


def _mh_metadata_present(tool_extra_fields: dict[str, Any]) -> float:
    return float(
        any(_plain_value(tool_extra_fields.get(key)) is not None for key in _MH_DIAGNOSTIC_KEYS)
    )


def _mh_reward_extra_info(
    tool_extra_fields: dict[str, Any],
    *,
    reward: float,
    score: float,
) -> dict[str, Any]:
    source = _plain_value(tool_extra_fields.get("mh_source"))
    info = {
        "mh_source": source if isinstance(source, str) else "missing",
        "mh_accepted": _bool_or_false(tool_extra_fields.get("mh_accepted")),
        "mh_metadata_present": _mh_metadata_present(tool_extra_fields),
    }
    info.update(
        {
            key: _numeric_or_sentinel(tool_extra_fields.get(key), sentinel)
            for key, sentinel in _MH_NUMERIC_SENTINELS.items()
        }
    )
    source = info.get("mh_source")
    for name in _MH_SOURCES:
        is_source = float(source == name)
        info[f"mh_source_{name}"] = is_source
        info[f"mh_reward_{name}"] = reward if is_source else 0.0
        info[f"mh_acc_{name}"] = score if is_source else 0.0
        info[f"mh_avg_logp_{name}"] = (
            _numeric_or_zero(info.get("mh_avg_token_logprob")) if is_source else 0.0
        )
        info[f"mh_len_{name}"] = (
            _numeric_or_zero(info.get("mh_response_length")) if is_source else 0.0
        )
    return info


class OverlongRewardManager:
    """Small VERL reward manager for exact response-token overlong penalty."""

    def __init__(
        self,
        config: Any = None,
        tokenizer: Any = None,
        num_examine: int = 0,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        max_resp_len: int | None = None,
        overlong_buffer_cfg: Any = None,
        **_: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)
        self.compute_score = compute_score or globals()["compute_score"]
        self.reward_fn_key = reward_fn_key

        if max_resp_len is None:
            max_resp_len = _get_config_value(config, "reward.reward_kwargs.max_resp_len")
        if overlong_buffer_cfg is None:
            overlong_buffer_cfg = _get_config_value(
                config, "reward.reward_kwargs.overlong_buffer_cfg"
            )
        if tokenizer is None:
            raise ValueError("OverlongRewardManager requires tokenizer")
        if max_resp_len is None:
            raise ValueError("OverlongRewardManager requires max_resp_len")
        if overlong_buffer_cfg is None:
            raise ValueError("OverlongRewardManager requires overlong_buffer_cfg")

        self.max_resp_len = int(max_resp_len)
        self.overlong_buffer_cfg = overlong_buffer_cfg

    def _overlong_cfg(self) -> tuple[bool, int, float]:
        cfg = self.overlong_buffer_cfg
        return (
            bool(_cfg_value(cfg, "enable", True)),
            int(_cfg_value(cfg, "len", 1024)),
            float(_cfg_value(cfg, "penalty_factor", 1.0)),
        )

    async def run_single(self, data: Any) -> dict[str, Any]:
        assert len(data) == 1, "Only support single data item"

        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(
            data_item.batch["attention_mask"][-response_length:].sum()
        )
        valid_response_ids = response_ids[:valid_response_length]

        response_str = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        data_source = data_item.non_tensor_batch.get("data_source")
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        result = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            overlong_penalty_enabled=False,
        )
        score = float(result["score"] if isinstance(result, dict) else result)
        enable_penalty, buffer_len, penalty_factor = self._overlong_cfg()
        penalty = (
            overlong_penalty(
                valid_response_length,
                self.max_resp_len,
                buffer_len,
                penalty_factor,
            )
            if enable_penalty
            else 0.0
        )
        reward = score + penalty
        tool_extra_fields = _tool_extra_fields(data_item.non_tensor_batch)
        mh_info = _mh_reward_extra_info(
            tool_extra_fields,
            reward=reward,
            score=score,
        )
        reward_extra_info = {
            "acc": score,
            "overlong_reward": penalty,
            "overlong": bool(penalty < 0),
            **mh_info,
        }

        return {
            "reward_score": reward,
            "reward_extra_info": _plain_value(reward_extra_info),
        }
