import asyncio

import pytest

from branch_grpo.reward_overlong import (
    OverlongRewardManager,
    compute_score,
    overlong_penalty,
)


class _TokenIds(list):
    @property
    def shape(self):
        return (len(self),)


class _Mask(list):
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return _Mask(result)
        return result

    def sum(self):
        return sum(self)


class _Tokenizer:
    def __init__(self, text):
        self.text = text

    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return self.text


class _DataItem:
    def __init__(
        self,
        text,
        ground_truth,
        valid_response_length,
        tool_extra_fields=None,
    ):
        response_length = 10000
        self.batch = {
            "responses": _TokenIds([0] * response_length),
            "attention_mask": _Mask([1] * valid_response_length),
        }
        self.non_tensor_batch = {
            "data_source": "math__test",
            "reward_model": {"ground_truth": ground_truth},
            "extra_info": {},
            "tool_extra_fields": tool_extra_fields or {},
        }
        self.tokenizer = _Tokenizer(text)


class _Data:
    def __init__(self, item):
        self.item = item

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.item


def _run_manager(text, ground_truth, valid_response_length, tool_extra_fields=None):
    item = _DataItem(text, ground_truth, valid_response_length, tool_extra_fields)
    manager = OverlongRewardManager(
        tokenizer=item.tokenizer,
        max_resp_len=10000,
        overlong_buffer_cfg={"enable": True, "len": 1024, "penalty_factor": 1.0},
    )
    return asyncio.run(manager.run_single(_Data(item)))


def test_correct_answer_scores_one_without_overlong_penalty():
    assert (
        compute_score(
            solution_str=r"The answer is \boxed{42}.",
            ground_truth="42",
            valid_response_length=8000,
            max_response_length=10000,
            overlong_buffer_len=1024,
        )
        == 1.0
    )


def test_incorrect_answer_scores_zero_without_overlong_penalty():
    assert (
        compute_score(
            solution_str=r"The answer is \boxed{41}.",
            ground_truth="42",
            valid_response_length=8000,
            max_response_length=10000,
            overlong_buffer_len=1024,
        )
        == 0.0
    )


def test_correct_overlong_answer_is_reduced_by_penalty():
    score = compute_score(
        solution_str=r"The answer is \boxed{42}.",
        ground_truth="42",
        valid_response_length=9488,
        max_response_length=10000,
        overlong_buffer_len=1024,
        overlong_penalty_factor=1.0,
    )
    assert score == 0.5


def test_incorrect_overlong_answer_is_negative_only_from_penalty():
    score = compute_score(
        solution_str=r"The answer is \boxed{41}.",
        ground_truth="42",
        valid_response_length=9488,
        max_response_length=10000,
        overlong_buffer_len=1024,
        overlong_penalty_factor=1.0,
    )
    assert score == -0.5


def test_overlong_penalty_matches_old_formula():
    assert overlong_penalty(8976, 10000, 1024, 1.0) == 0.0
    assert overlong_penalty(10000, 10000, 1024, 1.0) == -1.0


def test_reward_manager_correct_answer_scores_one_without_overlong_penalty():
    result = _run_manager(r"The answer is \boxed{42}.", "42", 8000)
    assert result["reward_score"] == 1.0
    assert result["reward_extra_info"]["acc"] == 1.0
    assert result["reward_extra_info"]["overlong_reward"] == 0.0
    assert result["reward_extra_info"]["overlong"] is False


def test_reward_manager_incorrect_answer_scores_zero_without_overlong_penalty():
    result = _run_manager(r"The answer is \boxed{41}.", "42", 8000)
    assert result["reward_score"] == 0.0
    assert result["reward_extra_info"]["acc"] == 0.0
    assert result["reward_extra_info"]["overlong_reward"] == 0.0
    assert result["reward_extra_info"]["overlong"] is False


def test_reward_manager_correct_overlong_answer_is_reduced_by_penalty():
    result = _run_manager(r"The answer is \boxed{42}.", "42", 9488)
    assert result["reward_score"] == 0.5
    assert result["reward_extra_info"]["acc"] == 1.0
    assert result["reward_extra_info"]["overlong_reward"] == -0.5
    assert result["reward_extra_info"]["overlong"] is True


def test_reward_manager_incorrect_overlong_answer_is_negative_only_from_penalty():
    result = _run_manager(r"The answer is \boxed{41}.", "42", 9488)
    assert result["reward_score"] == -0.5
    assert result["reward_extra_info"]["acc"] == 0.0
    assert result["reward_extra_info"]["overlong_reward"] == -0.5
    assert result["reward_extra_info"]["overlong"] is True


def test_reward_manager_copies_mh_diagnostics_and_source_counters():
    result = _run_manager(
        r"The answer is \boxed{42}.",
        "42",
        8000,
        {
            "mh_source": "proposal",
            "mh_accepted": False,
            "mh_sequence_logprob": -12.5,
            "mh_response_length": 8000,
            "mh_avg_token_logprob": -0.0015625,
        },
    )

    info = result["reward_extra_info"]
    assert info["mh_source"] == "proposal"
    assert info["mh_accepted"] is False
    assert info["mh_metadata_present"] == 1.0
    assert info["mh_sequence_logprob"] == -12.5
    assert info["mh_response_length"] == 8000
    assert info["mh_source_proposal"] == 1.0
    assert info["mh_source_initial"] == 0.0
    assert info["mh_reward_proposal"] == 1.0
    assert info["mh_acc_proposal"] == 1.0
    assert info["mh_avg_logp_proposal"] == -0.0015625
    assert info["mh_avg_logp_initial"] == 0.0
    assert info["mh_len_proposal"] == 8000.0
    assert info["mh_len_initial"] == 0.0


def test_reward_manager_mh_diagnostics_are_validation_safe_when_missing():
    result = _run_manager(r"The answer is \boxed{42}.", "42", 8000)

    info = result["reward_extra_info"]
    assert all(value is not None for value in info.values())
    assert info["mh_source"] == "missing"
    assert info["mh_metadata_present"] == 0.0
    assert info["mh_accepted"] is False
    assert info["mh_step"] == -1.0
    assert info["mh_branch_point"] == -1.0
    assert info["mh_sequence_logprob"] == 0.0
    assert info["mh_response_length"] == 0.0


def test_reward_manager_payload_passes_verl_validation_metric_aggregation():
    metric_utils = pytest.importorskip("verl.trainer.ppo.metric_utils")

    results = [
        _run_manager(r"The answer is \boxed{42}.", "42", 8000),
        _run_manager(
            r"The answer is \boxed{42}.",
            "42",
            8000,
            {
                "mh_source": "proposal",
                "mh_accepted": True,
                "mh_branch_point": 12,
                "mh_sequence_logprob": -3.5,
                "mh_response_length": 8000,
                "mh_avg_token_logprob": -0.0004375,
            },
        ),
    ]
    infos = [result["reward_extra_info"] for result in results]
    info_dict = {key: [info[key] for info in infos] for key in infos[0]}

    metrics = metric_utils.process_validation_metrics(
        data_sources=["math__test", "math__test"],
        sample_uids=["sample-0", "sample-0"],
        infos_dict=info_dict,
    )

    assert metrics["math__test"]["acc"]["mean@2"] == 1.0
