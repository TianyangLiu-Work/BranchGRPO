import math
import random

from branch_grpo.mh_sampling import (
    MHCandidate,
    choose_low_logprob_branch,
    exact_dedup,
    mh_accept_logprob,
    sequence_logprob,
    should_accept,
)


def test_sequence_logprob_treats_missing_values_as_zero():
    assert sequence_logprob([-1.0, None, -2.5]) == -3.5


def test_mh_accept_logprob_accepts_higher_power_score():
    assert mh_accept_logprob(current_logprob=-10.0, proposal_logprob=-8.0, alpha=1.5) == 0.0
    assert math.isclose(mh_accept_logprob(-8.0, -10.0, 1.5), -3.0)


def test_should_accept_uses_log_probability_threshold():
    rng = random.Random(0)
    assert should_accept(0.0, rng)
    rng = random.Random(0)
    assert not should_accept(math.log(0.1), rng)


def test_choose_low_logprob_branch_respects_bounds():
    assert choose_low_logprob_branch([-0.1, -3.0, -0.2, -4.0], min_prefix_tokens=1) == 3
    assert choose_low_logprob_branch([-0.1, -3.0, -0.2, -4.0], min_prefix_tokens=1, max_prefix_tokens=2) == 1
    assert choose_low_logprob_branch([-0.1], min_prefix_tokens=1) == 0


def test_exact_dedup_preserves_first_candidate():
    candidates = [
        MHCandidate([1, 2], [-0.1, -0.2], "initial", 0, True),
        MHCandidate([1, 2], [-9.0, -9.0], "proposal", 1, False),
        MHCandidate([3], [-0.3], "proposal", 1, True),
    ]
    assert [candidate.source for candidate in exact_dedup(candidates)] == ["initial", "proposal"]
