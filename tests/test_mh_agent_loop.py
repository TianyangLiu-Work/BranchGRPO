import asyncio
import random
from types import SimpleNamespace

from branch_grpo.mh_agent_loop import MHPowerAgentLoop
from branch_grpo.mh_sampling import MHCandidate


def test_all_proposals_chain_respects_max_candidates():
    loop = object.__new__(MHPowerAgentLoop)
    loop.variant = "all_proposals"
    loop.steps = 4
    loop.alpha = 1.0
    loop.rng = random.Random(0)

    async def fake_generate(*args, **kwargs):
        return SimpleNamespace(num_preempted=-1)

    def fake_candidate_from_output(*args, **kwargs):
        return MHCandidate(
            response_ids=[1, 2, 3],
            logprobs=[-0.3, -0.2, -0.1],
            source="initial",
            mh_step=0,
            accepted=True,
        )

    async def fake_propose(**kwargs):
        step = kwargs["step"]
        proposal = MHCandidate(
            response_ids=[step, step + 1, step + 2],
            logprobs=[-0.1, -0.1, -0.1],
            source="proposal",
            mh_step=step,
            accepted=False,
        )
        return proposal, -1

    loop._generate = fake_generate
    loop._candidate_from_output = fake_candidate_from_output
    loop._propose = fake_propose

    candidates, _current, _num_preempted = asyncio.run(
        loop._run_chain(
            prompt_ids=[0],
            sampling_params={},
            images=None,
            videos=None,
            chain_id=0,
            max_candidates=9,
        )
    )

    assert len(candidates) == 9


def test_agent_loop_output_includes_mh_diagnostics():
    loop = object.__new__(MHPowerAgentLoop)
    loop.response_length = 10
    loop.variant = "all_proposals"
    loop.alpha = 2.0
    loop.top_logprobs = 20

    candidate = MHCandidate(
        response_ids=[1, 2, 3, 4],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        source="proposal",
        mh_step=1,
        accepted=False,
        chain_id=0,
        accept_logprob=-0.7,
        branch_point=2,
        branch_entropy=1.2,
        branch_strategy="topk_entropy",
    )

    output = loop._to_agent_loop_output(
        candidate,
        prompt_ids=[0],
        multi_modal_data={},
        elapsed=0.1,
        num_preempted=-1,
    )

    assert output.extra_fields["mh_source"] == "proposal"
    assert output.extra_fields["mh_accepted"] is False
    assert output.extra_fields["mh_step"] == 1
    assert output.extra_fields["mh_branch_point"] == 2
    assert output.extra_fields["mh_accept_logprob"] == -0.7
    assert output.extra_fields["mh_branch_entropy"] == 1.2
    assert output.extra_fields["mh_sequence_logprob"] == -1.0
    assert output.extra_fields["mh_response_length"] == 4
    assert output.extra_fields["mh_avg_token_logprob"] == -0.25
    assert "mh_suffix_length" not in output.extra_fields
    assert "mh_behavior_correction_exact" not in output.extra_fields


def test_agent_loop_output_mh_diagnostics_are_validation_safe_when_missing():
    loop = object.__new__(MHPowerAgentLoop)
    loop.response_length = 10
    loop.variant = "all_proposals"
    loop.alpha = 2.0
    loop.top_logprobs = 20

    candidate = MHCandidate(
        response_ids=[1, 2, 3, 4],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        source="initial",
        mh_step=0,
        accepted=True,
    )

    output = loop._to_agent_loop_output(
        candidate,
        prompt_ids=[0],
        multi_modal_data={},
        elapsed=0.1,
        num_preempted=-1,
    )

    mh_fields = {
        key: value for key, value in output.extra_fields.items() if key.startswith("mh_")
    }
    assert all(value is not None for value in mh_fields.values())
    assert mh_fields["mh_accept_logprob"] == 0.0
    assert mh_fields["mh_branch_point"] == -1.0
    assert mh_fields["mh_branch_entropy"] == 0.0
