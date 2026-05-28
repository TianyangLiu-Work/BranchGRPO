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
