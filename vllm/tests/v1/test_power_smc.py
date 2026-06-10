# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.power_smc as power_smc_module
from vllm.exceptions import VLLMValidationError
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import PowerSMCParentRequest
from vllm.v1.outputs import PowerSMCLogprobTensors
from vllm.v1.power_smc import (
    PowerSMCConfig,
    PowerSMCGroupManager,
    PowerSMCParticleState,
    PowerSMCResamplePlan,
    _normalize_log_weights_and_ess,
    alpha_ramp,
    effective_sample_size,
    make_power_smc_child_request_id,
    normalize_log_weights,
    parse_power_smc_child_request_id,
    proposal_temperature,
    systematic_resample,
    update_log_weight,
    validate_power_smc_engine_features,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

pytestmark = pytest.mark.skip_global_cleanup


def power_smc_args(**overrides):
    cfg = {
        "enabled": True,
        "alpha": 4.0,
        "particles": 32,
        "block_size": 64,
        "ess_threshold": 0.5,
        "alpha_ramp_tokens": 400,
        "proposal": "power_temperature",
        "return_diagnostics": True,
    }
    cfg.update(overrides)
    return {"power_smc": cfg}


def make_engine_core_request(
    sampling_params: SamplingParams,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="parent",
        external_req_id="external-parent",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def test_power_smc_config_from_sampling_params() -> None:
    params = SamplingParams(max_tokens=16, extra_args=power_smc_args())

    cfg = PowerSMCConfig.from_sampling_params(params)

    assert cfg == PowerSMCConfig(
        enabled=True,
        alpha=4.0,
        particles=32,
        block_size=64,
        ess_threshold=0.5,
        alpha_ramp_tokens=400,
        proposal="power_temperature",
        return_diagnostics=True,
        kv_cow=True,
        kv_pool_diagnostics=False,
    )


def test_request_stores_power_smc_config() -> None:
    params = SamplingParams(max_tokens=16, extra_args=power_smc_args(alpha=1.0))

    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
    )

    assert request.power_smc_config is not None
    assert request.power_smc_config.alpha == 1.0


def test_power_smc_config_accepts_subunit_alpha() -> None:
    params = SamplingParams(max_tokens=16, extra_args=power_smc_args(alpha=0.8))

    cfg = PowerSMCConfig.from_sampling_params(params)

    assert cfg is not None
    assert cfg.alpha == 0.8


def test_power_smc_config_can_disable_kv_cow() -> None:
    params = SamplingParams(
        max_tokens=16,
        extra_args=power_smc_args(kv_cow=False),
    )

    cfg = PowerSMCConfig.from_sampling_params(params)

    assert cfg is not None
    assert cfg.kv_cow is False


def test_power_smc_config_can_enable_kv_pool_diagnostics() -> None:
    params = SamplingParams(
        max_tokens=16,
        extra_args=power_smc_args(kv_pool_diagnostics=True),
    )

    cfg = PowerSMCConfig.from_sampling_params(params)

    assert cfg is not None
    assert cfg.kv_pool_diagnostics is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"particles": 0}, "particles"),
        ({"block_size": 0}, "block_size"),
        ({"ess_threshold": 0.0}, "ess_threshold"),
        ({"alpha_ramp_tokens": 0}, "alpha_ramp_tokens"),
        ({"proposal": "base"}, "proposal"),
    ],
)
def test_power_smc_config_rejects_invalid_values(kwargs, match) -> None:
    params = SamplingParams(max_tokens=16, extra_args=power_smc_args(**kwargs))

    with pytest.raises(VLLMValidationError, match=match):
        PowerSMCConfig.from_sampling_params(params)


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(max_tokens=16, top_p=0.9),
        SamplingParams(max_tokens=16, top_k=10),
        SamplingParams(max_tokens=16, min_p=0.1),
        SamplingParams(max_tokens=16, repetition_penalty=1.1),
        SamplingParams(max_tokens=16, frequency_penalty=0.1),
        SamplingParams(max_tokens=16, presence_penalty=0.1),
        SamplingParams(max_tokens=16, n=2),
    ],
)
def test_power_smc_rejects_unsupported_sampling_features(params) -> None:
    params.extra_args = power_smc_args()

    with pytest.raises(VLLMValidationError, match="Unsupported"):
        PowerSMCConfig.from_sampling_params(params)


def test_power_smc_rejects_user_temperature_override() -> None:
    params = SamplingParams(
        max_tokens=16,
        temperature=0.7,
        extra_args=power_smc_args(),
    )

    with pytest.raises(VLLMValidationError, match="temperature"):
        PowerSMCConfig.from_sampling_params(params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stream_input": True},
        {"is_encoder_decoder": True},
        {"speculative_config": object()},
        {"kv_block_size": 16},
    ],
)
def test_power_smc_rejects_engine_level_unsupported_features(kwargs) -> None:
    config = PowerSMCConfig(enabled=True, block_size=24)

    with pytest.raises(VLLMValidationError, match="Unsupported"):
        validate_power_smc_engine_features(config, **kwargs)


def test_power_smc_accepts_lora_request() -> None:
    config = PowerSMCConfig(enabled=True, block_size=24)

    validate_power_smc_engine_features(config, lora_request=object())


def test_power_smc_accepts_block_aligned_kv_boundary() -> None:
    config = PowerSMCConfig(enabled=True, block_size=64)

    validate_power_smc_engine_features(config, kv_block_size=16)


def test_power_smc_child_request_id_round_trip() -> None:
    request_id = make_power_smc_child_request_id("parent-abc", 12)

    parsed = parse_power_smc_child_request_id(request_id)

    assert request_id == "psmc12_parent-abc"
    assert parsed is not None
    assert parsed.parent_id == "parent-abc"
    assert parsed.particle_idx == 12
    assert parse_power_smc_child_request_id("12_parent-abc") is None


def test_request_caches_power_smc_child_info() -> None:
    params = SamplingParams(max_tokens=16, extra_args=power_smc_args())
    request = Request(
        request_id=make_power_smc_child_request_id("parent", 3),
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
    )
    normal_request = Request(
        request_id="normal",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
    )

    assert request.power_smc_child_info is not None
    assert request.power_smc_child_info.parent_id == "parent"
    assert request.power_smc_child_info.particle_idx == 3
    assert normal_request.power_smc_child_info is None


def test_power_smc_alpha_and_proposal_temperature() -> None:
    assert alpha_ramp(0, alpha_final=4.0, ramp_tokens=4) == 1.75
    assert alpha_ramp(3, alpha_final=4.0, ramp_tokens=4) == 4.0
    assert proposal_temperature(4.0) == 0.25


def test_power_smc_group_manager_cached_alpha_schedule_matches_helper() -> None:
    ramped = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            alpha=4.0,
            alpha_ramp_tokens=4,
        ))
    assert [
        ramped._alpha_at_step(step) for step in range(6)
    ] == pytest.approx([
        alpha_ramp(step, alpha_final=4.0, ramp_tokens=4)
        for step in range(6)
    ])

    immediate = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            alpha=16.0,
            alpha_ramp_tokens=1,
        ))
    assert [immediate._alpha_at_step(step) for step in range(3)] == [
        16.0,
        16.0,
        16.0,
    ]


def test_power_smc_weight_update_matches_formula() -> None:
    log_weight, cum_logp, prev_alpha = update_log_weight(
        log_weight=0.0,
        cum_logp=-2.0,
        prev_alpha=1.0,
        alpha_t=2.0,
        base_logp=-0.25,
        proposal_logq=-0.10,
    )

    assert log_weight == pytest.approx(-2.0 + 2.0 * -0.25 - -0.10)
    assert cum_logp == pytest.approx(-2.25)
    assert prev_alpha == 2.0


def test_power_smc_group_manager_weight_update_matches_helper() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            alpha=4.0,
            particles=1,
            alpha_ramp_tokens=4,
        ))
    expected_log_weight = 0.0
    expected_cum_logp = 0.0
    expected_prev_alpha = 1.0
    updates = [
        (-2.0, -1.5),
        (-0.25, -0.1),
        (-1.25, -0.9),
        (-0.5, -0.2),
        (-0.75, -0.3),
    ]

    for step, (base_logp, proposal_logq) in enumerate(updates):
        alpha_t = alpha_ramp(step, alpha_final=4.0, ramp_tokens=4)
        (
            expected_log_weight,
            expected_cum_logp,
            expected_prev_alpha,
        ) = update_log_weight(
            log_weight=expected_log_weight,
            cum_logp=expected_cum_logp,
            prev_alpha=expected_prev_alpha,
            alpha_t=alpha_t,
            base_logp=base_logp,
            proposal_logq=proposal_logq,
        )
        manager.update_after_token(
            0,
            100 + step,
            base_logp=base_logp,
            proposal_logq=proposal_logq,
        )

    particle = manager.particles[0]
    assert particle.log_weight == pytest.approx(expected_log_weight)
    assert particle.cum_logp == pytest.approx(expected_cum_logp)
    assert particle.prev_alpha == pytest.approx(expected_prev_alpha)


def test_power_smc_ess_and_systematic_resample() -> None:
    weights = normalize_log_weights([0.0, math.log(3.0)])
    assert weights == [0.25, 0.75]
    assert effective_sample_size(weights) == pytest.approx(1.6)
    assert systematic_resample([0.1, 0.2, 0.7], random.Random(0)) == [1, 2, 2]


def test_power_smc_ess_matches_torch_reference() -> None:
    log_weights = torch.tensor(
        [-1000.0, -2.5, 0.0, 1.25, 4.0],
        dtype=torch.float64,
    )
    weights = normalize_log_weights(log_weights.tolist())

    torch_weights = torch.softmax(log_weights, dim=0)
    torch_ess = 1.0 / torch.sum(torch_weights.square())

    assert weights == pytest.approx(torch_weights.tolist())
    assert effective_sample_size(weights) == pytest.approx(torch_ess.item())


def test_power_smc_combined_normalize_and_ess_matches_public_helpers() -> None:
    log_weights = [-1000.0, -2.5, 0.0, 1.25, 4.0]

    weights, ess = _normalize_log_weights_and_ess(log_weights)
    expected_weights = normalize_log_weights(log_weights)

    assert weights == pytest.approx(expected_weights)
    assert ess == pytest.approx(effective_sample_size(expected_weights))


def test_power_smc_uniform_log_weights_fast_path() -> None:
    log_weights = [-12.5, -12.5, -12.5, -12.5]

    weights, ess = _normalize_log_weights_and_ess(log_weights)

    assert normalize_log_weights(log_weights) == [0.25, 0.25, 0.25, 0.25]
    assert weights == [0.25, 0.25, 0.25, 0.25]
    assert ess == 4.0


def test_power_smc_systematic_resample_matches_torch_reference() -> None:
    def torch_reference(weights: list[float], seed: int) -> list[int]:
        n = len(weights)
        generator = random.Random(seed)
        positions = (
            torch.arange(n, dtype=torch.float64) / n
            + generator.random() / n
        )
        cdf = torch.cumsum(torch.tensor(weights, dtype=torch.float64), dim=0)
        return torch.searchsorted(cdf, positions, right=False).tolist()

    cases = [
        [0.1, 0.2, 0.7],
        [0.25, 0.25, 0.25, 0.25],
        [0.01, 0.04, 0.15, 0.8],
    ]

    for weights in cases:
        for seed in range(16):
            assert systematic_resample(weights, random.Random(seed)) == (
                torch_reference(weights, seed)
            )


def test_power_smc_systematic_resample_empirical_distribution() -> None:
    weights = [0.1, 0.2, 0.7]
    draws = 5000
    counts = [0, 0, 0]

    for seed in range(draws):
        ancestors = systematic_resample(weights, random.Random(seed))
        for ancestor in ancestors:
            counts[ancestor] += 1

    total = draws * len(weights)
    frequencies = [count / total for count in counts]

    assert frequencies == pytest.approx(weights, abs=0.015)


def test_power_smc_sampler_gathers_base_and_proposal_logprobs() -> None:
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    alpha = torch.tensor([1.0, 2.0], dtype=torch.float32)
    sampled = torch.tensor([2, 0], dtype=torch.int64)

    tensors = Sampler.gather_power_smc_logprobs(logits, alpha, sampled)

    expected_base = logits.log_softmax(dim=-1)[[0, 1], sampled]
    expected_proposal = (logits * alpha.unsqueeze(-1)).log_softmax(dim=-1)[
        [0, 1], sampled
    ]
    assert tensors.base_logprobs == pytest.approx(expected_base)
    assert tensors.proposal_logprobs == pytest.approx(expected_proposal)
    assert tensors.base_logprobs[0] == pytest.approx(tensors.proposal_logprobs[0])


def test_power_smc_logprob_tensors_convert_to_scheduler_lists() -> None:
    tensors = PowerSMCLogprobTensors(
        base_logprobs=torch.tensor([-1.5, -0.25]),
        proposal_logprobs=torch.tensor([-1.0, -0.1]),
    )

    lists = tensors.to_cpu_nonblocking().tolists()

    assert lists.slice_request(0) == pytest.approx((-1.5, -1.0))
    assert lists.slice_request(1) == pytest.approx((-0.25, -0.1))


def test_power_smc_group_manager_alpha_one_keeps_uniform_weights() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            alpha=1.0,
            particles=2,
            block_size=2,
            alpha_ramp_tokens=4,
        ),
        random.Random(0),
    )

    manager.update_after_token(0, 10, base_logp=-2.0, proposal_logq=-2.0)
    manager.update_after_token(1, 11, base_logp=-0.5, proposal_logq=-0.5)

    assert manager.normalized_weights() == pytest.approx([0.5, 0.5])


def test_power_smc_group_manager_update_returns_block_boundary() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            particles=1,
            block_size=2,
        ))

    assert manager.update_after_token(
        0,
        10,
        base_logp=-1.0,
        proposal_logq=-1.0,
    ) is False
    assert manager.update_after_token(
        0,
        11,
        base_logp=-1.0,
        proposal_logq=-1.0,
    ) is True


def test_power_smc_group_manager_particles_one_never_resamples() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(enabled=True, particles=1, block_size=1),
        random.Random(0),
    )

    manager.update_after_token(0, 10, base_logp=-2.0, proposal_logq=-0.1)

    assert manager.maybe_resample() is False
    assert manager.resample_count == 0
    assert manager.diagnostics()["maybe_resample_calls"] == 1
    assert manager.diagnostics()["resample_skip_reasons"] == {
        "particles_one": 1,
    }


def test_power_smc_group_manager_resamples_at_block_boundary() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            particles=3,
            block_size=1,
            ess_threshold=0.99,
            alpha=2.0,
            alpha_ramp_tokens=1,
        ),
        random.Random(0),
    )

    manager.update_after_token(0, 10, base_logp=-6.0, proposal_logq=-0.1)
    manager.update_after_token(1, 11, base_logp=-3.0, proposal_logq=-0.1)
    manager.update_after_token(2, 12, base_logp=-0.1, proposal_logq=-0.1)
    boundary_states = [particle.state_at(1) for particle in manager.particles]

    assert manager.maybe_resample() is True
    assert manager.resample_count == 1
    assert manager.diagnostics()["maybe_resample_calls"] == 1
    assert manager.diagnostics()["block_boundary_checks"] == 1
    assert manager.diagnostics()["last_boundary_check_length"] == 1
    assert manager.diagnostics()["resample_skip_reasons"] == {}
    assert len(manager.particles) == 3
    assert len(manager.unique_ancestors_per_resample) == 1
    assert len({tuple(p.token_ids) for p in manager.particles}) < 3
    assert all(p.log_weight == 0.0 for p in manager.particles)
    assert manager.last_resample_plan is not None
    assert manager.last_resample_plan.ancestors == manager.ancestor_history[-1]
    assert manager.unique_ancestors_per_resample == [
        len(set(manager.last_resample_plan.ancestors))
    ]
    assert manager.last_resample_plan.particle_token_ids == [
        particle.token_ids for particle in manager.particles
    ]
    assert all(
        plan_token_ids is particle.token_ids
        for plan_token_ids, particle in zip(
            manager.last_resample_plan.particle_token_ids,
            manager.particles,
            strict=True,
        ))
    assert len({
        id(plan_token_ids)
        for plan_token_ids in manager.last_resample_plan.particle_token_ids
    }) == len(manager.last_resample_plan.particle_token_ids)
    assert all(p.history_start_length == 1 for p in manager.particles)
    assert [
        p.history_start_state for p in manager.particles
    ] == [
        (0.0, boundary_states[ancestor][1], boundary_states[ancestor][2])
        for ancestor in manager.last_resample_plan.ancestors
    ]
    assert all(p.boundary_state_history == [] for p in manager.particles)


def test_power_smc_group_manager_reuses_boundary_states_for_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            particles=3,
            block_size=1,
            ess_threshold=0.99,
            alpha=2.0,
            alpha_ramp_tokens=1,
        ),
        random.Random(0),
    )

    manager.update_after_token(0, 10, base_logp=-6.0, proposal_logq=-0.1)
    manager.update_after_token(1, 11, base_logp=-3.0, proposal_logq=-0.1)
    manager.update_after_token(2, 12, base_logp=-0.1, proposal_logq=-0.1)

    calls: list[int] = []
    original_state_at = PowerSMCParticleState.state_at

    def counting_state_at(
        self: PowerSMCParticleState,
        output_length: int,
    ) -> tuple[float, float, float]:
        calls.append(output_length)
        return original_state_at(self, output_length)

    monkeypatch.setattr(
        PowerSMCParticleState,
        "state_at",
        counting_state_at,
    )

    assert manager.maybe_resample() is True
    assert calls == [1, 1, 1]


def test_power_smc_group_manager_skips_uniform_boundary_without_normalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            particles=3,
            block_size=1,
            ess_threshold=1.0,
            alpha=2.0,
            alpha_ramp_tokens=1,
        ),
        random.Random(0),
    )

    for particle_idx in range(3):
        manager.update_after_token(
            particle_idx,
            10 + particle_idx,
            base_logp=-0.5,
            proposal_logq=-0.25,
        )

    def fail_normalize(*args, **kwargs):
        raise AssertionError("uniform boundary should skip normalization")

    monkeypatch.setattr(
        power_smc_module,
        "_normalize_log_weights_and_ess_from_max",
        fail_normalize,
    )

    assert manager.maybe_resample() is False
    assert manager.ess_history == [3.0]
    assert manager.diagnostics()["resample_skip_reasons"] == {
        "ess_above_threshold": 1,
    }


def test_power_smc_group_manager_uses_compact_resampled_history() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            particles=2,
            block_size=2,
            ess_threshold=1.0,
            alpha=2.0,
            alpha_ramp_tokens=1,
        ),
        random.Random(0),
    )

    manager.update_after_token(0, 10, base_logp=-6.0, proposal_logq=-0.1)
    manager.update_after_token(0, 11, base_logp=-6.0, proposal_logq=-0.1)
    manager.update_after_token(1, 20, base_logp=-0.1, proposal_logq=-0.1)
    manager.update_after_token(1, 21, base_logp=-0.1, proposal_logq=-0.1)

    assert manager.maybe_resample() is True
    assert all(p.history_start_length == 2 for p in manager.particles)
    assert all(p.boundary_state_history == [] for p in manager.particles)

    for particle_idx in range(2):
        manager.update_after_token(
            particle_idx,
            100 + particle_idx,
            base_logp=-0.2,
            proposal_logq=-0.1,
        )
        manager.update_after_token(
            particle_idx,
            200 + particle_idx,
            base_logp=-0.2,
            proposal_logq=-0.1,
        )

    assert all(
        [length for length, _ in p.boundary_state_history] == [4]
        for p in manager.particles)
    assert manager.maybe_resample() is False
    assert manager.last_boundary_check_length == 4


def test_power_smc_group_manager_resamples_at_common_crossed_boundary() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            particles=2,
            block_size=2,
            ess_threshold=1.0,
            alpha=2.0,
            alpha_ramp_tokens=1,
        ),
        random.Random(0),
    )

    manager.update_after_token(0, 10, base_logp=-6.0, proposal_logq=-0.1)
    manager.update_after_token(0, 11, base_logp=-6.0, proposal_logq=-0.1)
    manager.update_after_token(1, 20, base_logp=-0.1, proposal_logq=-0.1)
    manager.update_after_token(1, 21, base_logp=-0.1, proposal_logq=-0.1)
    manager.update_after_token(1, 22, base_logp=-0.1, proposal_logq=-0.1)

    assert manager.maybe_resample() is True
    assert manager.block_boundary_checks == 1
    assert manager.last_boundary_check_length == 2
    assert manager.last_resample_plan is not None
    assert all(len(tokens) == 2
               for tokens in manager.last_resample_plan.particle_token_ids)
    assert all(22 not in tokens
               for tokens in manager.last_resample_plan.particle_token_ids)
    assert all(len(particle.token_ids) == 2 for particle in manager.particles)
    assert all(particle.log_weight == 0.0 for particle in manager.particles)


def test_power_smc_group_manager_rejects_updates_after_done() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(enabled=True, particles=1),
        random.Random(0),
    )

    manager.update_after_token(
        0,
        10,
        base_logp=-1.0,
        proposal_logq=-1.0,
        done=True,
        finish_reason="stop",
    )

    with pytest.raises(ValueError, match="already done"):
        manager.update_after_token(0, 11, base_logp=-1.0, proposal_logq=-1.0)


def test_power_smc_group_manager_does_not_resample_done_particles() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(enabled=True, particles=2, block_size=1),
        random.Random(0),
    )

    manager.update_after_token(
        0,
        10,
        base_logp=-3.0,
        proposal_logq=-0.1,
        done=True,
        finish_reason="stop",
    )
    manager.update_after_token(1, 11, base_logp=-0.1, proposal_logq=-0.1)

    assert manager.maybe_resample() is False
    assert manager.last_resample_plan is None
    diagnostics = manager.diagnostics()
    assert diagnostics["particle_lengths"] == [1, 1]
    assert diagnostics["min_particle_length"] == 1
    assert diagnostics["max_particle_length"] == 1
    assert diagnostics["done_count"] == 1
    assert diagnostics["finish_reason_counts"] == {"stop": 1}
    assert diagnostics["stop_reason_counts"] == {}
    assert diagnostics["maybe_resample_calls"] == 1
    assert diagnostics["block_boundary_checks"] == 0
    assert diagnostics["last_boundary_check_length"] == 0
    assert diagnostics["resample_skip_reasons"] == {"done_particle": 1}


def test_power_smc_group_manager_final_select_and_diagnostics() -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(enabled=True, particles=2, alpha=2.0),
        random.Random(0),
    )
    manager.update_after_token(0, 10, base_logp=-2.0, proposal_logq=-0.5)
    manager.update_after_token(1, 11, base_logp=-0.5, proposal_logq=-0.5)

    chosen = manager.final_select()
    diagnostics = manager.diagnostics()

    assert chosen in (0, 1)
    assert diagnostics["chosen_particle"] == chosen
    assert diagnostics["particles"] == 2
    assert diagnostics["final_ess"] <= 2.0
    assert diagnostics["ess_history"] == []
    assert diagnostics["ancestor_history"] == []
    assert diagnostics["maybe_resample_calls"] == 0
    assert diagnostics["block_boundary_checks"] == 0
    assert diagnostics["last_boundary_check_length"] == 0
    assert diagnostics["resample_skip_reasons"] == {}
    assert diagnostics["particle_lengths"] == [1, 1]
    assert diagnostics["min_particle_length"] == 1
    assert diagnostics["max_particle_length"] == 1
    assert diagnostics["done_count"] == 0
    assert diagnostics["finish_reason_counts"] == {}
    assert diagnostics["stop_reason_counts"] == {}
    assert "avg_log_weight" in diagnostics
    assert diagnostics["kv_resample_events"] == []
    assert diagnostics["kv_alias_success_count"] == 0
    assert diagnostics["kv_alias_fallback_count"] == 0
    assert diagnostics["kv_aliased_blocks"] == 0
    assert diagnostics["kv_aliased_tokens"] == 0
    assert diagnostics["kv_cow_physical_blocks"] == 0
    assert diagnostics["kv_cow_saved_blocks"] == 0
    assert diagnostics["kv_cow_saved_tokens"] == 0


def test_power_smc_group_manager_reuses_final_select_ess_for_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PowerSMCGroupManager(
        PowerSMCConfig(enabled=True, particles=2, alpha=2.0),
        random.Random(0),
    )
    manager.update_after_token(0, 10, base_logp=-2.0, proposal_logq=-0.5)
    manager.update_after_token(1, 11, base_logp=-0.5, proposal_logq=-0.5)

    calls = 0
    original = _normalize_log_weights_and_ess

    def counting_normalize(
        log_weights: list[float],
    ) -> tuple[list[float], float]:
        nonlocal calls
        calls += 1
        return original(log_weights)

    monkeypatch.setattr(
        "vllm.v1.power_smc._normalize_log_weights_and_ess",
        counting_normalize,
    )

    manager.final_select()
    assert calls == 1
    diagnostics = manager.diagnostics()

    assert calls == 1
    assert diagnostics["final_ess"] <= 2.0


def test_power_smc_parent_request_selects_one_internal_particle() -> None:
    params = SamplingParams(
        max_tokens=1,
        seed=0,
        extra_args=power_smc_args(
            alpha=2.0,
            particles=2,
            alpha_ramp_tokens=1,
            return_diagnostics=True,
        ),
    )
    config = PowerSMCConfig.from_sampling_params(params)
    assert config is not None
    parent = PowerSMCParentRequest(make_engine_core_request(params), config)

    child0, child0_params = parent.get_child_info(0)
    child1, child1_params = parent.get_child_info(1)

    assert child0.startswith("psmc0_")
    assert child1.startswith("psmc1_")
    assert child0_params.n == 1
    assert child1_params.n == 1

    parent.observe_power_smc_step(
        child0,
        [10],
        (math.log(0.1), math.log(0.5)),
        "length",
    )
    parent.observe_power_smc_step(
        child1,
        [11],
        (math.log(0.9), math.log(0.5)),
        "length",
    )

    outputs0, finished0 = parent.get_outputs(
        child0,
        CompletionOutput(
            index=0,
            text="low",
            token_ids=[10],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="length",
        ),
    )
    outputs1, finished1 = parent.get_outputs(
        child1,
        CompletionOutput(
            index=1,
            text="high",
            token_ids=[11],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="length",
        ),
    )

    assert outputs0 == []
    assert finished0 is False
    assert finished1 is True
    assert len(outputs1) == 1
    assert outputs1[0].index == 0
    assert outputs1[0].text == "high"
    chosen_particle = parent.manager.chosen_particle
    assert chosen_particle is not None
    assert outputs1[0].token_ids is parent.manager.particles[
        chosen_particle].token_ids
    token_ids_before_rewrite = outputs1[0].token_ids
    parent.rewrite_power_smc_outputs(
        outputs1,
        lambda token_ids: "|".join(str(token_id) for token_id in token_ids),
    )
    assert outputs1[0].token_ids is token_ids_before_rewrite
    assert outputs1[0].token_ids == [11]
    assert outputs1[0].text == "11"
    assert parent.get_power_smc_diagnostics() == {
        **parent.manager.diagnostics(),
    }


def test_power_smc_parent_request_caches_seedless_child_sampling_params() -> None:
    seedless_params = SamplingParams(
        max_tokens=1,
        extra_args=power_smc_args(particles=2),
    )
    seedless_config = PowerSMCConfig.from_sampling_params(seedless_params)
    assert seedless_config is not None
    seedless_parent = PowerSMCParentRequest(
        make_engine_core_request(seedless_params),
        seedless_config,
    )

    _, child0_params = seedless_parent.get_child_info(0)
    _, child1_params = seedless_parent.get_child_info(1)

    assert child0_params is child1_params
    assert child0_params.n == 1
    assert child0_params.output_kind == RequestOutputKind.FINAL_ONLY
    assert child0_params.seed is None

    seeded_params = SamplingParams(
        max_tokens=1,
        seed=7,
        extra_args=power_smc_args(particles=2),
    )
    seeded_config = PowerSMCConfig.from_sampling_params(seeded_params)
    assert seeded_config is not None
    seeded_parent = PowerSMCParentRequest(
        make_engine_core_request(seeded_params),
        seeded_config,
    )

    _, seeded_child0_params = seeded_parent.get_child_info(0)
    _, seeded_child1_params = seeded_parent.get_child_info(1)

    assert seeded_child0_params is not seeded_child1_params
    assert seeded_child0_params.seed == 7
    assert seeded_child1_params.seed == 8


def test_power_smc_parent_request_particles_one_returns_diagnostics() -> None:
    params = SamplingParams(
        max_tokens=1,
        seed=0,
        extra_args=power_smc_args(
            alpha=1.0,
            particles=1,
            return_diagnostics=True,
        ),
    )
    config = PowerSMCConfig.from_sampling_params(params)
    assert config is not None
    parent = PowerSMCParentRequest(make_engine_core_request(params), config)
    child_id, _ = parent.get_child_info(0)

    parent.observe_power_smc_step(
        child_id,
        [10],
        (-1.0, -1.0),
        "length",
    )
    outputs, finished = parent.get_outputs(
        child_id,
        CompletionOutput(
            index=0,
            text="token",
            token_ids=[10],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="length",
        ),
    )

    assert finished is True
    assert outputs[0].token_ids == [10]
    diagnostics = parent.get_power_smc_diagnostics()
    assert diagnostics is not None
    assert diagnostics["particles"] == 1
    assert diagnostics["resample_count"] == 0
    assert diagnostics["final_ess"] == 1.0


def test_power_smc_parent_request_queues_resample_only_at_boundaries() -> None:
    params = SamplingParams(
        max_tokens=8,
        seed=0,
        extra_args=power_smc_args(
            particles=2,
            block_size=4,
            return_diagnostics=True,
        ),
    )
    config = PowerSMCConfig.from_sampling_params(params)
    assert config is not None
    parent = PowerSMCParentRequest(make_engine_core_request(params), config)
    child_id, _ = parent.get_child_info(0)

    for token_id in (10, 11, 12):
        assert parent.observe_power_smc_step(
            child_id,
            [token_id],
            (-1.0, -1.0),
            None,
        ) is False

    assert parent.manager.maybe_resample_calls == 0
    assert len(parent.manager.particles[0].token_ids) == 3
    assert parent.observe_power_smc_step(
        child_id,
        [13],
        (-1.0, -1.0),
        None,
    ) is True
    assert parent.observe_power_smc_step(
        child_id,
        [14],
        (-1.0, -1.0),
        "length",
    ) is True


def test_power_smc_parent_request_records_resampling_diagnostics() -> None:
    params = SamplingParams(
        max_tokens=1,
        seed=0,
        extra_args=power_smc_args(
            alpha=2.0,
            particles=3,
            block_size=1,
            ess_threshold=0.99,
            alpha_ramp_tokens=1,
            return_diagnostics=True,
        ),
    )
    config = PowerSMCConfig.from_sampling_params(params)
    assert config is not None
    parent = PowerSMCParentRequest(make_engine_core_request(params), config)
    child_ids = [parent.get_child_info(idx)[0] for idx in range(3)]

    parent.observe_power_smc_step(child_ids[0], [10], (-6.0, -0.1), None)
    parent.observe_power_smc_step(child_ids[1], [11], (-3.0, -0.1), None)
    parent.observe_power_smc_step(child_ids[2], [12], (-0.1, -0.1), None)

    assert parent.manager.resample_count == 0
    parent.maybe_resample_power_smc()
    assert parent.manager.resample_count == 1
    assert parent.manager.last_resample_plan is not None
    parent.observe_power_smc_kv_event({
        "resample_index": 1,
        "kv_mode": "snapshot_alias_replay",
        "alias_success_count": 2,
        "fallback_count": 0,
        "aliased_blocks": 3,
        "aliased_tokens": 48,
        "cow_physical_blocks": 2,
        "cow_saved_blocks": 1,
        "cow_saved_tokens": 16,
    })
    assert parent.manager.diagnostics()["resample_count"] == 1
    assert parent.manager.diagnostics()["kv_resample_events"] == [{
        "resample_index": 1,
        "kv_mode": "snapshot_alias_replay",
        "alias_success_count": 2,
        "fallback_count": 0,
        "aliased_blocks": 3,
        "aliased_tokens": 48,
        "cow_physical_blocks": 2,
        "cow_saved_blocks": 1,
        "cow_saved_tokens": 16,
    }]
    assert parent.manager.diagnostics()["kv_alias_success_count"] == 2
    assert parent.manager.diagnostics()["kv_aliased_tokens"] == 48
    assert parent.manager.diagnostics()["kv_cow_saved_blocks"] == 1


def test_power_smc_external_abort_removes_child_and_parent_state() -> None:
    params = SamplingParams(
        max_tokens=1,
        extra_args=power_smc_args(particles=2),
    )
    config = PowerSMCConfig.from_sampling_params(params)
    assert config is not None
    parent = PowerSMCParentRequest(make_engine_core_request(params), config)
    processor = OutputProcessor(tokenizer=None, log_stats=False)

    child_ids = []
    for idx in range(config.particles):
        child_id, child_params = parent.get_child_info(idx)
        child_request = make_engine_core_request(child_params)
        child_request.request_id = child_id
        processor.add_request(child_request, "prompt", parent, idx)
        child_ids.append(child_id)

    aborted = processor.abort_requests(["external-parent"], internal=False)

    assert set(aborted) == set(child_ids)
    assert processor.request_states == {}
    assert processor.parent_requests == {}
    assert "external-parent" not in processor.external_req_ids


def test_power_smc_external_abort_does_not_remove_normal_request() -> None:
    power_params = SamplingParams(
        max_tokens=1,
        extra_args=power_smc_args(particles=2),
    )
    config = PowerSMCConfig.from_sampling_params(power_params)
    assert config is not None
    parent = PowerSMCParentRequest(make_engine_core_request(power_params),
                                   config)
    processor = OutputProcessor(tokenizer=None, log_stats=False)

    child_ids = []
    for idx in range(config.particles):
        child_id, child_params = parent.get_child_info(idx)
        child_request = make_engine_core_request(child_params)
        child_request.request_id = child_id
        processor.add_request(child_request, "prompt", parent, idx)
        child_ids.append(child_id)

    normal_request = make_engine_core_request(SamplingParams(max_tokens=1))
    normal_request.request_id = "normal-internal"
    normal_request.external_req_id = "normal-external"
    processor.add_request(normal_request, "normal prompt")

    aborted = processor.abort_requests(["external-parent"], internal=False)

    assert set(aborted) == set(child_ids)
    assert set(processor.request_states) == {"normal-internal"}
    assert processor.parent_requests == {}
    assert "external-parent" not in processor.external_req_ids
    assert processor.external_req_ids["normal-external"] == ["normal-internal"]


def test_request_reset_output_token_ids_rebuilds_views_and_hashes() -> None:
    def block_hasher(request: Request):
        return [tuple(request.all_token_ids)]

    params = SamplingParams(max_tokens=8)
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2],
        sampling_params=params,
        pooling_params=None,
        block_hasher=block_hasher,
    )
    request.append_output_token_ids([10, 11, 12])

    request.reset_output_token_ids([20, 21])

    assert list(request.output_token_ids) == [20, 21]
    assert list(request.all_token_ids) == [1, 2, 20, 21]
    assert request.has_output_token_ids([20, 21])
    assert not request.has_output_token_ids([20, 22])
    assert request.block_hashes == [(1, 2, 20, 21)]


def test_scheduler_power_smc_state_isolated_from_normal_requests() -> None:
    normal_request = Request(
        request_id="normal",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    power_params = SamplingParams(
        max_tokens=8,
        extra_args=power_smc_args(particles=2, block_size=1),
    )
    power_request = Request(
        request_id=make_power_smc_child_request_id("parent", 0),
        prompt_token_ids=[3, 4],
        sampling_params=power_params,
        pooling_params=None,
    )

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {
        normal_request.request_id: normal_request,
        power_request.request_id: power_request,
    }
    scheduler.power_smc_group_managers = {}
    scheduler.power_smc_group_children = {}

    assert Scheduler._observe_power_smc_request_output(
        scheduler,
        normal_request,
        [10],
        (-1.0, -1.0),
        None,
    ) is None
    assert scheduler.power_smc_group_managers == {}
    assert scheduler.power_smc_group_children == {}
    assert normal_request.request_id in scheduler.requests

    assert Scheduler._observe_power_smc_request_output(
        scheduler,
        power_request,
        [20],
        (-1.0, -1.0),
        None,
    ) == "parent"
    assert set(scheduler.requests) == {
        normal_request.request_id,
        power_request.request_id,
    }
    assert list(scheduler.power_smc_group_managers) == ["parent"]
    assert scheduler.power_smc_group_children == {
        "parent": {
            0: power_request.request_id,
        }
    }


def test_scheduler_defers_power_smc_resample_until_batch_boundary() -> None:
    params = SamplingParams(
        max_tokens=8,
        extra_args=power_smc_args(
            alpha=2.0,
            particles=3,
            block_size=1,
            ess_threshold=0.99,
            alpha_ramp_tokens=1,
        ),
    )
    requests = []
    for particle_idx in range(3):
        requests.append(
            Request(
                request_id=make_power_smc_child_request_id("parent", particle_idx),
                prompt_token_ids=[1, 2],
                sampling_params=params,
                pooling_params=None,
            ))

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.power_smc_group_managers = {}
    scheduler.power_smc_group_children = {}
    scheduler._apply_power_smc_resample_plan = lambda parent_id, manager: []

    for particle_idx, request in enumerate(requests):
        parent_id = Scheduler._observe_power_smc_request_output(
            scheduler,
            request,
            [10 + particle_idx],
            ([-6.0, -3.0, -0.1][particle_idx], -0.1),
            None,
        )
        assert parent_id == "parent"

    manager = scheduler.power_smc_group_managers["parent"]
    assert manager.resample_count == 0
    assert manager.maybe_resample_calls == 0

    assert Scheduler._maybe_resample_power_smc_parent(scheduler, "parent") == []
    assert manager.resample_count == 1
    assert manager.block_boundary_checks == 1


def test_scheduler_only_queues_power_smc_resample_at_boundaries() -> None:
    params = SamplingParams(
        max_tokens=8,
        extra_args=power_smc_args(
            particles=2,
            block_size=4,
        ),
    )
    request = Request(
        request_id=make_power_smc_child_request_id("parent", 0),
        prompt_token_ids=[1, 2],
        sampling_params=params,
        pooling_params=None,
    )

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.power_smc_group_managers = {}
    scheduler.power_smc_group_children = {}

    for token_id in (10, 11, 12):
        assert Scheduler._observe_power_smc_request_output(
            scheduler,
            request,
            [token_id],
            (-1.0, -1.0),
            None,
        ) is None

    manager = scheduler.power_smc_group_managers["parent"]
    assert manager.maybe_resample_calls == 0

    assert Scheduler._observe_power_smc_request_output(
        scheduler,
        request,
        [13],
        (-1.0, -1.0),
        None,
    ) == "parent"


def test_scheduler_pauses_fast_power_smc_child_at_group_boundary() -> None:
    params = SamplingParams(
        max_tokens=8,
        extra_args=power_smc_args(particles=2, block_size=2),
    )
    fast = Request(
        request_id=make_power_smc_child_request_id("parent", 0),
        prompt_token_ids=[1, 2],
        sampling_params=params,
        pooling_params=None,
    )
    lagging = Request(
        request_id=make_power_smc_child_request_id("parent", 1),
        prompt_token_ids=[1, 2],
        sampling_params=params,
        pooling_params=None,
    )
    fast.append_output_token_ids([10, 11])
    lagging.append_output_token_ids([20])
    fast.num_computed_tokens = fast.num_tokens
    lagging.num_computed_tokens = lagging.num_tokens

    scheduler = object.__new__(Scheduler)
    scheduler.power_smc_group_managers = {
        "parent": PowerSMCGroupManager(
            PowerSMCConfig(enabled=True, particles=2, block_size=2))
    }
    manager = scheduler.power_smc_group_managers["parent"]
    manager.particles[0].token_ids = [10, 11]
    manager.particles[1].token_ids = [20]

    assert Scheduler._power_smc_should_pause_at_group_boundary(
        scheduler, fast) is True
    assert Scheduler._power_smc_should_pause_at_group_boundary(
        scheduler, lagging) is False

    manager.particles[1].token_ids = [20, 21]
    lagging.append_output_token_ids(21)
    assert Scheduler._power_smc_should_pause_at_group_boundary(
        scheduler, fast) is False


def test_scheduler_does_not_pause_power_smc_replay_at_boundary() -> None:
    params = SamplingParams(
        max_tokens=8,
        extra_args=power_smc_args(particles=2, block_size=2),
    )
    request = Request(
        request_id=make_power_smc_child_request_id("parent", 0),
        prompt_token_ids=[1, 2],
        sampling_params=params,
        pooling_params=None,
    )
    request.append_output_token_ids([10, 11])
    request.num_computed_tokens = request.num_tokens - 1

    scheduler = object.__new__(Scheduler)
    scheduler.power_smc_group_managers = {
        "parent": PowerSMCGroupManager(
            PowerSMCConfig(enabled=True, particles=2, block_size=2))
    }
    manager = scheduler.power_smc_group_managers["parent"]
    manager.particles[0].token_ids = [10, 11]
    manager.particles[1].token_ids = [20]

    assert Scheduler._power_smc_should_pause_at_group_boundary(
        scheduler, request) is False


def test_scheduler_applies_power_smc_resample_plan_to_request_state() -> None:
    class FakeCacheManager:
        def __init__(self) -> None:
            self.freed: list[str] = []

        def free(self, request: Request) -> None:
            self.freed.append(request.request_id)

    params = SamplingParams(max_tokens=8, extra_args=power_smc_args())
    request = Request(
        request_id=make_power_smc_child_request_id("parent", 0),
        prompt_token_ids=[1, 2],
        sampling_params=params,
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING
    request.append_output_token_ids([10, 11, 12])
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1
    request.async_tokens_to_discard = 1
    request.spec_token_ids = [99]
    request.is_prefill_chunk = True

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.kv_cache_manager = FakeCacheManager()
    scheduler.encoder_cache_manager = FakeCacheManager()
    scheduler._inflight_prefills = {request}
    scheduler.power_smc_group_children = {"parent": {0: request.request_id}}
    manager = PowerSMCGroupManager(PowerSMCConfig(enabled=True, particles=1))
    manager.last_resample_plan = PowerSMCResamplePlan(
        ancestors=[0],
        particle_token_ids=[[20, 21]],
    )

    resampled = Scheduler._apply_power_smc_resample_plan(
        scheduler,
        "parent",
        manager,
    )

    assert resampled == [request]
    assert scheduler.kv_cache_manager.freed == [request.request_id]
    assert scheduler.encoder_cache_manager.freed == [request.request_id]
    assert request not in scheduler._inflight_prefills
    assert request.status == RequestStatus.PREEMPTED
    assert list(request.output_token_ids) == [20, 21]
    assert list(request.all_token_ids) == [1, 2, 20, 21]
    assert request.num_computed_tokens == 0
    assert request.num_output_placeholders == 0
    assert request.async_tokens_to_discard == 0
    assert request.spec_token_ids == []
    assert request.is_prefill_chunk is False


def test_scheduler_aliases_power_smc_resample_from_block_snapshots() -> None:
    class FakeBlockPool:
        num_gpu_blocks = 100

        def __init__(self) -> None:
            self.free_blocks = 80

        def get_num_free_blocks(self) -> int:
            return self.free_blocks

        def get_usage(self) -> float:
            return (self.num_gpu_blocks - self.get_num_free_blocks()
                    ) / self.num_gpu_blocks

    class FakeCacheManager:
        def __init__(self) -> None:
            self.snapshots = {
                make_power_smc_child_request_id("parent", 0): "blocks0",
                make_power_smc_child_request_id("parent", 1): "blocks1",
            }
            self.cached = {
                make_power_smc_child_request_id("parent", 0): [2],
                make_power_smc_child_request_id("parent", 1): [2],
            }
            self.aliases: list[tuple[str, str, list[int], int]] = []
            self.get_blocks_calls: list[str] = []
            self.freed: list[str] = []
            self.block_pool = FakeBlockPool()

        def get_blocks(self, request_id: str) -> str:
            self.get_blocks_calls.append(request_id)
            return self.snapshots[request_id]

        def get_num_cached_blocks(self, request_id: str) -> list[int]:
            return self.cached[request_id]

        def alias_request_blocks_from_snapshot(
            self,
            *,
            dst_request_id: str,
            src_blocks: str,
            src_num_cached_blocks: list[int],
            num_prefix_blocks: int,
        ) -> None:
            self.aliases.append((
                dst_request_id,
                src_blocks,
                src_num_cached_blocks,
                num_prefix_blocks,
            ))
            self.snapshots[dst_request_id] = f"mutated-{dst_request_id}"
            self.block_pool.free_blocks = 78

        def free(self, request: Request) -> None:
            self.freed.append(request.request_id)

    class FakeEncoderCacheManager:
        def __init__(self) -> None:
            self.freed: list[str] = []

        def free(self, request: Request) -> None:
            self.freed.append(request.request_id)

    params = SamplingParams(max_tokens=8, extra_args=power_smc_args())
    requests = []
    for particle_idx in range(2):
        request = Request(
            request_id=make_power_smc_child_request_id("parent", particle_idx),
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=params,
            pooling_params=None,
        )
        request.status = RequestStatus.RUNNING
        request.append_output_token_ids([10, 11, 12, 13, 14])
        request.num_computed_tokens = request.num_tokens
        requests.append(request)

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.kv_cache_manager = FakeCacheManager()
    scheduler.encoder_cache_manager = FakeEncoderCacheManager()
    scheduler._inflight_prefills = set(requests)
    scheduler.power_smc_group_children = {
        "parent": {
            0: requests[0].request_id,
            1: requests[1].request_id,
        }
    }
    scheduler.block_size = 4
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4)),
        ])

    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            particles=2,
            kv_pool_diagnostics=True,
        ))
    manager.resample_count = 1
    manager.last_resample_plan = PowerSMCResamplePlan(
        ancestors=[1, 1],
        particle_token_ids=[
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
        ],
    )

    resampled = Scheduler._apply_power_smc_resample_plan(
        scheduler,
        "parent",
        manager,
    )

    assert resampled == requests
    assert scheduler.kv_cache_manager.freed == []
    assert scheduler.kv_cache_manager.aliases == [
        (requests[0].request_id, "blocks1", [2], 2),
        (requests[1].request_id, "blocks1", [2], 2),
    ]
    assert scheduler.kv_cache_manager.get_blocks_calls == [
        requests[1].request_id,
    ]
    assert manager.diagnostics()["kv_resample_events"] == [{
        "resample_index": 1,
        "kv_cow_enabled": True,
        "kv_alias_supported": True,
        "kv_mode": "snapshot_alias_replay",
        "block_size": 4,
        "child_count": 2,
        "snapshot_count": 1,
        "alias_attempt_count": 2,
        "alias_success_count": 2,
        "fallback_count": 0,
        "aliased_blocks": 4,
        "aliased_tokens": 16,
        "cow_physical_blocks": 2,
        "cow_saved_blocks": 2,
        "cow_saved_tokens": 8,
        "replay_tokens": 2,
        "identity_noop_count": 0,
        "kv_pool_total_blocks_before": 100,
        "kv_pool_free_blocks_before": 80,
        "kv_pool_used_blocks_before": 20,
        "kv_pool_usage_before": 0.2,
        "kv_pool_total_blocks_after": 100,
        "kv_pool_free_blocks_after": 78,
        "kv_pool_used_blocks_after": 22,
        "kv_pool_usage_after": 0.22,
    }]
    assert manager.diagnostics()["kv_alias_success_count"] == 2
    assert manager.diagnostics()["kv_aliased_blocks"] == 4
    assert manager.diagnostics()["kv_aliased_tokens"] == 16
    assert manager.diagnostics()["kv_cow_physical_blocks"] == 2
    assert manager.diagnostics()["kv_cow_saved_blocks"] == 2
    assert manager.diagnostics()["kv_cow_saved_tokens"] == 8
    assert scheduler.encoder_cache_manager.freed == [
        requests[0].request_id,
        requests[1].request_id,
    ]
    assert [request.num_computed_tokens for request in requests] == [8, 8]
    assert [list(request.output_token_ids) for request in requests] == [
        [20, 21, 22, 23, 24],
        [30, 31, 32, 33, 34],
    ]


def test_scheduler_skips_identity_power_smc_resample_plan() -> None:
    class FakeBlockPool:
        num_gpu_blocks = 100

        def get_num_free_blocks(self) -> int:
            return 80

        def get_usage(self) -> float:
            return 0.2

    class FakeCacheManager:
        def __init__(self) -> None:
            self.freed: list[str] = []
            self.block_pool = FakeBlockPool()

        def free(self, request: Request) -> None:
            self.freed.append(request.request_id)

    class FakeEncoderCacheManager:
        def __init__(self) -> None:
            self.freed: list[str] = []

        def free(self, request: Request) -> None:
            self.freed.append(request.request_id)

    params = SamplingParams(max_tokens=8, extra_args=power_smc_args())
    requests = []
    for particle_idx, token_ids in enumerate(([10, 11], [20, 21])):
        request = Request(
            request_id=make_power_smc_child_request_id("parent", particle_idx),
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=params,
            pooling_params=None,
        )
        request.status = RequestStatus.RUNNING
        request.append_output_token_ids(list(token_ids))
        request.num_computed_tokens = request.num_tokens
        requests.append(request)

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.kv_cache_manager = FakeCacheManager()
    scheduler.encoder_cache_manager = FakeEncoderCacheManager()
    scheduler._inflight_prefills = set(requests)
    scheduler.power_smc_group_children = {
        "parent": {
            0: requests[0].request_id,
            1: requests[1].request_id,
        }
    }
    scheduler.block_size = 4
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4)),
        ])

    manager = PowerSMCGroupManager(PowerSMCConfig(enabled=True, particles=2))
    manager.resample_count = 1
    manager.last_resample_plan = PowerSMCResamplePlan(
        ancestors=[0, 1],
        particle_token_ids=[
            [10, 11],
            [20, 21],
        ],
    )

    resampled = Scheduler._apply_power_smc_resample_plan(
        scheduler,
        "parent",
        manager,
    )

    assert resampled == []
    assert scheduler.kv_cache_manager.freed == []
    assert scheduler.encoder_cache_manager.freed == []
    assert requests[0].status == RequestStatus.RUNNING
    assert requests[1].status == RequestStatus.RUNNING
    assert scheduler._last_power_smc_kv_event is not None
    assert manager.diagnostics()["kv_resample_events"] == [{
        "resample_index": 1,
        "kv_cow_enabled": True,
        "kv_alias_supported": True,
        "kv_mode": "identity_noop",
        "block_size": 4,
        "child_count": 2,
        "snapshot_count": 0,
        "alias_attempt_count": 0,
        "alias_success_count": 0,
        "fallback_count": 0,
        "aliased_blocks": 0,
        "aliased_tokens": 0,
        "cow_physical_blocks": 0,
        "cow_saved_blocks": 0,
        "cow_saved_tokens": 0,
        "replay_tokens": 0,
        "identity_noop_count": 2,
    }]
    assert all(
        not key.startswith("kv_pool_")
        for key in manager.diagnostics()["kv_resample_events"][0])


def test_scheduler_skips_mixed_self_ancestor_resample_children() -> None:
    class FakeBlockPool:
        num_gpu_blocks = 100

        def __init__(self) -> None:
            self.free_blocks = 80

        def get_num_free_blocks(self) -> int:
            return self.free_blocks

        def get_usage(self) -> float:
            return (self.num_gpu_blocks - self.free_blocks) / self.num_gpu_blocks

    class FakeCacheManager:
        def __init__(self) -> None:
            self.snapshots = {
                make_power_smc_child_request_id("parent", 0): "blocks0",
                make_power_smc_child_request_id("parent", 1): "blocks1",
                make_power_smc_child_request_id("parent", 2): "blocks2",
            }
            self.cached = {
                make_power_smc_child_request_id("parent", 0): [2],
                make_power_smc_child_request_id("parent", 1): [2],
                make_power_smc_child_request_id("parent", 2): [2],
            }
            self.get_blocks_calls: list[str] = []
            self.aliases: list[tuple[str, str, list[int], int]] = []
            self.freed: list[str] = []
            self.block_pool = FakeBlockPool()

        def get_blocks(self, request_id: str) -> str:
            self.get_blocks_calls.append(request_id)
            return self.snapshots[request_id]

        def get_num_cached_blocks(self, request_id: str) -> list[int]:
            return self.cached[request_id]

        def alias_request_blocks_from_snapshot(
            self,
            *,
            dst_request_id: str,
            src_blocks: str,
            src_num_cached_blocks: list[int],
            num_prefix_blocks: int,
        ) -> None:
            self.aliases.append((
                dst_request_id,
                src_blocks,
                src_num_cached_blocks,
                num_prefix_blocks,
            ))
            self.block_pool.free_blocks = 79

        def free(self, request: Request) -> None:
            self.freed.append(request.request_id)

    class FakeEncoderCacheManager:
        def __init__(self) -> None:
            self.freed: list[str] = []

        def free(self, request: Request) -> None:
            self.freed.append(request.request_id)

    params = SamplingParams(max_tokens=8, extra_args=power_smc_args(particles=3))
    requests = []
    for particle_idx, token_ids in enumerate(([10, 11], [20, 21], [30, 31])):
        request = Request(
            request_id=make_power_smc_child_request_id("parent", particle_idx),
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=params,
            pooling_params=None,
        )
        request.status = RequestStatus.RUNNING
        request.append_output_token_ids(list(token_ids))
        request.num_computed_tokens = request.num_tokens
        requests.append(request)

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.kv_cache_manager = FakeCacheManager()
    scheduler.encoder_cache_manager = FakeEncoderCacheManager()
    scheduler._inflight_prefills = set(requests)
    scheduler.power_smc_group_children = {
        "parent": {
            0: requests[0].request_id,
            1: requests[1].request_id,
            2: requests[2].request_id,
        }
    }
    scheduler.block_size = 4
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4)),
        ])

    manager = PowerSMCGroupManager(
        PowerSMCConfig(
            enabled=True,
            particles=3,
            kv_pool_diagnostics=True,
        ))
    manager.resample_count = 1
    manager.last_resample_plan = PowerSMCResamplePlan(
        ancestors=[0, 0, 2],
        particle_token_ids=[
            [10, 11],
            [10, 11],
            [30, 31],
        ],
    )

    resampled = Scheduler._apply_power_smc_resample_plan(
        scheduler,
        "parent",
        manager,
    )

    assert resampled == [requests[1]]
    assert scheduler.kv_cache_manager.get_blocks_calls == [
        requests[0].request_id,
    ]
    assert scheduler.kv_cache_manager.aliases == [
        (requests[1].request_id, "blocks0", [2], 1),
    ]
    assert scheduler.kv_cache_manager.freed == []
    assert scheduler.encoder_cache_manager.freed == [requests[1].request_id]
    assert [request.status for request in requests] == [
        RequestStatus.RUNNING,
        RequestStatus.PREEMPTED,
        RequestStatus.RUNNING,
    ]
    assert [list(request.output_token_ids) for request in requests] == [
        [10, 11],
        [10, 11],
        [30, 31],
    ]
    assert manager.diagnostics()["kv_resample_events"] == [{
        "resample_index": 1,
        "kv_cow_enabled": True,
        "kv_alias_supported": True,
        "kv_mode": "snapshot_alias_replay",
        "block_size": 4,
        "child_count": 3,
        "snapshot_count": 1,
        "alias_attempt_count": 1,
        "alias_success_count": 1,
        "fallback_count": 0,
        "aliased_blocks": 1,
        "aliased_tokens": 4,
        "cow_physical_blocks": 1,
        "cow_saved_blocks": 0,
        "cow_saved_tokens": 0,
        "replay_tokens": 2,
        "identity_noop_count": 2,
        "kv_pool_total_blocks_before": 100,
        "kv_pool_free_blocks_before": 80,
        "kv_pool_used_blocks_before": 20,
        "kv_pool_usage_before": 0.2,
        "kv_pool_total_blocks_after": 100,
        "kv_pool_free_blocks_after": 79,
        "kv_pool_used_blocks_after": 21,
        "kv_pool_usage_after": 0.21,
    }]


def test_scheduler_power_smc_resample_respects_kv_cow_disabled() -> None:
    class FakeCacheManager:
        def __init__(self) -> None:
            self.freed: list[str] = []

        def get_blocks(self, request_id: str) -> str:
            return "blocks"

        def get_num_cached_blocks(self, request_id: str) -> list[int]:
            return [2]

        def alias_request_blocks_from_snapshot(self, **kwargs) -> None:
            raise AssertionError("kv_cow=False should not alias KV blocks")

        def free(self, request: Request) -> None:
            self.freed.append(request.request_id)

    class FakeEncoderCacheManager:
        def free(self, request: Request) -> None:
            pass

    params = SamplingParams(max_tokens=8, extra_args=power_smc_args(kv_cow=False))
    request = Request(
        request_id=make_power_smc_child_request_id("parent", 0),
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=params,
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING
    request.append_output_token_ids([10, 11, 12, 13, 14])
    request.num_computed_tokens = request.num_tokens

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.kv_cache_manager = FakeCacheManager()
    scheduler.encoder_cache_manager = FakeEncoderCacheManager()
    scheduler._inflight_prefills = {request}
    scheduler.power_smc_group_children = {"parent": {0: request.request_id}}
    scheduler.block_size = 4
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4)),
        ])

    manager = PowerSMCGroupManager(
        PowerSMCConfig(enabled=True, particles=1, kv_cow=False))
    manager.resample_count = 1
    manager.last_resample_plan = PowerSMCResamplePlan(
        ancestors=[0],
        particle_token_ids=[[20, 21, 22, 23, 24]],
    )

    resampled = Scheduler._apply_power_smc_resample_plan(
        scheduler,
        "parent",
        manager,
    )

    assert resampled == [request]
    assert scheduler.kv_cache_manager.freed == [request.request_id]
    assert request.num_computed_tokens == 0
    assert manager.diagnostics()["kv_resample_events"] == [{
        "resample_index": 1,
        "kv_cow_enabled": False,
        "kv_alias_supported": True,
        "kv_mode": "disabled_reset_recompute",
        "block_size": 4,
        "child_count": 1,
        "snapshot_count": 0,
        "alias_attempt_count": 0,
        "alias_success_count": 0,
        "fallback_count": 1,
        "aliased_blocks": 0,
        "aliased_tokens": 0,
        "cow_physical_blocks": 0,
        "cow_saved_blocks": 0,
        "cow_saved_tokens": 0,
        "replay_tokens": 9,
        "identity_noop_count": 0,
    }]
    assert manager.diagnostics()["kv_alias_fallback_count"] == 1


def test_power_smc_input_batch_sets_alpha_and_proposal_temperature() -> None:
    params = SamplingParams(
        max_tokens=8,
        extra_args=power_smc_args(alpha=4.0, alpha_ramp_tokens=4),
    )
    request = CachedRequestState(
        req_id="power-smc-req",
        prompt_token_ids=[1, 2],
        mm_features=[],
        sampling_params=params,
        generator=None,
        block_ids=([],),
        num_computed_tokens=2,
        output_token_ids=[],
    )
    input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=32,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )

    input_batch.add_request(request)
    input_batch.refresh_metadata()

    assert input_batch.sampling_metadata.power_smc_alpha is not None
    expected_alpha = alpha_ramp(0, alpha_final=4.0, ramp_tokens=4)
    torch.testing.assert_close(
        input_batch.sampling_metadata.power_smc_alpha,
        torch.tensor([expected_alpha]),
    )
    torch.testing.assert_close(
        input_batch.sampling_metadata.temperature,
        torch.tensor([proposal_temperature(expected_alpha)]),
    )


def test_power_smc_input_batch_uses_scheduler_config_without_extra_args() -> None:
    config = PowerSMCConfig(
        enabled=True,
        alpha=4.0,
        particles=8,
        block_size=64,
        ess_threshold=0.5,
        alpha_ramp_tokens=4,
    )
    params = SamplingParams(max_tokens=8)
    request = CachedRequestState(
        req_id="power-smc-req",
        prompt_token_ids=[1, 2],
        mm_features=[],
        sampling_params=params,
        generator=None,
        block_ids=([],),
        num_computed_tokens=2,
        output_token_ids=[],
        power_smc_config=config,
    )
    input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=32,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )

    input_batch.add_request(request)
    input_batch.refresh_metadata()

    assert input_batch.sampling_metadata.power_smc_alpha is not None
    expected_alpha = alpha_ramp(0, alpha_final=4.0, ramp_tokens=4)
    torch.testing.assert_close(
        input_batch.sampling_metadata.power_smc_alpha,
        torch.tensor([expected_alpha]),
    )
    torch.testing.assert_close(
        input_batch.sampling_metadata.temperature,
        torch.tensor([proposal_temperature(expected_alpha)]),
    )

    request.output_token_ids.extend([10, 11])
    input_batch.update_power_smc_sampling_state()

    expected_alpha = alpha_ramp(2, alpha_final=4.0, ramp_tokens=4)
    torch.testing.assert_close(
        input_batch.sampling_metadata.power_smc_alpha,
        torch.tensor([expected_alpha]),
    )
    torch.testing.assert_close(
        input_batch.sampling_metadata.temperature,
        torch.tensor([proposal_temperature(expected_alpha)]),
    )
