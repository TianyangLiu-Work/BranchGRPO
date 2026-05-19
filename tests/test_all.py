#!/usr/bin/env python3
"""Comprehensive smoke test for BranchGRPO codebase.
Runs on MacBook with MPS, using gpt2 (124M params) as a lightweight substitute."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")
    return condition

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ============================================================
# Test 1: Config loading
# ============================================================
section("1. Config loading")

from src.config import ExperimentConfig, MethodConfig, MHConfig, TrainingConfig

check("ExperimentConfig import", True)
check("MethodConfig import", True)
check("MHConfig import", True)

cfg = ExperimentConfig()
check("Default config created", cfg is not None)
check("Default method", cfg.method.method == "mh_all_proposals")
check("Default model", cfg.training.model_name == "Qwen/Qwen2.5-Math-7B")
check("Default lora", cfg.training.use_lora == True)

cfg_mh = ExperimentConfig(method=MethodConfig(method="mh_all_proposals", mh=MHConfig()))
check("MH default alpha", cfg_mh.method.mh.alpha == 1.5)
check("MH default mh_steps", cfg_mh.method.mh.mh_steps == 4)
check("MH default span_len", cfg_mh.method.mh.span_len == 16)
check("Default model", cfg.training.model_name == "Qwen/Qwen2.5-Math-7B")
check("Default lora", cfg.training.use_lora == True)
check("Method name", "mh_all_proposals" in cfg_mh.get_method_name())

cfg2 = ExperimentConfig(method=MethodConfig(method="standard_grpo", num_rollouts_per_prompt=8))
check("Standard GRPO method name", "standard_grpo" in cfg2.get_method_name())

check("YAML roundtrip check skipped (needs file)", True)

# ============================================================
# Test 2: Verifier
# ============================================================
section("2. Verifier (exact_match_reward)")

from src.verifier import extract_boxed_answer, normalize_answer, exact_match_reward, compute_rewards

check("extract_boxed_answer basic",
      extract_boxed_answer(r"The answer is \boxed{42}") == "42")
check("extract_boxed_answer nested",
      extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}")
check("extract_boxed_answer last",
      extract_boxed_answer(r"\boxed{3} and \boxed{5}") == "5")
check("extract_boxed_answer no box",
      extract_boxed_answer("answer is 42") == "answer is 42")

check("normalize_answer",
      normalize_answer("  Hello, World!  ") == "helloworld!")
check("normalize_answer math",
      normalize_answer(r"\frac{1}{2}") == r"\frac{1}{2}")

check("exact_match correct",
      exact_match_reward(r"The answer is \boxed{x=3}", r"\boxed{x = 3}") == 1.0)
check("exact_match wrong",
      exact_match_reward(r"\boxed{4}", r"\boxed{5}") == 0.0)

pool = [
    {"response_text": r"\boxed{42}"},
    {"response_text": r"\boxed{7}"},
    {"response_text": r"\boxed{42}"},
]
rewards = compute_rewards(pool, r"\boxed{42}")
check("compute_rewards", rewards == [1.0, 0.0, 1.0])

# ============================================================
# Test 3: Advantage computation
# ============================================================
section("3. Advantage computation")

from src.grpo import compute_advantages

adv = compute_advantages([1.0, 0.0, 1.0, 0.0])
check("Advantage shape", len(adv) == 4)

import torch as t
rewards_t = t.tensor([1.0, 0.0, 1.0, 0.0], dtype=t.float32)
mu = rewards_t.mean()
sigma = rewards_t.std() + 1e-8
expected = ((rewards_t - mu) / sigma).tolist()
for i, (a, e) in enumerate(zip(adv, expected)):
    check(f"Advantage[{i}] ~= {e:.4f}", abs(a - e) < 1e-5)

# ============================================================
# Test 4: GRPO loss with synthetic data (requires tokenizer)
# ============================================================
section("4. GRPO loss computation (synthetic)")

from src.grpo import compute_grpo_loss

# Skip full GRPO loss test without tokenizer - tested with gpt2 in test 6
print("  ⏭️  Skipped (needs tokenizer, tested in section 6 with gpt2)")

# ============================================================
# Test 5: Data loading (try MATH dataset if network available)
# ============================================================
section("5. Data loading")

try:
    from src.data import load_train_data
    from src.config import DataConfig
    cfg_data = ExperimentConfig()
    cfg_data.data = DataConfig()
    data = load_train_data(cfg_data, num_prompts=10)
    check("MATH train loaded", len(data) == 10)
    check("Has prompt key", "prompt" in data[0])
    check("Has problem key", "problem" in data[0])
    check("Prompt non-empty", len(data[0]["prompt"]) > 10)
except Exception as e:
    err_msg = str(e)[:80]
    if "Couldn't find" in err_msg or "LocalEntryNotFound" in err_msg or "Connection" in err_msg:
        print(f"  ⏭️  MATH data loading skipped (HF Hub not reachable)")
    else:
        check("MATH data loading", False, err_msg)

try:
    from src.data import load_val_data
    val_data = load_val_data(cfg_data, num_samples=10)
    check("MATH val loaded", len(val_data) == 10)
except Exception as e:
    err_msg = str(e)[:80]
    if "Couldn't find" in err_msg or "LocalEntryNotFound" in err_msg or "Connection" in err_msg:
        print(f"  ⏭️  MATH val loading skipped (HF Hub not reachable)")
    else:
        check("MATH val loading", False, err_msg)

# ============================================================
# Test 6: Full pipeline with gpt2
# ============================================================
section("6. Full pipeline with gpt2 (mini smoke test)")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import ExperimentConfig, MethodConfig, MHConfig

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    check("GPT-2 loaded", True)

    # Build a minimal config
    mh_cfg = MHConfig(alpha=1.5, mh_steps=2, span_len=8)
    method_cfg = MethodConfig(
        method="mh_all_proposals",
        num_rollouts_per_prompt=1,
        temperature=1.0,
        mh=mh_cfg,
        include_rejected=True,
    )
    cfg = ExperimentConfig(method=method_cfg)
    cfg.training.max_response_length = 64
    cfg.training.model_name = "gpt2"
    cfg.loss_mask = "full_response"

    from src.mh_sampling import mh_power_sampling
    prompt = "Question: What is 2+2?\nAnswer:"

    with torch.no_grad():
        pool = mh_power_sampling(model, tokenizer, prompt, cfg)
    check(f"Generated {len(pool)} candidates", len(pool) > 0)

    for i, c in enumerate(pool):
        src = c.get("source", "?")
        acc = c.get("accepted", "?")
        has_old_lp = "old_token_logprobs" in c
        n_tokens = len(c["response_ids"])
        check(f"Candidate[{i}] source={src}, accepted={acc}, tokens={n_tokens}, old_lp={has_old_lp}",
              n_tokens > 0 and has_old_lp)

    # Test standard_grpo
    method_cfg2 = MethodConfig(method="standard_grpo", num_rollouts_per_prompt=3, temperature=1.0)
    cfg2 = ExperimentConfig(method=method_cfg2)
    cfg2.training.max_response_length = 64
    with torch.no_grad():
        pool2 = mh_power_sampling(model, tokenizer, prompt, cfg2)
    check(f"Standard GRPO: {len(pool2)} candidates", len(pool2) == 3)

    # Test low_temp_grpo
    method_cfg3 = MethodConfig(method="low_temp_grpo", num_rollouts_per_prompt=2, temperature=0.5)
    cfg3 = ExperimentConfig(method=method_cfg3)
    cfg3.training.max_response_length = 64
    with torch.no_grad():
        pool3 = mh_power_sampling(model, tokenizer, prompt, cfg3)
    check(f"Low-temp GRPO: {len(pool3)} candidates", len(pool3) == 2)

    # Test mh_final_only
    mh_cfg4 = MHConfig(alpha=1.5, mh_steps=2, span_len=4)
    method_cfg4 = MethodConfig(method="mh_final_only", num_rollouts_per_prompt=2, mh=mh_cfg4)
    cfg4 = ExperimentConfig(method=method_cfg4)
    cfg4.training.max_response_length = 64
    with torch.no_grad():
        pool4 = mh_power_sampling(model, tokenizer, prompt, cfg4)
    check(f"MH Final Only: {len(pool4)} candidates", len(pool4) == 2)

    # Test mh_chain_only
    mh_cfg5 = MHConfig(alpha=1.5, mh_steps=2, span_len=4)
    method_cfg5 = MethodConfig(method="mh_chain_only", num_rollouts_per_prompt=1, mh=mh_cfg5,
                                include_chain_states=True, include_rejected=False)
    cfg5 = ExperimentConfig(method=method_cfg5)
    cfg5.training.max_response_length = 64
    with torch.no_grad():
        pool5 = mh_power_sampling(model, tokenizer, prompt, cfg5)
    check(f"MH Chain Only: {len(pool5)} candidates", len(pool5) > 0)
    all_accepted = all(c.get("accepted", False) for c in pool5)
    check("All chain-only candidates accepted", all_accepted)

    # Test mh_all_proposals_dedup
    mh_cfg6 = MHConfig(alpha=1.5, mh_steps=2, span_len=4)
    method_cfg6 = MethodConfig(method="mh_all_proposals_dedup", num_rollouts_per_prompt=1, mh=mh_cfg6,
                                exact_dedup=True, include_rejected=True)
    cfg6 = ExperimentConfig(method=method_cfg6)
    cfg6.training.max_response_length = 64
    with torch.no_grad():
        pool6 = mh_power_sampling(model, tokenizer, prompt, cfg6)
    check(f"MH All Proposals Dedup: {len(pool6)} candidates", len(pool6) > 0)

    # Test GRPO loss with actual gpt2 model
    from src.verifier import compute_rewards
    rewards6 = [1.0 if i == 0 else 0.0 for i in range(len(pool6))]
    model.train()
    try:
        loss, metrics = compute_grpo_loss(
            model, tokenizer, [prompt], [pool6], [rewards6],
            clip_epsilon=0.2, loss_mask="full_response"
        )
        check("GRPO loss with gpt2 works", True, f"loss={loss.item():.4f}")
        check("Loss is finite", torch.isfinite(loss))
        check("KL approx", isinstance(metrics["approx_kl"], (int, float)))
    except Exception as e:
        check("GRPO loss with gpt2", False, str(e))

    # Test branch-after-only loss mask
    test_response_ids = torch.tensor([101, 102, 103], device=device)
    branch_pool = [{
        "response_ids": test_response_ids,
        "response_text": "test",
        "source": "initial",
        "branch_point": 1,
        "old_token_logprobs": [-0.5, -0.3, -0.7],
    }]
    try:
        loss2, _ = compute_grpo_loss(
            model, tokenizer, ["prompt"], [branch_pool], [[1.0]],
            clip_epsilon=0.2, loss_mask="branch_after_only"
        )
        check("GRPO loss branch_after_only works", True)
    except Exception as e:
        check("GRPO loss branch_after_only", False, str(e))

    model.to("cpu")
    del model, tokenizer

except ImportError as e:
    check("GPT-2 test skipped", False, str(e))
except Exception as e:
    check("GPT-2 test", False, f"{type(e).__name__}: {str(e)[:120]}")

# ============================================================
# Test 7: Proposal kernel logic (unit tests)
# ============================================================
section("7. Proposal kernel (unit tests)")

from src.proposal_kernel import exact_dedup

pool_data = [
    {"response_ids": torch.tensor([1, 2, 3]), "response_text": "a", "source": "x"},
    {"response_ids": torch.tensor([1, 2, 3]), "response_text": "a", "source": "y"},
    {"response_ids": torch.tensor([4, 5, 6]), "response_text": "b", "source": "z"},
]
deduped = exact_dedup(pool_data)
check("exact_dedup removes duplicates", len(deduped) == 2)
check("exact_dedup keeps first occurrence", deduped[0]["source"] == "x")

pool_unique = [
    {"response_ids": torch.tensor([1, 2]), "response_text": "a"},
    {"response_ids": torch.tensor([3, 4]), "response_text": "b"},
]
check("exact_dedup no change on unique", len(exact_dedup(pool_unique)) == 2)

# ============================================================
# Test 8: Diagnostics output
# ============================================================
section("8. Diagnostics")

from src.diagnostics import log_mh_diagnostics, log_group_quality

pool_diag = [
    [
        {"source": "initial", "reward": 1.0},
        {"source": "proposal", "reward": 0.0, "accepted": True, "accept_logprob": -0.3},
        {"source": "proposal", "reward": 1.0, "accepted": False, "accept_logprob": -2.0},
    ]
]
try:
    mh_metrics = log_mh_diagnostics(pool_diag, step=0)
    check("MH diagnostics runs", len(mh_metrics) > 0)
    check("Acceptance rate", abs(mh_metrics.get("mh/acceptance_rate", 0) - 0.5) < 0.01)
except Exception as e:
    check("MH diagnostics", False, str(e))

try:
    group_metrics = log_group_quality(pool_diag, step=0)
    check("Group quality runs", len(group_metrics) > 0)
except Exception as e:
    check("Group quality", False, str(e))

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*60}")
if FAIL > 0:
    print("❌ SOME TESTS FAILED")
    sys.exit(1)
else:
    print("✅ ALL TESTS PASSED")
    sys.exit(0)
