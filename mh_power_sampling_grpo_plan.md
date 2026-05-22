# MH Power Sampling as GRPO Rollout Generator — Experiment Plan

## 1. Goal

Evaluate whether **2510-style MH Power Sampling** can be used as a rollout generator for GRPO, especially whether **accepted + rejected MH proposals** should all be included in the GRPO candidate group.

The central hypothesis is:

\[
\text{MH proposal process}
\rightarrow
\text{more local counterfactual rollouts}
\rightarrow
\text{higher verifier reward contrast}
\rightarrow
\text{better GRPO updates}
\]

The key method under test is:

> Run MH power sampling, collect every complete rollout/proposal generated during the MH process, score all of them with the verifier, and use all scored candidates in the per-prompt GRPO group.

This should be treated as **proposal-pool GRPO**, not standard on-policy GRPO.

---

## 2. Background

2510-style MH Power Sampling targets a sequence-level power distribution:

\[
\pi_\alpha(y \mid x) \propto \pi_{\text{old}}(y \mid x)^\alpha, \quad \alpha > 1
\]

A Metropolis-Hastings step proposes a complete candidate response:

\[
y' \sim q(y' \mid y)
\]

and accepts it with probability:

\[
a(y \to y') = \min\left(1,
\frac{\pi_\alpha(y' \mid x) q(y \mid y')}
{\pi_\alpha(y \mid x) q(y' \mid y)}
\right)
\]

If the proposal kernel is treated approximately symmetric, use the simplified accept probability:

\[
a(y \to y') = \min\left(1,
\exp\left[\alpha\left(\log\pi_{\text{old}}(y' \mid x)-\log\pi_{\text{old}}(y \mid x)\right)\right]
\right)
\]

Important distinction:

- The **post-accept/reject chain state** approaches the power distribution after burn-in.
- The **raw rejected proposal** is not a power-distribution sample.
- However, a rejected proposal can still be valuable as a verifier-scored counterfactual candidate for GRPO.

---

## 3. Main Research Questions

1. Does MH-generated rollout data improve GRPO compared with normal policy sampling?
2. Do rejected proposals provide useful learning signal when scored by a verifier?
3. Is `all proposals` better than `accepted/current-chain only`?
4. Does the method improve sample efficiency under equal generated-token or wall-clock budget?
5. Does MH proposal generation increase group reward variance and reduce all-correct/all-wrong groups?

---

## 4. Method Variants

### A. Standard GRPO

Standard baseline:

\[
y_1, \ldots, y_G \sim \pi_{\text{old}}(\cdot \mid x)
\]

Suggested config:

```yaml
method: standard_grpo
num_rollouts_per_prompt: 8
temperature: 1.0
top_p: 1.0
```

---

### B. Low-Temperature GRPO

Control for simple likelihood sharpening:

\[
y_i \sim \pi_{\text{old}, \tau}(\cdot \mid x)
\]

Suggested config:

```yaml
method: low_temp_grpo
num_rollouts_per_prompt: 8
temperature: 0.5  # or 0.7
top_p: 1.0
```

Purpose: determine whether any improvement comes merely from sharper sampling rather than MH proposal structure.

---

### C. MH Final-Only GRPO

Run MH for `K` steps and use only the final chain state.

```text
initial rollout -> MH K steps -> final state only
```

This tests MH as a stronger sampler without using intermediate candidates.

---

### D. MH Chain-States GRPO

Use the MH chain state after every accept/reject step:

\[
Y_1, Y_2, \ldots, Y_K
\]

If a proposal is rejected, the previous state is repeated.

```text
include post-accept/reject chain states
exclude raw rejected proposals
```

This is the closest variant to valid MH sampling semantics.

---

### E. MH All-Proposals GRPO

Core proposed method.

Include:

```text
initial rollout
accepted proposals
rejected proposals
optionally post-accept/reject chain states
```

All complete responses are scored by the verifier and placed into the GRPO group.

This should be described as:

```text
MH proposal-pool GRPO
```

not as pure power-distribution GRPO.

---

### F. MH All-Proposals + Exact Dedup

Same as E, but exact duplicate token sequences are collapsed.

Purpose: test whether duplicate chain states/proposals are useful implicit weighting or just wasteful repetition.

---

## 5. Recommended Main Comparison

| Variant | Rollout Source | Includes Rejected? | Main Purpose |
|---|---|---:|---|
| A | Normal policy samples | No | Standard baseline |
| B | Low-temperature samples | No | Control for sharper sampling |
| C | Final MH chain states | No | Test MH sampler only |
| D | Post-accept chain states | No | Test valid MH chain samples |
| E | All raw proposals | Yes | Test rejected-proposal value |
| F | All raw proposals + dedup | Yes | Test duplicate effects |

---

## 6. Proposal Kernel Design

### Preferred First Version: Entropy-Branch Local Proposal

For each prompt:

1. Generate an initial root rollout.
2. Compute token entropy over the root rollout.
3. Pick a high-entropy branch point `b`.
4. Propose a new complete response by resampling from that point.

Two possible implementations:

#### Option 1: Suffix Resampling

\[
y' = y_{<b} + \tilde y_{\ge b}
\]

where:

\[
\tilde y_{\ge b} \sim \pi_{\text{old}}(\cdot \mid x, y_{<b})
\]

Pros: simple.
Cons: expensive because each proposal may be a long full suffix.

#### Option 2: Short-Span Resampling + Normal Continuation

1. Resample a short span of length `L` after branch point:

\[
z \sim \pi_{\text{old}}(\cdot \mid x, y_{<b})
\]

2. Continue normally to EOS:

\[
y' = y_{<b} + z + \text{continue}(x, y_{<b}, z)
\]

Recommended first choice:

```yaml
proposal: entropy_branch_span_then_continue
span_len: 16  # try 16 or 32 first
```

---

## 7. MH-All-Proposals Rollout Algorithm

```python
for prompt in batch:
    candidate_pool = []

    # 1. Initial rollout
    y = sample(policy_old, prompt)
    candidate_pool.append({
        "response": y,
        "source": "initial",
        "mh_step": 0,
        "accepted": True,
    })

    # 2. Optional entropy branch-point selection
    entropies = compute_token_entropy(policy_old, prompt, y)
    branch_point = select_top_entropy_position(entropies)

    # 3. MH proposal loop
    for k in range(1, K + 1):
        y_prop, proposal_info = propose(
            policy=policy_old,
            prompt=prompt,
            current_response=y,
            branch_point=branch_point,
            span_len=L,
        )

        logp_y = sequence_logprob(policy_old, prompt, y)
        logp_prop = sequence_logprob(policy_old, prompt, y_prop)

        accept_logprob = min(0.0, alpha * (logp_prop - logp_y))
        accept = log_uniform() < accept_logprob

        # Store raw proposal no matter whether accepted or rejected
        candidate_pool.append({
            "response": y_prop,
            "source": "proposal",
            "mh_step": k,
            "accepted": accept,
            "accept_logprob": accept_logprob,
            "proposal_info": proposal_info,
        })

        # Update chain state
        if accept:
            y = y_prop

        # Optional: store post-accept/reject chain state
        candidate_pool.append({
            "response": y,
            "source": "chain_state",
            "mh_step": k,
            "accepted": True,
        })

    # 4. Optional exact dedup
    # candidate_pool = exact_dedup(candidate_pool)

    # 5. Verifier scoring
    rewards = verifier(prompt, [c["response"] for c in candidate_pool])

    # 6. GRPO group advantage over all candidates for this prompt
    advantages = normalize_per_prompt(rewards)

    # 7. Recompute old-policy logprobs for every final candidate
    old_logprobs = recompute_logprobs(policy_old, prompt, candidate_pool)

    # 8. Train with prompt-level normalized GRPO loss
    loss += grpo_loss(candidate_pool, advantages, old_logprobs)
```

---

## 8. GRPO Objective

For each prompt \(x\), candidate pool:

\[
\mathcal C(x)=\{y_1, \ldots, y_M\}
\]

Verifier rewards:

\[
r_i = R(x, y_i)
\]

Per-prompt group advantage:

\[
A_i = \frac{r_i - \mu_x}{\sigma_x + \epsilon}
\]

where:

\[
\mu_x = \frac{1}{M}\sum_{i=1}^{M} r_i
\]

\[
\sigma_x^2 = \frac{1}{M}\sum_{i=1}^{M}(r_i - \mu_x)^2
\]

PPO-style clipped GRPO loss:

\[
\rho_{i,t}=\exp\left(
\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
-
\log\pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})
\right)
\]

\[
\mathcal L_x =
-\frac{1}{M}\sum_{i=1}^{M}\sum_t m_{i,t}
\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
\]

Prompt-level batch loss:

\[
\mathcal L = \frac{1}{B}\sum_{x\in\text{batch}} \mathcal L_x
\]

Important: use prompt-level normalization so prompts with more MH proposals do not dominate the batch.

---

## 9. Loss Masking Ablation

Run two loss-mask variants.

### Full-Response Loss

Train on all generated response tokens.

```yaml
loss_mask: full_response
```

### Branch-After-Only Loss

Train only on tokens after the branch point:

\[
m_{i,t}=1[t \ge b]
\]

```yaml
loss_mask: branch_after_only
```

Expected: branch-after-only may improve credit assignment when proposals are local edits.

---

## 10. Fair Budgeting

Use two reporting modes.

### 10.1 Equal Prompt Budget

Each method sees the same number of prompts.

Pros: simple.
Cons: MH methods may use more generation compute.

### 10.2 Equal Generated-Token Budget

Compare methods under approximately equal total generated tokens:

\[
\text{generated tokens per prompt}
\]

For normal GRPO:

\[
G \times \text{avg response length}
\]

For MH-all-proposals:

\[
1 \times \text{initial response length} + K \times \text{proposal response length}
\]

Report all results against:

```text
optimizer steps
wall-clock time
generated tokens
verifier calls
```

---

## 11. Hyperparameters

### First-pass Defaults

```yaml
model: Qwen2.5-Math-7B
train_prompts: 500-1000
val: MATH500
reward: exact_match
alpha: 1.5
mh_steps: 4
proposal: entropy_branch_span_then_continue
span_len: 16 or 32
max_response_length: 2048 or 4096
group_advantage: per_prompt
loss_mask: branch_after_only
```

### Sweep After Smoke Test

```yaml
alpha: [1.2, 1.5, 2.0]
mh_steps: [4, 8, 16]
span_len: [16, 32, 64]
branch_selection:
  - entropy_top1
  - entropy_top4_random_one
  - random_position
include_variants:
  - chain_only
  - accepted_only
  - all_proposals
  - all_proposals_dedup
loss_mask:
  - full_response
  - branch_after_only
```

Avoid full Cartesian product initially. First identify whether `all_proposals` helps.

---

## 12. Minimal Single-GPU Experiment

Designed for one RTX PRO 6000.

```yaml
experiment: mh_power_grpo_minimal
model: Qwen2.5-Math-7B
train_prompts: 500
val: MATH500_subset_then_full
max_response_length: 2048
methods:
  - standard_grpo_g8
  - mh_chain_only_k4
  - mh_all_proposals_k4
  - mh_all_proposals_k4_exact_dedup
alpha: 1.5
mh_steps: 4
span_len: 16
proposal: entropy_branch_span_then_continue
reward: exact_match
loss_mask:
  - branch_after_only
normalize_loss: per_prompt
```

Primary diagnostic goal:

```text
Does mh_all_proposals increase group_reward_std and validation accuracy relative to standard_grpo and mh_chain_only?
```

---

## 13. Diagnostics to Log

### 13.1 MH Sampler Diagnostics

```text
acceptance_rate
mean_accept_logprob
mh_step_reward_mean
mh_step_reward_std
reward_by_step
logprob_by_step
length_by_step
unique_response_count
duplicate_rate
```

Important interpretation:

- If logprob increases but reward does not, MH may be amplifying confident wrong trajectories.
- If reward increases by MH step, power sampling is likely helping.

---

### 13.2 Accepted vs Rejected Diagnostics

```text
accepted_reward_mean
rejected_reward_mean
accepted_correct_rate
rejected_correct_rate
accepted_length_mean
rejected_length_mean
accepted_old_logprob_mean
rejected_old_logprob_mean
```

Important metric:

```text
rejected_correct_rate
```

If rejected proposals often contain correct answers, including them is likely valuable.

---

### 13.3 GRPO Group Quality

```text
group_size_M
group_reward_mean
group_reward_std
all_correct_rate
all_wrong_rate
num_correct_per_prompt
num_unique_answers
num_unique_traces
```

Most important:

```text
group_reward_std
```

GRPO needs reward contrast. If all proposals are all correct or all wrong, the learning signal is weak.

---

### 13.4 Training Stability

```text
policy_loss
clip_fraction
approx_kl
entropy
response_length
grad_norm
ratio_mean
ratio_max
```

Warning signs:

```text
clip_fraction too high
approx_kl spikes
ratio_max exploding
response_length drifting upward
```

---

## 14. Evaluation Metrics

Primary:

```text
MATH500 accuracy
AIME accuracy if available
pass@1
pass@k
```

Efficiency-normalized:

```text
accuracy vs optimizer step
accuracy vs generated tokens
accuracy vs wall-clock time
accuracy vs verifier calls
```

Training-signal:

```text
group_reward_std
all_correct_rate
all_wrong_rate
rejected_correct_rate
unique_response_count
```

---

## 15. Expected Outcomes

### Outcome A: MH-All-Proposals Wins

Interpretation:

```text
Rejected proposals provide useful verifier-scored counterfactuals.
MH proposal pool improves GRPO reward contrast.
```

Claim:

> Rejected MH proposals are not power-distribution samples, but they are valuable complete counterfactual trajectories for verifier-supervised GRPO.

---

### Outcome B: MH-Chain-Only Wins

Interpretation:

```text
MH correction matters.
Raw proposals are too noisy or distort group baseline.
```

Next step: use rejected proposals only for auxiliary ranking or hard negatives.

---

### Outcome C: Low-Temperature GRPO Matches MH

Interpretation:

```text
Most benefit comes from sharper sampling, not MH proposal dynamics.
```

Next step: strengthen branch-local proposal analysis or use harder tasks.

---

### Outcome D: Standard GRPO Wins

Interpretation:

```text
MH proposals reduce exploration or over-focus on confident wrong basins.
```

Check:

```text
old_logprob vs reward correlation
rejected_correct_rate
group_reward_std
unique_answer_count
```

---

## 16. Suggested Result Tables

### Main Accuracy Table

| Method | Val Acc | Generated Tokens | Wall Time | Group Reward Std | Unique Responses |
|---|---:|---:|---:|---:|---:|
| Standard GRPO | | | | | |
| Low-temp GRPO | | | | | |
| MH Final Only | | | | | |
| MH Chain States | | | | | |
| MH All Proposals | | | | | |
| MH All Proposals Dedup | | | | | |

### Accepted vs Rejected Table

| Source | Count | Correct Rate | Avg Reward | Avg Old Logprob | Avg Length |
|---|---:|---:|---:|---:|---:|
| Initial | | | | | |
| Accepted Proposal | | | | | |
| Rejected Proposal | | | | | |
| Chain State | | | | | |

### Ablation Table

| Variant | Alpha | MH Steps | Span Len | Loss Mask | Val Acc | Group Reward Std |
|---|---:|---:|---:|---|---:|---:|
| | | | | | | |

---

## 17. Recommended Implementation Notes

1. Recompute `old_logprobs` under `pi_old` for every final candidate response.
2. Do not use proposal sampling logprob as PPO denominator.
3. Keep `accepted/rejected/source/mh_step/accept_logprob` as metadata only.
4. Do not use accepted/rejected as reward labels.
5. Use verifier reward for all candidates.
6. Use prompt-level loss normalization.
7. Start with short max response length before running long-CoT settings.
8. Log exact duplicate rate before deciding whether to dedup.
9. Compare both raw-all and exact-dedup variants.
10. Use branch-after-only loss for branch-local proposals as a key ablation.

---

## 18. Suggested Claim If Successful

Use careful wording:

> We use MH not only as a sampler, but as a generator of complete counterfactual rollout candidates. Although rejected proposals are not samples from the MH stationary distribution, they are complete trajectories produced by the proposal kernel. Once scored by a verifier, they provide useful relative reward contrast for GRPO.

Avoid claiming:

> Rejected proposals are power-distribution samples.

---

## 19. One-Day RTX PRO 6000 Run Plan

Approximate feasible single-GPU run:

```yaml
model: Qwen2.5-Math-7B
train_prompts: 500
max_response_length: 2048
mh_steps: 4
span_len: 16
alpha: 1.5
methods:
  - standard_grpo_g8
  - mh_chain_only_k4
  - mh_all_proposals_k4
validation: MATH500 subset during training, full MATH500 at end
```

Expected output after this run:

```text
Does all-proposals improve reward contrast?
Does rejected_correct_rate justify keeping rejected proposals?
Does validation improve per wall-clock or generated token?
Is training stable under proposal-pool GRPO?
```

---

## 20. Next Steps After Minimal Run

If `MH All Proposals` improves reward contrast and validation:

1. Add exact-dedup ablation.
2. Sweep alpha: `[1.2, 1.5, 2.0]`.
3. Sweep MH steps: `[4, 8, 16]`.
4. Compare full-response vs branch-after-only loss.
5. Scale train prompts from 500 to 1k, then 5k.
6. Run equal generated-token comparison against standard GRPO.

If it does not improve:

1. Check whether proposals are too duplicate-heavy.
2. Check whether rejected proposals are almost all wrong and identical.
3. Try lower alpha.
4. Try random branch vs entropy branch.
5. Try shorter span length.
6. Try using rejected proposals only for auxiliary pairwise loss.
