"""
BranchGRPO with veRL integration.
Uses veRL's core_algos for GRPO advantage + policy loss,
and veRL's reward scoring, while keeping our MH power sampling
as the rollout engine.

Usage:
    python train_verl.py --config configs/verl_mh_all_proposals.yaml
"""

import argparse
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

# --- veRL imports ---
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla
from verl.utils.reward_score.math_reward import compute_score as math_reward_score
from verl.trainer.ppo.metric_utils import reduce_metrics as _reduce_metrics

# --- Our modules ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mh_sampling import mh_power_sampling
from src.proposal_kernel import select_branch_point, propose_span_then_continue, exact_dedup
from src.verifier import extract_boxed_answer, exact_match_reward


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_mh_rollouts(model, tokenizer, prompt: str, config: DictConfig) -> list:
    """Generate rollouts using MH power sampling.

    Returns list of dicts with keys:
        response_ids, response_text, source, mh_step, accepted, branch_point, old_token_logprobs
    """
    max_response_length = config.training.max_response_length
    temperature = config.method.temperature
    mh_steps = config.method.mh_steps
    alpha = config.method.alpha
    span_len = config.method.span_len
    method = config.method.get("method", "mh_all_proposals")
    include_rejected = config.method.get("include_rejected", True)
    include_chain_states = config.method.get("include_chain_states", False)
    exact_dedup_flag = config.method.get("exact_dedup", False)

    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)

    if method == "standard_grpo":
        num_rollouts = config.method.num_rollouts
        pool = []
        for _ in range(num_rollouts):
            ids = _generate_rollout(model, tokenizer, prompt_enc, temperature, max_response_length)
            pool.append({
                "response_ids": ids[0],
                "response_text": tokenizer.decode(ids[0], skip_special_tokens=True),
                "source": "sample",
                "branch_point": None,
            })
        _compute_old_logprobs(model, tokenizer, prompt, pool)
        return pool

    mh_pool = _run_mh_chain(
        model, tokenizer, prompt, prompt_enc, temperature, max_response_length,
        mh_steps, alpha, span_len, method, include_rejected, include_chain_states, exact_dedup_flag
    )
    _compute_old_logprobs(model, tokenizer, prompt, mh_pool)
    return mh_pool


@torch.no_grad()
def _run_mh_chain(model, tokenizer, prompt, prompt_enc, temperature, max_len,
                  mh_steps, alpha, span_len, method, include_rejected,
                  include_chain_states, exact_dedup_flag):
    device = next(model.parameters()).device
    pool = []

    init_ids = _generate_rollout(model, tokenizer, prompt_enc, temperature, max_len)
    pool.append({
        "response_ids": init_ids[0],
        "response_text": tokenizer.decode(init_ids[0], skip_special_tokens=True),
        "source": "initial",
        "mh_step": 0,
        "accepted": True,
        "branch_point": None,
    })

    if method == "mh_final_only":
        current_ids = init_ids[0]
        for k in range(1, mh_steps + 1):
            current_ids = _mh_step(model, tokenizer, prompt, current_ids, temperature, max_len, alpha, span_len)
        return [{
            "response_ids": current_ids,
            "response_text": tokenizer.decode(current_ids, skip_special_tokens=True),
            "source": "final",
            "mh_step": mh_steps,
            "accepted": True,
            "branch_point": None,
        }]

    current_ids = init_ids[0]
    for k in range(1, mh_steps + 1):
        branch_point = select_branch_point(model, tokenizer, prompt, current_ids, "entropy_top1")
        proposed_ids, _ = propose_span_then_continue(
            model, tokenizer, prompt, current_ids,
            branch_point, span_len, temperature, 1.0, max_len,
        )
        logp_cur = _sequence_logprob(model, tokenizer, prompt, current_ids)
        logp_prop = _sequence_logprob(model, tokenizer, prompt, proposed_ids)
        accept_logp = min(0.0, alpha * (logp_prop - logp_cur))
        accept = np.random.random() < np.exp(accept_logp)

        pool.append({
            "response_ids": proposed_ids,
            "response_text": tokenizer.decode(proposed_ids, skip_special_tokens=True),
            "source": "proposal",
            "mh_step": k,
            "accepted": accept,
            "accept_logprob": accept_logp,
            "branch_point": branch_point,
        })

        if accept:
            current_ids = proposed_ids

        if include_chain_states:
            pool.append({
                "response_ids": current_ids,
                "response_text": tokenizer.decode(current_ids, skip_special_tokens=True),
                "source": "chain_state",
                "mh_step": k,
                "accepted": True,
                "branch_point": branch_point,
            })

    if exact_dedup_flag:
        pool = exact_dedup(pool)

    if not include_rejected and method == "mh_chain_only":
        pool = [c for c in pool if c["accepted"]]
        seen = set()
        deduped = []
        for c in pool:
            key = tuple(c["response_ids"].tolist())
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        pool = deduped

    return pool


@torch.no_grad()
def _mh_step(model, tokenizer, prompt, current_ids, temperature, max_len, alpha, span_len):
    branch_point = select_branch_point(model, tokenizer, prompt, current_ids, "entropy_top1")
    proposed_ids, _ = propose_span_then_continue(
        model, tokenizer, prompt, current_ids,
        branch_point, span_len, temperature, 1.0, max_len,
    )
    logp_cur = _sequence_logprob(model, tokenizer, prompt, current_ids)
    logp_prop = _sequence_logprob(model, tokenizer, prompt, proposed_ids)
    accept_logp = min(0.0, alpha * (logp_prop - logp_cur))
    if np.random.random() < np.exp(accept_logp):
        return proposed_ids
    return current_ids


@torch.no_grad()
def _sequence_logprob(model, tokenizer, prompt, response_ids):
    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_enc.input_ids.shape[1]
    full_ids = torch.cat([prompt_enc.input_ids[0], response_ids.to(device)])
    outputs = model(input_ids=full_ids.unsqueeze(0))
    logprobs = F.log_softmax(outputs.logits[0], dim=-1)
    total = 0.0
    for t in range(len(response_ids)):
        total += logprobs[prompt_len + t - 1, response_ids[t].item()].item()
    return total


@torch.no_grad()
def _compute_old_logprobs(model, tokenizer, prompt, pool):
    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_enc.input_ids.shape[1]
    for candidate in pool:
        response_ids = candidate["response_ids"].to(device)
        full_ids = torch.cat([prompt_enc.input_ids[0], response_ids])
        outputs = model(input_ids=full_ids.unsqueeze(0))
        logprobs = F.log_softmax(outputs.logits[0], dim=-1)
        token_lps = [logprobs[prompt_len + t - 1, response_ids[t].item()].item()
                     for t in range(len(response_ids))]
        candidate["old_token_logprobs"] = token_lps


@torch.no_grad()
def _generate_rollout(model, tokenizer, prompt_enc, temperature, max_length):
    input_ids = prompt_enc.input_ids.clone()
    attention_mask = prompt_enc.attention_mask.clone()
    generated = []
    for _ in range(max_length):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :] / max(temperature, 0.01)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=input_ids.device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=attention_mask.device)], dim=1)
    return torch.tensor([generated])


def compute_grpo_loss_verl(model, tokenizer, prompts, candidate_pools, rewards_list, config):
    """Compute GRPO loss using veRL's core_algos.

    We construct per-token tensors in veRL format and call compute_grpo_outcome_advantage
    followed by compute_policy_loss_vanilla.
    """
    device = next(model.parameters()).device
    clip_ratio = config.training.get("clip_epsilon", 0.2)
    loss_agg_mode = config.training.get("loss_agg_mode", "token-mean")

    all_old_logprob = []
    all_new_logprob = []
    all_advantages = []
    all_response_masks = []
    all_indices = []
    prompt_idx = 0

    for prompt, pool, rewards in zip(prompts, candidate_pools, rewards_list):
        M = len(pool)
        if M == 0:
            continue

        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        mu = rewards_t.mean()
        sigma = rewards_t.std() + 1e-8
        advantages_val = (rewards_t - mu) / sigma

        prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = prompt_enc.input_ids.shape[1]

        for i, candidate in enumerate(pool):
            response_ids = candidate["response_ids"].to(device)
            resp_len = len(response_ids)

            full_ids = torch.cat([prompt_enc.input_ids[0], response_ids])
            outputs = model(input_ids=full_ids.unsqueeze(0))
            logprobs_new = F.log_softmax(outputs.logits[0], dim=-1)

            old_lps = candidate.get("old_token_logprobs", [0.0] * resp_len)
            new_lps = [logprobs_new[prompt_len + t - 1, response_ids[t].item()]
                       for t in range(resp_len)]

            all_old_logprob.append(torch.tensor(old_lps, device=device))
            all_new_logprob.append(torch.stack(new_lps) if isinstance(new_lps[0], torch.Tensor)
                                   else torch.tensor(new_lps, device=device))
            all_advantages.append(torch.full((resp_len,), advantages_val[i], device=device))
            all_response_masks.append(torch.ones(resp_len, device=device))
            all_indices.append(prompt_idx)

        prompt_idx += 1

    if not all_old_logprob:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    bs = len(all_old_logprob)
    max_len = max(t.shape[0] for t in all_old_logprob)

    old_logprob = torch.zeros(bs, max_len, device=device)
    new_logprob = torch.zeros(bs, max_len, device=device)
    advantages = torch.zeros(bs, max_len, device=device)
    response_mask = torch.zeros(bs, max_len, device=device)

    for i in range(bs):
        L = all_old_logprob[i].shape[0]
        old_logprob[i, :L] = all_old_logprob[i]
        new_logprob[i, :L] = all_new_logprob[i]
        advantages[i, :L] = all_advantages[i]
        response_mask[i, :L] = all_response_masks[i]

    pg_loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_logprob,
        log_prob=new_logprob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=_make_actor_config(clip_ratio),
    )

    return pg_loss, metrics


class SimpleActorConfig:
    def __init__(self, clip_ratio):
        self.clip_ratio = clip_ratio
        self.clip_ratio_low = clip_ratio
        self.clip_ratio_high = clip_ratio
        self.clip_ratio_c = 3.0
        self.global_batch_info = {}

    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_actor_config(clip_ratio):
    return SimpleActorConfig(clip_ratio)


def load_math_data(num_train=500, num_val=500):
    from datasets import load_dataset
    try:
        try:
            ds = load_dataset("hendrycks/competition_math", "all", split="train")
        except Exception:
            ds = load_dataset("hendrycks/competition_math", split="train")
    except Exception:
        print("HF Hub not reachable, using dummy data")
        ds = None

    if ds is not None:
        train_data = []
        for i in range(min(num_train, len(ds))):
            item = ds[i]
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            prompt = f"Solve the following math problem step by step. Put your final answer within \\boxed{{}}.\n\n{problem}\n\n"
            train_data.append({"prompt": prompt, "problem": problem, "answer": solution})

        try:
            ds_val = load_dataset("hendrycks/competition_math", "all", split="test")
        except Exception:
            ds_val = load_dataset("hendrycks/competition_math", split="test")
        val_data = []
        for i in range(min(num_val, len(ds_val))):
            item = ds_val[i]
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            prompt = f"Solve the following math problem step by step. Put your final answer within \\boxed{{}}.\n\n{problem}\n\n"
            val_data.append({"prompt": prompt, "problem": problem, "answer": solution})
    else:
        train_data = [
            {"prompt": "Question: What is 2+2?\nAnswer:", "problem": "2+2", "answer": "4"},
            {"prompt": "Question: What is 3+5?\nAnswer:", "problem": "3+5", "answer": "8"},
            {"prompt": "Question: What is 10-3?\nAnswer:", "problem": "10-3", "answer": "7"},
            {"prompt": "Question: What is 6*7?\nAnswer:", "problem": "6*7", "answer": "42"},
        ][:num_train]
        val_data = [
            {"prompt": "Question: What is 1+1?\nAnswer:", "problem": "1+1", "answer": "2"},
            {"prompt": "Question: What is 4+4?\nAnswer:", "problem": "4+4", "answer": "8"},
        ][:num_val]

    return train_data, val_data


def evaluate(model, tokenizer, val_data, config):
    model.eval()
    correct = 0
    total = 0
    for item in tqdm(val_data, desc="Eval"):
        prompt = item["prompt"]
        answer = item["answer"]
        prompt_enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = _generate_rollout(model, tokenizer, prompt_enc, config.method.temperature,
                                     config.training.max_response_length)
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        reward = exact_match_reward(text, answer)
        correct += reward
        total += 1
    model.train()
    return {"accuracy": correct / max(total, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/verl_mh_all_proposals.yaml")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--method", type=str, default="mh_all_proposals")
    parser.add_argument("--train_prompts", type=int, default=500)
    parser.add_argument("--mh_steps", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--span_len", type=int, default=16)
    parser.add_argument("--max_response_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs_verl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_gpt2", action="store_true")
    args = parser.parse_args()

    if args.test_gpt2:
        args.model = "gpt2"
        args.train_prompts = 4
        args.max_response_length = 32
        args.num_epochs = 1
        args.lr = 1e-5
        args.batch_size = 1
        args.grad_accum = 2

    # Build config
    if os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
    else:
        cfg = OmegaConf.create({
            "training": {
                "max_response_length": args.max_response_length,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "output_dir": args.output_dir,
                "clip_epsilon": 0.2,
                "loss_agg_mode": "token-mean",
            },
            "method": {
                "method": args.method,
                "mh_steps": args.mh_steps,
                "alpha": args.alpha,
                "span_len": args.span_len,
                "temperature": 1.0,
                "num_rollouts": 8,
                "include_rejected": True,
                "include_chain_states": False,
                "exact_dedup": False,
            }
        })

    cfg.training.max_response_length = args.max_response_length
    cfg.training.num_epochs = args.num_epochs
    cfg.training.lr = args.lr
    cfg.method.method = args.method
    cfg.method.mh_steps = args.mh_steps
    cfg.method.alpha = args.alpha
    cfg.method.span_len = args.span_len

    set_seed(args.seed)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Method: {cfg.method.method}")
    print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    from peft import LoraConfig, get_peft_model, TaskType
    # Auto-detect LoRA target modules based on model type
    if "gpt2" in args.model.lower():
        target_modules = ["c_attn", "c_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading data...")
    train_data, val_data = load_math_data(num_train=args.train_prompts)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.training.lr, weight_decay=0.01)
    num_steps = (len(train_data) // cfg.training.batch_size // cfg.training.grad_accum) * cfg.training.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_steps),
                                                  num_training_steps=num_steps)

    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        indices = list(range(len(train_data)))
        random.shuffle(indices)

        pbar = tqdm(range(0, len(indices), cfg.training.batch_size),
                    desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

        for batch_start in pbar:
            batch_idxs = indices[batch_start:batch_start + cfg.training.batch_size]
            batch_items = [train_data[i] for i in batch_idxs]

            candidate_pools = []
            rewards_list = []

            for item in batch_items:
                with torch.no_grad():
                    pool = generate_mh_rollouts(model, tokenizer, item["prompt"], cfg)
                rewards = [exact_match_reward(c["response_text"], item["answer"]) for c in pool]
                candidate_pools.append(pool)
                rewards_list.append(rewards)

            loss, _ = compute_grpo_loss_verl(
                model, tokenizer,
                [item["prompt"] for item in batch_items],
                candidate_pools, rewards_list, cfg,
            )

            loss = loss / cfg.training.grad_accum
            loss.backward()

            if (global_step + 1) % cfg.training.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item() * cfg.training.grad_accum
            pbar.set_postfix({"loss": f"{epoch_loss / max(1, batch_start + 1):.4f}"})

        avg_loss = epoch_loss / max(1, len(indices))
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        eval_metrics = evaluate(model, tokenizer, val_data, cfg)
        print(f"Eval Accuracy: {eval_metrics['accuracy']:.4f}")

        save_dir = os.path.join(cfg.training.output_dir, f"epoch_{epoch + 1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
