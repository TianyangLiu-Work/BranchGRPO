"""Evaluation utilities for BranchGRPO (MATH500)."""

import torch
from tqdm import tqdm
from typing import Dict
from .verifier import exact_match_reward
from .mh_sampling import _generate_rollout


@torch.no_grad()
def evaluate_on_math500(model, tokenizer, config, val_data) -> Dict:
    model.eval()
    correct = 0
    total = 0
    generated_tokens = 0

    for item in tqdm(val_data, desc="Evaluating"):
        prompt = item["prompt"]
        answer = item["answer"]

        prompt_enc = tokenizer(prompt, return_tensors="pt").to(model.device)

        response_ids = _generate_rollout(
            model, tokenizer, prompt_enc,
            temperature=config.method.temperature,
            top_p=config.method.top_p,
            max_length=config.training.max_response_length,
        )
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        reward = exact_match_reward(response_text, answer)
        correct += reward
        total += 1
        generated_tokens += len(response_ids[0])

    accuracy = correct / max(total, 1)
    metrics = {
        "accuracy": accuracy,
        "num_samples": total,
        "generated_tokens": generated_tokens,
    }

    model.train()
    return metrics
