#!/usr/bin/env python3
"""Batch generation throughput benchmark — extended sizes."""

import torch, time, sys
sys.path.insert(0, '/home/tyliu/ghworkspace/BranchGRPO')
from src.model_utils import load_model_and_tokenizer
from src.config import ExperimentConfig
from src.mh_sampling import _generate_rollout_batch

config = ExperimentConfig()
config.training.model_name = "Qwen/Qwen2.5-Math-7B"
config.training.use_lora = False
print("Loading model...")
model, tokenizer = load_model_and_tokenizer(config)
model.eval()
device = next(model.parameters()).device
prompt = "Solve: Find the sum of all positive integers n such that n^2 - n + 1 is a perfect square.\n\n"
prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)

max_length, temperature, top_p = 1024, 1.0, 1.0
batch_sizes = [32, 64, 128]
warmup, measure = 1, 2

print(f"Prompt: {prompt_enc.input_ids.shape[1]} tokens, max_len={max_length}")

for bs in batch_sizes:
    try:
        print(f"\nBS={bs} warmup...", flush=True)
        _ = _generate_rollout_batch(model, tokenizer, prompt_enc, temperature, top_p, max_length, bs)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        total_tokens = 0
        for _ in range(measure):
            ids_list = _generate_rollout_batch(model, tokenizer, prompt_enc, temperature, top_p, max_length, bs)
            torch.cuda.synchronize()
            for ids in ids_list:
                total_tokens += len(ids)
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        avg_time = elapsed / measure
        avg_tokens = total_tokens / measure
        tps = avg_tokens / avg_time
        print(f"BS={bs:3d} | {avg_time:.1f}s | {avg_tokens:.0f} tok/run | {tps:.0f} tok/s | peak_mem={peak_mem:.1f}GiB", flush=True)
    except torch.OutOfMemoryError:
        print(f"BS={bs:3d} OOM", flush=True)
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"BS={bs:3d} error: {e}", flush=True)
        break

model.to("cpu")
