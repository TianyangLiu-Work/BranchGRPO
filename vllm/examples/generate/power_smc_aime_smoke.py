import json
import math
import os
import re
import statistics
import time
from pathlib import Path

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
NUM_PROBLEMS = int(os.environ.get("NUM_PROBLEMS", "3"))
SAMPLES = int(os.environ.get("SAMPLES_PER_PROBLEM", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
PARTICLES = int(os.environ.get("PARTICLES", "32"))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "64"))
ESS = float(os.environ.get("ESS_THRESHOLD", "0.5"))
ALPHA_RAMP = int(os.environ.get("ALPHA_RAMP_TOKENS", "400"))
GPU_MEM = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
ALPHAS = [
    float(a) for a in os.environ.get("ALPHAS", "2.0,4.0").split(",")
    if a.strip()
]
ENABLE_BL = os.environ.get("ENABLE_BASELINE", "1") == "1"
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "1") == "1"
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "FLASHINFER")
K_VALUES = [1, 2, 4, 8, 16, 32]
OUT = Path("outputs/power_smc/aime2025")


def log(msg):
    print(f"[aime-smoke] {msg}", flush=True)


problems = []
with open("aime2025.jsonl", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            problems.append(json.loads(ln))
problems = problems[:NUM_PROBLEMS]
log(f"loaded {len(problems)} problems")


def extract_answer(text):
    for m in re.findall(r"\\boxed\{([^{}]+)\}", text):
        try:
            v = int(m.strip())
            if 0 <= v <= 999:
                return v
        except ValueError:
            pass
    for m in re.findall(r"\b(\d{1,3})\b", text[-500:]):
        v = int(m)
        if 0 <= v <= 999:
            return v
    return None


def pass_at_k(corrects, k):
    n = len(corrects)
    if k > n:
        raise ValueError(f"pass@k requires k <= n, got k={k}, n={n}.")
    c = sum(corrects)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_or_none(corrects, k):
    if k > len(corrects):
        return None
    return pass_at_k(corrects, k)


def mean_pass_at_k(results, k):
    values = [
        pass_k for result in results
        if (pass_k := result["pass_at_k"].get(k)) is not None
    ]
    if not values:
        return None
    return statistics.fmean(values)


def format_metric(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main():
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    log(f"loading {MODEL}")
    attention_config = (
        {"backend": ATTENTION_BACKEND} if ATTENTION_BACKEND else None
    )
    llm = LLM(
        model=MODEL, gpu_memory_utilization=GPU_MEM, dtype="auto",
        trust_remote_code=True, enable_prefix_caching=True,
        async_scheduling=False,
        logprobs_mode="raw_logprobs",
        attention_config=attention_config,
    )
    # Pre-build the tokenizer for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    log("LLM ready")

    def build_messages(problem_text):
        return [{"role": "user", "content": (
            "Please solve the following AIME math competition problem step by "
            "step. Think carefully, show all your work, and put your final "
            "answer (an integer between 0 and 999) within \\boxed{}.\n\n"
            f"Problem:\n{problem_text}"
        )}]

    def format_prompt(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            enable_thinking=ENABLE_THINKING, add_generation_prompt=True)

    def run_psmc(messages, alpha, seed):
        prompt = format_prompt(messages)
        params = SamplingParams(
            max_tokens=MAX_TOKENS, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
            seed=seed,
            extra_args={"power_smc": {
                "enabled": True, "alpha": alpha, "particles": PARTICLES,
                "block_size": BLOCK_SIZE, "ess_threshold": ESS,
                "alpha_ramp_tokens": ALPHA_RAMP, "proposal": "power_temperature",
                "return_diagnostics": True, "kv_cow": True,
            }},
        )
        t0 = time.perf_counter()
        out = llm.generate([prompt], params, use_tqdm=False)[0]
        dt = time.perf_counter() - t0
        comp = out.outputs[0]
        return {
            "text": comp.text, "token_count": len(comp.token_ids),
            "latency_s": dt, "diagnostics": getattr(out, "power_smc", None),
        }

    def run_bl(messages, seed):
        prompt = format_prompt(messages)
        params = SamplingParams(
            max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
            seed=seed,
        )
        t0 = time.perf_counter()
        out = llm.generate([prompt], params, use_tqdm=False)[0]
        dt = time.perf_counter() - t0
        comp = out.outputs[0]
        return {"text": comp.text, "token_count": len(comp.token_ids), "latency_s": dt}

    results = {
        "config": {
            "model": MODEL, "alphas": ALPHAS, "k_values": K_VALUES,
            "samples_per_problem": SAMPLES, "num_problems": NUM_PROBLEMS,
            "particles": PARTICLES, "max_tokens": MAX_TOKENS,
            "block_size": BLOCK_SIZE, "ess_threshold": ESS,
            "alpha_ramp_tokens": ALPHA_RAMP,
            "enable_thinking": ENABLE_THINKING,
            "attention_backend": ATTENTION_BACKEND,
        },
        "baseline": {},
        "power_smc": {},
    }

    if ENABLE_BL:
        log("=== BASELINE (T=0) ===")
        bl_results = []
        for pi, prob in enumerate(problems):
            msgs = build_messages(prob["problem"])
            gold = int(prob["answer"])
            corrects = []
            rows = []
            for i in range(SAMPLES):
                r = run_bl(msgs, i)
                ans = extract_answer(r["text"])
                ok = ans == gold if ans is not None else False
                corrects.append(ok)
                rows.append({"sample_idx": i, "text": r["text"],
                             "extracted_answer": ans, "correct": ok,
                             "latency_s": r["latency_s"],
                             "token_count": r["token_count"]})
                log(f"[baseline] Q{pi+1}/{len(problems)} s{i+1}/{SAMPLES} "
                    f"correct={ok} ans={ans} gold={gold} {r['latency_s']:.1f}s")
            passk = {k: pass_at_k_or_none(corrects, k) for k in K_VALUES}
            bl_results.append({"problem_id": prob["id"], "gold_answer": gold,
                               "correct_count": int(sum(corrects)),
                               "corrects": corrects, "pass_at_k": passk,
                               "samples": rows})
        results["baseline"] = bl_results

    for alpha in ALPHAS:
        label = f"alpha_{alpha:.1f}"
        log(f"=== Power-SMC alpha={alpha:.1f} ===")
        ar = []
        for pi, prob in enumerate(problems):
            msgs = build_messages(prob["problem"])
            gold = int(prob["answer"])
            corrects = []
            rows = []
            lats = []
            toks = []
            for i in range(SAMPLES):
                r = run_psmc(msgs, alpha, i)
                ans = extract_answer(r["text"])
                ok = ans == gold if ans is not None else False
                corrects.append(ok)
                lats.append(r["latency_s"])
                toks.append(r["token_count"])
                rows.append({"sample_idx": i, "text": r["text"],
                             "extracted_answer": ans, "correct": ok,
                             "latency_s": r["latency_s"],
                             "token_count": r["token_count"],
                             "diagnostics": r.get("diagnostics")})
                log(f"[a={alpha:.1f}] Q{pi+1}/{len(problems)} s{i+1}/{SAMPLES} "
                    f"correct={ok} ans={ans} gold={gold} {r['latency_s']:.1f}s "
                    f"tok={r['token_count']}")
            passk = {k: pass_at_k_or_none(corrects, k) for k in K_VALUES}
            ar.append({"problem_id": prob["id"], "gold_answer": gold,
                       "correct_count": int(sum(corrects)),
                       "corrects": corrects, "pass_at_k": passk,
                       "mean_latency_s": statistics.fmean(lats),
                       "mean_token_count": statistics.fmean(toks),
                       "total_resamples": sum(
                           (s.get("diagnostics") or {}).get("resample_count", 0)
                           for s in rows),
                       "samples": rows})
        results["power_smc"][label] = ar

    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / "aime2025_smoke.json"
    jp.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                  encoding="utf-8")
    log(f"saved {jp}")

    lines = [
        "# AIME 2025 pass@k — Smoke Test",
        "",
        f"**Model:** `{MODEL}`",
        f"**Problems:** {len(problems)}",
        f"**Samples/problem:** {SAMPLES}",
        f"**Particles:** {PARTICLES}",
        f"**Alphas:** {[f'{a:.1f}' for a in ALPHAS]}",
        "",
        "## Pass@k",
        "",
        "| Alpha | pass@1 | pass@2 | pass@4 | Mean latency (s) | "
        "Mean tokens | Total resamples |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha in ALPHAS:
        label = f"alpha_{alpha:.1f}"
        ar = results["power_smc"].get(label, [])
        if not ar:
            continue
        all_l = []
        all_t = []
        all_rs = 0
        for r in ar:
            all_l.append(r["mean_latency_s"])
            all_t.append(r["mean_token_count"])
            all_rs += r.get("total_resamples", 0)
        pks = [format_metric(mean_pass_at_k(ar, k)) for k in [1, 2, 4]]
        lines.append(
            f"| {alpha:.1f} | {pks[0]} | {pks[1]} | {pks[2]} | "
            f"{statistics.fmean(all_l):.1f} | {statistics.fmean(all_t):.0f} | "
            f"{all_rs} |"
        )

    if ENABLE_BL and results.get("baseline"):
        all_l = []
        for r in results["baseline"]:
            all_l.extend(s["latency_s"] for s in r["samples"])
        pks = [
            format_metric(mean_pass_at_k(results["baseline"], k))
            for k in [1, 2, 4]
        ]
        lines.append(
            f"| baseline (T=0) | {pks[0]} | {pks[1]} | {pks[2]} | "
            f"{statistics.fmean(all_l):.1f} | - | - |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Smoke test: reduced problems/samples to verify pipeline.",
        "- Power-SMC: temperature=1.0, particles=32, "
        "proposal=power_temperature, kv_cow=True.",
        "- Baseline: temperature=0.0 (greedy).",
        "",
    ]
    mp = OUT / "aime2025_smoke.md"
    mp.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved {mp}")
    log("DONE")

if __name__ == "__main__":
    main()
