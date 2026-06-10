# SPDX-License-Identifier: Apache-2.0
"""AIME Power-SMC alpha/particle sweep.

For each AIME problem and each ``alpha x particles`` configuration, this
script runs several independent Power-SMC attempts as one vLLM batch. It writes
attempt-level CSV rows, per-problem summaries, aggregate summaries, and a
Markdown report with plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ALPHAS = "1.2,1.4"
DEFAULT_PARTICLES = "4,8,16"
DEFAULT_BATCH_SIZES = "4,8,16"
DEFAULT_PROBLEM_INDICES = "7,18,20"
DEFAULT_K_VALUES = (1, 4, 8, 16)


@dataclass
class Config:
    model: str
    prompt_file: Path
    output_dir: Path
    alphas: list[float]
    particles: list[int]
    batch_sizes: list[int]
    attempts_per_problem: int
    max_tokens: int
    block_size: int
    ess_threshold: float
    alpha_ramp_tokens: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    dtype: str
    attention_backend: str | None
    enable_thinking: bool
    async_scheduling: bool
    enforce_eager: bool
    trust_remote_code: bool
    limit: int | None
    problem_indices: list[int] | None
    checkpoint_every_batch: bool


def log(message: str) -> None:
    print(f"[aime-sweep] {message}", flush=True)


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one alpha")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one particles value")
    return values


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt-file", type=Path, default=Path("aime2025.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/power_smc/aime2025"))
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--particles", default=DEFAULT_PARTICLES)
    parser.add_argument("--batch-sizes", default=None,
                        help=("Comma-separated external batch sizes. Defaults "
                              "to --attempts-per-problem when omitted."))
    parser.add_argument("--attempts-per-problem", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--alpha-ramp-tokens", type=int, default=400)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--async-scheduling", action="store_true",
                        help=("Enable vLLM async scheduling. Disabled by "
                              "default because Power-SMC needs per-step "
                              "base/proposal logprobs synchronously."))
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of AIME problems.")
    parser.add_argument(
        "--problem-indices",
        default=DEFAULT_PROBLEM_INDICES,
        help=("Comma-separated zero-based problem indices. Overrides --limit "
              "when provided."),
    )
    parser.add_argument("--checkpoint-every-batch",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()

    batch_sizes = (
        parse_int_list(args.batch_sizes)
        if args.batch_sizes is not None else [args.attempts_per_problem]
    )
    if args.attempts_per_problem < 1:
        raise ValueError("--attempts-per-problem must be >= 1")
    if any(batch_size < 1 for batch_size in batch_sizes):
        raise ValueError("--batch-sizes values must be >= 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")
    if args.block_size < 1:
        raise ValueError("--block-size must be >= 1")
    if not 0.0 < args.ess_threshold <= 1.0:
        raise ValueError("--ess-threshold must be in (0, 1]")

    return Config(
        model=args.model,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        alphas=parse_float_list(args.alphas),
        particles=parse_int_list(args.particles),
        batch_sizes=batch_sizes,
        attempts_per_problem=args.attempts_per_problem,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
        ess_threshold=args.ess_threshold,
        alpha_ramp_tokens=args.alpha_ramp_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        enable_thinking=args.enable_thinking,
        async_scheduling=args.async_scheduling,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        limit=args.limit,
        problem_indices=parse_int_list(args.problem_indices)
        if args.problem_indices else None,
        checkpoint_every_batch=args.checkpoint_every_batch,
    )


def load_problems(path: Path, limit: int | None,
                  problem_indices: list[int] | None) -> list[dict[str, Any]]:
    all_problems: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if "problem" not in item or "answer" not in item:
                raise ValueError(f"{path}:{line_no} requires problem and answer")
            item.setdefault("_source_index", line_no - 1)
            all_problems.append(item)
    if problem_indices is not None:
        problems = []
        for index in problem_indices:
            if index < 0 or index >= len(all_problems):
                raise ValueError(
                    f"problem index {index} is out of range for {path}")
            problems.append(all_problems[index])
    else:
        problems = all_problems
    if limit is not None and problem_indices is None:
        problems = problems[:limit]
    if not problems:
        raise ValueError(f"No problems loaded from {path}")
    log(f"loaded {len(problems)} AIME problems")
    return problems


def extract_aime_answer(text: str) -> int | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    for content in reversed(boxed):
        try:
            value = int(content.strip())
        except ValueError:
            continue
        if 0 <= value <= 999:
            return value

    numbers = re.findall(r"\b(\d{1,3})\b", text[-500:])
    for number in reversed(numbers):
        value = int(number)
        if 0 <= value <= 999:
            return value
    return None


def pass_at_k(corrects: list[bool], k: int) -> float:
    n = len(corrects)
    if k > n:
        raise ValueError(f"pass@k requires k <= n, got k={k}, n={n}.")
    c = sum(corrects)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_or_none(corrects: list[bool], k: int) -> float | None:
    if k > len(corrects):
        return None
    return pass_at_k(corrects, k)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def format_metric(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def build_messages(problem_text: str) -> list[dict[str, str]]:
    return [{
        "role": "user",
        "content": (
            "Please solve the following AIME math competition problem step by "
            "step. Think carefully, show all your work, and put your final "
            "answer (an integer between 0 and 999) within \\boxed{}.\n\n"
            f"Problem:\n{problem_text}"
        ),
    }]


def format_prompt(tokenizer: Any, messages: list[dict[str, str]],
                  enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def make_sampling_params(
    *,
    alpha: float,
    particles: int,
    seed: int,
    config: Config,
) -> Any:
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "max_tokens": config.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "seed": seed,
    }
    if alpha != 1.0:
        kwargs["extra_args"] = {
            "power_smc": {
                "enabled": True,
                "alpha": float(alpha),
                "particles": particles,
                "block_size": config.block_size,
                "ess_threshold": config.ess_threshold,
                "alpha_ramp_tokens": config.alpha_ramp_tokens,
                "proposal": "power_temperature",
                "return_diagnostics": True,
                "kv_cow": True,
            },
        }
    return SamplingParams(**kwargs)


def canonical_particles(alpha: float, particles: int) -> int:
    return 0 if alpha == 1.0 else particles


def effective_particle_batch(alpha: float, particles: int, batch_size: int) -> int:
    return batch_size if alpha == 1.0 else batch_size * particles


def run_problem_attempt_batch(
    *,
    llm: Any,
    prompt: str,
    problem: dict[str, Any],
    problem_index: int,
    alpha: float,
    particles: int,
    batch_size: int,
    config: Config,
) -> list[dict[str, Any]]:
    prompts = [prompt] * batch_size
    params = [
        make_sampling_params(
            alpha=alpha,
            particles=particles,
            seed=attempt_idx,
            config=config,
        )
        for attempt_idx in range(batch_size)
    ]

    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    batch_latency = time.perf_counter() - started
    gold_answer = int(problem["answer"])
    rows: list[dict[str, Any]] = []
    for attempt_idx, output in enumerate(outputs):
        completion = output.outputs[0]
        text = completion.text
        extracted = extract_aime_answer(text)
        diagnostics = getattr(output, "power_smc", None)
        passed = extracted == gold_answer if extracted is not None else False
        row_particles = canonical_particles(alpha, particles)
        rows.append({
            "alpha": alpha,
            "particles": row_particles,
            "external_batch_size": batch_size,
            "effective_particle_batch": effective_particle_batch(
                alpha, particles, batch_size),
            "problem_index": problem_index,
            "problem_id": problem.get("id", problem_index),
            "year": problem.get("year"),
            "attempt": attempt_idx,
            "seed": attempt_idx,
            "gold_answer": gold_answer,
            "extracted_answer": extracted,
            "passed": passed,
            "correct": passed,
            "token_count": len(completion.token_ids),
            "batch_latency_s": batch_latency,
            "latency_per_attempt_s": batch_latency / batch_size,
            "latency_s": batch_latency / batch_size,
            "resample_count": (
                diagnostics or {}).get("resample_count") if diagnostics else None,
            "final_ess": (
                diagnostics or {}).get("final_ess") if diagnostics else None,
            "kv_cow_saved_blocks": (
                diagnostics or {}).get("kv_cow_saved_blocks")
            if diagnostics else None,
            "kv_cow_saved_tokens": (
                diagnostics or {}).get("kv_cow_saved_tokens")
            if diagnostics else None,
            "diagnostics_json": json.dumps(
                diagnostics,
                ensure_ascii=False,
                default=str,
            ) if diagnostics is not None else "",
            "text": text,
        })
    return rows


def iter_sweep_batches(
    *,
    problems: list[dict[str, Any]],
    prompts: list[str],
    config: Config,
) -> Iterator[tuple[int, dict[str, Any], str, int, float, int]]:
    for local_index, (problem, prompt) in enumerate(
            zip(problems, prompts, strict=True)):
        problem_index = int(problem.get("_source_index", local_index))
        for batch_size in config.batch_sizes:
            for alpha in config.alphas:
                if alpha == 1.0:
                    yield problem_index, problem, prompt, batch_size, alpha, 0
                    continue
                for particles in config.particles:
                    yield problem_index, problem, prompt, batch_size, alpha, particles


def summarize_problem(rows: list[dict[str, Any]]) -> dict[str, Any]:
    corrects = [bool(row.get("passed", row["correct"])) for row in rows]
    latencies = [float(row["latency_per_attempt_s"]) for row in rows]
    token_counts = [int(row["token_count"]) for row in rows]
    first = rows[0]
    summary = {
        "alpha": first["alpha"],
        "particles": first["particles"],
        "external_batch_size": first["external_batch_size"],
        "effective_particle_batch": first["effective_particle_batch"],
        "problem_index": first["problem_index"],
        "problem_id": first["problem_id"],
        "attempts": len(rows),
        "correct_count": sum(corrects),
        "problem_passed_at_batch": sum(corrects) > 0,
        "accuracy": sum(corrects) / len(corrects),
        "batch_latency_s": float(first["batch_latency_s"]),
        "mean_latency_s": statistics.fmean(latencies),
        "latency_per_attempt_s": statistics.fmean(latencies),
        "mean_token_count": statistics.fmean(token_counts),
        "total_tokens": sum(token_counts),
    }
    for k in DEFAULT_K_VALUES:
        summary[f"pass_at_{k}"] = pass_at_k_or_none(corrects, k)
    return summary


def summarize_config(rows: list[dict[str, Any]],
                     problem_summaries: list[dict[str, Any]],
                     planned_problems: int | None = None) -> dict[str, Any]:
    first = rows[0]
    pass_ks: dict[int, float | None] = {}
    for k in DEFAULT_K_VALUES:
        values = [
            float(summary[f"pass_at_{k}"])
            for summary in problem_summaries
            if summary[f"pass_at_{k}"] is not None
        ]
        pass_ks[k] = mean_or_none(values)

    correct_count = sum(bool(row.get("passed", row["correct"])) for row in rows)
    token_counts = [int(row["token_count"]) for row in rows]
    latencies = [float(row["latency_per_attempt_s"]) for row in rows]
    batch_latencies = [
        float(summary["batch_latency_s"]) for summary in problem_summaries
    ]
    resamples = [
        int(row["resample_count"]) for row in rows
        if row["resample_count"] is not None
    ]
    saved_blocks = [
        int(row["kv_cow_saved_blocks"]) for row in rows
        if row["kv_cow_saved_blocks"] is not None
    ]
    summary: dict[str, Any] = {
        "alpha": first["alpha"],
        "particles": first["particles"],
        "external_batch_size": first["external_batch_size"],
        "effective_particle_batch": first["effective_particle_batch"],
        "completed_problems": len(problem_summaries),
        "planned_problems": planned_problems or len(problem_summaries),
        "problems": len(problem_summaries),
        "attempts_per_problem": len(rows) // len(problem_summaries),
        "total_attempts": len(rows),
        "correct_count": correct_count,
        "attempt_accuracy": correct_count / len(rows),
        "problem_pass_rate": (
            sum(bool(summary["problem_passed_at_batch"])
                for summary in problem_summaries) / len(problem_summaries)
        ),
        "mean_batch_latency_s": statistics.fmean(batch_latencies),
        "median_batch_latency_s": statistics.median(batch_latencies),
        "p90_batch_latency_s": percentile(batch_latencies, 0.9),
        "mean_latency_s": statistics.fmean(latencies),
        "latency_per_attempt_s": statistics.fmean(latencies),
        "mean_token_count": statistics.fmean(token_counts),
        "total_tokens": sum(token_counts),
        "tokens_per_second": (
            sum(token_counts) / math.fsum(batch_latencies)
            if math.fsum(batch_latencies) > 0.0 else None
        ),
        "mean_resample_count": mean_or_none(resamples),
        "total_resamples": sum(resamples),
        "total_kv_cow_saved_blocks": sum(saved_blocks),
    }
    for k in DEFAULT_K_VALUES:
        summary[f"mean_pass_at_{k}"] = pass_ks[k]
    return summary


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config_key(row: dict[str, Any]) -> tuple[int, float, int]:
    return (
        int(row["external_batch_size"]),
        float(row["alpha"]),
        int(row["particles"]),
    )


def summarize_completed_configs(
    *,
    attempts_by_config: dict[tuple[int, float, int], list[dict[str, Any]]],
    problems_by_config: dict[tuple[int, float, int], list[dict[str, Any]]],
    planned_problems: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(attempts_by_config):
        rows.append(
            summarize_config(
                attempts_by_config[key],
                problems_by_config[key],
                planned_problems=planned_problems,
            ))
    return rows


def planned_config_keys(config: Config) -> list[tuple[int, float, int]]:
    keys: list[tuple[int, float, int]] = []
    for batch_size in config.batch_sizes:
        for alpha in config.alphas:
            if alpha == 1.0:
                keys.append((batch_size, alpha, 0))
                continue
            for particles in config.particles:
                keys.append((batch_size, alpha, particles))
    return keys


def key_to_state(key: tuple[int, float, int]) -> dict[str, Any]:
    batch_size, alpha, particles = key
    return {
        "external_batch_size": batch_size,
        "alpha": alpha,
        "particles": particles,
        "effective_particle_batch": (
            batch_size if alpha == 1.0 else batch_size * particles
        ),
    }


def write_run_state(
    *,
    path: Path,
    config: Config,
    planned_keys: list[tuple[int, float, int]],
    completed: dict[tuple[int, float, int], int],
    failures: list[dict[str, Any]],
    started_at: float,
    last_update_at: float,
    planned_problems: int,
) -> None:
    total_batches = len(planned_keys) * planned_problems
    completed_batches = sum(completed.values())
    completed_set = {key for key, count in completed.items()
                     if count >= planned_problems}
    failed_keys = {
        (int(item["external_batch_size"]), float(item["alpha"]),
         int(item["particles"]))
        for item in failures
    }
    pending = [
        key_to_state(key) for key in planned_keys
        if key not in completed_set and key not in failed_keys
    ]
    state = {
        "model": config.model,
        "prompt_file": str(config.prompt_file),
        "problem_indices": config.problem_indices,
        "planned_problem_count": planned_problems,
        "planned_config_count": len(planned_keys),
        "planned_batch_count": total_batches,
        "completed_batch_count": completed_batches,
        "completion_fraction": (
            completed_batches / total_batches if total_batches else 1.0
        ),
        "started_at_unix": started_at,
        "last_update_at_unix": last_update_at,
        "completed_configs": [
            key_to_state(key) for key in sorted(completed_set)
        ],
        "pending_configs": pending,
        "failed_configs": failures,
    }
    path.write_text(json.dumps(state, indent=2, sort_keys=True),
                    encoding="utf-8")


def write_search_csvs(
    *,
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    fastest_path = output_dir / "fastest_by_pass_requirement.csv"
    pareto_path = output_dir / "pareto_configs.csv"
    write_csv(fastest_path, fastest_by_pass_requirement(summary_rows))
    write_csv(pareto_path, pareto_configs(summary_rows))
    return fastest_path, pareto_path


def fastest_by_pass_requirement(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    thresholds = {
        "problem_pass_rate": [0.34, 0.67, 1.0],
        "attempt_accuracy": [0.2, 0.3, 0.4, 0.5],
        "mean_pass_at_4": [0.5, 0.67, 1.0],
        "mean_pass_at_8": [0.67, 1.0],
        "mean_pass_at_16": [0.67, 1.0],
    }
    rows: list[dict[str, Any]] = []
    for metric, values in thresholds.items():
        for threshold in values:
            candidates = [
                row for row in summary_rows
                if row.get(metric) is not None and float(row[metric]) >= threshold
            ]
            if not candidates:
                rows.append({
                    "metric": metric,
                    "threshold": threshold,
                    "found": False,
                })
                continue
            best = min(candidates, key=lambda row: float(row["mean_batch_latency_s"]))
            rows.append({
                "metric": metric,
                "threshold": threshold,
                "found": True,
                "alpha": best["alpha"],
                "particles": best["particles"],
                "external_batch_size": best["external_batch_size"],
                "effective_particle_batch": best["effective_particle_batch"],
                "value": best[metric],
                "mean_batch_latency_s": best["mean_batch_latency_s"],
                "latency_per_attempt_s": best["latency_per_attempt_s"],
                "completed_problems": best["completed_problems"],
                "planned_problems": best["planned_problems"],
            })
    return rows


def pareto_configs(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for row in summary_rows:
        latency = float(row["mean_batch_latency_s"])
        score = float(row["problem_pass_rate"])
        dominated = False
        for other in summary_rows:
            if other is row:
                continue
            other_latency = float(other["mean_batch_latency_s"])
            other_score = float(other["problem_pass_rate"])
            if ((other_latency <= latency and other_score >= score)
                    and (other_latency < latency or other_score > score)):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier,
                  key=lambda row: (float(row["mean_batch_latency_s"]),
                                   -float(row["problem_pass_rate"])))


def plot_summary(summary_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib is not installed; skipping plots")
        return []

    plots: list[Path] = []

    def save(fig, filename: str) -> None:
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(path)

    power_rows = [row for row in summary_rows if int(row["particles"]) > 0]
    if power_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        for batch_size in sorted(
                {int(row["external_batch_size"]) for row in power_rows}):
            rows = [
                row for row in power_rows
                if int(row["external_batch_size"]) == batch_size
            ]
            grouped: dict[int, list[float]] = {}
            for row in rows:
                grouped.setdefault(int(row["particles"]), []).append(
                    float(row["mean_batch_latency_s"]))
            xs = sorted(grouped)
            ys = [statistics.fmean(grouped[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=f"batch={batch_size}")
        ax.set_xlabel("particles")
        ax.set_ylabel("mean batch latency (s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save(fig, "latency_vs_particles_by_batch.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        for particles in sorted({int(row["particles"]) for row in power_rows}):
            rows = [
                row for row in power_rows
                if int(row["particles"]) == particles
            ]
            grouped: dict[int, list[float]] = {}
            for row in rows:
                grouped.setdefault(int(row["external_batch_size"]), []).append(
                    float(row["mean_batch_latency_s"]))
            xs = sorted(grouped)
            ys = [statistics.fmean(grouped[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=f"particles={particles}")
        ax.set_xlabel("external batch size")
        ax.set_ylabel("mean batch latency (s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save(fig, "latency_vs_batch_by_particles.png")

    scatter_specs = [
        ("effective_particle_batch", "mean_batch_latency_s",
         "effective particle batch", "mean batch latency (s)",
         "latency_vs_effective_batch.png"),
        ("effective_particle_batch", "tokens_per_second",
         "effective particle batch", "tokens / second",
         "throughput_vs_effective_batch.png"),
        ("mean_batch_latency_s", "problem_pass_rate",
         "mean batch latency (s)", "problem pass rate",
         "pass_rate_vs_latency.png"),
        ("mean_batch_latency_s", "mean_pass_at_8",
         "mean batch latency (s)", "mean pass@8",
         "pass_at_k_vs_latency.png"),
    ]
    for x_key, y_key, xlabel, ylabel, filename in scatter_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for batch_size in sorted(
                {int(row["external_batch_size"]) for row in summary_rows}):
            rows = [
                row for row in summary_rows
                if int(row["external_batch_size"]) == batch_size
                and row.get(y_key) is not None
            ]
            if not rows:
                continue
            xs = [float(row[x_key]) for row in rows]
            ys = [float(row[y_key]) for row in rows]
            ax.scatter(xs, ys, label=f"batch={batch_size}", alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        save(fig, filename)
    return plots


def write_report(
    *,
    config: Config,
    summary_rows: list[dict[str, Any]],
    plots: list[Path],
    output_dir: Path,
) -> Path:
    report_path = output_dir / "aime2025_power_smc_sweep.md"
    complete = all(
        int(row["completed_problems"]) >= int(row["planned_problems"])
        for row in summary_rows
    ) if summary_rows else False
    lines = [
        "# AIME 2025 Power-SMC Sweep",
        "",
        f"- Model: `{config.model}`",
        f"- Problems: `{config.problem_indices or config.limit or 'all'}`",
        f"- Complete: `{complete}`",
        f"- Batch sizes: `{config.batch_sizes}`",
        f"- Alphas: `{config.alphas}`",
        f"- Particles: `{config.particles}`",
        f"- Max tokens: `{config.max_tokens}`",
        f"- Block size: `{config.block_size}`",
        f"- ESS threshold: `{config.ess_threshold}`",
        f"- Alpha ramp tokens: `{config.alpha_ramp_tokens}`",
        "",
        "## Summary",
        "",
        "| Batch | Alpha | Particles | Eff. batch | Coverage | Problem pass | "
        "pass@4 | pass@8 | pass@16 | Accuracy | Batch latency (s) | "
        "Latency/attempt (s) | Tok/s | Resamples |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows,
                      key=lambda item: (int(item["external_batch_size"]),
                                        float(item["alpha"]),
                                        int(item["particles"]))):
        lines.append(
            f"| {int(row['external_batch_size'])} | "
            f"{float(row['alpha']):.1f} | {int(row['particles'])} | "
            f"{int(row['effective_particle_batch'])} | "
            f"{int(row['completed_problems'])}/{int(row['planned_problems'])} | "
            f"{format_metric(row['problem_pass_rate'])} | "
            f"{format_metric(row['mean_pass_at_4'])} | "
            f"{format_metric(row['mean_pass_at_8'])} | "
            f"{format_metric(row['mean_pass_at_16'])} | "
            f"{format_metric(row['attempt_accuracy'])} | "
            f"{format_metric(row['mean_batch_latency_s'], 2)} | "
            f"{format_metric(row['latency_per_attempt_s'], 2)} | "
            f"{format_metric(row['tokens_per_second'], 1)} | "
            f"{format_metric(row['total_resamples'], 0)} |"
        )

    fastest_rows = fastest_by_pass_requirement(summary_rows)
    if fastest_rows:
        lines += [
            "",
            "## Fastest By Pass Requirement",
            "",
            "| Metric | Threshold | Found | Batch | Alpha | Particles | "
            "Eff. batch | Value | Batch latency (s) | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in fastest_rows:
            lines.append(
                f"| {row['metric']} | {format_metric(row['threshold'])} | "
                f"{row['found']} | {row.get('external_batch_size', '-')} | "
                f"{row.get('alpha', '-')} | {row.get('particles', '-')} | "
                f"{row.get('effective_particle_batch', '-')} | "
                f"{format_metric(row.get('value'))} | "
                f"{format_metric(row.get('mean_batch_latency_s'), 2)} | "
                f"{row.get('completed_problems', '-')}/"
                f"{row.get('planned_problems', '-')} |"
            )

    if plots:
        lines += ["", "## Plots", ""]
        for plot in plots:
            lines.append(f"![{plot.stem}]({plot.name})")
            lines.append("")

    lines += [
        "## Output Files",
        "",
        "- `aime2025_power_smc_attempts.csv`: one row per generated attempt.",
        "- `aime2025_power_smc_problem_summary.csv`: one row per problem/config.",
        "- `aime2025_power_smc_summary.csv`: one row per alpha/particles config.",
        "- `search_config_summary.csv`: search summary with batch-size metrics.",
        "- `fastest_by_pass_requirement.csv`: fastest configs by pass threshold.",
        "- `pareto_configs.csv`: non-dominated problem-pass/latency configs.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems(config.prompt_file, config.limit,
                             config.problem_indices)

    from transformers import AutoTokenizer

    from vllm import LLM

    log(f"loading model: {config.model}")
    attention_config = (
        {"backend": config.attention_backend}
        if config.attention_backend else None
    )
    llm = LLM(
        model=config.model,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        async_scheduling=config.async_scheduling,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=True,
        logprobs_mode="raw_logprobs",
        attention_config=attention_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        trust_remote_code=config.trust_remote_code,
    )
    log("LLM ready")

    attempt_rows: list[dict[str, Any]] = []
    problem_summary_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prompts_by_problem = [
        format_prompt(tokenizer, build_messages(problem["problem"]),
                      config.enable_thinking)
        for problem in problems
    ]

    attempts_by_config: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    problems_by_config: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    completed_by_config: dict[tuple[int, float, int], int] = {}
    failures: list[dict[str, Any]] = []
    planned_keys = planned_config_keys(config)
    seen_configs: set[tuple[int, float, int]] = set()
    started_at = time.time()

    def write_partial_outputs() -> None:
        partial_summary_rows = summarize_completed_configs(
            attempts_by_config=attempts_by_config,
            problems_by_config=problems_by_config,
            planned_problems=len(problems),
        )
        write_csv(config.output_dir / "partial_attempts.csv", attempt_rows)
        write_csv(config.output_dir / "partial_problem_summary.csv",
                  problem_summary_rows)
        write_csv(config.output_dir / "partial_config_summary.csv",
                  partial_summary_rows)
        write_csv(config.output_dir / "search_config_summary.csv",
                  partial_summary_rows)
        write_search_csvs(output_dir=config.output_dir,
                          summary_rows=partial_summary_rows)
        plots = plot_summary(partial_summary_rows, config.output_dir)
        report_path = write_report(
            config=config,
            summary_rows=partial_summary_rows,
            plots=plots,
            output_dir=config.output_dir,
        )
        interim_path = config.output_dir / "interim_report.md"
        interim_path.write_text(report_path.read_text(encoding="utf-8"),
                                encoding="utf-8")
        write_run_state(
            path=config.output_dir / "partial_run_state.json",
            config=config,
            planned_keys=planned_keys,
            completed=completed_by_config,
            failures=failures,
            started_at=started_at,
            last_update_at=time.time(),
            planned_problems=len(problems),
        )

    for (problem_index, problem, prompt, batch_size, alpha,
         particles) in iter_sweep_batches(
            problems=problems,
            prompts=prompts_by_problem,
            config=config,
    ):
        config_key = (batch_size, alpha, canonical_particles(alpha, particles))
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            log(
                f"=== batch={batch_size} alpha={alpha:.1f} "
                f"particles={canonical_particles(alpha, particles)} ==="
            )
        try:
            rows = run_problem_attempt_batch(
                llm=llm,
                prompt=prompt,
                problem=problem,
                problem_index=problem_index,
                alpha=alpha,
                particles=particles,
                batch_size=batch_size,
                config=config,
            )
        except Exception as exc:
            failures.append({
                "external_batch_size": batch_size,
                "alpha": alpha,
                "particles": canonical_particles(alpha, particles),
                "problem_index": problem_index,
                "error": repr(exc),
            })
            if config.checkpoint_every_batch:
                write_partial_outputs()
            raise
        attempt_rows.extend(rows)
        attempts_by_config.setdefault(config_key, []).extend(rows)
        problem_summary = summarize_problem(rows)
        problem_summary_rows.append(problem_summary)
        problems_by_config.setdefault(config_key, []).append(problem_summary)
        completed_by_config[config_key] = completed_by_config.get(
            config_key, 0) + 1
        log(
            f"problem={problem.get('id', problem_index)} "
            f"batch={batch_size} "
            f"alpha={alpha:.1f} particles={particles} "
            f"correct={problem_summary['correct_count']}/"
            f"{problem_summary['attempts']} "
            f"pass@8={format_metric(problem_summary['pass_at_8'])}"
        )
        if config.checkpoint_every_batch:
            write_partial_outputs()

    summary_rows = summarize_completed_configs(
        attempts_by_config=attempts_by_config,
        problems_by_config=problems_by_config,
        planned_problems=len(problems),
    )

    attempts_csv = config.output_dir / "aime2025_power_smc_attempts.csv"
    problem_csv = config.output_dir / "aime2025_power_smc_problem_summary.csv"
    summary_csv = config.output_dir / "aime2025_power_smc_summary.csv"
    write_csv(attempts_csv, attempt_rows)
    write_csv(problem_csv, problem_summary_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(config.output_dir / "search_config_summary.csv", summary_rows)
    fastest_path, pareto_path = write_search_csvs(
        output_dir=config.output_dir,
        summary_rows=summary_rows,
    )
    log(f"saved {attempts_csv}")
    log(f"saved {problem_csv}")
    log(f"saved {summary_csv}")
    log(f"saved {fastest_path}")
    log(f"saved {pareto_path}")

    plots = plot_summary(summary_rows, config.output_dir)
    report_path = write_report(
        config=config,
        summary_rows=summary_rows,
        plots=plots,
        output_dir=config.output_dir,
    )
    log(f"saved {report_path}")
    write_run_state(
        path=config.output_dir / "partial_run_state.json",
        config=config,
        planned_keys=planned_keys,
        completed=completed_by_config,
        failures=failures,
        started_at=started_at,
        last_update_at=time.time(),
        planned_problems=len(problems),
    )
    log("DONE")


if __name__ == "__main__":
    main()
