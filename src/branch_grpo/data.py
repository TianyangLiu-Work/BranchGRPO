"""Data conversion helpers for VeRL GRPO training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .reward import extract_answer


DEFAULT_INSTRUCTION = "Solve the problem. Keep the reasoning concise. Put the final answer in \\boxed{}."


def load_json_examples(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of math examples")
    return data


def split_examples(
    examples: list[dict[str, Any]],
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        raise ValueError("No examples provided")
    val_size = val_limit if val_limit is not None else max(1, min(64, len(examples) // 10))
    val_size = min(val_size, max(1, len(examples) - 1))
    train = examples[:-val_size]
    val = examples[-val_size:]
    if train_limit is not None:
        train = train[:train_limit]
    if val_limit is not None:
        val = val[:val_limit]
    return train, val


def build_verl_record(
    example: dict[str, Any],
    index: int,
    split: str,
    data_source: str,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    problem = example.get("problem") or example.get("question") or example.get("prompt")
    solution = example.get("solution") or example.get("answer") or example.get("ground_truth")
    if not problem:
        raise ValueError(f"Example {index} is missing problem/question/prompt")
    if solution is None:
        raise ValueError(f"Example {index} is missing solution/answer/ground_truth")

    content = f"{str(problem).strip()} {instruction}".strip()
    ground_truth = extract_answer(solution)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": int(index)},
    }


def build_records(
    examples: Iterable[dict[str, Any]],
    split: str,
    data_source: str,
    instruction: str = DEFAULT_INSTRUCTION,
) -> list[dict[str, Any]]:
    return [
        build_verl_record(example, index, split, data_source, instruction)
        for index, example in enumerate(examples)
    ]


def write_json_reference(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
