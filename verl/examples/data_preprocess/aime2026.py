# Copyright 2026 The veRL-PowerSMC contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
Preprocess AIME2026-style math data to veRL parquet format.

The raw dataset is expected to provide a problem statement and a final answer.
JSON/JSONL/CSV files are supported directly; a Hugging Face dataset path can
also be passed through datasets.load_dataset.
"""

import argparse
import os
import re
from collections.abc import Mapping

import datasets


def _last_boxed_only_string(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    idx = start + len(marker)
    depth = 1
    while idx < len(text):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
        idx += 1
    return None


def _remove_boxed(text: str) -> str:
    marker = "\\boxed{"
    if text.startswith(marker) and text.endswith("}"):
        return text[len(marker) : -1]
    return text


def _load_dataset(path: str, train_split: str, val_split: str):
    if os.path.isfile(os.path.expanduser(path)):
        expanded = os.path.expanduser(path)
        ext = os.path.splitext(expanded)[1].lower()
        if ext in {".json", ".jsonl"}:
            dataset = datasets.load_dataset("json", data_files={train_split: expanded})
        elif ext == ".csv":
            dataset = datasets.load_dataset("csv", data_files={train_split: expanded})
        else:
            raise ValueError(
                "Unsupported local dataset extension. Use .json, .jsonl, or .csv, "
                f"got {ext!r}."
            )
    else:
        dataset = datasets.load_dataset(path)

    if train_split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Missing train split {train_split!r}; available splits: {available}")

    if val_split not in dataset:
        print(
            f"Warning: val split {val_split!r} not found. Reusing {train_split!r} "
            "for validation.",
            flush=True,
        )
        dataset[val_split] = dataset[train_split]

    return dataset[train_split], dataset[val_split]


def _extract_answer(answer: object) -> str:
    if isinstance(answer, Mapping):
        for key in ("answer", "final_answer", "ground_truth", "solution"):
            if key in answer:
                return _extract_answer(answer[key])
    text = str(answer).strip()
    boxed = _last_boxed_only_string(text)
    if boxed:
        return _remove_boxed(boxed)
    number = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return number[-1] if number else text


def _make_map_fn(args, split: str):
    instruction = args.instruction

    def process_fn(example, idx):
        question = example.get(args.question_key)
        if question is None:
            question = example.get("problem", example.get("prompt"))
        if question is None:
            raise KeyError(
                f"Could not find question field {args.question_key!r}; "
                "also tried 'problem' and 'prompt'."
            )

        answer = example.get(args.answer_key)
        if answer is None:
            answer = example.get("solution", example.get("ground_truth"))
        if answer is None:
            raise KeyError(
                f"Could not find answer field {args.answer_key!r}; "
                "also tried 'solution' and 'ground_truth'."
            )

        content = str(question).strip()
        if instruction:
            content = f"{content} {instruction}"

        return {
            "data_source": args.data_source,
            "prompt": [{"role": "user", "content": content}],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": _extract_answer(answer),
            },
            "extra_info": {"split": split, "index": idx},
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", required=True)
    parser.add_argument("--local_save_dir", default="~/data/aime2026")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="test")
    parser.add_argument("--question_key", default="question")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--data_source", default="aime2026")
    parser.add_argument(
        "--instruction",
        default="Let's think step by step and output the final answer within \\boxed{}.",
    )
    args = parser.parse_args()

    train_dataset, val_dataset = _load_dataset(
        args.local_dataset_path, args.train_split, args.val_split
    )
    train_dataset = train_dataset.map(
        function=_make_map_fn(args, "train"),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        function=_make_map_fn(args, "test"),
        with_indices=True,
        remove_columns=val_dataset.column_names,
    )

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"Wrote {local_dir}/train.parquet and {local_dir}/test.parquet", flush=True)


if __name__ == "__main__":
    main()
