#!/usr/bin/env python3
"""Prepare MATH-style data in VeRL parquet format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from branch_grpo.data import build_records, load_json_examples, split_examples, write_json_reference


def _limit_dataset(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit < 0:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _load_hf_split(dataset_name: str, subset: str | None, split: str) -> Any:
    if subset:
        return load_dataset(dataset_name, subset, split=split)
    return load_dataset(dataset_name, split=split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/math")
    parser.add_argument("--dataset", default="DigitalLearningGmbH/MATH-lighteval")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--local-json", default=None)
    parser.add_argument("--local-val-json", default=None)
    parser.add_argument("--data-source", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_source = args.data_source or ("math_smoketest" if args.local_json else args.dataset)

    if args.local_json:
        examples = load_json_examples(args.local_json)
        if args.local_val_json:
            train_examples = examples[: args.train_limit] if args.train_limit else examples
            val_examples = load_json_examples(args.local_val_json)
            val_examples = val_examples[: args.val_limit] if args.val_limit else val_examples
        else:
            train_examples, val_examples = split_examples(examples, args.train_limit, args.val_limit)
    else:
        train_examples = list(
            _limit_dataset(_load_hf_split(args.dataset, args.subset, args.train_split), args.train_limit)
        )
        val_examples = list(
            _limit_dataset(_load_hf_split(args.dataset, args.subset, args.val_split), args.val_limit)
        )

    train_records = build_records(train_examples, "train", data_source)
    val_records = build_records(val_examples, "test", data_source)

    Dataset.from_list(train_records).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(val_records).to_parquet(str(output_dir / "test.parquet"))
    write_json_reference(output_dir / "train_example.json", train_records[0])
    write_json_reference(output_dir / "test_example.json", val_records[0])

    print(f"Wrote {len(train_records)} train rows to {output_dir / 'train.parquet'}")
    print(f"Wrote {len(val_records)} validation rows to {output_dir / 'test.parquet'}")


if __name__ == "__main__":
    main()
