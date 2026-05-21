from branch_grpo.data import build_records, split_examples


def test_build_records_uses_verl_schema():
    records = build_records(
        [{"problem": "What is 2+2?", "solution": r"\boxed{4}"}],
        split="train",
        data_source="unit",
    )
    record = records[0]
    assert record["data_source"] == "unit"
    assert record["prompt"][0]["role"] == "user"
    assert record["reward_model"]["style"] == "rule"
    assert record["reward_model"]["ground_truth"] == "4"
    assert record["extra_info"]["split"] == "train"


def test_split_examples_keeps_train_and_validation():
    examples = [{"problem": str(i), "solution": str(i)} for i in range(10)]
    train, val = split_examples(examples, train_limit=4, val_limit=2)
    assert len(train) == 4
    assert len(val) == 2

