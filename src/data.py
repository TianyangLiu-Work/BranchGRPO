from datasets import load_dataset
from typing import List, Dict


MATH500_INDICES = None


def load_math500_indices():
    global MATH500_INDICES
    if MATH500_INDICES is not None:
        return MATH500_INDICES
    try:
        ds = load_dataset("hendrycks/competition_math", "all", split="test")
        MATH500_INDICES = list(range(min(500, len(ds))))
    except Exception:
        MATH500_INDICES = list(range(500))
    return MATH500_INDICES


def load_train_data(config, num_prompts: int = 500) -> List[Dict[str, str]]:
    ds = load_dataset(
        config.data.train_dataset,
        config.data.train_subset,
        split=config.data.train_split,
    )
    indices = list(range(min(num_prompts, len(ds))))
    ds = ds.select(indices)
    prompts = []
    for item in ds:
        problem = item["problem"]
        prompt = config.data.prompt_template.format(problem=problem)
        prompts.append({"prompt": prompt, "problem": problem, "answer": item.get("solution", "")})
    return prompts


def load_val_data(config, num_samples: int = 500) -> List[Dict[str, str]]:
    ds = load_dataset(
        config.data.val_dataset,
        config.data.val_subset,
        split=config.data.val_split,
    )
    load_math500_indices()
    indices = MATH500_INDICES[:num_samples]
    ds = ds.select(indices)
    prompts = []
    for item in ds:
        problem = item["problem"]
        prompt = config.data.prompt_template.format(problem=problem)
        prompts.append({"prompt": prompt, "problem": problem, "answer": item.get("solution", "")})
    return prompts
