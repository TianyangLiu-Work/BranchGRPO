from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from typing import List, Dict


MATH_SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]
MATH500_INDICES = None


def _load_math_dataset(dataset_name: str, subset: str, split: str):
    """Load a MATH dataset, handling 'all' subset by concatenating all subsets."""
    if subset == "all":
        all_ds = []
        available = get_dataset_config_names(dataset_name)
        for s in MATH_SUBSETS:
            if s in available:
                ds = load_dataset(dataset_name, s, split=split)
                all_ds.append(ds)
        if not all_ds:
            raise ValueError(f"No MATH subsets found for {dataset_name}")
        return concatenate_datasets(all_ds)
    else:
        return load_dataset(dataset_name, subset, split=split)


def load_math500_indices():
    global MATH500_INDICES
    if MATH500_INDICES is not None:
        return MATH500_INDICES
    try:
        ds = _load_math_dataset("EleutherAI/hendrycks_math", "all", split="test")
        MATH500_INDICES = list(range(min(500, len(ds))))
    except Exception:
        MATH500_INDICES = list(range(500))
    return MATH500_INDICES


def load_train_data(config, num_prompts: int = 500) -> List[Dict[str, str]]:
    ds = _load_math_dataset(
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
    ds = _load_math_dataset(
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
