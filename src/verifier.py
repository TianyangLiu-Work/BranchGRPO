import re


def extract_boxed_answer(text: str) -> str:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(" ", "")
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    return s


def exact_match_reward(predicted: str, ground_truth: str) -> float:
    pred_ans = extract_boxed_answer(predicted)
    gt_ans = extract_boxed_answer(ground_truth)
    return 1.0 if normalize_answer(pred_ans) == normalize_answer(gt_ans) else 0.0


def compute_rewards(candidate_pool: list, ground_truth: str) -> list:
    rewards = []
    for c in candidate_pool:
        r = exact_match_reward(c["response_text"], ground_truth)
        rewards.append(r)
    return rewards
