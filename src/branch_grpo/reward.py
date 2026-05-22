"""Rule-based math reward for VeRL.

VeRL loads this function through custom_reward_function.path/name. The signature
matches VeRL custom rewards while staying usable in local tests.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


def _last_boxed(text: str) -> str | None:
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None

    brace_start = text.find("{", idx)
    if brace_start < 0:
        match = re.search(r"\\(?:boxed|fbox)\s+([^\s$]+)", text[idx:])
        return match.group(1).strip() if match else None

    depth = 0
    for pos in range(brace_start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : pos].strip()
    return None


def extract_answer(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, dict):
        text = text.get("ground_truth") or text.get("answer") or text.get("solution") or ""
    text = str(text)
    boxed = _last_boxed(text)
    return boxed if boxed is not None else text.strip()


def normalize_answer(answer: Any) -> str:
    text = extract_answer(answer).lower().strip()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "$": "",
        "%": "",
        ",": "",
        " ": "",
        "\n": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=", 1)[1]
    return text


def exact_match_reward(solution_str: Any, ground_truth: Any) -> float:
    return float(normalize_answer(solution_str) == normalize_answer(ground_truth))


def _numeric_value(answer: Any) -> Fraction | None:
    text = extract_answer(answer).lower().strip()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
        "$": "",
        ",": "",
        " ": "",
        "\n": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.rstrip(".")

    frac_match = re.fullmatch(r"\\frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}", text)
    if frac_match:
        numerator, denominator = frac_match.groups()
        denominator_value = Fraction(denominator)
        if denominator_value == 0:
            return None
        return Fraction(numerator) / denominator_value

    slash_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", text)
    if slash_match:
        numerator, denominator = slash_match.groups()
        denominator_value = Fraction(denominator)
        if denominator_value == 0:
            return None
        return Fraction(numerator) / denominator_value

    decimal_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    if decimal_match:
        return Fraction(text)

    return None


def numeric_match_reward(solution_str: Any, ground_truth: Any) -> float:
    candidate = _numeric_value(solution_str)
    target = _numeric_value(ground_truth)
    if candidate is None or target is None:
        return 0.0
    return float(abs(float(candidate - target)) <= 1e-9)


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """Return 1.0 for a correct final math answer, otherwise 0.0."""
    del data_source, extra_info

    candidate = solution_str or ""
    target = extract_answer(ground_truth)

    try:
        from verl.utils.reward_score.math_reward import compute_score as verl_math_score

        score = float(verl_math_score(candidate, target))
        if score > 0:
            return score
    except Exception:
        pass

    return max(exact_match_reward(candidate, target), numeric_match_reward(candidate, target))
