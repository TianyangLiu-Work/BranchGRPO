"""Rule-based math reward for BranchGRPO-style parquet datasets."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


def _last_boxed(text: str) -> str | None:
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        match = re.search(r"\\(?:boxed|fbox)\s+([^\s$]+)", text[idx:])
        return match.group(1).strip() if match else None
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:pos].strip()
    return None


def extract_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("ground_truth") or value.get("answer") or value.get(
            "solution", "")
    text = str(value)
    return _last_boxed(text) or text.strip()


def _normalize(value: Any) -> str:
    text = extract_answer(value).lower().strip()
    for old, new in {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "$": "",
        "%": "",
        ",": "",
        " ": "",
        "\n": "",
    }.items():
        text = text.replace(old, new)
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=", 1)[1]
    return text


def _numeric_value(value: Any) -> Fraction | None:
    text = extract_answer(value).lower().strip()
    for old, new in {
        "\\left": "",
        "\\right": "",
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
        "$": "",
        ",": "",
        " ": "",
        "\n": "",
    }.items():
        text = text.replace(old, new)
    text = text.rstrip(".")
    if len(text) > 128:
        return None
    frac = re.fullmatch(
        r"\\frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}",
        text,
    )
    if frac:
        numerator, denominator = frac.groups()
        denominator_value = Fraction(denominator)
        if denominator_value == 0:
            return None
        return Fraction(numerator) / denominator_value
    slash = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", text)
    if slash:
        numerator, denominator = slash.groups()
        denominator_value = Fraction(denominator)
        if denominator_value == 0:
            return None
        return Fraction(numerator) / denominator_value
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return Fraction(text)
    return None


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
        from verl.utils.reward_score.math_reward import compute_score as math_score

        score = float(math_score(candidate, target))
        if score > 0:
            return score
    except Exception:
        pass
    if _normalize(candidate) == _normalize(target):
        return 1.0
    candidate_num = _numeric_value(candidate)
    target_num = _numeric_value(target)
    if candidate_num is None or target_num is None:
        return 0.0
    return float(abs(candidate_num - target_num) <= Fraction(1, 1_000_000_000))
