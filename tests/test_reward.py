from branch_grpo.reward import compute_score, exact_match_reward, extract_answer, normalize_answer


def test_extract_answer_uses_last_boxed_value():
    assert extract_answer(r"first \boxed{3}, final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_exact_match_reward_normalizes_simple_answers():
    assert exact_match_reward(r"The answer is \boxed{x = 3}", r"\boxed{3}") == 1.0
    assert exact_match_reward(r"\boxed{4}", r"\boxed{5}") == 0.0


def test_compute_score_matches_verl_signature():
    score = compute_score(
        data_source="math",
        solution_str=r"We get \boxed{187.5}.",
        ground_truth="187.5",
        extra_info={"index": 0},
    )
    assert score == 1.0


def test_normalize_answer_handles_ground_truth_dict():
    assert normalize_answer({"ground_truth": r"\boxed{42}"}) == "42"

