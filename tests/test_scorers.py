from framework.accuracy_scorer import AccuracyScorer
from framework.efficiency_scorer import EfficiencyScorer


def test_accuracy_numeric_match_returns_full_score():
    scorer = AccuracyScorer()
    score = scorer.score(response="The answer is 42.", expected="42")
    assert score == 1.0


def test_accuracy_prefers_explicit_final_answer_over_intermediate_values():
    scorer = AccuracyScorer()
    response = (
        "From the setup we get 5c = 15a and therefore c/a = 3. "
        "After substitution, c/b = 1. Final answer: c/b = 1."
    )

    score = scorer.score(response=response, expected="15")
    assert score == 0.0


def test_accuracy_numeric_match_does_not_use_fraction_numerator_only():
    scorer = AccuracyScorer()
    score = scorer.score(response="Result: 15/5", expected="15")
    assert score == 0.0


def test_accuracy_uses_previous_math_line_when_final_answer_payload_is_blank():
    scorer = AccuracyScorer()
    response = (
        "(m - 4)(2n + 7) = m(2n + 7) - 4(2n + 7)\n"
        "= 2mn + 7m - 8n - 28\n"
        "Final answer:"
    )

    score = scorer.score(response=response, expected="2mn + 7m - 8n - 28")
    assert score == 1.0


def test_accuracy_extracts_expression_from_narrative_final_line():
    scorer = AccuracyScorer()
    response = (
        "To expand the expression (m - 4)(2n + 7), I will use distribution:\n"
        "(m - 4)(2n + 7) = m(2n + 7) - 4(2n + 7)\n"
        "= 2mn + 7m - 8n - 28\n"
        "So, the expanded expression is: 2mn + 7m - 8n - 28."
    )

    score = scorer.score(response=response, expected="2mn + 7m - 8n - 28")
    assert score == 1.0


def test_efficiency_score_is_bounded():
    scorer = EfficiencyScorer()
    score = scorer.score(
        response="Compute quickly and return final answer.",
        metrics={
            "elapsed_time": 2.0,
            "total_tokens": 40,
            "prompt_tokens": 14,
            "completion_tokens": 26,
        },
    )
    assert 0.0 <= score <= 1.0


def test_accuracy_multi_value_expected_rejects_partial_single_root_answer():
    scorer = AccuracyScorer()
    expected = "x = 1, x = -1, x = 3, x = -3"
    response = "Final answer: +√3"

    score = scorer.score(response=response, expected=expected)
    assert score == 0.0


def test_accuracy_multi_value_expected_accepts_reordered_complete_set():
    scorer = AccuracyScorer()
    expected = "x = 1, x = -1, x = 3, x = -3"
    response = "Final answer: x = -3, x = 1, x = -1, x = 3"

    score = scorer.score(response=response, expected=expected)
    assert score == 1.0
