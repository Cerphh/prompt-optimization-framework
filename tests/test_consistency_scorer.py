from framework.consistency_scorer import ConsistencyScorer


def test_consistency_is_provisional_for_first_run():
    scorer = ConsistencyScorer()
    normalized = [scorer.normalize_output("The answer is 2")]

    state = scorer.compute_consistency(normalized)

    assert state["is_provisional"] is True
    assert state["value"] is None
    assert state["runs_used"] == 1
    assert state["matching_runs"] is None


def test_consistency_treats_numeric_equivalents_as_identical():
    scorer = ConsistencyScorer()
    outputs = [
        scorer.normalize_output("Ans: 2/1."),
        scorer.normalize_output("The answer is 2"),
        scorer.normalize_output("result = 02.0"),
    ]

    state = scorer.compute_consistency(outputs)

    assert state["is_provisional"] is False
    assert state["value"] == 1.0
    assert state["matching_runs"] == 3


def test_consistency_treats_fraction_decimal_equivalence_as_identical():
    scorer = ConsistencyScorer()
    outputs = [
        scorer.normalize_output("0.5"),
        scorer.normalize_output("1/2"),
    ]

    state = scorer.compute_consistency(outputs)

    assert state["value"] == 1.0


def test_consistency_treats_symbolic_equivalents_as_identical():
    scorer = ConsistencyScorer()
    outputs = [
        scorer.normalize_output("The answer is n(n-1)/2."),
        scorer.normalize_output("Ans: n*(n-1)/2"),
    ]

    state = scorer.compute_consistency(outputs)

    assert state["value"] == 1.0


def test_consistency_uses_majority_match_ratio():
    scorer = ConsistencyScorer()
    outputs = [
        scorer.normalize_output("2"),
        scorer.normalize_output("The answer is 2."),
        scorer.normalize_output("3"),
    ]

    state = scorer.compute_consistency(outputs)

    assert state["value"] == 2 / 3
    assert state["matching_runs"] == 2
