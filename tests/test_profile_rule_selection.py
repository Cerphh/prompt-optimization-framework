import copy

import main


def _mock_result(best_technique: str = "few_shot"):
    zero_shot_result = {
        "technique": "zero_shot",
        "success": True,
        "prompt": "Zero-shot prompt",
        "response": "Response",
        "metrics": {
            "elapsed_time": 0.5,
            "total_tokens": 20,
            "prompt_tokens": 10,
            "completion_tokens": 10,
        },
        "scores": {
            "accuracy": 0.9,
            "consistency": 0.7,
            "efficiency": 0.8,
            "overall": 0.83,
        },
    }

    few_shot_result = {
        "technique": "few_shot",
        "success": True,
        "prompt": "Few-shot prompt",
        "response": "Response",
        "metrics": {
            "elapsed_time": 0.6,
            "total_tokens": 24,
            "prompt_tokens": 12,
            "completion_tokens": 12,
        },
        "scores": {
            "accuracy": 0.92,
            "consistency": 0.78,
            "efficiency": 0.74,
            "overall": 0.85,
        },
    }

    all_results = {
        "zero_shot": zero_shot_result,
        "few_shot": few_shot_result,
    }

    return {
        "problem": "Solve for all real solutions: x^2 - 9x + 20 = 0.",
        "all_results": all_results,
        "best_technique": best_technique,
        "best_result": all_results[best_technique],
        "comparison": [
            {
                "technique": "few_shot",
                "accuracy": 0.92,
                "consistency": 0.78,
                "efficiency": 0.74,
                "overall": 0.85,
            },
            {
                "technique": "zero_shot",
                "accuracy": 0.9,
                "consistency": 0.7,
                "efficiency": 0.8,
                "overall": 0.83,
            },
        ],
    }


def test_profile_rules_override_domain_average_when_confident(monkeypatch):
    monkeypatch.setenv("DB_EXPLORATION_RATE", "0")
    monkeypatch.setenv("DB_PROFILE_MIN_SAMPLES_PER_TECHNIQUE", "2")
    monkeypatch.setenv("DB_PROFILE_MIN_AVG_SCORE_GAP", "0.01")
    monkeypatch.setenv("DB_PROFILE_MIN_SIMILARITY", "0.2")

    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_profile",
        lambda *args, **kwargs: {
            "success": True,
            "best_technique": "few_shot",
            "ranking": [
                {"technique": "few_shot", "average_overall": 0.91, "samples": 6},
                {"technique": "zero_shot", "average_overall": 0.82, "samples": 6},
            ],
            "match_type": "profile_rule",
        },
    )

    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_domain",
        lambda *args, **kwargs: {
            "success": True,
            "best_technique": "zero_shot",
            "ranking": [
                {"technique": "zero_shot", "average_overall": 0.89, "samples": 10},
                {"technique": "few_shot", "average_overall": 0.86, "samples": 10},
            ],
        },
    )

    result = main._apply_db_based_selection(
        result=copy.deepcopy(_mock_result(best_technique="zero_shot")),
        domain="algebra",
        difficulty="basic",
    )

    assert result["best_technique"] == "few_shot"
    assert result["selection_source"] == "db_profile_rules"
    assert result["selection_details"]["profile_selection"]["best_technique"] == "few_shot"


def test_profile_rules_fallback_to_domain_average_when_profile_low_confidence(monkeypatch):
    monkeypatch.setenv("DB_EXPLORATION_RATE", "0")
    monkeypatch.setenv("DB_MIN_SAMPLES_PER_TECHNIQUE", "10")
    monkeypatch.setenv("DB_PROFILE_MIN_SAMPLES_PER_TECHNIQUE", "3")
    monkeypatch.setenv("DB_PROFILE_MIN_AVG_SCORE_GAP", "0.03")

    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_profile",
        lambda *args, **kwargs: {
            "success": True,
            "best_technique": "few_shot",
            "ranking": [
                {"technique": "few_shot", "average_overall": 0.811, "samples": 6},
                {"technique": "zero_shot", "average_overall": 0.806, "samples": 6},
            ],
        },
    )

    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_domain",
        lambda *args, **kwargs: {
            "success": True,
            "best_technique": "zero_shot",
            "ranking": [
                {"technique": "zero_shot", "average_overall": 0.89, "samples": 12},
                {"technique": "few_shot", "average_overall": 0.83, "samples": 12},
            ],
        },
    )

    result = main._apply_db_based_selection(
        result=copy.deepcopy(_mock_result(best_technique="few_shot")),
        domain="algebra",
        difficulty="basic",
    )

    assert result["best_technique"] == "zero_shot"
    assert result["selection_source"] == "db_history"
    assert result["selection_details"]["profile_decision_reason"] == "low_confidence_gap"


def test_profile_and_domain_no_history_falls_back_to_runtime_scores(monkeypatch):
    monkeypatch.setenv("DB_EXPLORATION_RATE", "0")

    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_profile",
        lambda *args, **kwargs: {
            "success": False,
            "reason": "no_profile_match",
            "error": "No profile data",
        },
    )

    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_domain",
        lambda *args, **kwargs: {
            "success": False,
            "reason": "no_data",
            "error": "No domain data",
        },
    )

    baseline = copy.deepcopy(_mock_result(best_technique="few_shot"))
    result = main._apply_db_based_selection(
        result=baseline,
        domain="algebra",
        difficulty="basic",
    )

    assert result["best_technique"] == "few_shot"
    assert result["selection_source"] == "runtime_scores"
