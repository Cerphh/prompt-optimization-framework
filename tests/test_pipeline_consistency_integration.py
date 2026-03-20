import pytest

from framework.pipeline import BenchmarkPipeline


def _success_payload(response: str):
    return {
        "success": True,
        "response": response,
        "metrics": {
            "elapsed_time": 1.2,
            "total_tokens": 40,
            "prompt_tokens": 18,
            "completion_tokens": 22,
        },
    }


def test_pipeline_consistency_updates_incrementally_with_all_previous_runs():
    pipeline = BenchmarkPipeline(
        model_name="llama3",
        accuracy_weight=1.0,
        consistency_weight=1.0,
        efficiency_weight=1.0,
        runs_per_technique=3,
    )

    outputs = iter([
        _success_payload("Ans: 1/2"),
        _success_payload("The answer is 0.5"),
        _success_payload("Final answer: 0.6"),
    ])
    pipeline.model_runner.run = lambda prompt: next(outputs)

    result = pipeline._evaluate_technique_runs(
        technique_name="zero_shot",
        prompt="Q: test\nA:",
        problem="What is 1/2?",
        ground_truth="0.5",
        runs_per_technique=3,
    )

    assert result["scores"]["consistency"] == pytest.approx(2 / 3, abs=1e-3)
    assert result["scores"]["consistency_runs_used"] == 3
    assert len(result["run_history"]) == 3
    assert result["run_history"][0]["scores"]["consistency"] is None
    assert result["run_history"][1]["scores"]["consistency"] == 1.0
    assert result["run_history"][2]["scores"]["consistency"] == pytest.approx(2 / 3, abs=1e-3)


def test_pipeline_reports_provisional_consistency_for_single_run():
    pipeline = BenchmarkPipeline(
        model_name="llama3",
        accuracy_weight=1.0,
        consistency_weight=1.0,
        efficiency_weight=1.0,
        runs_per_technique=1,
    )

    pipeline.model_runner.run = lambda prompt: _success_payload("Final answer: 4")

    result = pipeline._evaluate_technique_runs(
        technique_name="zero_shot",
        prompt="Q: 2+2\nA:",
        problem="What is 2+2?",
        ground_truth="4",
        runs_per_technique=1,
    )

    assert result["scores"]["consistency"] is None
    assert result["scores"]["consistency_is_provisional"] is True
    assert result["scores"]["overall_is_provisional"] is True
