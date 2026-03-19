import json

from fastapi.testclient import TestClient

import main


def _mock_benchmark_result(problem: str, ground_truth: str | None = None):
    zero_shot_result = {
        "technique": "zero_shot",
        "success": True,
        "prompt": "Solve step-by-step, be concise.\n\n" + problem,
        "response": "The answer is 4.",
        "metrics": {
            "elapsed_time": 0.5,
            "total_tokens": 24,
            "prompt_tokens": 12,
            "completion_tokens": 12,
        },
        "scores": {
            "accuracy": 1.0,
            "completeness": 0.7,
            "efficiency": 0.9,
            "overall": 0.89,
        },
    }

    few_shot_result = {
        "technique": "few_shot",
        "success": True,
        "prompt": "Q: 1+1\nA: 2\n\nQ: " + problem + "\nA:",
        "response": "Final answer: 4",
        "metrics": {
            "elapsed_time": 0.6,
            "total_tokens": 30,
            "prompt_tokens": 16,
            "completion_tokens": 14,
        },
        "scores": {
            "accuracy": 1.0,
            "completeness": 0.8,
            "efficiency": 0.8,
            "overall": 0.9,
        },
    }

    return {
        "problem": problem,
        "ground_truth": ground_truth,
        "all_results": {
            "zero_shot": zero_shot_result,
            "few_shot": few_shot_result,
        },
        "best_technique": "few_shot",
        "best_result": few_shot_result,
        "comparison": [
            {
                "technique": "few_shot",
                "accuracy": 1.0,
                "completeness": 0.8,
                "efficiency": 0.8,
                "overall": 0.9,
            },
            {
                "technique": "zero_shot",
                "accuracy": 1.0,
                "completeness": 0.7,
                "efficiency": 0.9,
                "overall": 0.89,
            },
        ],
        "weights": {
            "accuracy": 0.5,
            "completeness": 0.3,
            "efficiency": 0.2,
        },
    }


def _mock_storage(*args, **kwargs):
    return {
        "success": True,
        "collection": "benchmark_results",
        "domain": "algebra",
        "difficulty": "basic",
        "document_id": "doc_123",
    }


def _mock_no_history(*args, **kwargs):
    return {
        "success": False,
        "reason": "no_data",
        "error": "No historical data",
    }


def test_benchmark_response_does_not_include_storage_until_manual_save(monkeypatch):
    def fake_benchmark(problem: str, ground_truth=None, subject: str = "general"):
        return _mock_benchmark_result(problem=problem, ground_truth=ground_truth)

    monkeypatch.setattr(main.pipeline, "benchmark", fake_benchmark)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_profile", _mock_no_history)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_domain", _mock_no_history)

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={"problem": "What is 2 + 2?", "subject": "algebra", "difficulty": "basic"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "storage" not in payload


def test_stream_complete_event_does_not_include_storage_until_manual_save(monkeypatch):
    def fake_stream_events(problem: str, ground_truth=None, subject: str = "general"):
        yield {
            "type": "complete",
            "result": _mock_benchmark_result(problem=problem, ground_truth=ground_truth),
        }

    monkeypatch.setattr(main.pipeline, "benchmark_stream_events", fake_stream_events)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_profile", _mock_no_history)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_domain", _mock_no_history)

    client = TestClient(main.app)
    response = client.post(
        "/benchmark/stream",
        json={"problem": "What is 2 + 2?", "subject": "algebra", "difficulty": "basic"},
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events
    assert events[-1]["type"] == "complete"
    assert "storage" not in events[-1]["result"]


def test_benchmark_tolerates_invalid_db_policy_env_values(monkeypatch):
    def fake_benchmark(problem: str, ground_truth=None, subject: str = "general"):
        return _mock_benchmark_result(problem=problem, ground_truth=ground_truth)

    monkeypatch.setattr(main.pipeline, "benchmark", fake_benchmark)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_profile", _mock_no_history)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_domain", _mock_no_history)

    monkeypatch.setenv("DB_MIN_SAMPLES_PER_TECHNIQUE", "invalid")
    monkeypatch.setenv("DB_MIN_AVG_SCORE_GAP", "invalid")
    monkeypatch.setenv("DB_EXPLORATION_RATE", "2.5")

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={"problem": "What is 2 + 2?", "subject": "algebra", "difficulty": "basic"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "storage" not in payload


def test_manual_save_persists_result(monkeypatch):
    monkeypatch.setattr(main.firestore_store, "save_benchmark_result", _mock_storage)

    client = TestClient(main.app)
    response = client.post(
        "/results/save",
        json={
            "result": _mock_benchmark_result(problem="What is 2 + 2?"),
            "source": "frontend_manual_save",
            "metadata": {"subject": "algebra", "difficulty": "basic"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["storage"]["success"] is True
    assert payload["storage"]["document_id"] == "doc_123"
