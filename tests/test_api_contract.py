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
            "consistency": 0.7,
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
            "consistency": 0.8,
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
                "consistency": 0.8,
                "efficiency": 0.8,
                "overall": 0.9,
            },
            {
                "technique": "zero_shot",
                "accuracy": 1.0,
                "consistency": 0.7,
                "efficiency": 0.9,
                "overall": 0.89,
            },
        ],
        "weights": {
            "accuracy": 0.5,
            "consistency": 0.3,
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
            "metadata": {"subject": "algebra", "difficulty": "basic", "run_mode": "benchmark"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["storage"]["success"] is True
    assert payload["storage"]["document_id"] == "doc_123"


def test_manual_save_rejects_out_of_scope_payload(monkeypatch):
    called = {"save_called": False}

    def _should_not_save(*args, **kwargs):
        called["save_called"] = True
        return {"success": True}

    monkeypatch.setattr(main.firestore_store, "save_benchmark_result", _should_not_save)

    client = TestClient(main.app)
    response = client.post(
        "/results/save",
        json={
            "result": {
                **_mock_benchmark_result(problem="Who wrote Hamlet?"),
                "run_mode": "benchmark",
            },
            "source": "frontend_manual_save",
            "metadata": {"subject": "history", "difficulty": "basic", "run_mode": "benchmark"},
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]
    assert called["save_called"] is False


def test_benchmark_skips_weakest_technique_when_history_has_15_samples(monkeypatch):
    captured = {}

    def fake_benchmark(problem: str, ground_truth=None, subject: str = "general", techniques_to_run=None):
        captured["techniques_to_run"] = techniques_to_run
        result = _mock_benchmark_result(problem=problem, ground_truth=ground_truth)
        if techniques_to_run:
            allowed = set(techniques_to_run)
            result["all_results"] = {
                technique: payload
                for technique, payload in result["all_results"].items()
                if technique in allowed
            }
            result["comparison"] = [
                record for record in result["comparison"] if record["technique"] in allowed
            ]
            best = next(iter(result["all_results"]))
            result["best_technique"] = best
            result["best_result"] = result["all_results"][best]
        return result

    monkeypatch.setattr(main.pipeline, "benchmark", fake_benchmark)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_profile", _mock_no_history)
    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_domain",
        lambda *args, **kwargs: {
            "success": True,
            "best_technique": "few_shot",
            "ranking": [
                {"technique": "few_shot", "average_overall": 0.91, "samples": 15},
                {"technique": "zero_shot", "average_overall": 0.82, "samples": 15},
            ],
        },
    )

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={"problem": "What is 2 + 2?", "subject": "algebra", "difficulty": "basic"},
    )

    assert response.status_code == 200
    assert captured["techniques_to_run"] == ["few_shot"]
    payload = response.json()
    assert payload["pre_execution_policy"]["skipped_technique"] == "zero_shot"


def test_benchmark_does_not_skip_weakest_technique_when_samples_below_min(monkeypatch):
    captured = {}

    def fake_benchmark(problem: str, ground_truth=None, subject: str = "general", techniques_to_run=None):
        captured["techniques_to_run"] = techniques_to_run
        return _mock_benchmark_result(problem=problem, ground_truth=ground_truth)

    monkeypatch.setattr(main.pipeline, "benchmark", fake_benchmark)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_profile", _mock_no_history)
    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_domain",
        lambda *args, **kwargs: {
            "success": True,
            "best_technique": "few_shot",
            "ranking": [
                {"technique": "few_shot", "average_overall": 0.91, "samples": 2},
                {"technique": "zero_shot", "average_overall": 0.82, "samples": 2},
            ],
        },
    )

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={"problem": "What is 2 + 2?", "subject": "algebra", "difficulty": "basic"},
    )

    assert response.status_code == 200
    assert captured["techniques_to_run"] is None
    payload = response.json()
    assert payload["pre_execution_policy"]["reason"] == "insufficient_samples"


def test_benchmark_mode_requires_ground_truth(monkeypatch):
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "What is 2 + 2?",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "benchmark",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert "ground_truth" in payload["detail"]


def test_benchmark_stream_mode_requires_ground_truth(monkeypatch):
    client = TestClient(main.app)
    response = client.post(
        "/benchmark/stream",
        json={
            "problem": "What is 2 + 2?",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "benchmark",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert "ground_truth" in payload["detail"]


def test_benchmark_mode_runs_all_techniques_ignoring_history(monkeypatch):
    captured = {}

    def fake_benchmark(problem: str, ground_truth=None, subject: str = "general", techniques_to_run=None):
        captured["techniques_to_run"] = techniques_to_run
        return _mock_benchmark_result(problem=problem, ground_truth=ground_truth)

    monkeypatch.setattr(main.pipeline, "benchmark", fake_benchmark)
    monkeypatch.setattr(
        main.firestore_store,
        "get_best_technique_by_domain",
        lambda *args, **kwargs: {
            "success": True,
            "best_technique": "few_shot",
            "ranking": [
                {"technique": "few_shot", "average_overall": 0.91, "samples": 25},
                {"technique": "zero_shot", "average_overall": 0.82, "samples": 25},
            ],
        },
    )
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_profile", _mock_no_history)

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "What is 2 + 2?",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "benchmark",
            "ground_truth": "4",
        },
    )

    assert response.status_code == 200
    # Benchmark mode must run ALL techniques (no preselection)
    assert captured["techniques_to_run"] is None
    payload = response.json()
    assert payload["run_mode"] == "benchmark"
    assert payload["pre_execution_policy"]["reason"] == "benchmark_mode_runs_all"


def test_normal_mode_rejects_explicit_unsupported_subject():
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "Who wrote Hamlet?",
            "subject": "history",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_normal_mode_rejects_non_math_prompt_even_with_supported_subject(monkeypatch):
    monkeypatch.setattr(main.pipeline.prompt_generator, "classify_subject", lambda _: "general")

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "What is life",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_normal_mode_rejects_non_math_prompt_even_with_supported_subject_without_mocking_classifier():
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "What is life",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_normal_mode_rejects_out_of_domain_math_even_with_supported_subject():
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "Find the area of a triangle with base 10 and height 6.",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_normal_mode_rejects_ladder_geometry_word_problem_with_supported_subject():
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": (
                "A 15-meter ladder leans against a wall. The foot of the ladder is 9 meters "
                "away from the wall. Find the angle theta that the ladder makes with the ground. "
                "Find the height at which the ladder touches the wall."
            ),
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_normal_mode_rejects_subject_mismatch_for_supported_domains(monkeypatch):
    monkeypatch.setattr(main.pipeline.prompt_generator, "classify_subject", lambda _: "counting-probability")

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "A fair die is rolled twice. What is the probability the sum is 7?",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_normal_mode_rejects_subject_mismatch_from_signal_when_classifier_is_general(monkeypatch):
    monkeypatch.setattr(main.pipeline.prompt_generator, "classify_subject", lambda _: "general")

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "A fair die is rolled twice. What is the probability that the sum is 7?",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_normal_mode_allows_supported_subject_when_classifier_is_general_but_signal_is_clear(monkeypatch):
    monkeypatch.setattr(main.pipeline.prompt_generator, "classify_subject", lambda _: "general")
    monkeypatch.setattr(main.pipeline, "benchmark", lambda **kwargs: _mock_benchmark_result(problem=kwargs["problem"], ground_truth=kwargs.get("ground_truth")))
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_profile", _mock_no_history)
    monkeypatch.setattr(main.firestore_store, "get_best_technique_by_domain", _mock_no_history)

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "Solve for x: 2x + 3 = 11",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_mode"] == "normal"
    assert payload["best_technique"] in {"few_shot", "zero_shot"}


def test_normal_mode_rejects_out_of_scope_problem_when_subject_is_general(monkeypatch):
    monkeypatch.setattr(main.pipeline.prompt_generator, "classify_subject", lambda _: "general")

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "Who wrote Hamlet?",
            "subject": "general",
            "difficulty": "basic",
            "run_mode": "normal",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_benchmark_mode_rejects_explicit_unsupported_subject_even_with_ground_truth():
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "Who wrote Hamlet?",
            "subject": "history",
            "difficulty": "basic",
            "run_mode": "benchmark",
            "ground_truth": "Shakespeare",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_benchmark_mode_rejects_non_math_prompt_even_with_supported_subject(monkeypatch):
    monkeypatch.setattr(main.pipeline.prompt_generator, "classify_subject", lambda _: "general")

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "What is life",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "benchmark",
            "ground_truth": "life",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_benchmark_mode_rejects_non_math_prompt_even_with_supported_subject_without_mocking_classifier():
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "What is life",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "benchmark",
            "ground_truth": "life",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_benchmark_mode_rejects_out_of_domain_math_even_with_supported_subject():
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "Find the area of a triangle with base 10 and height 6.",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "benchmark",
            "ground_truth": "30",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_benchmark_mode_rejects_ladder_geometry_word_problem_with_supported_subject():
    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": (
                "A 15-meter ladder leans against a wall. The foot of the ladder is 9 meters "
                "away from the wall. Find the angle theta that the ladder makes with the ground. "
                "Find the height at which the ladder touches the wall."
            ),
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "benchmark",
            "ground_truth": "12 meters",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_benchmark_stream_rejects_out_of_scope_problem_when_subject_is_general(monkeypatch):
    monkeypatch.setattr(main.pipeline.prompt_generator, "classify_subject", lambda _: "general")

    client = TestClient(main.app)
    response = client.post(
        "/benchmark/stream",
        json={
            "problem": "Who wrote Hamlet?",
            "subject": "general",
            "difficulty": "basic",
            "run_mode": "benchmark",
            "ground_truth": "Shakespeare",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "only supports the following 3 domains" in payload["detail"]


def test_benchmark_mode_accepts_single_technique_when_preselected(monkeypatch):
    def fake_benchmark(problem: str, ground_truth=None, subject: str = "general", techniques_to_run=None):
        result = _mock_benchmark_result(problem=problem, ground_truth=ground_truth)
        result["all_results"] = {
            "zero_shot": result["all_results"]["zero_shot"],
        }
        result["comparison"] = [
            {
                "technique": "zero_shot",
                "accuracy": 1.0,
                "consistency": 0.7,
                "efficiency": 0.9,
                "overall": 0.89,
            }
        ]
        result["best_technique"] = "zero_shot"
        result["best_result"] = result["all_results"]["zero_shot"]
        return result

    monkeypatch.setattr(main.pipeline, "benchmark", fake_benchmark)

    client = TestClient(main.app)
    response = client.post(
        "/benchmark",
        json={
            "problem": "What is 2 + 2?",
            "subject": "algebra",
            "difficulty": "basic",
            "run_mode": "benchmark",
            "ground_truth": "4",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_mode"] == "benchmark"
