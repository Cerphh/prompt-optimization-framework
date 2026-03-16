from framework.model_runner import ModelRunner


class FakeJsonResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, non_stream_payloads=None):
        self.non_stream_payloads = list(non_stream_payloads or [])

    def post(self, url, json=None, stream=False, timeout=None):
        _ = (url, json, stream, timeout)
        if not self.non_stream_payloads:
            raise AssertionError("No non-stream payloads left for test session")
        return FakeJsonResponse(self.non_stream_payloads.pop(0))


def _payload(response_text: str, done_reason: str = "stop"):
    return {
        "response": response_text,
        "done_reason": done_reason,
        "prompt_eval_count": 5,
        "eval_count": 50,
        "load_duration": 100_000_000,
        "prompt_eval_duration": 100_000_000,
        "eval_duration": 200_000_000,
    }


def test_run_retries_once_when_verifier_fails():
    initial = _payload("Steps but no final sentence")
    verifier = _payload("VERDICT: FAIL")
    retry = _payload("Recomputed carefully.\nFinal Answer: 12")

    runner = ModelRunner(model_name="llama3")
    runner.session = FakeSession(non_stream_payloads=[initial, verifier, retry])
    runner.auto_continue_on_length = False
    runner.max_continue_rounds = 0
    runner.verifier_retry_enabled = True
    runner.verifier_retry_attempts = 1

    result = runner.run("Solve 7 + 5")

    assert result["success"] is True
    assert result["response"].endswith("Final Answer: 12")
    assert result["metrics"]["verifier_retry_applied"] is True
    assert result["metrics"]["verifier_verdict"] == "fail"


def test_run_skips_retry_when_verifier_passes_and_answer_is_clear():
    initial = _payload("Quick check.\nFinal Answer: 4")
    verifier = _payload("VERDICT: PASS")

    runner = ModelRunner(model_name="llama3")
    runner.session = FakeSession(non_stream_payloads=[initial, verifier])
    runner.auto_continue_on_length = False
    runner.max_continue_rounds = 0
    runner.verifier_retry_enabled = True
    runner.verifier_retry_attempts = 1

    result = runner.run("What is 2 + 2?")

    assert result["success"] is True
    assert result["response"].endswith("Final Answer: 4")
    assert result["metrics"]["verifier_retry_applied"] is False
    assert result["metrics"]["verifier_verdict"] == "pass"
