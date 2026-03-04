import json as jsonlib

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


class FakeStreamResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        for chunk in self.chunks:
            yield jsonlib.dumps(chunk)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    def __init__(self, non_stream_payloads=None, stream_payloads=None):
        self.non_stream_payloads = list(non_stream_payloads or [])
        self.stream_payloads = list(stream_payloads or [])

    def post(self, url, json=None, stream=False, timeout=None):
        if stream:
            if not self.stream_payloads:
                raise AssertionError("No stream payloads left for test session")
            return FakeStreamResponse(self.stream_payloads.pop(0))

        if not self.non_stream_payloads:
            raise AssertionError("No non-stream payloads left for test session")
        return FakeJsonResponse(self.non_stream_payloads.pop(0))


def test_run_auto_continues_when_length_limit_hit():
    first_response = {
        "response": "Step 1. Step 2.",
        "prompt_eval_count": 10,
        "eval_count": 1024,
        "load_duration": 1_000_000_000,
        "prompt_eval_duration": 500_000_000,
        "eval_duration": 2_000_000_000,
        "done_reason": "length",
    }
    continuation_response = {
        "response": " Final answer: 42.",
        "prompt_eval_count": 6,
        "eval_count": 80,
        "load_duration": 200_000_000,
        "prompt_eval_duration": 300_000_000,
        "eval_duration": 600_000_000,
        "done_reason": "stop",
    }

    runner = ModelRunner(model_name="llama3")
    runner.session = FakeSession(non_stream_payloads=[first_response, continuation_response])
    runner.auto_continue_on_length = True
    runner.max_continue_rounds = 1

    result = runner.run("Solve this hard problem")

    assert result["success"] is True
    assert result["response"].endswith("Final answer: 42.")
    assert result["metrics"]["continuation_rounds"] == 1
    assert result["metrics"]["truncated"] is False
    assert result["metrics"]["completion_tokens"] == 1104


def test_run_stream_auto_continues_after_length_limit():
    stream_chunks = [
        {
            "response": "Reasoning starts. ",
            "done": False,
        },
        {
            "response": "It follows that",
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 8,
            "eval_count": 1024,
            "load_duration": 500_000_000,
            "prompt_eval_duration": 200_000_000,
            "eval_duration": 1_200_000_000,
        },
    ]
    continuation_response = {
        "response": " the final answer is 9.",
        "prompt_eval_count": 5,
        "eval_count": 64,
        "load_duration": 100_000_000,
        "prompt_eval_duration": 100_000_000,
        "eval_duration": 300_000_000,
        "done_reason": "stop",
    }

    runner = ModelRunner(model_name="llama3")
    runner.session = FakeSession(
        non_stream_payloads=[continuation_response],
        stream_payloads=[stream_chunks],
    )
    runner.auto_continue_on_length = True
    runner.max_continue_rounds = 1

    events = list(runner.run_stream("Solve this harder problem"))

    done_events = [event for event in events if event.get("type") == "done"]
    token_events = [event for event in events if event.get("type") == "token"]

    assert done_events
    done_result = done_events[-1]["result"]

    assert "final answer is 9" in done_result["response"]
    assert done_result["metrics"]["continuation_rounds"] == 1
    assert done_result["metrics"]["truncated"] is False
    assert any("final answer is 9" in event.get("content", "") for event in token_events)
