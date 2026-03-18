import requests

from framework.model_runner import ModelRunner


MEMORY_ERROR = "model requires more system memory (4.6 GiB) than is available (3.8 GiB)"
CUDA_ERROR = "llama runner process has terminated: CUDA error"


class FakeErrorResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.payload = payload
        self.text = payload.get("error", "")

    def raise_for_status(self):
        raise requests.HTTPError("boom", response=self)

    def json(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSuccessJsonResponse:
    def __init__(self, payload: dict):
        self.payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSuccessStreamResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        _ = decode_unicode
        import json

        for chunk in self.chunks:
            yield json.dumps(chunk)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    def __init__(self, responses=None):
        self.responses = list(responses or [])

    def post(self, url, json=None, stream=False, timeout=None):
        _ = (url, json, stream, timeout)
        if not self.responses:
            raise AssertionError("No responses left for fake session")
        return self.responses.pop(0)

    def get(self, url, timeout=None):
        _ = (url, timeout)
        return FakeTagsResponse(["llama3:latest"])


class FakeFallbackSession:
    def __init__(self):
        self.post_calls = 0

    def post(self, url, json=None, stream=False, timeout=None):
        _ = (url, timeout)
        self.post_calls += 1
        model = (json or {}).get("model")

        if model == "llama3":
            return FakeErrorResponse(500, {"error": MEMORY_ERROR})

        if model == "phi3:mini":
            if stream:
                return FakeSuccessStreamResponse(
                    [
                        {"response": "fallback ", "done": False},
                        {
                            "response": "works",
                            "done": True,
                            "done_reason": "stop",
                            "prompt_eval_count": 3,
                            "eval_count": 4,
                            "load_duration": 0,
                            "prompt_eval_duration": 0,
                            "eval_duration": 0,
                        },
                    ]
                )

            return FakeSuccessJsonResponse(
                {
                    "response": "Final Answer: 4",
                    "done_reason": "stop",
                    "prompt_eval_count": 3,
                    "eval_count": 4,
                    "load_duration": 0,
                    "prompt_eval_duration": 0,
                    "eval_duration": 0,
                }
            )

        raise AssertionError(f"Unexpected model requested: {model}")

    def get(self, url, timeout=None):
        _ = (url, timeout)
        return FakeTagsResponse(["llama3:latest", "phi3:mini"])


class FakeCudaRecoverySession:
    def __init__(self):
        self.post_calls = 0
        self.last_num_gpu = None

    def post(self, url, json=None, stream=False, timeout=None):
        _ = (url, timeout)
        self.post_calls += 1

        if self.post_calls == 1:
            return FakeErrorResponse(500, {"error": CUDA_ERROR})

        self.last_num_gpu = (json or {}).get("options", {}).get("num_gpu")

        if stream:
            return FakeSuccessStreamResponse(
                [
                    {"response": "cpu ", "done": False},
                    {
                        "response": "fallback",
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 3,
                        "eval_count": 4,
                        "load_duration": 0,
                        "prompt_eval_duration": 0,
                        "eval_duration": 0,
                    },
                ]
            )

        return FakeSuccessJsonResponse(
            {
                "response": "Final Answer: 4",
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 4,
                "load_duration": 0,
                "prompt_eval_duration": 0,
                "eval_duration": 0,
            }
        )

    def get(self, url, timeout=None):
        _ = (url, timeout)
        return FakeTagsResponse(["llama3:latest"])


class FakeTagsResponse:
    def __init__(self, model_names):
        self.status_code = 200
        self._model_names = model_names

    def json(self):
        return {"models": [{"name": name} for name in self._model_names]}


def test_run_surfaces_insufficient_memory_error():
    runner = ModelRunner(model_name="llama3")
    runner.session = FakeSession([FakeErrorResponse(500, {"error": MEMORY_ERROR})])

    result = runner.run("Solve 2 + 2")

    assert result["success"] is False
    assert "cannot start in Ollama" in result["error"]
    assert MEMORY_ERROR in result["error"]
    assert "switch to a smaller model" in result["error"]


def test_run_stream_surfaces_insufficient_memory_error():
    runner = ModelRunner(model_name="llama3")
    runner.session = FakeSession([FakeErrorResponse(500, {"error": MEMORY_ERROR})])

    events = list(runner.run_stream("Solve 2 + 2"))

    assert events == [
        {
            "type": "error",
            "error": (
                "Model 'llama3' cannot start in Ollama: "
                f"{MEMORY_ERROR}. "
                "Free RAM, close other applications, reduce MODEL_NUM_CTX, or switch to a smaller model."
            ),
        }
    ]


def test_run_surfaces_cuda_error_guidance_when_cpu_fallback_disabled():
    runner = ModelRunner(model_name="llama3")
    runner.auto_cpu_fallback_on_cuda_error = False
    runner.session = FakeSession([FakeErrorResponse(500, {"error": CUDA_ERROR})])

    result = runner.run("Solve 2 + 2")

    assert result["success"] is False
    assert "failed to initialize CUDA" in result["error"]
    assert "MODEL_NUM_GPU=0" in result["error"]


def test_run_retries_in_cpu_mode_on_cuda_error():
    runner = ModelRunner(model_name="llama3")
    session = FakeCudaRecoverySession()
    runner.session = session
    runner.verifier_retry_enabled = False

    result = runner.run("What is 2 + 2?")

    assert result["success"] is True
    assert result["response"] == "Final Answer: 4"
    assert session.last_num_gpu == 0
    assert result["metrics"]["cpu_fallback_applied"] is True


def test_run_stream_retries_in_cpu_mode_on_cuda_error():
    runner = ModelRunner(model_name="llama3")
    session = FakeCudaRecoverySession()
    runner.session = session

    events = list(runner.run_stream("What is 2 + 2?"))

    assert any(event.get("technique") == "cpu_fallback" for event in events)
    done_events = [event for event in events if event.get("type") == "done"]
    assert done_events
    assert done_events[-1]["result"]["success"] is True
    assert "cpu fallback" in done_events[-1]["result"]["response"]
    assert session.last_num_gpu == 0


def test_validate_model_ready_reports_generation_failure():
    runner = ModelRunner(model_name="llama3")
    runner.session = FakeSession([FakeErrorResponse(500, {"error": MEMORY_ERROR})])

    ready, error = runner.validate_model_ready()

    assert ready is False
    assert error is not None
    assert MEMORY_ERROR in error


def test_run_switches_to_fallback_model_on_memory_error(monkeypatch):
    monkeypatch.setenv("MODEL_FALLBACK_MODELS", "phi3:mini")
    monkeypatch.setenv("MODEL_AUTO_FALLBACK_ON_MEMORY_ERROR", "true")

    runner = ModelRunner(model_name="llama3")
    runner.session = FakeFallbackSession()
    runner.verifier_retry_enabled = False

    result = runner.run("What is 2 + 2?")

    assert result["success"] is True
    assert result["response"] == "Final Answer: 4"
    assert result["model"] == "phi3:mini"
    assert runner.model_name == "phi3:mini"


def test_run_stream_switches_to_fallback_model_on_memory_error(monkeypatch):
    monkeypatch.setenv("MODEL_FALLBACK_MODELS", "phi3:mini")
    monkeypatch.setenv("MODEL_AUTO_FALLBACK_ON_MEMORY_ERROR", "true")

    runner = ModelRunner(model_name="llama3")
    runner.session = FakeFallbackSession()

    events = list(runner.run_stream("What is 2 + 2?"))

    assert any(event.get("type") == "status" for event in events)
    done_events = [event for event in events if event.get("type") == "done"]
    assert done_events
    assert done_events[-1]["result"]["model"] == "phi3:mini"
    assert "fallback works" in done_events[-1]["result"]["response"]