"""
Model Runner Module
Handles communication with language models (Ollama) and tracks execution metrics.
"""

import json
import os
import re
import requests
import time
from typing import Dict, Any, Generator, Optional, Tuple


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int, min_value: Optional[int] = None) -> int:
    raw_value = os.getenv(name)
    try:
        parsed = int(raw_value) if raw_value is not None else default
    except (TypeError, ValueError):
        parsed = default

    if min_value is not None and parsed < min_value:
        return min_value
    return parsed


def _env_float(name: str, default: float, min_value: Optional[float] = None) -> float:
    raw_value = os.getenv(name)
    try:
        parsed = float(raw_value) if raw_value is not None else default
    except (TypeError, ValueError):
        parsed = default

    if min_value is not None and parsed < min_value:
        return min_value
    return parsed

class ModelRunner:
    """
    Runs prompts through language models and returns responses with metrics.
    
    Tracks:
    - Response text
    - Token usage (prompt tokens, completion tokens, total tokens)
    - Latency (time to first token, total time)
    - Model metadata
    """
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://127.0.0.1:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

        self.auto_continue_on_length = _env_bool("MODEL_AUTO_CONTINUE_ON_LENGTH", True)
        self.max_continue_rounds = _env_int("MODEL_MAX_CONTINUE_ROUNDS", 4, min_value=0)
        self.continuation_tail_chars = _env_int("MODEL_CONTINUATION_TAIL_CHARS", 1200, min_value=200)

        self.verifier_retry_enabled = _env_bool("MODEL_VERIFIER_RETRY_ENABLED", True)
        self.verifier_retry_attempts = _env_int("MODEL_VERIFIER_RETRY_ATTEMPTS", 1, min_value=0)
        self.verifier_num_predict = _env_int("MODEL_VERIFIER_NUM_PREDICT", 256, min_value=64)
        self.verifier_min_chars = _env_int("MODEL_VERIFIER_MIN_CHARS", 48, min_value=10)
        self.verifier_require_final_answer = _env_bool("MODEL_VERIFIER_REQUIRE_FINAL_ANSWER", True)

        self.generation_options = {
            "temperature": _env_float("MODEL_TEMPERATURE", 0.0, min_value=0.0),
            "seed": _env_int("MODEL_SEED", 42),
            "num_predict": _env_int("MODEL_NUM_PREDICT", 2048, min_value=64),
            "num_ctx": _env_int("MODEL_NUM_CTX", 8192, min_value=512),
            "top_k": _env_int("MODEL_TOP_K", 10, min_value=1),
            "top_p": _env_float("MODEL_TOP_P", 0.9, min_value=0.0),
            "repeat_penalty": _env_float("MODEL_REPEAT_PENALTY", 1.1, min_value=0.1),
        }

    def _build_payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "5m"),
            "options": self.generation_options,
        }

    def _merge_with_overlap(self, existing_text: str, new_text: str) -> str:
        """Merge text chunks while removing duplicated overlap at boundaries."""
        if not existing_text:
            return new_text or ""
        if not new_text:
            return existing_text

        max_overlap = min(len(existing_text), len(new_text), 500)
        for overlap in range(max_overlap, 0, -1):
            if existing_text.endswith(new_text[:overlap]):
                return existing_text + new_text[overlap:]

        return existing_text + new_text

    def _build_continuation_prompt(self, original_prompt: str, partial_response: str) -> str:
        """Create a continuation prompt when generation is cut by token length."""
        tail = partial_response[-self.continuation_tail_chars:] if partial_response else ""
        return (
            f"{original_prompt}\n\n"
            "Your previous response was cut due token limit. "
            "Continue exactly where it stopped. Do not restart or repeat solved steps. "
            "Provide only the continuation and finish with a final answer if needed.\n\n"
            f"Last generated text:\n{tail}\n\n"
            "Continue:"
        )

    def _continue_generation(self, original_prompt: str, partial_response: str, done_reason: str) -> Dict[str, Any]:
        """Continue generation in additional rounds while model reports length truncation."""
        combined_response = partial_response
        continuation_rounds = 0
        continuation_error = None

        aggregate_metrics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "load_time": 0.0,
            "prompt_eval_time": 0.0,
            "eval_time": 0.0,
        }

        current_done_reason = done_reason

        while (
            self.auto_continue_on_length
            and current_done_reason == "length"
            and continuation_rounds < self.max_continue_rounds
        ):
            continuation_rounds += 1
            continuation_prompt = self._build_continuation_prompt(original_prompt, combined_response)

            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=self._build_payload(prompt=continuation_prompt, stream=False),
                    timeout=(5, 180),
                )
                response.raise_for_status()
                continuation_result = response.json()
            except Exception as exc:
                continuation_error = str(exc)
                break

            continuation_text = continuation_result.get("response", "")
            combined_response = self._merge_with_overlap(combined_response, continuation_text)

            prompt_eval_count = continuation_result.get("prompt_eval_count", 0)
            eval_count = continuation_result.get("eval_count", 0)

            aggregate_metrics["prompt_tokens"] += prompt_eval_count
            aggregate_metrics["completion_tokens"] += eval_count
            aggregate_metrics["total_tokens"] += prompt_eval_count + eval_count
            aggregate_metrics["load_time"] += continuation_result.get("load_duration", 0) / 1e9
            aggregate_metrics["prompt_eval_time"] += continuation_result.get("prompt_eval_duration", 0) / 1e9
            aggregate_metrics["eval_time"] += continuation_result.get("eval_duration", 0) / 1e9

            current_done_reason = continuation_result.get("done_reason", "stop")
            if not continuation_text:
                break

        return {
            "response": combined_response,
            "done_reason": current_done_reason,
            "continuation_rounds": continuation_rounds,
            "continuation_error": continuation_error,
            "metrics": aggregate_metrics,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "load_time": 0.0,
            "prompt_eval_time": 0.0,
            "eval_time": 0.0,
        }

    def _safe_int(self, value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _extract_metrics(self, ollama_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt_tokens = self._safe_int(ollama_result.get("prompt_eval_count", 0))
        completion_tokens = self._safe_int(ollama_result.get("eval_count", 0))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "load_time": float(ollama_result.get("load_duration", 0) or 0) / 1e9,
            "prompt_eval_time": float(ollama_result.get("prompt_eval_duration", 0) or 0) / 1e9,
            "eval_time": float(ollama_result.get("eval_duration", 0) or 0) / 1e9,
        }

    def _combine_metrics(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        target["prompt_tokens"] += int(source.get("prompt_tokens", 0) or 0)
        target["completion_tokens"] += int(source.get("completion_tokens", 0) or 0)
        target["total_tokens"] = target["prompt_tokens"] + target["completion_tokens"]
        target["load_time"] += float(source.get("load_time", 0.0) or 0.0)
        target["prompt_eval_time"] += float(source.get("prompt_eval_time", 0.0) or 0.0)
        target["eval_time"] += float(source.get("eval_time", 0.0) or 0.0)

    def _run_generation_once(self, prompt: str, timeout: Tuple[int, int] = (5, 180)) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=self._build_payload(prompt=prompt, stream=False),
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()

        response_text = result.get("response", "")
        done_reason = result.get("done_reason", "stop")
        metrics = self._extract_metrics(result)
        continuation_rounds = 0
        continuation_error = None

        if done_reason == "length" and self.auto_continue_on_length and self.max_continue_rounds > 0:
            continuation = self._continue_generation(
                original_prompt=prompt,
                partial_response=response_text,
                done_reason=done_reason,
            )
            response_text = continuation["response"]
            done_reason = continuation["done_reason"]
            continuation_rounds = continuation["continuation_rounds"]
            continuation_error = continuation["continuation_error"]
            self._combine_metrics(metrics, continuation["metrics"])

        return {
            "response": response_text,
            "done_reason": done_reason,
            "continuation_rounds": continuation_rounds,
            "continuation_error": continuation_error,
            "metrics": metrics,
        }

    def _has_final_answer(self, response_text: str) -> bool:
        if re.search(r"(?im)^\s*final\s+answer\s*[:=-]", response_text):
            return True

        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        if not lines:
            return False

        last_line = lines[-1]
        if re.search(r"(?i)^\s*answer\s*[:=-]", last_line):
            return True
        if re.search(r"(?i)\b([a-z]\s*=\s*)?[-+]?\d+(?:\.\d+)?(?:/\d+)?\b", last_line):
            return True
        return False

    def _response_needs_retry(self, response_text: str, done_reason: str) -> bool:
        normalized = (response_text or "").strip()
        if not normalized:
            return True
        if done_reason == "length":
            return True
        if len(normalized) < self.verifier_min_chars:
            return True
        if re.search(r"(?i)\b(i am not sure|cannot determine|insufficient information)\b", normalized):
            return True
        if self.verifier_require_final_answer and not self._has_final_answer(normalized):
            return True
        return False

    def _run_verifier_check(self, original_prompt: str, candidate_response: str) -> Tuple[Optional[bool], Dict[str, Any], Optional[str]]:
        verifier_prompt = (
            "You are a strict verifier for a math solution.\n"
            "Determine if the candidate answer is complete, self-consistent, and ends with a clear final answer.\n"
            "Return exactly one line: VERDICT: PASS or VERDICT: FAIL.\n\n"
            f"Problem:\n{original_prompt}\n\n"
            f"Candidate answer:\n{candidate_response}\n"
        )

        options = dict(self.generation_options)
        options["temperature"] = 0.0
        options["num_predict"] = self.verifier_num_predict

        payload = self._build_payload(prompt=verifier_prompt, stream=False)
        payload["options"] = options

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(5, 120),
            )
            response.raise_for_status()
            verifier_result = response.json()
            verdict_text = (verifier_result.get("response", "") or "").strip().lower()

            verdict: Optional[bool] = None
            if "verdict: pass" in verdict_text or verdict_text.startswith("pass"):
                verdict = True
            elif "verdict: fail" in verdict_text or verdict_text.startswith("fail"):
                verdict = False

            return verdict, self._extract_metrics(verifier_result), None
        except Exception as exc:
            return None, self._empty_metrics(), str(exc)

    def _build_retry_prompt(self, original_prompt: str, candidate_response: str) -> str:
        return (
            "Rewrite this math solution to ensure it is correct and complete.\n"
            "Keep steps concise and internally consistent.\n"
            "End with exactly one final line in this format: Final Answer: <answer>.\n\n"
            f"Problem:\n{original_prompt}\n\n"
            f"Previous answer:\n{candidate_response}\n"
        )

    def _maybe_retry_with_verifier(
        self,
        original_prompt: str,
        generation: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        verifier_state = {
            "heuristic_weak": False,
            "verdict": "disabled",
            "retry_applied": False,
            "verifier_error": None,
            "retry_error": None,
        }
        overhead_metrics = self._empty_metrics()

        if not self.verifier_retry_enabled or self.verifier_retry_attempts <= 0:
            return generation, overhead_metrics, verifier_state

        verifier_state["heuristic_weak"] = self._response_needs_retry(
            response_text=generation.get("response", ""),
            done_reason=generation.get("done_reason", "stop"),
        )

        verifier_verdict, verifier_metrics, verifier_error = self._run_verifier_check(
            original_prompt=original_prompt,
            candidate_response=generation.get("response", ""),
        )
        self._combine_metrics(overhead_metrics, verifier_metrics)
        verifier_state["verifier_error"] = verifier_error

        if verifier_verdict is True:
            verifier_state["verdict"] = "pass"
        elif verifier_verdict is False:
            verifier_state["verdict"] = "fail"
        else:
            verifier_state["verdict"] = "unknown"

        should_retry = verifier_state["heuristic_weak"] or verifier_verdict is False
        if not should_retry:
            return generation, overhead_metrics, verifier_state

        for _ in range(self.verifier_retry_attempts):
            retry_prompt = self._build_retry_prompt(
                original_prompt=original_prompt,
                candidate_response=generation.get("response", ""),
            )
            try:
                regenerated = self._run_generation_once(retry_prompt, timeout=(5, 240))
            except Exception as exc:
                verifier_state["retry_error"] = str(exc)
                break

            self._combine_metrics(overhead_metrics, regenerated.get("metrics", self._empty_metrics()))

            if (regenerated.get("response", "") or "").strip():
                generation = regenerated
                verifier_state["retry_applied"] = True
                break

        return generation, overhead_metrics, verifier_state
    
    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Run a prompt through the language model with detailed metrics.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Dictionary with response, metrics, and metadata
        """
        start_time = time.time()
        
        try:
            generation = self._run_generation_once(prompt=prompt, timeout=(5, 180))
            generation, overhead_metrics, verifier_state = self._maybe_retry_with_verifier(
                original_prompt=prompt,
                generation=generation,
            )

            end_time = time.time()
            merged_metrics = dict(generation["metrics"])
            self._combine_metrics(merged_metrics, overhead_metrics)
            done_reason = generation.get("done_reason", "stop")
            
            result_dict = {
                "response": generation.get("response", ""),
                "model": self.model_name,
                "success": True,
                "metrics": {
                    "prompt_tokens": merged_metrics["prompt_tokens"],
                    "completion_tokens": merged_metrics["completion_tokens"],
                    "total_tokens": merged_metrics["total_tokens"],
                    "elapsed_time": end_time - start_time,
                    "load_time": merged_metrics["load_time"],
                    "prompt_eval_time": merged_metrics["prompt_eval_time"],
                    "eval_time": merged_metrics["eval_time"],
                    "done_reason": done_reason,
                    "truncated": done_reason == "length",
                    "continuation_rounds": generation.get("continuation_rounds", 0),
                    "continuation_error": generation.get("continuation_error"),
                    "verifier_retry_applied": verifier_state["retry_applied"],
                    "verifier_verdict": verifier_state["verdict"],
                    "verifier_heuristic_weak": verifier_state["heuristic_weak"],
                    "verifier_error": verifier_state["verifier_error"],
                    "verifier_retry_error": verifier_state["retry_error"],
                }
            }
            
            return result_dict
        except requests.HTTPError as e:
            end_time = time.time()
            status_code = e.response.status_code if e.response is not None else None
            response_text = ""
            response_error = ""

            if e.response is not None:
                response_text = (e.response.text or "").strip()
                try:
                    response_error = (e.response.json() or {}).get("error", "")
                except Exception:
                    response_error = ""

            if status_code == 404:
                combined_error = (response_error or response_text).lower()
                if "model" in combined_error and "not found" in combined_error:
                    error_message = (
                        f"Model '{self.model_name}' not found in Ollama. "
                        f"Run: ollama pull {self.model_name}"
                    )
                else:
                    error_message = (
                        f"Ollama endpoint '/api/generate' not found at {self.base_url}. "
                        "Verify Ollama is running on this URL and supports the Ollama API."
                    )
            else:
                detail = response_error or response_text or str(e)
                error_message = f"Ollama request failed ({status_code}): {detail}"

            return {
                "response": "",
                "model": self.model_name,
                "success": False,
                "error": error_message,
                "metrics": {
                    "elapsed_time": end_time - start_time,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        except Exception as e:
            end_time = time.time()
            return {
                "response": "",
                "model": self.model_name,
                "success": False,
                "error": str(e),
                "metrics": {
                    "elapsed_time": end_time - start_time,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    def run_stream(self, prompt: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tokens from Ollama and emit structured events.

        Yields event dictionaries with one of these types:
        - token: incremental text chunk
        - done: final aggregated model result
        - error: stream failure
        """
        start_time = time.time()
        response_parts = []

        try:
            with self.session.post(
                f"{self.base_url}/api/generate",
                json=self._build_payload(prompt=prompt, stream=True),
                stream=True,
                timeout=(5, 300),
            ) as response:
                response.raise_for_status()

                final_chunk: Dict[str, Any] = {}
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    chunk = json.loads(raw_line)
                    text_piece = chunk.get("response", "")
                    if text_piece:
                        response_parts.append(text_piece)
                        yield {"type": "token", "content": text_piece}

                    if chunk.get("done", False):
                        final_chunk = chunk
                        break

            response_text = "".join(response_parts)
            done_reason = final_chunk.get("done_reason", "stop")
            prompt_tokens = final_chunk.get("prompt_eval_count", 0)
            completion_tokens = final_chunk.get("eval_count", 0)
            load_time = final_chunk.get("load_duration", 0) / 1e9
            prompt_eval_time = final_chunk.get("prompt_eval_duration", 0) / 1e9
            eval_time = final_chunk.get("eval_duration", 0) / 1e9
            continuation_rounds = 0
            continuation_error = None

            if done_reason == "length" and self.auto_continue_on_length and self.max_continue_rounds > 0:
                continuation = self._continue_generation(
                    original_prompt=prompt,
                    partial_response=response_text,
                    done_reason=done_reason,
                )

                extended_response = continuation["response"]
                if extended_response.startswith(response_text):
                    additional_text = extended_response[len(response_text):]
                    if additional_text:
                        yield {"type": "token", "content": additional_text}

                response_text = extended_response
                done_reason = continuation["done_reason"]
                continuation_rounds = continuation["continuation_rounds"]
                continuation_error = continuation["continuation_error"]

                continuation_metrics = continuation["metrics"]
                prompt_tokens += continuation_metrics["prompt_tokens"]
                completion_tokens += continuation_metrics["completion_tokens"]
                load_time += continuation_metrics["load_time"]
                prompt_eval_time += continuation_metrics["prompt_eval_time"]
                eval_time += continuation_metrics["eval_time"]

            end_time = time.time()
            result = {
                "response": response_text,
                "model": self.model_name,
                "success": True,
                "metrics": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "elapsed_time": end_time - start_time,
                    "load_time": load_time,
                    "prompt_eval_time": prompt_eval_time,
                    "eval_time": eval_time,
                    "done_reason": done_reason,
                    "truncated": done_reason == "length",
                    "continuation_rounds": continuation_rounds,
                    "continuation_error": continuation_error,
                    "verifier_retry_applied": False,
                    "verifier_verdict": "skipped_stream",
                    "verifier_heuristic_weak": False,
                    "verifier_error": None,
                    "verifier_retry_error": None,
                },
            }
            yield {"type": "done", "result": result}
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            response_text = ""
            response_error = ""

            if e.response is not None:
                response_text = (e.response.text or "").strip()
                try:
                    response_error = (e.response.json() or {}).get("error", "")
                except Exception:
                    response_error = ""

            if status_code == 404:
                combined_error = (response_error or response_text).lower()
                if "model" in combined_error and "not found" in combined_error:
                    detail = (
                        f"Model '{self.model_name}' not found in Ollama. "
                        f"Run: ollama pull {self.model_name}"
                    )
                else:
                    detail = (
                        f"Ollama endpoint '/api/generate' not found at {self.base_url}. "
                        "Verify Ollama is running on this URL and supports the Ollama API."
                    )
            else:
                detail = response_error or response_text or str(e)

            error_message = f"Ollama stream failed ({status_code}): {detail}"
            # Ensure error message is never empty
            if not error_message.strip():
                error_message = "Unknown streaming error from Ollama"

            yield {
                "type": "error",
                "error": error_message,
            }
        except Exception as e:
            error_message = str(e) if str(e).strip() else "Unknown streaming error"
            yield {
                "type": "error",
                "error": error_message,
            }
    
    def test_connection(self) -> bool:
        """Test if the model is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
