"""
Model Runner Module
Handles communication with language models (Ollama) and tracks execution metrics.
"""

import json
import os
import requests
import time
from typing import Dict, Any, Generator

class ModelRunner:
    """
    Runs prompts through language models and returns responses with metrics.
    
    Tracks:
    - Response text
    - Token usage (prompt tokens, completion tokens, total tokens)
    - Latency (time to first token, total time)
    - Model metadata
    """
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
        self.auto_continue_on_length = os.getenv("MODEL_AUTO_CONTINUE_ON_LENGTH", "true").lower() == "true"
        self.max_continue_rounds = max(0, int(os.getenv("MODEL_MAX_CONTINUE_ROUNDS", "2")))
        self.continuation_tail_chars = max(200, int(os.getenv("MODEL_CONTINUATION_TAIL_CHARS", "1200")))
        self.generation_options = {
            "temperature": float(os.getenv("MODEL_TEMPERATURE", "0")),
            "seed": int(os.getenv("MODEL_SEED", "42")),
            "num_predict": int(os.getenv("MODEL_NUM_PREDICT", "1024")),
            "num_ctx": int(os.getenv("MODEL_NUM_CTX", "2048")),
            "top_k": int(os.getenv("MODEL_TOP_K", "10")),
            "top_p": float(os.getenv("MODEL_TOP_P", "0.9")),
            "repeat_penalty": float(os.getenv("MODEL_REPEAT_PENALTY", "1.1")),
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
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=self._build_payload(prompt=prompt, stream=False),
                timeout=(5, 180)
            )
            response.raise_for_status()
            result = response.json()

            # Extract metrics from Ollama response
            response_text = result.get("response", "")
            done_reason = result.get("done_reason", "stop")
            
            # Token counts (if available in response)
            prompt_eval_count = result.get("prompt_eval_count", 0)
            eval_count = result.get("eval_count", 0)
            
            # Timing metrics
            load_duration = result.get("load_duration", 0) / 1e9  # Convert to seconds
            prompt_eval_duration = result.get("prompt_eval_duration", 0) / 1e9
            eval_duration = result.get("eval_duration", 0) / 1e9

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

                continuation_metrics = continuation["metrics"]
                prompt_eval_count += continuation_metrics["prompt_tokens"]
                eval_count += continuation_metrics["completion_tokens"]
                load_duration += continuation_metrics["load_time"]
                prompt_eval_duration += continuation_metrics["prompt_eval_time"]
                eval_duration += continuation_metrics["eval_time"]

            end_time = time.time()
            elapsed_time = end_time - start_time
            total_tokens = prompt_eval_count + eval_count
            truncated = done_reason == "length"
            
            result_dict = {
                "response": response_text,
                "model": self.model_name,
                "success": True,
                "metrics": {
                    "prompt_tokens": prompt_eval_count,
                    "completion_tokens": eval_count,
                    "total_tokens": total_tokens,
                    "elapsed_time": elapsed_time,
                    "load_time": load_duration,
                    "prompt_eval_time": prompt_eval_duration,
                    "eval_time": eval_duration,
                    "done_reason": done_reason,
                    "truncated": truncated,
                    "continuation_rounds": continuation_rounds,
                    "continuation_error": continuation_error,
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

            detail = response_error or response_text or str(e)
            yield {
                "type": "error",
                "error": f"Ollama stream failed ({status_code}): {detail}",
            }
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
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
