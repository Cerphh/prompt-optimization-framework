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
        self.generation_options = {
            "temperature": float(os.getenv("MODEL_TEMPERATURE", "0")),
            "seed": int(os.getenv("MODEL_SEED", "42")),
            "num_predict": int(os.getenv("MODEL_NUM_PREDICT", "256")),
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
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Extract metrics from Ollama response
            response_text = result.get("response", "")
            
            # Token counts (if available in response)
            prompt_eval_count = result.get("prompt_eval_count", 0)
            eval_count = result.get("eval_count", 0)
            total_tokens = prompt_eval_count + eval_count
            
            # Timing metrics
            load_duration = result.get("load_duration", 0) / 1e9  # Convert to seconds
            prompt_eval_duration = result.get("prompt_eval_duration", 0) / 1e9
            eval_duration = result.get("eval_duration", 0) / 1e9
            
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
                    "eval_time": eval_duration
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

            end_time = time.time()
            result = {
                "response": "".join(response_parts),
                "model": self.model_name,
                "success": True,
                "metrics": {
                    "prompt_tokens": final_chunk.get("prompt_eval_count", 0),
                    "completion_tokens": final_chunk.get("eval_count", 0),
                    "total_tokens": final_chunk.get("prompt_eval_count", 0)
                    + final_chunk.get("eval_count", 0),
                    "elapsed_time": end_time - start_time,
                    "load_time": final_chunk.get("load_duration", 0) / 1e9,
                    "prompt_eval_time": final_chunk.get("prompt_eval_duration", 0) / 1e9,
                    "eval_time": final_chunk.get("eval_duration", 0) / 1e9,
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
