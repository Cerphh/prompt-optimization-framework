"""
Model Runner Module
Handles communication with language models (Ollama) and tracks execution metrics.
"""

import requests
import time
from typing import Dict, Any

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
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300  # Extended timeout for complex prompts (5 minutes)
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
            
            return {
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
    
    def test_connection(self) -> bool:
        """Test if the model is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
