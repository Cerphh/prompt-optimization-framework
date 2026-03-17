"""
Pipeline Module
Research benchmarking pipeline for comparative prompt evaluation.

Implements:
1. Generate multiple prompts using different techniques
2. Execute each prompt independently through the model
3. Evaluate responses using multiple metrics
4. Use greedy algorithm to select optimal prompt
5. Return comprehensive results and recommendations
"""

from typing import Dict, List, Any, Optional
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prompt_generator import PromptGenerator
from .model_runner import ModelRunner
from .accuracy_scorer import AccuracyScorer
from .completeness_scorer import CompletenessScorer
from .efficiency_scorer import EfficiencyScorer

class BenchmarkPipeline:
    """
    Main pipeline for research-based prompt optimization.
    
    Evaluates multiple prompting techniques on the same problem
    and selects the optimal approach using a greedy algorithm.
    """
    
    def __init__(self, model_name: str = "llama3",
                 base_url: str = "http://127.0.0.1:11434",
                 accuracy_weight: float = 0.5,
                 completeness_weight: float = 0.3,
                 efficiency_weight: float = 0.2):
        """
        Initialize the benchmarking pipeline.
        
        Args:
            model_name: Name of the LLM to use
            base_url: Base URL for Ollama API
            accuracy_weight: Weight for accuracy score (0-1)
            completeness_weight: Weight for completeness score (0-1)
            efficiency_weight: Weight for efficiency score (0-1)
        """
        self.prompt_generator = PromptGenerator()
        self.model_runner = ModelRunner(model_name=model_name, base_url=base_url)
        self.accuracy_scorer = AccuracyScorer()
        self.completeness_scorer = CompletenessScorer()
        self.efficiency_scorer = EfficiencyScorer()
        
        # Metric weights for scoring
        self.weights = {
            'accuracy': accuracy_weight,
            'completeness': completeness_weight,
            'efficiency': efficiency_weight
        }
        self.weights = self._normalize_weights(self.weights)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize metric weights to sum to 1.0."""
        sanitized: Dict[str, float] = {}
        for name, value in weights.items():
            if value is None:
                raise ValueError(f"Weight '{name}' cannot be None")

            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Weight '{name}' must be a finite number")
            if numeric_value < 0:
                raise ValueError(f"Weight '{name}' must be >= 0")

            sanitized[name] = numeric_value

        total = sum(sanitized.values())
        if total <= 0:
            raise ValueError("At least one metric weight must be greater than 0")

        return {name: value / total for name, value in sanitized.items()}

    def _build_failed_result(self, technique_name: str, prompt: str, error: str) -> Dict[str, Any]:
        """Build a uniform failed technique payload."""
        return {
            "technique": technique_name,
            "success": False,
            "error": error,
            "prompt": prompt,
            "scores": {
                "accuracy": 0.0,
                "completeness": 0.0,
                "efficiency": 0.0,
                "overall": 0.0,
            },
        }
    
    def benchmark(self, problem: str, ground_truth: str = None, subject: str = "general") -> Dict[str, Any]:
        """
        Run comprehensive benchmark on a single problem.
        
        Evaluates all prompting techniques IN PARALLEL for faster results.
        
        Args:
            problem: The math problem to solve
            ground_truth: Expected answer for accuracy evaluation
            subject: Subject category for few-shot examples (algebra, statistics (Counting & Probability), calculus (Pre-calculus), general)
            
        Returns:
            Dictionary with results for all techniques and best selection
        """
        # Step 1: Generate prompts using all techniques
        prompts = self.prompt_generator.generate_all_techniques(problem, subject=subject)
        if not prompts:
            raise ValueError("No prompting techniques available.")
        
        # Step 2: Evaluate each technique IN PARALLEL
        results = {}
        with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as executor:
            # Submit all tasks
            future_to_technique = {
                executor.submit(
                    self._evaluate_single_prompt,
                    technique_name=technique_name,
                    prompt=prompt,
                    problem=problem,
                    ground_truth=ground_truth
                ): technique_name
                for technique_name, prompt in prompts.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_technique):
                technique_name = future_to_technique[future]
                try:
                    result = future.result()
                    results[technique_name] = result
                except Exception as e:
                    # Handle individual technique failure
                    results[technique_name] = self._build_failed_result(
                        technique_name=technique_name,
                        prompt=prompts[technique_name],
                        error=str(e),
                    )
        
        # Step 3: Greedy selection - pick best performing technique
        best_technique = self._greedy_select(results, problem)
        
        # Step 4: Compile comprehensive results
        return {
            "problem": problem,
            "ground_truth": ground_truth,
            "all_results": results,
            "best_technique": best_technique,
            "best_result": results[best_technique],
            "comparison": self._generate_comparison(results),
            "weights": self.weights
        }

    def benchmark_stream_events(
        self,
        problem: str,
        ground_truth: str = None,
        subject: str = "general",
    ):
        """
        Stream benchmark progress and partial output events.

        Emits event dicts:
        - status: progress updates
        - token: incremental response text for the preview technique
        - complete: final benchmark payload
        - error: terminal error event
        """
        prompts = self.prompt_generator.generate_all_techniques(problem, subject=subject)
        if not prompts:
            yield {"type": "error", "error": "No prompting techniques available."}
            return

        techniques = sorted(prompts.keys())
        preview_technique = techniques[0]
        remaining_techniques = [t for t in techniques if t != preview_technique]

        yield {
            "type": "status",
            "message": f"Streaming response for {preview_technique}...",
            "technique": preview_technique,
        }

        results: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max(1, len(remaining_techniques))) as executor:
            future_to_technique = {
                executor.submit(
                    self._evaluate_single_prompt,
                    technique_name=technique_name,
                    prompt=prompts[technique_name],
                    problem=problem,
                    ground_truth=ground_truth,
                ): technique_name
                for technique_name in remaining_techniques
            }

            model_result = None
            for event in self.model_runner.run_stream(prompts[preview_technique]):
                event_type = event.get("type")
                if event_type == "token":
                    yield {
                        "type": "token",
                        "technique": preview_technique,
                        "content": event.get("content", ""),
                    }
                elif event_type == "done":
                    model_result = event.get("result")
                elif event_type == "error":
                    model_result = {
                        "response": "",
                        "success": False,
                        "error": event.get("error", "Unknown stream error"),
                        "metrics": {
                            "elapsed_time": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }

            if not model_result or not model_result.get("success", False):
                results[preview_technique] = self._build_failed_result(
                    technique_name=preview_technique,
                    prompt=prompts[preview_technique],
                    error=(model_result or {}).get("error", "Streaming finished without a final model result."),
                )
            else:
                results[preview_technique] = self._build_scored_result(
                    technique_name=preview_technique,
                    prompt=prompts[preview_technique],
                    model_result=model_result,
                    problem=problem,
                    ground_truth=ground_truth,
                )

            yield {
                "type": "status",
                "message": "Finalizing remaining techniques...",
            }

            for future in as_completed(future_to_technique):
                technique_name = future_to_technique[future]
                try:
                    results[technique_name] = future.result()
                except Exception as e:
                    results[technique_name] = self._build_failed_result(
                        technique_name=technique_name,
                        prompt=prompts[technique_name],
                        error=str(e),
                    )

        best_technique = self._greedy_select(results, problem)
        final_result = {
            "problem": problem,
            "ground_truth": ground_truth,
            "all_results": results,
            "best_technique": best_technique,
            "best_result": results[best_technique],
            "comparison": self._generate_comparison(results),
            "weights": self.weights,
        }
        yield {"type": "complete", "result": final_result}
    
    def _evaluate_single_prompt(self, technique_name: str, prompt: str,
                                problem: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Evaluate a single prompt through the complete pipeline.
        
        Args:
            technique_name: Name of the prompting technique
            prompt: The generated prompt
            problem: Original problem
            ground_truth: Expected answer
            
        Returns:
            Evaluation results with scores and metadata
        """
        # Run prompt through model
        model_result = self.model_runner.run(prompt)
        
        if not model_result["success"]:
            return self._build_failed_result(
                technique_name=technique_name,
                prompt=prompt,
                error=model_result.get("error", "Unknown error"),
            )
        
        return self._build_scored_result(
            technique_name=technique_name,
            prompt=prompt,
            model_result=model_result,
            problem=problem,
            ground_truth=ground_truth,
        )

    def _build_scored_result(
        self,
        technique_name: str,
        prompt: str,
        model_result: Dict[str, Any],
        problem: str,
        ground_truth: str = None,
    ) -> Dict[str, Any]:
        response = model_result["response"]
        metrics = model_result["metrics"]

        accuracy = self.accuracy_scorer.score(response, ground_truth, problem)
        completeness = self.completeness_scorer.score(response, problem)
        efficiency = self.efficiency_scorer.score(response, metrics)

        overall = (
            accuracy * self.weights['accuracy'] +
            completeness * self.weights['completeness'] +
            efficiency * self.weights['efficiency']
        )

        return {
            "technique": technique_name,
            "success": True,
            "prompt": prompt,
            "response": response,
            "metrics": metrics,
            "scores": {
                "accuracy": round(accuracy, 3),
                "completeness": round(completeness, 3),
                "efficiency": round(efficiency, 3),
                "overall": round(overall, 3)
            }
        }
    
    def _greedy_select(self, results: Dict[str, Dict], problem: str) -> str:
        """
        Greedy algorithm to select best performing technique.
        
        Selects the technique with the highest overall score.
        Deterministic tie-breaking ensures same problem always picks same technique.
        
        Args:
            results: Dictionary of technique results
            problem: The problem text (for deterministic tie-breaking)
            
        Returns:
            Name of the best technique
        """
        if not results:
            raise ValueError("No technique results available for selection.")

        # Sort techniques deterministically by name for consistent iteration
        sorted_techniques = sorted(results.items(), key=lambda x: x[0])
        
        best_technique = None
        best_score = -1
        
        for technique_name, result in sorted_techniques:
            if not result.get("success", False):
                continue
            
            scores = result["scores"]
            overall = scores["overall"]
            
            # Pick the technique with highest score
            if overall > best_score:
                best_technique = technique_name
                best_score = overall
            elif overall == best_score:
                # Exact tie - deterministic hash (stable across process restarts)
                problem_hash = hashlib.sha256(f"{problem}|{technique_name}".encode("utf-8")).hexdigest()
                current_hash = hashlib.sha256(f"{problem}|{best_technique}".encode("utf-8")).hexdigest()
                if problem_hash > current_hash:
                    best_technique = technique_name
        
        return best_technique if best_technique else sorted_techniques[0][0]
    
    def _generate_comparison(self, results: Dict[str, Dict]) -> List[Dict]:
        """
        Generate comparison table of all techniques.
        
        Args:
            results: Dictionary of technique results
            
        Returns:
            List of comparison records sorted by overall score
        """
        comparison = []
        
        for technique_name, result in results.items():
            if result.get("success", False):
                scores = result["scores"]
                comparison.append({
                    "technique": technique_name,
                    "accuracy": scores["accuracy"],
                    "completeness": scores["completeness"],
                    "efficiency": scores["efficiency"],
                    "overall": scores["overall"],
                    "latency": result["metrics"].get("elapsed_time", 0),
                    "tokens": result["metrics"].get("total_tokens", 0)
                })
        
        # Sort by overall score (descending)
        comparison.sort(key=lambda x: x["overall"], reverse=True)
        
        return comparison
    
    def test_connection(self) -> bool:
        """Test if the model is accessible."""
        return self.model_runner.test_connection()
    
    def set_weights(self, accuracy: float = None, 
                   completeness: float = None, 
                   efficiency: float = None):
        """
        Update metric weights.
        
        Args:
            accuracy: New accuracy weight
            completeness: New completeness weight
            efficiency: New efficiency weight
        """
        new_weights = dict(self.weights)

        if accuracy is not None:
            new_weights['accuracy'] = accuracy
        if completeness is not None:
            new_weights['completeness'] = completeness
        if efficiency is not None:
            new_weights['efficiency'] = efficiency

        self.weights = self._normalize_weights(new_weights)


# Legacy compatibility wrapper
class OptimizationPipeline(BenchmarkPipeline):
    """
    Legacy wrapper for backward compatibility.
    
    Provides the old interface while using the new benchmark pipeline.
    """
    
    def run(self, query: str, task_type: str = "general", expected: any = None) -> dict:
        """
        Legacy run method for backward compatibility.
        
        Args:
            query: User's query
            task_type: Ignored (always treats as math)
            expected: Expected answer
            
        Returns:
            Simplified results in old format
        """
        result = self.benchmark(problem=query, ground_truth=expected)
        
        if not result["best_result"]["success"]:
            return {
                "success": False,
                "error": result["best_result"].get("error", "Unknown error"),
                "query": query
            }
        
        best = result["best_result"]
        
        return {
            "success": True,
            "query": query,
            "prompt": best["prompt"],
            "response": best["response"],
            "elapsed_time": best["metrics"].get("elapsed_time", 0),
            "scores": best["scores"]
        }
