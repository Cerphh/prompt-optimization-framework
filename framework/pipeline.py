"""
Pipeline Module
Research benchmarking pipeline for comparative prompt evaluation.

Implements:
1. Generate multiple prompts using different techniques
2. Execute each prompt independently through the model (optionally repeated runs)
3. Evaluate responses using multiple metrics
4. Use greedy algorithm to select optimal prompt
5. Return comprehensive results and recommendations
"""

from typing import Dict, List, Any, Optional, Tuple
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prompt_generator import PromptGenerator
from .model_runner import ModelRunner
from .accuracy_scorer import AccuracyScorer
from .consistency_scorer import ConsistencyScorer
from .efficiency_scorer import EfficiencyScorer

class BenchmarkPipeline:
    """
    Main pipeline for research-based prompt optimization.
    
    Evaluates multiple prompting techniques on the same problem
    and selects the optimal approach using a greedy algorithm.
    """
    
    def __init__(self, model_name: str = "llama3",
                 base_url: str = "http://127.0.0.1:11434",
                 accuracy_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 efficiency_weight: float = 1.0,
                 runs_per_technique: int = 3):
        """
        Initialize the benchmarking pipeline.
        
        Args:
            model_name: Name of the LLM to use
            base_url: Base URL for Ollama API
            accuracy_weight: Weight for accuracy score (0-1)
            consistency_weight: Weight for consistency score (0-1)
            efficiency_weight: Weight for efficiency score (0-1)
            runs_per_technique: Number of repeated runs per technique for consistency
        """
        self.prompt_generator = PromptGenerator()
        self.model_runner = ModelRunner(model_name=model_name, base_url=base_url)
        self.accuracy_scorer = AccuracyScorer()
        self.consistency_scorer = ConsistencyScorer()
        self.efficiency_scorer = EfficiencyScorer()
        self.default_runs_per_technique = self._sanitize_runs_per_technique(runs_per_technique)
        
        # Metric weights for scoring
        self.weights = {
            'accuracy': accuracy_weight,
            'consistency': consistency_weight,
            'efficiency': efficiency_weight
        }
        self.weights = self._normalize_weights(self.weights)

    def _sanitize_runs_per_technique(self, runs_per_technique: Any) -> int:
        """Validate and coerce run count to a positive integer."""
        try:
            value = int(runs_per_technique)
        except (TypeError, ValueError):
            raise ValueError("runs_per_technique must be an integer")

        if value < 1:
            raise ValueError("runs_per_technique must be >= 1")

        return value

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

    def _build_failed_result(
        self,
        technique_name: str,
        prompt: str,
        error: str,
        runs_configured: int = 0,
    ) -> Dict[str, Any]:
        """Build a uniform failed technique payload."""
        return {
            "technique": technique_name,
            "success": False,
            "error": error,
            "prompt": prompt,
            "response": "",
            "metrics": {
                "elapsed_time": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "runs_recorded": 0,
                "runs_succeeded": 0,
                "runs_failed": runs_configured,
            },
            "scores": {
                "accuracy": 0.0,
                "consistency": None,
                "efficiency": 0.0,
                "overall": 0.0,
                "consistency_is_provisional": True,
                "consistency_runs_used": 0,
                "consistency_matching_runs": None,
                "overall_is_provisional": True,
                "overall_note": "No successful runs available.",
            },
            "run_history": [],
            "audit": {
                "runs_configured": runs_configured,
                "runs_recorded": 0,
                "normalized_outputs": [],
                "output_counts": {},
                "canonical_output": None,
            },
        }
    
    def benchmark(
        self,
        problem: str,
        ground_truth: str = None,
        subject: str = "general",
        techniques_to_run: Optional[List[str]] = None,
        runs_per_technique: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        resolved_runs = self.default_runs_per_technique
        if runs_per_technique is not None:
            resolved_runs = self._sanitize_runs_per_technique(runs_per_technique)

        # Step 1: Generate prompts using all techniques
        prompts = self.prompt_generator.generate_all_techniques(problem, subject=subject)
        if techniques_to_run:
            allowed = set(techniques_to_run)
            prompts = {
                technique_name: prompt
                for technique_name, prompt in prompts.items()
                if technique_name in allowed
            }
        if not prompts:
            raise ValueError("No prompting techniques available.")
        
        # Step 2: Evaluate each technique IN PARALLEL
        results = {}
        with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as executor:
            # Submit all tasks
            future_to_technique = {
                executor.submit(
                    self._evaluate_technique_runs,
                    technique_name=technique_name,
                    prompt=prompt,
                    problem=problem,
                    ground_truth=ground_truth,
                    runs_per_technique=resolved_runs,
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
                        runs_configured=resolved_runs,
                    )
        
        # Step 3: Greedy selection - pick best performing technique
        best_technique = self._greedy_select(results, problem)
        
        # Step 4: Compile comprehensive results
        return {
            "problem": problem,
            "ground_truth": ground_truth,
            "runs_per_technique": resolved_runs,
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
        techniques_to_run: Optional[List[str]] = None,
        runs_per_technique: Optional[int] = None,
    ):
        """
        Stream benchmark progress and partial output events.

        Emits event dicts:
        - status: progress updates
        - token: incremental response text for the preview technique
        - complete: final benchmark payload
        - error: terminal error event
        """
        resolved_runs = self.default_runs_per_technique
        if runs_per_technique is not None:
            resolved_runs = self._sanitize_runs_per_technique(runs_per_technique)

        prompts = self.prompt_generator.generate_all_techniques(problem, subject=subject)
        if techniques_to_run:
            allowed = set(techniques_to_run)
            prompts = {
                technique_name: prompt
                for technique_name, prompt in prompts.items()
                if technique_name in allowed
            }
        if not prompts:
            yield {"type": "error", "error": "No prompting techniques available."}
            return

        techniques = sorted(prompts.keys())
        preview_technique = techniques[0]
        remaining_techniques = [t for t in techniques if t != preview_technique]

        yield {
            "type": "status",
            "message": (
                f"Streaming run 1/{resolved_runs} for {preview_technique}; "
                "remaining runs and techniques will finalize in background."
            ),
            "technique": preview_technique,
        }

        results: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max(1, len(remaining_techniques))) as executor:
            future_to_technique = {
                executor.submit(
                    self._evaluate_technique_runs,
                    technique_name=technique_name,
                    prompt=prompts[technique_name],
                    problem=problem,
                    ground_truth=ground_truth,
                    runs_per_technique=resolved_runs,
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
                    error_msg = event.get("error", "").strip()
                    if not error_msg:
                        error_msg = "Unknown streaming error from model runner"
                    model_result = {
                        "response": "",
                        "success": False,
                        "error": error_msg,
                        "metrics": {
                            "elapsed_time": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }

            if not model_result or not model_result.get("success", False):
                model_result = {
                    "response": "",
                    "success": False,
                    "error": (model_result or {}).get(
                        "error",
                        "Streaming finished without a final model result.",
                    ),
                    "metrics": {
                        "elapsed_time": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }

            # The streaming path skips the verifier retry.  Apply it now
            # so that run 1 has the same quality-check opportunity as the
            # non-streamed runs 2–N, preventing a systematic run-1 miss.
            if model_result.get("success", False):
                preview_prompt = prompts[preview_technique]
                generation_for_verify = {
                    "response": model_result.get("response", ""),
                    "done_reason": model_result.get("metrics", {}).get("done_reason", "stop"),
                    "metrics": model_result.get("metrics", {}),
                }
                verified, overhead, vstate = self.model_runner._maybe_retry_with_verifier(
                    original_prompt=preview_prompt,
                    generation=generation_for_verify,
                )
                if vstate.get("retry_applied"):
                    model_result["response"] = verified.get("response", model_result["response"])
                    merged = dict(model_result["metrics"])
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        merged[key] = merged.get(key, 0) + overhead.get(key, 0)
                    for key in ("load_time", "prompt_eval_time", "eval_time"):
                        merged[key] = merged.get(key, 0) + overhead.get(key, 0)
                    model_result["metrics"] = merged
                model_result["metrics"]["verifier_retry_applied"] = vstate.get("retry_applied", False)
                model_result["metrics"]["verifier_verdict"] = vstate.get("verdict", "skipped")
                model_result["metrics"]["verifier_heuristic_weak"] = vstate.get("heuristic_weak", False)
                model_result["metrics"]["verifier_error"] = vstate.get("verifier_error")
                model_result["metrics"]["verifier_retry_error"] = vstate.get("retry_error")

            results[preview_technique] = self._evaluate_technique_runs(
                technique_name=preview_technique,
                prompt=prompts[preview_technique],
                problem=problem,
                ground_truth=ground_truth,
                runs_per_technique=resolved_runs,
                first_run_model_result=model_result,
            )

            yield {
                "type": "status",
                "message": "Finalizing remaining runs and techniques...",
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
                        runs_configured=resolved_runs,
                    )

        best_technique = self._greedy_select(results, problem)
        final_result = {
            "problem": problem,
            "ground_truth": ground_truth,
            "runs_per_technique": resolved_runs,
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
        Evaluate one run for a technique through the complete pipeline.
        
        This wrapper is kept for backward compatibility. Internally, it uses the
        multi-run evaluator with one configured run.
        
        Args:
            technique_name: Name of the prompting technique
            prompt: The generated prompt
            problem: Original problem
            ground_truth: Expected answer
            
        Returns:
            Evaluation results with scores and metadata
        """
        return self._evaluate_technique_runs(
            technique_name=technique_name,
            prompt=prompt,
            problem=problem,
            ground_truth=ground_truth,
            runs_per_technique=1,
        )

    def _evaluate_technique_runs(
        self,
        technique_name: str,
        prompt: str,
        problem: str,
        ground_truth: str = None,
        runs_per_technique: Optional[int] = None,
        first_run_model_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate repeated runs for one technique and compute consistency."""
        resolved_runs = self.default_runs_per_technique
        if runs_per_technique is not None:
            resolved_runs = self._sanitize_runs_per_technique(runs_per_technique)

        run_history: List[Dict[str, Any]] = []
        normalized_outputs: List[str] = []

        for run_index in range(1, resolved_runs + 1):
            if run_index == 1 and first_run_model_result is not None:
                model_result = first_run_model_result
            else:
                model_result = self.model_runner.run(prompt)

            run_payload = self._build_single_run_payload(
                technique_name=technique_name,
                prompt=prompt,
                model_result=model_result,
                problem=problem,
                ground_truth=ground_truth,
                run_index=run_index,
                runs_per_technique=resolved_runs,
            )

            normalized_outputs.append(run_payload["normalized_output"])
            consistency_state = self.consistency_scorer.compute_consistency(normalized_outputs)
            consistency_value = consistency_state["value"]

            overall_score, overall_is_provisional = self._compute_overall_score(
                accuracy=run_payload["scores"]["accuracy"],
                efficiency=run_payload["scores"]["efficiency"],
                consistency=consistency_value,
            )

            run_payload["scores"].update(
                {
                    "consistency": round(consistency_value, 3) if consistency_value is not None else None,
                    "consistency_is_provisional": bool(consistency_state["is_provisional"]),
                    "consistency_runs_used": int(consistency_state["runs_used"]),
                    "consistency_matching_runs": consistency_state["matching_runs"],
                    "overall": round(overall_score, 3),
                    "overall_is_provisional": overall_is_provisional,
                    "overall_note": (
                        "Provisional overall score: consistency is not available until run 2."
                        if overall_is_provisional
                        else "Overall score includes accuracy, efficiency, and consistency."
                    ),
                }
            )

            run_history.append(run_payload)

        final_consistency_state = self.consistency_scorer.compute_consistency(normalized_outputs)
        successful_runs = [run for run in run_history if run.get("success", False)]
        last_successful_run = successful_runs[-1] if successful_runs else None

        accuracy = 0.0
        efficiency = 0.0
        if successful_runs:
            accuracy = sum(run["scores"]["accuracy"] for run in successful_runs) / len(successful_runs)
            efficiency = sum(run["scores"]["efficiency"] for run in successful_runs) / len(successful_runs)

        consistency = final_consistency_state["value"]
        overall, overall_is_provisional = self._compute_overall_score(
            accuracy=accuracy,
            efficiency=efficiency,
            consistency=consistency,
        )

        aggregated_metrics = self._aggregate_metrics(run_history)

        if successful_runs:
            success = True
            response = str(last_successful_run.get("response", ""))
            error = None
        else:
            success = False
            response = ""
            error = str(run_history[-1].get("error", "All runs failed")) if run_history else "All runs failed"

        result = {
            "technique": technique_name,
            "success": success,
            "prompt": prompt,
            "response": response,
            "metrics": aggregated_metrics,
            "scores": {
                "accuracy": round(accuracy, 3),
                "consistency": round(consistency, 3) if consistency is not None else None,
                "efficiency": round(efficiency, 3),
                "overall": round(overall, 3),
                "consistency_is_provisional": bool(final_consistency_state["is_provisional"]),
                "consistency_runs_used": int(final_consistency_state["runs_used"]),
                "consistency_matching_runs": final_consistency_state["matching_runs"],
                "overall_is_provisional": overall_is_provisional,
                "overall_note": (
                    "Provisional overall score: consistency is not available until run 2."
                    if overall_is_provisional
                    else "Overall score includes accuracy, efficiency, and consistency."
                ),
            },
            "run_history": run_history,
            "audit": {
                "runs_configured": resolved_runs,
                "runs_recorded": len(run_history),
                "normalized_outputs": normalized_outputs,
                "output_counts": final_consistency_state["output_counts"],
                "canonical_output": final_consistency_state["canonical_output"],
            },
        }

        if error:
            result["error"] = error

        return result

    def _build_single_run_payload(
        self,
        technique_name: str,
        prompt: str,
        model_result: Dict[str, Any],
        problem: str,
        ground_truth: str,
        run_index: int,
        runs_per_technique: int,
    ) -> Dict[str, Any]:
        """Build one run payload while preserving accuracy and efficiency formulas."""
        metrics = model_result.get("metrics") or {}
        metrics = {
            "elapsed_time": metrics.get("elapsed_time", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "prompt_tokens": metrics.get("prompt_tokens", 0),
            "completion_tokens": metrics.get("completion_tokens", 0),
            "done_reason": metrics.get("done_reason"),
            "truncated": metrics.get("truncated", False),
            "continuation_rounds": metrics.get("continuation_rounds", 0),
            "continuation_error": metrics.get("continuation_error"),
            "verifier_retry_applied": metrics.get("verifier_retry_applied", False),
            "verifier_verdict": metrics.get("verifier_verdict"),
        }

        success = bool(model_result.get("success", False))
        response = str(model_result.get("response", "") or "")

        if success:
            accuracy = self.accuracy_scorer.score(response, ground_truth, problem)
            efficiency = self.efficiency_scorer.score(response, metrics)
            normalized_output = self.consistency_scorer.normalize_output(response)
            error = None
        else:
            accuracy = 0.0
            efficiency = 0.0
            normalized_output = f"error:run_{run_index}"
            error = str(model_result.get("error", "Unknown error"))

        payload = {
            "technique": technique_name,
            "run_index": run_index,
            "runs_configured": runs_per_technique,
            "success": success,
            "prompt": prompt,
            "response": response,
            "metrics": metrics,
            "normalized_output": normalized_output,
            "scores": {
                "accuracy": round(accuracy, 3),
                "efficiency": round(efficiency, 3),
            },
        }

        if error:
            payload["error"] = error

        return payload

    def _aggregate_metrics(self, run_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across successful runs for summary reporting."""
        successful_runs = [run for run in run_history if run.get("success", False)]
        runs_recorded = len(run_history)
        runs_succeeded = len(successful_runs)
        runs_failed = runs_recorded - runs_succeeded

        if not successful_runs:
            return {
                "elapsed_time": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "runs_recorded": runs_recorded,
                "runs_succeeded": runs_succeeded,
                "runs_failed": runs_failed,
            }

        def _mean_metric(key: str) -> float:
            total = 0.0
            for run in successful_runs:
                value = run.get("metrics", {}).get(key, 0)
                try:
                    total += float(value)
                except (TypeError, ValueError):
                    continue
            return total / max(len(successful_runs), 1)

        return {
            "elapsed_time": round(_mean_metric("elapsed_time"), 4),
            "total_tokens": int(round(_mean_metric("total_tokens"))),
            "prompt_tokens": int(round(_mean_metric("prompt_tokens"))),
            "completion_tokens": int(round(_mean_metric("completion_tokens"))),
            "runs_recorded": runs_recorded,
            "runs_succeeded": runs_succeeded,
            "runs_failed": runs_failed,
        }

    def _compute_overall_score(
        self,
        *,
        accuracy: float,
        efficiency: float,
        consistency: Optional[float],
    ) -> Tuple[float, bool]:
        """Compute weighted overall score with provisional fallback when consistency is unavailable."""
        available_scores: Dict[str, float] = {
            "accuracy": float(accuracy),
            "efficiency": float(efficiency),
        }
        if consistency is not None:
            available_scores["consistency"] = float(consistency)

        weight_sum = sum(self.weights.get(name, 0.0) for name in available_scores)
        if weight_sum > 0:
            overall = sum(
                score * self.weights.get(name, 0.0)
                for name, score in available_scores.items()
            ) / weight_sum
        else:
            overall = sum(available_scores.values()) / max(len(available_scores), 1)

        return overall, consistency is None
    
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
            overall = float(scores.get("overall", 0.0) or 0.0)
            
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
                    "consistency": scores.get("consistency"),
                    "efficiency": scores["efficiency"],
                    "overall": scores["overall"],
                    "consistency_is_provisional": scores.get("consistency_is_provisional", False),
                    "consistency_runs_used": scores.get("consistency_runs_used", 0),
                    "consistency_matching_runs": scores.get("consistency_matching_runs"),
                    "overall_is_provisional": scores.get("overall_is_provisional", False),
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
                   consistency: float = None, 
                   efficiency: float = None):
        """
        Update metric weights.
        
        Args:
            accuracy: New accuracy weight
            consistency: New consistency weight
            efficiency: New efficiency weight
        """
        new_weights = dict(self.weights)

        if accuracy is not None:
            new_weights['accuracy'] = accuracy
        if consistency is not None:
            new_weights['consistency'] = consistency
        if efficiency is not None:
            new_weights['efficiency'] = efficiency

        self.weights = self._normalize_weights(new_weights)

    def set_runs_per_technique(self, runs_per_technique: int):
        """Update the default run count used for consistency estimation."""
        self.default_runs_per_technique = self._sanitize_runs_per_technique(runs_per_technique)


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
