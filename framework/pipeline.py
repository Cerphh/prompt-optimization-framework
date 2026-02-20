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
                 accuracy_weight: float = 0.5,
                 completeness_weight: float = 0.3,
                 efficiency_weight: float = 0.2):
        """
        Initialize the benchmarking pipeline.
        
        Args:
            model_name: Name of the LLM to use
            accuracy_weight: Weight for accuracy score (0-1)
            completeness_weight: Weight for completeness score (0-1)
            efficiency_weight: Weight for efficiency score (0-1)
        """
        self.prompt_generator = PromptGenerator()
        self.model_runner = ModelRunner(model_name=model_name)
        self.accuracy_scorer = AccuracyScorer()
        self.completeness_scorer = CompletenessScorer()
        self.efficiency_scorer = EfficiencyScorer()
        
        # Metric weights for scoring
        self.weights = {
            'accuracy': accuracy_weight,
            'completeness': completeness_weight,
            'efficiency': efficiency_weight
        }
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def benchmark(self, problem: str, ground_truth: str = None, subject: str = "general") -> Dict[str, Any]:
        """
        Run comprehensive benchmark on a single problem.
        
        Evaluates all prompting techniques IN PARALLEL for faster results.
        
        Args:
            problem: The math problem to solve
            ground_truth: Expected answer for accuracy evaluation
            subject: Subject category for few-shot examples (algebra, statistics, calculus, general)
            
        Returns:
            Dictionary with results for all techniques and best selection
        """
        # Step 1: Generate prompts using all techniques
        prompts = self.prompt_generator.generate_all_techniques(problem, subject=subject)
        
        # Step 2: Evaluate each technique IN PARALLEL
        results = {}
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
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
                    results[technique_name] = {
                        "technique": technique_name,
                        "success": False,
                        "error": str(e),
                        "prompt": prompts[technique_name],
                        "scores": {
                            "accuracy": 0.0,
                            "completeness": 0.0,
                            "efficiency": 0.0,
                            "overall": 0.0
                        }
                    }
        
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
            return {
                "technique": technique_name,
                "success": False,
                "error": model_result.get("error", "Unknown error"),
                "prompt": prompt,
                "scores": {
                    "accuracy": 0.0,
                    "completeness": 0.0,
                    "efficiency": 0.0,
                    "overall": 0.0
                }
            }
        
        response = model_result["response"]
        metrics = model_result["metrics"]
        
        # Calculate scores
        accuracy = self.accuracy_scorer.score(response, ground_truth, problem)
        completeness = self.completeness_scorer.score(response, problem)
        efficiency = self.efficiency_scorer.score(response, metrics)
        
        # Calculate weighted overall score
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
                # Exact tie - use problem hash for deterministic selection
                problem_hash = hash(problem + technique_name) % 100
                current_hash = hash(problem + best_technique) % 100
                # Higher hash wins (deterministic)
                if problem_hash > current_hash:
                    best_technique = technique_name
        
        return best_technique if best_technique else list(results.keys())[0]
    
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
        if accuracy is not None:
            self.weights['accuracy'] = accuracy
        if completeness is not None:
            self.weights['completeness'] = completeness
        if efficiency is not None:
            self.weights['efficiency'] = efficiency
        
        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}


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
