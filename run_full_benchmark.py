"""
Comprehensive Benchmark Test
Runs all dataset problems and generates summary report for thesis analysis.
"""

import time
from framework.pipeline import BenchmarkPipeline
from framework.dataset import get_sample_dataset

print("=" * 80)
print("COMPREHENSIVE BENCHMARK - ALL DATASET PROBLEMS")
print("=" * 80)
print()

# Initialize
pipeline = BenchmarkPipeline(
    model_name="llama3",
    accuracy_weight=0.5,
    completeness_weight=0.3,
    efficiency_weight=0.2
)

# Check connection
if not pipeline.test_connection():
    print("✗ Ollama not connected. Please start Ollama first.")
    exit(1)

print("✓ Pipeline ready")
print(f"  Metric weights: Accuracy={pipeline.weights['accuracy']:.2f}, "
      f"Completeness={pipeline.weights['completeness']:.2f}, "
      f"Efficiency={pipeline.weights['efficiency']:.2f}")
print()

# Load dataset
dataset = get_sample_dataset()
problems = dataset.get_problems()
print(f"✓ Loaded {len(problems)} problems")
print()

# Storage for results
all_results = []
technique_wins = {"zero_shot": 0, "few_shot": 0}
technique_scores = {"zero_shot": [], "few_shot": []}
category_results = {}

# Run benchmarks
print("=" * 80)
print("RUNNING BENCHMARKS")
print("=" * 80)
print()

start_time = time.time()

for i, problem in enumerate(problems, 1):
    print(f"[{i}/{len(problems)}] Testing: {problem['category']}")
    print(f"Problem: {problem['problem'][:60]}...")
    
    try:
        result = pipeline.benchmark(
            problem=problem["problem"],
            ground_truth=problem["answer"]
        )
        
        if result["best_result"]["success"]:
            # Store result
            all_results.append({
                "problem_id": problem["id"],
                "problem": problem["problem"],
                "category": problem["category"],
                "ground_truth": problem["answer"],
                "best_technique": result["best_technique"],
                "comparison": result["comparison"],
                "all_results": result["all_results"]
            })
            
            # Track wins
            best = result["best_technique"]
            technique_wins[best] += 1
            
            # Track scores by technique
            for comp in result["comparison"]:
                technique_scores[comp["technique"]].append(comp["overall"])
            
            # Track by category
            if problem["category"] not in category_results:
                category_results[problem["category"]] = []
            category_results[problem["category"]].append(result)
            
            print(f"  ✓ Best: {best} (score: {result['best_result']['scores']['overall']:.3f})")
            print(f"    Accuracy: {result['best_result']['scores']['accuracy']:.3f}")
        else:
            print(f"  ✗ Failed: {result['best_result'].get('error')}")
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

total_time = time.time() - start_time

# Generate Summary Report
print("=" * 80)
print("SUMMARY REPORT")
print("=" * 80)
print()

print(f"Total Problems Tested: {len(all_results)}")
print(f"Total Time: {total_time:.2f}s")
print(f"Average Time per Problem: {total_time/len(all_results):.2f}s")
print()

# Technique Wins
print("-" * 80)
print("TECHNIQUE WINS")
print("-" * 80)
for technique, wins in sorted(technique_wins.items(), key=lambda x: x[1], reverse=True):
    percentage = (wins / len(all_results)) * 100
    print(f"{technique:<20} {wins:>3} wins  ({percentage:>5.1f}%)")
print()

# Average Scores by Technique
print("-" * 80)
print("AVERAGE SCORES BY TECHNIQUE")
print("-" * 80)
print(f"{'Technique':<20} {'Avg Score':<12} {'Min':<8} {'Max':<8}")
print("-" * 80)
for technique, scores in technique_scores.items():
    if scores:
        avg = sum(scores) / len(scores)
        print(f"{technique:<20} {avg:<12.3f} {min(scores):<8.3f} {max(scores):<8.3f}")
print()

# Detailed Metrics Breakdown
print("-" * 80)
print("DETAILED METRICS - ALL PROBLEMS")
print("-" * 80)

# Aggregate metrics for each technique
technique_metrics = {
    "zero_shot": {"accuracy": [], "completeness": [], "efficiency": [], "overall": []},
    "few_shot": {"accuracy": [], "completeness": [], "efficiency": [], "overall": []}
}

for result in all_results:
    for comp in result["comparison"]:
        tech = comp["technique"]
        technique_metrics[tech]["accuracy"].append(comp["accuracy"])
        technique_metrics[tech]["completeness"].append(comp["completeness"])
        technique_metrics[tech]["efficiency"].append(comp["efficiency"])
        technique_metrics[tech]["overall"].append(comp["overall"])

print(f"{'Technique':<15} {'Metric':<15} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
print("-" * 80)

for technique in ["zero_shot", "few_shot"]:
    for metric in ["accuracy", "completeness", "efficiency", "overall"]:
        values = technique_metrics[technique][metric]
        if values:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = variance ** 0.5
            print(f"{technique:<15} {metric:<15} {mean:<8.3f} {std:<8.3f} "
                  f"{min(values):<8.3f} {max(values):<8.3f}")
    print()

# Performance by Category
print("-" * 80)
print("PERFORMANCE BY PROBLEM CATEGORY")
print("-" * 80)
for category, results in sorted(category_results.items()):
    print(f"\n{category.upper()}:")
    category_wins = {"zero_shot": 0, "few_shot": 0}
    for result in results:
        category_wins[result["best_technique"]] += 1
    
    for technique, wins in category_wins.items():
        print(f"  {technique:<20} {wins} / {len(results)}")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
print()
print("Results collected for thesis analysis.")
print("Ready to implement database storage for permanent record keeping.")
print()
