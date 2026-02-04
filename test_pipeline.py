"""
Test Pipeline - Research Benchmarking Framework
Demonstrates comparative evaluation of prompting techniques on math problems.
"""

print("=" * 70)
print("PROMPT OPTIMIZATION FRAMEWORK - RESEARCH BENCHMARK")
print("=" * 70)
print()

# Test imports
try:
    from framework.pipeline import BenchmarkPipeline
    from framework.dataset import get_sample_dataset
    print("✓ Framework imports successful")
except Exception as e:
    print(f"✗ Framework import failed: {e}")
    exit(1)

# Initialize pipeline
try:
    pipeline = BenchmarkPipeline(
        model_name="llama3",
        accuracy_weight=0.5,
        completeness_weight=0.3,
        efficiency_weight=0.2
    )
    print("✓ Benchmark pipeline initialized")
    print(f"  Weights: Accuracy={pipeline.weights['accuracy']:.1f}, "
          f"Completeness={pipeline.weights['completeness']:.1f}, "
          f"Efficiency={pipeline.weights['efficiency']:.1f}")
except Exception as e:
    print(f"✗ Pipeline initialization failed: {e}")
    exit(1)

# Test model connection
print()
print("-" * 70)
print("Testing Ollama connection...")
print("-" * 70)
if pipeline.test_connection():
    print("✓ Ollama is connected and ready")
else:
    print("⚠ Ollama is not connected. Make sure Ollama is running.")
    print("  Run: ollama serve")
    print("  Then: ollama pull llama3")
    exit(1)

# Load sample dataset
print()
print("-" * 70)
print("Loading sample dataset...")
print("-" * 70)
dataset = get_sample_dataset()
print(f"✓ Loaded {dataset.size()} math problems")

# Run benchmark on a sample problem
print()
print("=" * 70)
print("RUNNING COMPARATIVE BENCHMARK")
print("=" * 70)

# Select a test problem - algebra problem
test_problem = dataset.get_problem(3)  # Algebra problem
print(f"\nProblem: {test_problem['problem']}")
print(f"Expected Answer: {test_problem['answer']}")
print(f"Category: {test_problem['category']}")
print()
print("Evaluating all prompting techniques...")
print("-" * 70)

try:
    # Run comprehensive benchmark
    result = pipeline.benchmark(
        problem=test_problem['problem'],
        ground_truth=test_problem['answer']
    )
    
    if result["best_result"]["success"]:
        print("\n✓ Benchmark completed successfully!\n")
        
        # Display comparison table
        print("=" * 70)
        print("RESULTS COMPARISON")
        print("=" * 70)
        print(f"{'Technique':<20} {'Accuracy':<10} {'Complete':<10} {'Efficiency':<10} {'Overall':<10}")
        print("-" * 70)
        
        for comp in result["comparison"]:
            print(f"{comp['technique']:<20} "
                  f"{comp['accuracy']:<10.3f} "
                  f"{comp['completeness']:<10.3f} "
                  f"{comp['efficiency']:<10.3f} "
                  f"{comp['overall']:<10.3f}")
        
        # Display best technique
        print()
        print("=" * 70)
        print("OPTIMAL TECHNIQUE SELECTED (GREEDY ALGORITHM)")
        print("=" * 70)
        print(f"Best Technique: {result['best_technique']}")
        print(f"Overall Score: {result['best_result']['scores']['overall']:.3f}")
        print()
        print("Scores:")
        print(f"  Accuracy:     {result['best_result']['scores']['accuracy']:.3f}")
        print(f"  Completeness: {result['best_result']['scores']['completeness']:.3f}")
        print(f"  Efficiency:   {result['best_result']['scores']['efficiency']:.3f}")
        print()
        print("Metrics:")
        print(f"  Latency:      {result['best_result']['metrics']['elapsed_time']:.2f}s")
        print(f"  Total Tokens: {result['best_result']['metrics']['total_tokens']}")
        print()
        print("Response (first 200 chars):")
        print("-" * 70)
        response = result['best_result']['response']
        print(response[:200] + ("..." if len(response) > 200 else ""))
        print()
        
        # Show all responses for comparison (abbreviated)
        print("=" * 70)
        print("ALL RESPONSES (abbreviated)")
        print("=" * 70)
        for technique in ["zero_shot", "few_shot"]:
            if technique in result["all_results"] and result["all_results"][technique]["success"]:
                resp = result["all_results"][technique]["response"]
                print(f"\n{technique.upper()}:")
                print(f"{resp[:150]}...")
                print(f"Score: {result['all_results'][technique]['scores']['overall']:.3f}")
        
    else:
        print(f"✗ Benchmark failed: {result['best_result'].get('error')}")
        
except Exception as e:
    print(f"✗ Benchmark error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("SETUP COMPLETE!")
print("=" * 70)
print()
print("Next steps:")
print("1. Run more benchmarks on different problems")
print("2. Adjust metric weights using pipeline.set_weights()")
print("3. Add custom problems to the dataset")
print("4. Start the API: uvicorn main:app --reload")
print("5. Visit: http://127.0.0.1:8000/docs")
print()
print("=" * 70)
