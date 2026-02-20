"""Test why conditional probability examples aren't being selected"""

from framework.prompt_generator import PromptGenerator

# Initialize
pg = PromptGenerator()

# User's problem
problem = """In a class of 50 students:

30 students take Math

25 students take Physics

10 students take both Math and Physics

A student is chosen at random. What is the probability that the student takes Math given that they take Physics?"""

# Detect keywords
keywords = pg._detect_problem_keywords(problem)
print(f"\n🔍 Detected Keywords: {keywords}\n")

# Get statistics examples
stats_examples = pg.example_dataset.get("statistics", [])
print(f"📊 Total statistics examples: {len(stats_examples)}\n")

# Score each example
print("📈 Example Relevance Scores:")
print("-" * 80)
for i, example in enumerate(stats_examples, 1):
    score = pg._score_example_relevance(example, problem, keywords)
    ex_text = example['problem'][:60] + "..." if len(example['problem']) > 60 else example['problem']
    print(f"{i}. [{score:.3f}] {ex_text}")

print("\n" + "=" * 80)

# Test selection
selected = pg._select_relevant_examples(stats_examples, problem, num_examples=2, seed=hash(problem) % (2**32))
print(f"\n✅ Selected Examples (num=2):")
for i, ex in enumerate(selected, 1):
    print(f"\n{i}. {ex['problem']}")
    print(f"   Solution: {ex['solution'][:50]}...")
