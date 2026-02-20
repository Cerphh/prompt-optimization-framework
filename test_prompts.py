from framework.prompt_generator import PromptGenerator

gen = PromptGenerator()

problems = [
    "Find x: 3x = 9",
    "Calculate x: 3x = 9",
    "Solve for x: 3x = 9"
]

for problem in problems:
    print(f"\n{'='*60}")
    print(f"PROBLEM: {problem}")
    print('='*60)
    
    print("\n--- ZERO-SHOT ---")
    print(gen.generate_zero_shot(problem))
    
    print("\n--- FEW-SHOT ---")
    print(gen.generate_few_shot(problem, subject="algebra", num_examples=2))

