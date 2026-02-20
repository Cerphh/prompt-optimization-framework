from framework.prompt_generator import PromptGenerator

gen = PromptGenerator()

test_cases = [
    ("Find the derivative of x³ + 2x", "calculus"),
    ("What is the mean of 5, 10, 15, 20?", "statistics"),
    ("Solve for x: 2x + 5 = 13", "algebra"),
    ("Calculate the integral of 3x²", "calculus"),
    ("Find the probability of rolling a 6", "statistics"),
    ("Factor x² - 16", "algebra")
]

for problem, subject in test_cases:
    print(f"\n{'='*70}")
    print(f"PROBLEM: {problem}")
    print(f"SUBJECT: {subject}")
    print('='*70)
    
    few_shot = gen.generate_few_shot(problem, subject=subject, num_examples=2)
    
    # Extract just the examples part
    lines = few_shot.split('\n')
    print("\nSELECTED EXAMPLES:")
    in_examples = False
    for line in lines:
        if line.startswith("Problem:"):
            in_examples = True
            print(f"  • {line}")
        elif in_examples and line.startswith("Solution:"):
            continue
        elif in_examples and line.strip() == "":
            continue
        elif line.startswith("Now solve"):
            break
