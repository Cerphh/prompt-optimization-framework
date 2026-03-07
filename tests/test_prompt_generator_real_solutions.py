from framework.prompt_generator import PromptGenerator


def test_detect_primary_intent_real_solutions_variants():
    generator = PromptGenerator()

    assert generator._detect_primary_intent("Get the real solutions of x^4 - 5x^2 + 4 = 0") == "real_solutions"
    assert generator._detect_primary_intent("Find all real roots of x^4 - 16 = 0") == "real_solutions"


def test_few_shot_prioritizes_real_solution_examples_for_algebra():
    generator = PromptGenerator()

    prompt = generator.generate_few_shot(
        "Get the real solutions of x^4 - 5x^2 + 4 = 0",
        subject="algebra",
        num_examples=2,
    )

    assert "Q: Solve for all real solutions:" in prompt


def test_few_shot_prioritizes_real_values_examples_for_statistics():
    generator = PromptGenerator()

    prompt = generator.generate_few_shot(
        "Find all real values of p if P(A)=p and P(B)=2p with disjoint events and P(A union B)=1",
        subject="statistics",
        num_examples=2,
    )

    assert "Q: Find all real values of p" in prompt


def test_few_shot_prioritizes_real_values_examples_for_calculus():
    generator = PromptGenerator()

    prompt = generator.generate_few_shot(
        "Find all real values where f'(x)=0 for f(x)=x^3-3x",
        subject="calculus",
        num_examples=2,
    )

    assert "Q: Find all real values of x where f'(x)=0 for f(x)=x³-3x." in prompt


def test_few_shot_prioritizes_conditional_probability_real_values_example():
    generator = PromptGenerator()

    prompt = generator.generate_few_shot(
        "Find all real values of p given that P(A|B)=1/2 and P(A ∩ B)=1/6",
        subject="statistics",
        num_examples=1,
    )

    assert "Q: Find all real values of p given that P(A|B)=1/2" in prompt
