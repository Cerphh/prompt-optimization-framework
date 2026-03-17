from framework.prompt_generator import PromptGenerator


def test_few_shot_falls_back_to_zero_shot_when_subject_examples_missing():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [],
        "general": [{"problem": "What is 1 + 1?", "solution": "2"}],
    }

    prompt = generator.generate_few_shot(
        "Solve for x: 2x + 6 = 10",
        subject="algebra",
        num_examples=2,
    )

    assert prompt.startswith("Solve the following math problem and end with a concise final answer.")
    assert "Q: Solve for x: 2x + 6 = 10" in prompt
    assert "A:" in prompt


def test_hard_problem_uses_more_examples_when_num_examples_not_provided():
    generator = PromptGenerator()
    generator.few_shot_min_examples = 1
    generator.few_shot_max_examples = 4
    generator.few_shot_medium_examples = 2
    generator.few_shot_hard_examples = 4
    generator.example_dataset = {
        "calculus": [
            {"problem": "Find derivative of x^4", "solution": "4x^3"},
            {"problem": "Use chain rule on (2x+1)^5", "solution": "10(2x+1)^4"},
            {"problem": "Find limit of (sin x)/x", "solution": "1"},
            {"problem": "Evaluate integral of x^2", "solution": "x^3/3 + C"},
            {"problem": "Find derivative of e^(3x)", "solution": "3e^(3x)"},
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Find the derivative using chain rule and evaluate a related limit.",
        subject="calculus",
    )

    assert prompt.startswith("Solve the following math problems and give the final answer.")
    assert prompt.count("\nQ: ") >= 4


def test_few_shot_accepts_nested_subject_difficulty_examples():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": {
            "basic": [
                {"problem": "Solve for x: x + 2 = 5", "solution": "x = 3"},
                {"problem": "Expand (x + 1)^2", "solution": "x^2 + 2x + 1"},
            ]
        },
        "general": [
            {"problem": "What is 1 + 1?", "solution": "2"}
        ],
    }

    prompt = generator.generate_few_shot(
        "Solve for x: x + 4 = 7",
        subject="algebra",
        num_examples=1,
    )

    assert prompt.startswith("Solve the following math problems and give the final answer.")
    assert "Q: Solve for x: x + 2 = 5" in prompt
    assert "Q: Solve for x: x + 4 = 7" in prompt
