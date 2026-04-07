from framework.prompt_generator import PromptGenerator, FewShotUnavailableError
import pytest


def test_generate_few_shot_auto_detects_subject_from_question():
    generator = PromptGenerator()
    generator.example_dataset = {
        "general": [
            {"problem": "What is 2 + 2?", "solution": "4"}
        ],
        "pre-calculus": [
            {
                "problem": "Find the derivative of f(x) = x^3.",
                "solution": "f'(x) = 3x^2",
            }
        ],
    }

    prompt = generator.generate_few_shot(
        "Find the derivative of x^4 + 2x.",
        subject="general",
        num_examples=1,
    )

    assert "Use the following examples only as style references." in prompt
    assert "Q: Find the derivative of f(x) = x^3." in prompt
    assert "Q: Find the derivative of x^4 + 2x." in prompt


def test_select_relevant_examples_ranks_conditional_probability_examples():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "Given that probability of rain is 0.4, what is the probability of no rain?",
            "solution": "Use complement rule.",
        },
        {
            "problem": "Given that P(A|B)=1/2 and P(A intersection B)=1/6, find P(B).",
            "solution": "Use P(A|B)=P(A intersection B)/P(B).",
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "Given that P(A|B)=1/2 and P(A intersection B)=1/6, find P(B).",
        num_examples=1,
    )

    assert selected[0]["problem"] == available_examples[1]["problem"]


def test_select_relevant_examples_ranks_real_solution_examples():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "Solve for all real solutions: x^2 + 1 = 0",
            "solution": "No real solution.",
        },
        {
            "problem": "Solve for all real solutions: x^4 - 5x^2 + 4 = 0",
            "solution": "Factor and solve.",
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "Solve for all real solutions: x^4 - 5x^2 + 4 = 0",
        num_examples=1,
    )

    assert selected[0]["problem"] == available_examples[1]["problem"]


def test_detect_primary_intent_for_value_with_assignments():
    generator = PromptGenerator()

    problem = "What is the positive value of the expression x^3 - 2y when x = 5 and y = 2?"

    assert generator._detect_primary_intent(problem) == "evaluate_substitution"
    features = generator._extract_math_features(problem)
    assert "substitution" in features
    assert "system" not in features


def test_few_shot_prefers_multi_assignment_substitution_examples():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "What is the value of x^3 when x = 5?",
                "solution": "x^3 = 5^3 = 125",
            },
            {
                "problem": "What is the value of a^3 - 2b when a = 3 and b = 2?",
                "solution": "a^3 - 2b = 3^3 - 2(2) = 27 - 4 = 23",
            },
            {
                "problem": "Solve x^3 - 2x = 0.",
                "solution": "x(x^2 - 2) = 0",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "What is the positive value of the expression x^3 - 2y when x = 5 and y = 2?",
        subject="algebra",
        num_examples=1,
    )

    assert "Use the following examples only as style references." in prompt
    assert "Q: What is the value of a^3 - 2b when a = 3 and b = 2?" in prompt
    assert "Q: What is the positive value of the expression x^3 - 2y when x = 5 and y = 2?" in prompt


def test_few_shot_strict_match_in_counting_probability_domain():
    generator = PromptGenerator()
    generator.example_dataset = {
        "counting-probability": [
            {
                "problem": "A die is rolled 5 times. What is the probability of getting exactly two 6s?",
                "solution": "Use binomial counting.",
                "type": "probability",
            },
            {
                "problem": "In how many ways can 5 students be arranged in a row?",
                "solution": "Use 5!.",
                "type": "counting_arrangements",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "A die is rolled 3 times. What is the probability of getting exactly one 6?",
        subject="counting-probability",
        num_examples=1,
    )

    assert "use the following examples only as style references." in prompt.lower()
    assert "probability of getting exactly two 6s" in prompt.lower()
    assert "probability of getting exactly one 6" in prompt.lower()
    assert "arranged in a row" not in prompt.lower()


def test_few_shot_strict_match_in_precalculus_domain():
    generator = PromptGenerator()
    generator.example_dataset = {
        "pre-calculus": [
            {
                "problem": "Find the derivative of f(t) = t^4 + t.",
                "solution": "f'(t) = 4t^3 + 1",
                "type": "derivative",
            },
            {
                "problem": "Evaluate the integral of 2x dx.",
                "solution": "x^2 + C",
                "type": "integral",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Find the derivative of x^4 + x.",
        subject="pre-calculus",
        num_examples=1,
    )

    assert "use the following examples only as style references." in prompt.lower()
    assert "find the derivative of f(t) = t^4 + t" in prompt.lower()
    assert "q: find the derivative of x^4 + x." in prompt.lower()
    assert "evaluate the integral" not in prompt.lower()


def test_few_shot_prompt_explicitly_targets_input_problem_answer():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "Solve for y: 3y - 12 = 0.",
                "solution": "y = 4",
                "type": "solve_equation",
            }
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Solve for x: 3x - 12 = 0.",
        subject="algebra",
        num_examples=1,
    )

    assert "do not repeat or copy any example answer" in prompt.lower()
    assert "target problem" in prompt.lower()
    assert "Q: Solve for x: 3x - 12 = 0." in prompt


def test_few_shot_target_problem_text_is_identical_to_input():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "Solve for x: x^2 + 2x - 8 = 0.",
                "solution": "x = 2 or x = -4",
                "type": "solve_equation",
            }
        ],
        "general": [],
    }

    raw_problem = "Solve for x: x^2 + 4x + 4 = 0"
    prompt = generator.generate_few_shot(raw_problem, subject="algebra", num_examples=1)

    assert f"Q: {raw_problem}" in prompt
    assert "Q: Solve for x: x² + 4x + 4 = 0" not in prompt


def test_few_shot_does_not_select_identical_problem_from_example_bank():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "Solve for x: 3x - 12 = 0.",
                "solution": "x = 4",
                "type": "solve_equation",
            },
            {
                "problem": "Solve for y: 3y - 12 = 0.",
                "solution": "x = 4",
                "type": "solve_equation",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Solve for x: 3x - 12 = 0.",
        subject="algebra",
        num_examples=1,
    )

    # Strict mode should reject identical problem text and use a related structure match.
    assert "Q: Solve for y: 3y - 12 = 0." in prompt
    assert prompt.count("Q: Solve for x: 3x - 12 = 0.") == 1


def test_few_shot_prefers_same_structure_algebra_with_variable_change():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "Solve for y: 4y + 8 = 20.",
                "solution": "y = 3",
                "type": "solve_equation",
            },
            {
                "problem": "Solve for y: y^2 - 9 = 0.",
                "solution": "y = 3 or y = -3",
                "type": "solve_equation",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Solve for x: 4x + 8 = 20.",
        subject="algebra",
        num_examples=1,
    )

    assert "Q: Solve for y: 4y + 8 = 20." in prompt
    assert "Q: Solve for y: y^2 - 9 = 0." not in prompt


def test_few_shot_prefers_same_structure_probability_with_value_change():
    generator = PromptGenerator()
    generator.example_dataset = {
        "counting-probability": [
            {
                "problem": "A coin is flipped 5 times. What is the probability of exactly 2 heads?",
                "solution": "Use binomial distribution.",
                "type": "probability",
            },
            {
                "problem": "A coin is flipped 5 times. What is the probability of at least 2 heads?",
                "solution": "Use complement and binomial.",
                "type": "probability",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "A coin is flipped 7 times. What is the probability of exactly 3 heads?",
        subject="counting-probability",
        num_examples=1,
    )

    assert "probability of exactly 2 heads" in prompt.lower()
    assert "probability of at least 2 heads" not in prompt.lower()


def test_few_shot_retrieves_factorial_digit_example_for_similar_problem():
    generator = PromptGenerator()
    generator.example_dataset = {
        "counting-probability": [
            {
                "problem": "What is the tens digit of 7! + 8! + 9! + ... + 400!?",
                "solution": "Only 7!, 8!, and 9! matter for the tens digit.",
                "type": "factorial_number_theory",
            },
            {
                "problem": "What is the smallest power of 7 that is NOT a palindrome?",
                "solution": "7^2 = 49 is the first non-palindrome.",
                "type": "palindrome_number_theory",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "What is the tens digit of 6! + 7! + 8! + ... + 200!?",
        subject="counting-probability",
        num_examples=1,
    )

    assert "Q: What is the tens digit of 7! + 8! + 9! + ... + 400!?" in prompt
    assert "smallest power of 7" not in prompt.lower()


def test_few_shot_prefers_same_structure_precalculus_with_variable_change():
    generator = PromptGenerator()
    generator.example_dataset = {
        "pre-calculus": [
            {
                "problem": "Find the derivative of f(t) = t^3 + t.",
                "solution": "f'(t) = 3t^2 + 1",
                "type": "derivative",
            },
            {
                "problem": "Find the derivative of f(t) = sin(t).",
                "solution": "f'(t) = cos(t)",
                "type": "derivative",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Find the derivative of f(x) = x^3 + x.",
        subject="pre-calculus",
        num_examples=1,
    )

    assert "Q: Find the derivative of f(t) = t^3 + t." in prompt
    assert "Q: Find the derivative of f(t) = sin(t)." not in prompt


def test_few_shot_falls_back_when_no_same_structure_in_bank():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "Solve for y: 5^y + 4 = 125^y.",
                "solution": "Rewrite the base and solve for y.",
                "type": "solve_equation",
            }
        ],
        "general": [],
    }

    # No same-structure example exists; strict matching raises error.
    with pytest.raises(FewShotUnavailableError):
        generator.generate_few_shot(
            "Solve for x: x + 7 = 18",
            subject="algebra",
            num_examples=1,
        )


def test_few_shot_excludes_spacing_variant_identical_example():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "Solve for x: x + 7 = 18.",
                "solution": "x = 11",
                "type": "solve_equation",
            },
            {
                "problem": "Solve for y: y + 7 = 18.",
                "solution": "y = 11",
                "type": "solve_equation",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Solve for x: x+7=18",
        subject="algebra",
        num_examples=1,
    )

    assert "Q: Solve for y: y + 7 = 18." in prompt
    assert "Q: Solve for x: x + 7 = 18." not in prompt
    assert "Q: Solve for x: x+7=18" in prompt


def test_few_shot_uses_related_example_for_linear_equation_variant():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "Solve for y: y/4 - 3 = 2.",
                "solution": "Add 3 then multiply by 4 to get y = 20.",
                "type": "solve_equation",
            },
            {
                "problem": "Factor x^2 + 5x + 6.",
                "solution": "(x + 2)(x + 3)",
                "type": "factor",
            },
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Solve: x/6 - 4 = 1",
        subject="algebra",
        num_examples=1,
    )

    assert "Use the following examples only as style references." in prompt
    assert "Q: Solve for y: y/4 - 3 = 2." in prompt
    assert "Q: Solve: x/6 - 4 = 1" in prompt
