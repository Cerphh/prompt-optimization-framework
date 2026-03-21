from framework.prompt_generator import PromptGenerator


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

    assert "Q: Find the derivative of f(x) = x^3." in prompt


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
                "problem": "If x = 3 and y = 2, then what is the value of 2x^3 - 3y^2?",
                "solution": "2(3^3) - 3(2^2) = 54 - 12 = 42",
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

    assert "Q: If x = 3 and y = 2, then what is the value of 2x^3 - 3y^2?" in prompt


def test_few_shot_strict_match_in_counting_probability_domain():
    generator = PromptGenerator()
    generator.example_dataset = {
        "counting-probability": [
            {
                "problem": "A coin is flipped 4 times. What is the probability of exactly 2 heads?",
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

    assert "probability of exactly 2 heads" in prompt.lower()
    assert "arranged in a row" not in prompt.lower()


def test_few_shot_strict_match_in_precalculus_domain():
    generator = PromptGenerator()
    generator.example_dataset = {
        "pre-calculus": [
            {
                "problem": "Find the derivative of f(x) = x^3.",
                "solution": "f'(x) = 3x^2",
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

    assert "find the derivative of f(x) = x^3" in prompt.lower()
    assert "evaluate the integral" not in prompt.lower()
