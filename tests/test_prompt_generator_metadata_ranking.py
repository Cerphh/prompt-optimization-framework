from framework.prompt_generator import PromptGenerator


def test_normalize_example_dataset_keeps_legacy_and_metadata_fields():
    generator = PromptGenerator()

    normalized = generator._normalize_example_dataset(
        {
            "algebra": {
                "basic": [
                    {
                        "problem": "Solve for x: x + 2 = 5",
                        "solution": "x = 3",
                    },
                    {
                        "problem": "Given x = 3 and y = 2, evaluate x^2 + y.",
                        "solution": "3^2 + 2 = 11",
                        "type": "substitution",
                        "concept": "expression evaluation",
                        "skills": ["substitution", "arithmetic"],
                        "format": {
                            "Template": "Expression With Assignments",
                            "Has Assignment": True,
                            "Assigned Variable Count": 2,
                        },
                        "difficulty": "easy",
                        "tags": ["value", "assignment"],
                        "constraints": ["positive_value"],
                        "anchor_priority": 0.9,
                    },
                ]
            }
        }
    )

    assert len(normalized["algebra"]) == 2

    legacy = normalized["algebra"][0]
    assert legacy["problem"] == "Solve for x: x + 2 = 5"
    assert legacy["solution"] == "x = 3"
    assert legacy["difficulty"] == "basic"

    upgraded = normalized["algebra"][1]
    assert upgraded["type"] == "evaluate_substitution"
    assert upgraded["concept"] == "expression_evaluation"
    assert upgraded["skills"] == ["substitution", "arithmetic"]
    assert upgraded["format"]["template"] == "expression_with_assignments"
    assert upgraded["format"]["has_assignment"] is True
    assert upgraded["difficulty"] == "basic"
    assert upgraded["constraints"] == ["positive_values"]
    assert upgraded["anchor_priority"] == 0.9


def test_select_relevant_examples_prefers_type_matched_metadata_pool():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "Given x = 4 and y = 3, evaluate x^2 + y.",
            "solution": "4^2 + 3 = 19",
            "type": "substitution",
            "format": "expression_with_assignments",
        },
        {
            "problem": "A quantity varies directly with z and equals 10 when z = 2.",
            "solution": "Use direct variation.",
            "type": "variation",
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "What is the value of x^2 + y when x = 4 and y = 3?",
        num_examples=1,
    )

    assert selected[0]["problem"] == "Given x = 4 and y = 3, evaluate x^2 + y."


def test_select_relevant_examples_prefers_constraint_aligned_examples():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "What is the positive difference between 120% of 30 and 130% of 20?",
            "solution": "36 - 26 = 10",
            "type": "variation",
            "constraints": ["positive_values"],
        },
        {
            "problem": "Compare 120% of 30 and 130% of 20.",
            "solution": "Compute both and compare.",
            "type": "variation",
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "What is the positive difference between 120% of 30 and 130% of 20?",
        num_examples=1,
    )

    assert selected[0]["constraints"] == ["positive_values"]


def test_select_relevant_examples_positive_value_substitution_not_overfiltered():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "If x = 3 and y = 2, evaluate x^3 - 2y.",
            "solution": "3^3 - 2(2) = 23",
            "type": "evaluate_substitution",
            "constraints": [],
        },
        {
            "problem": "Find the positive value of t where t^2 - 9 = 0.",
            "solution": "t = 3",
            "type": "solve_equation",
            "constraints": ["positive_values"],
        },
        {
            "problem": "Find the positive difference in x-coordinates of two lines.",
            "solution": "Compute each x, then subtract.",
            "type": "solve_equation",
            "constraints": ["positive_values"],
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "What is the positive value of the expression x^3 - 2y when x = 5 and y = 2?",
        num_examples=1,
    )

    assert selected[0]["type"] == "evaluate_substitution"


def test_select_relevant_examples_rate_word_problem_not_misread_as_counting():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "A train travels 1 mile in 2 minutes. At this rate, how many miles in 1 hour?",
            "solution": "Use unit rate and proportion.",
            "type": "ratio_proportion",
            "constraints": [],
        },
        {
            "problem": "How many ways can 6 students be arranged in a row?",
            "solution": "Use permutations.",
            "type": "counting_arrangements",
            "constraints": [],
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "A freight train travels 1 mile in 1 minute 30 seconds. At this rate, how many miles will the train travel in 1 hour?",
        num_examples=1,
    )

    assert selected[0]["type"] == "ratio_proportion"


def test_select_relevant_examples_rate_query_allows_mislabeled_speed_examples():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "A cyclist travels 1 mile in 3 minutes. At this rate, how many miles in 1 hour?",
            "solution": "Use unit rate and multiply by 60 minutes.",
            "type": "counting_arrangements",
            "constraints": [],
        },
        {
            "problem": "At constant temperature, pressure is inversely proportional to volume.",
            "solution": "Use inverse variation.",
            "type": "ratio_proportion",
            "constraints": [],
        },
        {
            "problem": "How many ways can 7 students sit in a row?",
            "solution": "Use permutations.",
            "type": "counting_arrangements",
            "constraints": [],
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "A freight train travels 1 mile in 1 minute 30 seconds. At this rate, how many miles will the train travel in 1 hour?",
        num_examples=1,
    )

    assert "mile" in selected[0]["problem"].lower()
    assert "hour" in selected[0]["problem"].lower()


def test_select_relevant_examples_prefers_compare_values_templates():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "Which of the following has the least value? A = 3/2, B = 5/4, C = 9/8.",
            "solution": "Compare each value: A = 1.5, B = 1.25, C = 1.125, so C is least.",
            "type": "comparison",
            "constraints": [],
        },
        {
            "problem": "If x = 3 and y = 2, evaluate x^2 + y.",
            "solution": "3^2 + 2 = 11",
            "type": "evaluate_substitution",
            "constraints": [],
        },
        {
            "problem": "Solve for x: 2x + 9 = 21.",
            "solution": "x = 6",
            "type": "solve_equation",
            "constraints": [],
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "Which of the following has the least value? A = 2, B = 4/4, C = 8/8.",
        num_examples=1,
    )

    assert "least value" in selected[0]["problem"].lower()


def test_select_relevant_examples_compare_values_returns_available_when_no_match():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "Solve for x: 3x + 9 = 24.",
            "solution": "x = 5",
            "type": "solve_equation",
        }
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "Which of the following has the least value? A = 2, B = 4/4, C = 8/8.",
        num_examples=1,
    )

    # Without template fallback, only the available example is returned
    assert len(selected) == 1
    assert selected[0]["type"] == "solve_equation"


def test_select_relevant_examples_enforces_linear_equation_family_match():
    generator = PromptGenerator()

    available_examples = [
        {
            "problem": "Solve for x: 2x + 9 = 21.",
            "solution": "x = 6",
            "type": "solve_equation",
        },
        {
            "problem": "Solve for x: x^2 - 5x + 6 = 0.",
            "solution": "(x-2)(x-3)=0, so x=2 or x=3",
            "type": "solve_equation",
        },
        {
            "problem": "Solve the system: x + y = 8 and x - y = 2.",
            "solution": "x = 5, y = 3",
            "type": "solve_equation",
        },
    ]

    selected = generator._select_relevant_examples(
        available_examples,
        "Solve for x: 3x + 12 = 0.",
        num_examples=1,
        subject="algebra",
    )

    assert selected[0]["problem"] == "Solve for x: 2x + 9 = 21."
