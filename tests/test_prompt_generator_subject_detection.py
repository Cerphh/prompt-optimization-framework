from framework.prompt_generator import PromptGenerator


def test_classify_subject_detects_precalculus_trigonometry():
    generator = PromptGenerator()

    subject = generator.classify_subject("Find the period of y = sin(2x) and state its range.")

    assert subject == "pre-calculus"


def test_classify_subject_detects_counting_arrangements():
    generator = PromptGenerator()

    subject = generator.classify_subject(
        "How many ways can 6 students be arranged in a row if two specific students must sit together?"
    )

    assert subject == "counting-probability"


def test_detect_primary_intent_counting_and_expected_value_goals():
    generator = PromptGenerator()

    assert (
        generator._detect_primary_intent("How many ways can 5 books be arranged on a shelf?")
        == "counting_arrangements"
    )
    assert (
        generator._detect_primary_intent("Find the expected value of the number shown on a fair die.")
        == "expected_value"
    )


def test_detect_primary_intent_precalculus_goal_patterns():
    generator = PromptGenerator()

    assert generator._detect_primary_intent("Find sin(75 degrees).") == "trigonometric"
    assert (
        generator._detect_primary_intent("Find the domain and asymptote of f(x) = 1/(x-2).")
        == "function_analysis"
    )
    assert (
        generator._detect_primary_intent("Find the 12th term of a geometric sequence.")
        == "sequence_series"
    )
