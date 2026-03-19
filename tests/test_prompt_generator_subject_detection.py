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


def test_detect_primary_intent_rate_word_problem_maps_to_ratio_proportion():
    generator = PromptGenerator()

    question = (
        "A freight train travels 1 mile in 1 minute 30 seconds. "
        "At this rate, how many miles will the train travel in 1 hour?"
    )

    assert generator._detect_primary_intent(question) == "ratio_proportion"


def test_classify_subject_rate_word_problem_as_algebra():
    generator = PromptGenerator()

    question = (
        "A freight train travels 1 mile in 1 minute 30 seconds. "
        "At this rate, how many miles will the train travel in 1 hour?"
    )

    assert generator.classify_subject(question) == "algebra"


def test_detect_primary_intent_compare_values_multiple_choice():
    generator = PromptGenerator()

    question = "Which of the following has the least value? A = 2, B = 4/4, C = 8/8."

    assert generator._detect_primary_intent(question) == "compare_values"
    assert generator.classify_subject(question) == "algebra"


def test_classify_subject_quadratic_real_solutions_prompt_as_algebra():
    generator = PromptGenerator()

    question = "Solve for all real solutions: x^2 - 9x + 20 = 0."

    assert generator._detect_primary_intent(question) == "real_solutions"
    assert generator.classify_subject(question) == "algebra"


def test_classify_subject_unicode_quadratic_real_solutions_as_algebra():
    generator = PromptGenerator()

    question = "Solve for all real solutions: x² - 9x + 20 = 0."

    assert generator._detect_primary_intent(question) == "real_solutions"
    assert generator.classify_subject(question) == "algebra"


def test_classify_subject_equation_only_prompt_defaults_to_algebra():
    generator = PromptGenerator()

    question = "x² - 9x + 20 = 0"

    assert generator._detect_primary_intent(question) == "solve_equation"
    assert generator.classify_subject(question) == "algebra"


def test_classify_subject_function_analysis_prompt_stays_precalculus():
    generator = PromptGenerator()

    question = "Find the domain and range of f(x) = (x^2 - 9x + 20)/(x - 4)."

    assert generator._detect_primary_intent(question) == "function_analysis"
    assert generator.classify_subject(question) == "pre-calculus"
