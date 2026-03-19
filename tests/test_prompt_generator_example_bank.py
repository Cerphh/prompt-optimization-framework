from framework.prompt_generator import PromptGenerator


def test_zero_shot_is_bank_anchored_for_balanced_two_techniques():
    generator = PromptGenerator()

    problem = "Solve for x: 2x + 6 = 10"
    prompts = generator.generate_all_techniques(problem, subject="algebra")

    assert "zero_shot" in prompts
    assert "few_shot" in prompts
    assert "Style anchor:" not in prompts["zero_shot"]
    assert f"Q: {problem}" in prompts["zero_shot"]
    assert "A:" in prompts["zero_shot"]
    assert "Q:" in prompts["few_shot"]


def test_prepare_example_text_for_prompt_flattens_fragmented_lines():
    generator = PromptGenerator()

    noisy_problem = "Q: Simplify:\n\n1\n2\n×\n1024\n0.125\n×\n2\n12"

    cleaned = generator._prepare_example_text_for_prompt(noisy_problem, is_solution=False)

    assert cleaned.startswith("Simplify:")
    assert "\n" not in cleaned
    assert "1024" in cleaned


def test_generate_few_shot_cleans_latex_and_asy_noise_in_examples():
    generator = PromptGenerator()
    generator.example_dataset = {
        "algebra": [
            {
                "problem": "Simplify: \\displaystyle \\frac{\\frac 12\\times 1024}{0.125\\times 2^{12}}.",
                "solution": (
                    "Use exponent laws. \\[\\displaystyle \\frac{\\frac 12\\times 1024}{0.125\\times 2^{12}}"
                    " = \\boxed{1}.\\]\\n[asy]\\nsize(60);\\ndraw((0,0)--(1,1));\\n[/asy]"
                ),
            }
        ],
        "general": [],
    }

    prompt = generator.generate_few_shot(
        "Simplify: (1/2 * 1024)/(0.125 * 2^12).",
        subject="algebra",
        num_examples=1,
    )

    assert "\\displaystyle" not in prompt
    assert "[asy]" not in prompt
    assert "\\boxed" not in prompt
    assert "\\frac" not in prompt
    assert "Q: Simplify:" in prompt


def test_prepare_example_text_for_prompt_keeps_fraction_semantics():
    generator = PromptGenerator()

    text = "Simplify: \\displaystyle \\frac{\\frac 12\\times 1024}{0.125\\times 2^{12}}."
    cleaned = generator._prepare_example_text_for_prompt(text, is_solution=False)

    assert "(1)/(2)" in cleaned
    assert "(12)/(x)" not in cleaned
