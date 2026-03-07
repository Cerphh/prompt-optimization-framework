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
