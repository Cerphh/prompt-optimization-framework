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
