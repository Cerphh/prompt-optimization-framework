from framework.dataset import MathDataset


def test_dataset_get_problem_returns_expected_record():
    dataset = MathDataset()
    dataset.add_problem("What is 1 + 1?", "2", "arithmetic")
    dataset.add_problem("What is 2 + 2?", "4", "arithmetic")

    problem = dataset.get_problem(1)

    assert problem is not None
    assert problem["problem"] == "What is 2 + 2?"
    assert problem["answer"] == "4"


def test_dataset_get_problem_returns_none_for_missing_id():
    dataset = MathDataset()
    dataset.add_problem("What is 1 + 1?", "2", "arithmetic")

    assert dataset.get_problem(999) is None


def test_dataset_load_from_nested_category_and_difficulty_mapping():
    dataset = MathDataset()
    dataset.load_from_dict(
        {
            "algebra": {
                "basic": [
                    {
                        "problem": "Solve for x: x + 1 = 2",
                        "solution": "x = 1",
                    }
                ]
            },
            "counting-probability": {
                "advanced": [
                    {
                        "problem": "What is P(A|B)?",
                        "answer": "1/2",
                    }
                ]
            },
        }
    )

    assert dataset.size() == 2
    assert dataset.get_problem(0)["category"] == "algebra"
    assert dataset.get_problem(0)["difficulty"] == "basic"
    assert dataset.get_problem(0)["answer"] == "x = 1"
    assert dataset.get_problem(1)["category"] == "counting-probability"
    assert dataset.get_problem(1)["difficulty"] == "advanced"
    assert dataset.get_problem(1)["answer"] == "1/2"
