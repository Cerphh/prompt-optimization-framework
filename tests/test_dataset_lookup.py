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
