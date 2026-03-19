"""
Dataset Module
Manages math problems with ground truth answers for benchmarking.
"""

from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import json

class MathDataset:
    """Dataset handler for math problems with ground truth answers."""
    
    def __init__(self):
        self.problems: List[Dict[str, Any]] = []
        self._problems_by_id: Dict[int, Dict[str, Any]] = {}
    
    def add_problem(
        self,
        problem: str,
        answer: str,
        category: str = "general",
        difficulty: Optional[str] = None,
    ):
        """
        Add a math problem to the dataset.
        
        Args:
            problem: The math problem text
            answer: Ground truth answer
            category: Problem category (arithmetic, algebra, pre-calculus, counting & probability, etc.)
        """
        record = {
            "problem": problem,
            "answer": answer,
            "category": category,
            "id": len(self.problems)
        }
        if difficulty is not None:
            record["difficulty"] = difficulty
        self.problems.append(record)
        self._problems_by_id[record["id"]] = record
    
    def get_problems(self) -> List[Dict[str, Any]]:
        """Get all problems in the dataset."""
        return self.problems
    
    def get_problem(self, problem_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific problem by ID."""
        return self._problems_by_id.get(problem_id)
    
    def _iter_problem_records(self, data: Any) -> Iterable[Dict[str, Any]]:
        """Flatten old and new dataset JSON shapes into normalized problem records."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return

        if not isinstance(data, dict):
            raise ValueError("Dataset must be a list of problems or a category mapping")

        for category, category_value in data.items():
            if isinstance(category_value, list):
                for item in category_value:
                    if isinstance(item, dict):
                        yield {
                            **item,
                            "category": item.get("category", category),
                        }
                continue

            if not isinstance(category_value, dict):
                continue

            for difficulty, problems in category_value.items():
                if not isinstance(problems, list):
                    continue

                for item in problems:
                    if isinstance(item, dict):
                        yield {
                            **item,
                            "category": item.get("category", category),
                            "difficulty": item.get("difficulty", difficulty),
                        }

    def load_from_dict(self, data: Any):
        """
        Load problems from JSON-like structures.
        
        Args:
            data: List of problem dicts or category/difficulty mapping
        """
        for item in self._iter_problem_records(data):
            answer = item.get("answer") or item.get("solution")
            if not item.get("problem") or answer is None:
                continue

            self.add_problem(
                problem=item["problem"],
                answer=answer,
                category=item.get("category", "general"),
                difficulty=item.get("difficulty"),
            )
    
    def save_to_file(self, filepath: str):
        """Save dataset to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.problems, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load dataset from JSON file."""
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            self.load_from_dict(data)
    
    def size(self) -> int:
        """Get the number of problems in the dataset."""
        return len(self.problems)


def get_sample_dataset() -> MathDataset:
    """
    Create a sample dataset of math problems for testing.
    
    Returns:
        MathDataset with sample problems
    """
    dataset = MathDataset()

    json_path = Path(__file__).with_name("example_problems.json")
    if json_path.exists():
        dataset.load_from_file(str(json_path))
        return dataset
    
    # Arithmetic problems
    dataset.add_problem("What is 15 + 27?", "42", "arithmetic")
    dataset.add_problem("Calculate 144 / 12", "12", "arithmetic")
    dataset.add_problem("What is 7 * 8?", "56", "arithmetic")
    
    # Algebra problems
    dataset.add_problem("Solve for x: 2x + 5 = 15", "5", "algebra")
    dataset.add_problem("If 3x = 21, what is x?", "7", "algebra")
    dataset.add_problem("Solve: x^2 = 25", "5", "algebra")  # Accept both +5 and -5
    
    # Word problems
    dataset.add_problem(
        "A train travels 120 miles in 2 hours. What is its average speed in miles per hour?",
        "60",
        "word_problem"
    )
    dataset.add_problem(
        "If a pizza is cut into 8 slices and you eat 3, what fraction of the pizza remains?",
        "5/8",
        "word_problem"
    )
    
    # More complex problems
    dataset.add_problem(
        "What is the sum of the first 10 positive integers?",
        "55",
        "arithmetic"
    )
    dataset.add_problem(
        "A rectangle has length 8 and width 5. What is its area?",
        "40",
        "geometry"
    )
    
    return dataset
