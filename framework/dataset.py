"""
Dataset Module
Manages math problems with ground truth answers for benchmarking.
"""

from typing import List, Dict, Any
import json

class MathDataset:
    """Dataset handler for math problems with ground truth answers."""
    
    def __init__(self):
        self.problems = []
    
    def add_problem(self, problem: str, answer: str, category: str = "general"):
        """
        Add a math problem to the dataset.
        
        Args:
            problem: The math problem text
            answer: Ground truth answer
            category: Problem category (arithmetic, algebra, calculus, etc.)
        """
        self.problems.append({
            "problem": problem,
            "answer": answer,
            "category": category,
            "id": len(self.problems)
        })
    
    def get_problems(self) -> List[Dict[str, Any]]:
        """Get all problems in the dataset."""
        return self.problems
    
    def get_problem(self, problem_id: int) -> Dict[str, Any]:
        """Get a specific problem by ID."""
        for problem in self.problems:
            if problem["id"] == problem_id:
                return problem
        return None
    
    def load_from_dict(self, data: List[Dict[str, str]]):
        """
        Load problems from a list of dictionaries.
        
        Args:
            data: List of dicts with 'problem', 'answer', and optional 'category'
        """
        for item in data:
            self.add_problem(
                problem=item["problem"],
                answer=item["answer"],
                category=item.get("category", "general")
            )
    
    def save_to_file(self, filepath: str):
        """Save dataset to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.problems, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
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
