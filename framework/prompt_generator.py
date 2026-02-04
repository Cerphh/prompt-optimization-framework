"""
Prompt Generator Module
Generates multiple prompting strategies for comparative benchmarking.
"""

from typing import List, Dict

class PromptGenerator:
    """
    Generates prompts using multiple techniques for research benchmarking.
    
    Implements two prompting strategies:
    1. Zero-shot: Direct question without examples
    2. Few-shot: Includes examples before the question
    """
    
    def __init__(self):
        # Few-shot examples for math problems
        self.few_shot_examples = [
            {
                "problem": "What is 12 + 8?",
                "solution": "12 + 8 = 20"
            },
            {
                "problem": "Calculate 3 * 7",
                "solution": "3 * 7 = 21"
            }
        ]
    
    def generate_zero_shot(self, problem: str) -> str:
        """
        Generate zero-shot prompt: direct question without context.
        
        Args:
            problem: The math problem
            
        Returns:
            Zero-shot prompt
        """
        return f"{problem}"
    
    def generate_few_shot(self, problem: str) -> str:
        """
        Generate few-shot prompt: includes examples before the question.
        
        Args:
            problem: The math problem
            
        Returns:
            Few-shot prompt with examples
        """
        examples_text = "\n\n".join([
            f"Problem: {ex['problem']}\nSolution: {ex['solution']}"
            for ex in self.few_shot_examples
        ])
        
        return f"""Here are some examples:

{examples_text}

Now solve this problem:
Problem: {problem}
Solution:"""
    
    def generate_all_techniques(self, problem: str) -> Dict[str, str]:
        """
        Generate prompts using all techniques.
        
        Args:
            problem: The math problem
            
        Returns:
            Dictionary mapping technique name to prompt
        """
        return {
            "zero_shot": self.generate_zero_shot(problem),
            "few_shot": self.generate_few_shot(problem)
        }
    
    def get_technique_names(self) -> List[str]:
        """Get list of all available prompting techniques."""
        return ["zero_shot", "few_shot"]
