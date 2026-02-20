"""
Prompt Generator Module
Generates multiple prompting strategies for comparative benchmarking.
"""

from typing import List, Dict
import random
import json
import os

class PromptGenerator:
    """
    Generates prompts using multiple techniques for research benchmarking.
    
    Implements two prompting strategies:
    1. Zero-shot: Direct question without examples
    2. Few-shot: Includes examples before the question
    """
    
    def __init__(self):
        """Initialize the prompt generator and load example dataset from JSON."""
        # Get the path to the JSON file (in the same directory as this module)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "example_problems.json")
        
        # Load examples from JSON file
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.example_dataset = json.load(f)
            print(f"✓ Loaded example dataset from {json_path}")
        except FileNotFoundError:
            print(f"⚠ Warning: Could not find {json_path}, using minimal fallback examples")
            # Fallback to minimal examples if JSON file not found
            self.example_dataset = {
                "general": [
                    {"problem": "What is 12 + 8?", "solution": "12 + 8 = 20"},
                    {"problem": "Calculate 3 × 7", "solution": "3 × 7 = 21"}
                ],
                "algebra": [
                    {"problem": "Solve for x: 3x + 7 = 22", "solution": "3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 15/3\nx = 5"}
                ],
                "statistics": [
                    {"problem": "Find the mean of: 4, 8, 12, 16, 20", "solution": "Mean = (4 + 8 + 12 + 16 + 20)/5\n= 60/5\n= 12"}
                ],
                "calculus": [
                    {"problem": "Find the derivative: f(x) = x³", "solution": "f(x) = x³\nf'(x) = 3x⁽³⁻¹⁾\nf'(x) = 3x²"}
                ]
            }
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Error parsing JSON file: {e}")
            # Use minimal fallback if JSON is malformed
            self.example_dataset = {
                "general": [
                    {"problem": "What is 12 + 8?", "solution": "12 + 8 = 20"}
                ]
            }
    
    
    def generate_zero_shot(self, problem: str) -> str:
        """
        Generate zero-shot prompt: direct question without context.
        
        Args:
            problem: The math problem
            
        Returns:
            Zero-shot prompt
        """
        return f"{problem}"
    
    def generate_few_shot(self, problem: str, subject: str = "general", num_examples: int = 2) -> str:
        """
        Generate few-shot prompt: includes examples from the specified subject.
        Uses problem text as seed for deterministic example selection.
        
        Args:
            problem: The math problem
            subject: Subject category (algebra, statistics, calculus, general)
            num_examples: Number of examples to include (default: 2)
            
        Returns:
            Few-shot prompt with subject-specific examples
        """
        # Get examples from the specified subject, default to general if not found
        if subject not in self.example_dataset:
            print(f"⚠ Warning: Subject '{subject}' not found in dataset, using 'general'")
            subject = "general"
        
        available_examples = self.example_dataset.get(subject, [])
        
        if not available_examples:
            print(f"⚠ Warning: No examples found for subject '{subject}'")
            # Return zero-shot if no examples
            return f"{problem}"
        
        # Use problem text as seed for deterministic selection
        # Same problem always gets same examples
        seed = hash(problem) % (2**32)
        rng = random.Random(seed)
        
        # Select random examples (or first N if fewer examples available)
        num_to_select = min(num_examples, len(available_examples))
        selected_examples = rng.sample(available_examples, num_to_select)
        
        # Format examples
        examples_text = "\n\n".join([
            f"Problem: {ex['problem']}\nSolution: {ex['solution']}"
            for ex in selected_examples
        ])
        
        return f"""Here are some {subject} examples:

{examples_text}

Now solve this problem:
Problem: {problem}
Solution:"""
    
    def generate_all_techniques(self, problem: str, subject: str = "general") -> Dict[str, str]:
        """
        Generate prompts using all techniques.
        
        Args:
            problem: The math problem
            subject: Subject category for few-shot examples
            
        Returns:
            Dictionary mapping technique name to prompt
        """
        return {
            "zero_shot": self.generate_zero_shot(problem),
            "few_shot": self.generate_few_shot(problem, subject=subject)
        }
    
    def get_technique_names(self) -> List[str]:
        """Get list of all available prompting techniques."""
        return ["zero_shot", "few_shot"]
    
    def get_available_subjects(self) -> List[str]:
        """Get list of available subject categories."""
        return list(self.example_dataset.keys())
