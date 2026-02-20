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
        return f"Solve step-by-step, be concise.\n\n{problem}"

    def _normalize_problem_text(self, problem: str) -> str:
        """Normalize user input to avoid duplicated Q:/A: wrappers in prompts."""
        normalized = problem.strip()
        if normalized.lower().startswith("q:"):
            normalized = normalized[2:].lstrip()

        lines = normalized.splitlines()
        while lines and lines[-1].strip().lower() == "a:":
            lines.pop()

        return "\n".join(lines).strip()

    def _is_conditional_probability_problem(self, problem: str) -> bool:
        """Detect conditional probability phrasing to prioritize matching examples."""
        text = problem.lower()
        conditional_markers = [
            "given that",
            "given ",
            "|",
            "conditional",
            "p("
        ]
        probability_markers = ["probability", "random", "chance", "odds"]

        has_conditional = any(marker in text for marker in conditional_markers)
        has_probability = any(marker in text for marker in probability_markers)
        return has_conditional and has_probability

    def _is_conditional_probability_example(self, example: dict) -> bool:
        """Identify examples that teach conditional probability patterns."""
        example_text = (example["problem"] + " " + example["solution"]).lower()
        return (
            "given that" in example_text
            or "|" in example_text
        )
    
    def _detect_problem_keywords(self, problem: str) -> set:
        """
        Extract keywords from a problem to help match relevant examples.
        
        Args:
            problem: The math problem text
            
        Returns:
            Set of relevant keywords found
        """
        problem_lower = problem.lower()
        
        # Define keyword categories
        keywords = {
            # Algebra - solving
            'solve', 'find', 'calculate', 'determine', 'equation', 'equals', '=',
            # Algebra - operations
            'factor', 'expand', 'simplify', 'system',
            # Calculus
            'derivative', 'differentiate', 'integrate', 'integral', 'limit', 'lim',
            'd/dx', 'dy/dx', '∫', '∂',
            # Statistics - central tendency
            'mean', 'median', 'mode', 'average',
            # Statistics - spread
            'variance', 'standard deviation', 'range', 'deviation',
            # Probability - general
            'probability', 'p(', 'chance', 'odds', 'likely',
            # Probability - conditional
            'given that', 'given', '|', 'conditional',
            # Probability - objects
            'coin', 'coins', 'dice', 'die', 'card', 'ball', 'balls', 'bag',
            # Probability - concepts
            'flip', 'flipped', 'roll', 'rolled', 'draw', 'drawn',
            'heads', 'tails', 'outcome', 'outcomes', 'event', 'favorable',
            # Combinatorics
            'combinations', 'permutations', 'c(', 'p(', 'factorial',
            # Distributions
            'random', 'distribution', 'expected value', 'sample'
        }
        
        # Find matching keywords
        found_keywords = set()
        for keyword in keywords:
            if keyword in problem_lower:
                found_keywords.add(keyword)
        
        return found_keywords
    
    def _score_example_relevance(self, example: dict, problem: str, problem_keywords: set) -> float:
        """
        Score how relevant an example is to the given problem.
        
        Args:
            example: Example dict with 'problem' and 'solution'
            problem: The user's problem
            problem_keywords: Keywords extracted from user's problem
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0
        example_text = (example['problem'] + ' ' + example['solution']).lower()
        problem_lower = problem.lower()
        
        # Check keyword overlap (weighted by relevance)
        keyword_matches = sum(1 for kw in problem_keywords if kw in example_text)
        if problem_keywords:
            score += (keyword_matches / len(problem_keywords)) * 0.6
        
        # Bonus for similar operation words in problem statement
        example_lower = example['problem'].lower()
        
        operation_words = {
            'solve', 'find', 'calculate', 'factor', 'expand', 'simplify',
            'derivative', 'integrate', 'limit', 
            'mean', 'median', 'mode', 'probability', 'variance',
            'roll', 'flip', 'draw', 'combinations', 'permutations'
        }
        
        for word in operation_words:
            if word in problem_lower and word in example_lower:
                score += 0.3
                break
        
        # Extra bonus for matching specific objects (dice, coins, balls, etc.)
        specific_objects = ['dice', 'die', 'coin', 'ball', 'card', 'bag']
        for obj in specific_objects:
            if obj in problem_lower and obj in example_lower:
                score += 0.2
                break
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _select_relevant_examples(self, available_examples: list, problem: str, 
                                  num_examples: int, seed: int) -> list:
        """
        Select the most relevant examples for the given problem.
        
        Args:
            available_examples: List of available examples
            problem: The user's problem
            num_examples: Number of examples to select
            seed: Random seed for deterministic fallback
            
        Returns:
            List of selected examples
        """
        # Explicitly prioritize conditional probability examples when detected
        if self._is_conditional_probability_problem(problem):
            conditional_examples = [
                ex for ex in available_examples
                if isinstance(ex, dict) and self._is_conditional_probability_example(ex)
            ]
            if len(conditional_examples) >= num_examples:
                return conditional_examples[:num_examples]

        # Extract keywords from the problem
        problem_keywords = self._detect_problem_keywords(problem)
        
        # Score all examples
        scored_examples = []
        for example in available_examples:
            relevance = self._score_example_relevance(example, problem, problem_keywords)
            scored_examples.append((relevance, example))
        
        # Sort by relevance (highest first)
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # If we have highly relevant examples (score > 0.4), prioritize them
        highly_relevant = [ex for score, ex in scored_examples if score > 0.4]
        
        if len(highly_relevant) >= num_examples:
            # Use a seeded random selection from highly relevant examples
            # This maintains determinism while preferring relevant ones
            rng = random.Random(seed)
            return rng.sample(highly_relevant[:min(len(highly_relevant), num_examples * 3)], 
                            num_examples)
        else:
            # Fall back to top-ranked examples by relevance
            return [ex for _, ex in scored_examples[:num_examples]]
    
    def generate_few_shot(self, problem: str, subject: str = "general", num_examples: int = None) -> str:
        """
        Generate few-shot prompt: includes examples from the specified subject.
        Uses semantic matching to select the most relevant examples.
        
        Args:
            problem: The math problem
            subject: Subject category (algebra, statistics, calculus, general)
            num_examples: Number of examples to include (auto-determined by subject if None)
            
        Returns:
            Few-shot prompt with subject-specific, relevant examples
        """
        # Normalize user problem text (e.g., remove leading Q: and trailing A:)
        normalized_problem = self._normalize_problem_text(problem)

        # Auto-determine number of examples based on subject if not specified
        if num_examples is None:
            # Statistics and calculus benefit more from multiple examples
            num_examples = 2 if subject in ['statistics', 'calculus'] else 1
        
        # Get examples from the specified subject, default to general if not found
        if subject not in self.example_dataset:
            print(f"⚠ Warning: Subject '{subject}' not found in dataset, using 'algebra'")
            subject = "algebra"
        
        available_examples = self.example_dataset.get(subject, [])
        
        if not available_examples:
            print(f"⚠ Warning: No examples found for subject '{subject}'")
            # Return zero-shot if no examples
            return f"{problem}"
        
        # Use problem text as seed for deterministic selection
        # Same problem always gets same examples
        seed = hash(normalized_problem) % (2**32)
        
        # Select relevant examples (smart selection based on problem content)
        num_to_select = min(num_examples, len(available_examples))
        selected_examples = self._select_relevant_examples(
            available_examples, normalized_problem, num_to_select, seed
        )
        
        # Format examples (concise format for speed)
        examples_text = "\n\n".join([
            f"Q: {ex['problem']}\nA: {ex['solution']}"
            for ex in selected_examples
        ])
        
        return (
            "Solve step-by-step, be concise.\n\n"
            f"{examples_text}\n\n"
            f"Q: {normalized_problem}\n"
            "A:"
        )
    
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
