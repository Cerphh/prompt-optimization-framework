"""
Accuracy Scorer Module
Evaluates the accuracy of model responses using exact and symbolic matching.
"""

import re
from typing import Any, Optional
from sympy import sympify, simplify, parse_expr
from sympy.core.sympify import SympifyError

class AccuracyScorer:
    """
    Scores the accuracy of model responses against ground truth.
    
    Uses multiple matching strategies:
    1. Exact string match (case-insensitive)
    2. Numeric extraction and comparison
    3. Symbolic math evaluation (using SymPy)
    4. Fraction and expression matching
    5. Auto-solving simple arithmetic problems
    """
    
    def score(self, response: str, expected: Any, problem: str = None) -> float:
        """
        Score the accuracy of a response against ground truth.
        
        Args:
            response: Model's response
            expected: Expected/ground truth answer
            problem: Original problem (for context)
            
        Returns:
            Accuracy score between 0 and 1
        """
        if not response or len(response.strip()) == 0:
            return 0.0
        
        # If no ground truth provided, try to auto-solve simple problems
        if expected is None and problem:
            expected = self._auto_solve_simple_problem(problem)
        
        if expected is None:
            # Heuristic scoring when no ground truth
            return self._heuristic_score(response, problem)
        
        # Extract candidate answers from response
        candidates = self._extract_answers(response)
        expected_str = str(expected).strip()
        
        # Try multiple matching strategies
        for candidate in candidates:
            # Strategy 1: Exact match
            if self._exact_match(candidate, expected_str):
                return 1.0
            
            # Strategy 2: Numeric match
            if self._numeric_match(candidate, expected_str):
                return 1.0
            
            # Strategy 3: Symbolic math match
            if self._symbolic_match(candidate, expected_str):
                return 1.0
        
        # Partial credit for close matches
        for candidate in candidates:
            if self._partial_match(candidate, expected_str):
                return 0.5
        
        return 0.0
    
    def _auto_solve_simple_problem(self, problem: str) -> Optional[str]:
        """
        Automatically solve simple arithmetic problems to get ground truth.
        
        Handles:
        - Basic arithmetic: "1+1=?", "2*3", "10/5"
        - "What is X+Y" format
        - Direct expressions without question marks
        
        Args:
            problem: The problem statement
            
        Returns:
            The calculated answer as a string, or None if can't solve
        """
        if not problem:
            return None
        
        problem = problem.strip()
        
        # Pattern 1: "What is X op Y" format
        what_is_pattern = r'what\s+is\s+([\d\s+\-*/().]+)[?\s]*$'
        match = re.match(what_is_pattern, problem, re.IGNORECASE)
        if match:
            expression = match.group(1).strip()
            return self._evaluate_expression(expression)
        
        # Pattern 2: Direct arithmetic "X op Y = ?" or "X op Y?"
        simple_pattern = r'^([\d\s+\-*/().]+)\s*[=?]*\s*$'
        match = re.match(simple_pattern, problem)
        if match:
            expression = match.group(1).strip()
            # Only solve if it contains an operator
            if any(op in expression for op in ['+', '-', '*', '/', '×', '÷']):
                return self._evaluate_expression(expression)
        
        return None
    
    def _evaluate_expression(self, expression: str) -> Optional[str]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression like "1+1" or "2*3"
            
        Returns:
            Result as string, or None if can't evaluate
        """
        try:
            # Replace common symbols
            expression = expression.replace('×', '*').replace('÷', '/')
            expression = expression.replace('^', '**')
            
            # Use SymPy for safe evaluation
            result = sympify(expression)
            
            # Convert to float/int
            result_float = float(result)
            
            # Return as integer if it's a whole number
            if result_float.is_integer():
                return str(int(result_float))
            else:
                return str(result_float)
        except (SympifyError, TypeError, ValueError, AttributeError, ZeroDivisionError):
            return None
    
    def _extract_answers(self, response: str) -> list:
        """
        Extract potential answers from response.
        
        Looks for:
        - Numbers (integers, decimals, fractions)
        - "Answer:" lines
        - Final lines
        - Mathematical expressions
        """
        candidates = []
        
        # Look for "Answer:" pattern
        answer_pattern = r'(?:answer|result|solution)\s*[:\-=]\s*([^\n]+)'
        matches = re.findall(answer_pattern, response, re.IGNORECASE)
        candidates.extend([m.strip() for m in matches])
        
        # Extract all numbers
        numbers = re.findall(r'-?\d+\.?\d*(?:/\d+)?', response)
        candidates.extend(numbers)
        
        # Last line (often contains final answer)
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            candidates.append(lines[-1])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        return unique_candidates
    
    def _exact_match(self, candidate: str, expected: str) -> bool:
        """Check if candidate exactly matches expected (case-insensitive)."""
        return candidate.lower().strip() == expected.lower().strip()
    
    def _numeric_match(self, candidate: str, expected: str) -> bool:
        """Check if candidate numerically matches expected."""
        try:
            # Extract numbers
            candidate_nums = re.findall(r'-?\d+\.?\d*', candidate)
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            
            if candidate_nums and expected_nums:
                c_val = float(candidate_nums[0])
                e_val = float(expected_nums[0])
                # Allow small floating point differences
                return abs(c_val - e_val) < 0.0001
        except (ValueError, IndexError):
            pass
        return False
    
    def _symbolic_match(self, candidate: str, expected: str) -> bool:
        """
        Check if candidate symbolically matches expected using SymPy.
        
        Handles:
        - Algebraic expressions
        - Fractions (3/4 == 0.75)
        - Equations in different forms
        """
        try:
            # Clean the strings
            candidate_clean = self._clean_for_sympy(candidate)
            expected_clean = self._clean_for_sympy(expected)
            
            # Parse as symbolic expressions
            candidate_expr = sympify(candidate_clean)
            expected_expr = sympify(expected_clean)
            
            # Check if they simplify to the same thing
            diff = simplify(candidate_expr - expected_expr)
            return diff == 0
        except (SympifyError, TypeError, ValueError, AttributeError):
            pass
        return False
    
    def _clean_for_sympy(self, text: str) -> str:
        """Clean text for SymPy parsing."""
        # Remove common words
        text = re.sub(r'\b(is|equals|the|answer)\b', '', text, flags=re.IGNORECASE)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Extract mathematical expression
        # Look for numbers, operators, and parentheses
        match = re.search(r'[-+*/().\d\s^x]+', text)
        if match:
            return match.group(0).strip()
        return text.strip()
    
    def _partial_match(self, candidate: str, expected: str) -> bool:
        """Check for partial matches (substring containment)."""
        return expected.lower() in candidate.lower() or candidate.lower() in expected.lower()
    
    def _heuristic_score(self, response: str, problem: str = None) -> float:
        """
        Calculate heuristic accuracy score when no ground truth available.
        
        Based on response quality indicators.
        """
        if not response or len(response.strip()) == 0:
            return 0.0
        
        score = 0.5  # Base score
        
        # Has numerical answer
        if re.search(r'\d+', response):
            score += 0.2
        
        # Contains reasoning words
        if any(word in response.lower() for word in ['therefore', 'because', 'thus', 'so']):
            score += 0.1
        
        # Has clear answer indicator
        if re.search(r'(?:answer|result|solution)\s*[:\-=]', response, re.IGNORECASE):
            score += 0.2
        
        return min(score, 1.0)
