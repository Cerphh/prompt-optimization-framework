"""
Accuracy Scorer Module
Evaluates the accuracy of model responses using exact and symbolic matching.
"""

import re
from typing import Any, Optional
from sympy import sympify, simplify, parse_expr, symbols, solve
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

        # If still no ground truth, try symbolic equation solving
        if expected is None and problem:
            expected = self._auto_solve_equation_problem(problem)

        # Special handling for solved root sets
        if isinstance(expected, dict) and expected.get("type") == "equation_roots":
            return self._score_equation_roots(response, expected)
        
        if expected is None:
            # Heuristic scoring when no ground truth
            return self._heuristic_score(response, problem)

        expected_str = str(expected).strip()

        # Prefer explicit/final answer lines over intermediate reasoning steps.
        # This avoids false positives where a correct intermediate number appears
        # before an incorrect final answer.
        priority_candidates = self._extract_priority_answers(response)
        if priority_candidates and self._has_explicit_answer_signal(response):
            for candidate in priority_candidates:
                if self._strong_match(candidate, expected_str):
                    return 1.0

            # Explicit final-answer signaling exists but none matched strongly.
            # Do not award partial credit from intermediate overlaps.
            return 0.0
        
        # Extract candidate answers from response
        candidates = self._extract_answers(response)
        
        # Try multiple matching strategies
        for candidate in candidates:
            if self._strong_match(candidate, expected_str):
                return 1.0
        
        # Partial credit for close matches
        for candidate in candidates:
            if self._partial_match(candidate, expected_str):
                return 0.5
        
        return 0.0

    def _strong_match(self, candidate: str, expected: str) -> bool:
        """Apply strict match strategies in priority order."""
        if self._exact_match(candidate, expected):
            return True
        if self._numeric_match(candidate, expected):
            return True
        if self._symbolic_match(candidate, expected):
            return True
        return False

    def _unique_preserve_order(self, values: list) -> list:
        """Remove duplicates while preserving order."""
        seen = set()
        unique_values = []
        for value in values:
            item = str(value).strip()
            if item and item not in seen:
                seen.add(item)
                unique_values.append(item)
        return unique_values

    def _extract_priority_answers(self, response: str) -> list:
        """Extract high-confidence final-answer candidates from response text."""
        candidates = []

        # Explicit answer lines have highest confidence.
        answer_pattern = r'(?im)^\s*(?:final\s+answer|answer|result|solution)\s*[:\-=]\s*([^\n]+)'
        matches = re.findall(answer_pattern, response, re.IGNORECASE)
        for match in matches:
            value = str(match).strip()
            if not value:
                continue
            candidates.append(value)
            inline_expression = self._extract_inline_math_candidate(value)
            if inline_expression:
                candidates.append(inline_expression)

        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            if self._is_blank_explicit_answer_line(lines[-1]):
                # If model ends with "Final answer:" but no payload, salvage
                # the nearest prior math-like line as the likely final expression.
                for previous_line in reversed(lines[:-1]):
                    if self._looks_like_answer_candidate(previous_line):
                        candidates.append(previous_line)
                        inline_expression = self._extract_inline_math_candidate(previous_line)
                        if inline_expression:
                            candidates.append(inline_expression)
                        break

            candidates.append(lines[-1])
            inline_expression = self._extract_inline_math_candidate(lines[-1])
            if inline_expression:
                candidates.append(inline_expression)

            # Include a penultimate line if it looks like a concluding statement.
            if len(lines) >= 2 and re.search(r'(?i)\b(therefore|thus|hence|so|final)\b', lines[-2]):
                candidates.append(lines[-2])
                inline_expression = self._extract_inline_math_candidate(lines[-2])
                if inline_expression:
                    candidates.append(inline_expression)

        return self._unique_preserve_order(candidates)

    def _has_explicit_answer_signal(self, response: str) -> bool:
        """Determine whether response provides an explicit final-answer target."""
        return bool(
            re.search(
                r'(?i)\b(?:final\s+answer|answer|result|solution)\s*[:\-=]',
                response,
            )
        )

    def _extract_inline_math_candidate(self, text: str) -> Optional[str]:
        """Extract a likely math expression from narrative text."""
        value = str(text or "").strip()
        if not value:
            return None

        value = value.rstrip('.;,')

        if ':' in value:
            right = value.rsplit(':', 1)[-1].strip().rstrip('.;,')
            if self._looks_math_like(right):
                return right

        lead_in_match = re.search(
            r'(?i)\b(?:is|equals|becomes|gives)\s*[:=]\s*([^\n]+)$',
            value,
        )
        if lead_in_match:
            tail = lead_in_match.group(1).strip().rstrip('.;,')
            if self._looks_math_like(tail):
                return tail

        return None

    def _looks_math_like(self, text: str) -> bool:
        """Heuristic check for whether text resembles math content."""
        value = str(text or "").strip()
        if not value:
            return False

        has_digit = bool(re.search(r'\d', value))
        has_variable = bool(re.search(r'[A-Za-z]', value))
        has_operator = bool(re.search(r'[+\-*/^=()]', value))

        if has_digit and (has_operator or has_variable):
            return True
        if has_variable and has_operator:
            return True

        return bool(re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?', value))

    def _is_blank_explicit_answer_line(self, line: str) -> bool:
        """Check whether a line is an explicit answer label with empty payload."""
        return bool(
            re.match(
                r'(?i)^\s*(?:final\s+answer|answer|result|solution)\s*[:\-=]\s*$',
                line or "",
            )
        )

    def _looks_like_answer_candidate(self, text: str) -> bool:
        """Heuristic check for whether a line resembles a final answer."""
        value = (text or "").strip().lower()
        if not value:
            return False

        if "=" in value:
            return True
        if re.search(r'-?\d', value):
            return True

        return False
    
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

        # Start with high-confidence final candidates.
        candidates.extend(self._extract_priority_answers(response))
        
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
        
        return self._unique_preserve_order(candidates)
    
    def _exact_match(self, candidate: str, expected: str) -> bool:
        """Check if candidate exactly matches expected (case-insensitive)."""
        candidate_norm = self._normalize_answer_text(candidate).lower()
        expected_norm = self._normalize_answer_text(expected).lower()

        if candidate_norm == expected_norm:
            return True

        # Also compare compact forms to ignore spacing differences.
        candidate_compact = re.sub(r'\s+', '', candidate_norm)
        expected_compact = re.sub(r'\s+', '', expected_norm)
        return candidate_compact == expected_compact

    def _normalize_answer_text(self, text: str) -> str:
        """Normalize answer text for resilient exact matching."""
        value = str(text or "").strip()
        value = re.sub(
            r'(?i)^\s*(?:final\s+answer|answer|result|solution)\s*[:\-=]\s*',
            '',
            value,
        ).strip()

        inline_expression = self._extract_inline_math_candidate(value)
        if inline_expression:
            value = inline_expression

        if '=' in value:
            value = value.split('=')[-1].strip()

        value = value.rstrip('.;,')
        return value
    
    def _numeric_match(self, candidate: str, expected: str) -> bool:
        """Check if candidate numerically matches expected."""
        try:
            c_val = self._parse_numeric_value(candidate)
            e_val = self._parse_numeric_value(expected)

            if c_val is not None and e_val is not None:
                # Allow small floating point differences
                return abs(c_val - e_val) < 0.0001
        except (ValueError, IndexError):
            pass
        return False

    def _parse_numeric_value(self, text: str) -> Optional[float]:
        """Parse a likely numeric answer (integer/decimal/fraction) from text."""
        if text is None:
            return None

        value = str(text).strip()
        if not value:
            return None

        # Strip common answer prefixes.
        value = re.sub(
            r'(?i)^\s*(?:final\s+answer|answer|result|solution)\s*[:\-=]\s*',
            '',
            value,
        ).strip()

        # Prefer the right-hand side for assignments like "x = 5".
        if '=' in value:
            value = value.split('=')[-1].strip()

        # Reject tokens like "15a" or "x15" that are not standalone numbers.
        if re.search(r'\d[a-zA-Z]|[a-zA-Z]\d', value):
            return None

        fraction_pattern = r'[-+]?\d+(?:\.\d+)?\s*/\s*[-+]?\d+(?:\.\d+)?'
        decimal_pattern = r'[-+]?\d+(?:\.\d+)?'

        if re.fullmatch(fraction_pattern, value):
            left, right = re.split(r'\s*/\s*', value, maxsplit=1)
            denominator = float(right)
            if abs(denominator) < 1e-12:
                return None
            return float(left) / denominator

        if re.fullmatch(decimal_pattern, value):
            return float(value)

        # Fallback: use a single numeric token when unambiguous.
        tokens = re.findall(rf'{fraction_pattern}|{decimal_pattern}', value)
        if len(tokens) != 1:
            return None

        token = tokens[0].strip()
        if '/' in token:
            left, right = re.split(r'\s*/\s*', token, maxsplit=1)
            denominator = float(right)
            if abs(denominator) < 1e-12:
                return None
            return float(left) / denominator

        return float(token)
    
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
        cleaned = self._normalize_answer_text(text)

        inline_expression = self._extract_inline_math_candidate(cleaned)
        if inline_expression:
            cleaned = inline_expression

        cleaned = self._normalize_math_text(cleaned)
        cleaned = cleaned.replace('^', '**')

        # Normalize implicit multiplication: 2x, x(, )x, and mn -> 2*x, x*(, )*x, m*n
        cleaned = re.sub(r'(?<=\d)\s*(?=[A-Za-z(])', '*', cleaned)
        cleaned = re.sub(r'(?<=[A-Za-z)])\s*(?=\d|\()', '*', cleaned)
        cleaned = re.sub(r'(?<=[A-Za-z])\s*(?=[A-Za-z])', '*', cleaned)

        # Keep only chars relevant to symbolic parsing.
        cleaned = re.sub(r'[^A-Za-z0-9+\-*/()._=\s*]', ' ', cleaned)

        if '=' in cleaned:
            cleaned = cleaned.split('=')[-1]

        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
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

    def _normalize_math_text(self, text: str) -> str:
        """Normalize unicode math symbols for parsing."""
        replacements = {
            "−": "-",
            "–": "-",
            "×": "*",
            "÷": "/",
            "⁰": "^0",
            "¹": "^1",
            "²": "^2",
            "³": "^3",
            "⁴": "^4",
            "⁵": "^5",
            "⁶": "^6",
            "⁷": "^7",
            "⁸": "^8",
            "⁹": "^9",
        }
        normalized = text
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)
        return normalized

    def _auto_solve_equation_problem(self, problem: str) -> Optional[dict]:
        """Attempt to solve a single-variable equation and return expected real roots."""
        if not problem:
            return None

        normalized = self._normalize_math_text(problem)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        equation_line = None
        for line in lines:
            if "=" in line and "x" in line.lower():
                equation_line = line
                break

        if equation_line is None:
            match = re.search(r"([0-9xX\^\-+*/().\s]+=[0-9xX\^\-+*/().\s]+)", normalized)
            if match:
                equation_line = match.group(1).strip()

        if equation_line is None:
            return None

        try:
            left, right = equation_line.split("=", 1)
            left_expr = sympify(left.replace("^", "**"))
            right_expr = sympify(right.replace("^", "**"))
            x = symbols("x", real=True)
            equation_expr = simplify(left_expr - right_expr)

            roots = solve(equation_expr, x)
            real_roots = []
            for root in roots:
                try:
                    root_val = complex(root.evalf())
                    if abs(root_val.imag) < 1e-9:
                        real_roots.append(float(root_val.real))
                except Exception:
                    continue

            if not real_roots:
                return None

            return {
                "type": "equation_roots",
                "variable": x,
                "equation": equation_expr,
                "roots": self._unique_numeric(real_roots),
            }
        except Exception:
            return None

    def _unique_numeric(self, values: list, tolerance: float = 1e-6) -> list:
        """Deduplicate numeric list with tolerance."""
        unique = []
        for value in sorted(values):
            if not any(abs(value - existing) <= tolerance for existing in unique):
                unique.append(value)
        return unique

    def _extract_numeric_candidates(self, response: str) -> list:
        """Extract numeric candidates from response text."""
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
        parsed = []
        for value in numbers:
            try:
                parsed.append(float(value))
            except ValueError:
                continue
        return self._unique_numeric(parsed)

    def _score_equation_roots(self, response: str, expected: dict) -> float:
        """Score response by matching predicted real roots against expected root set."""
        equation = expected.get("equation")
        variable = expected.get("variable")
        expected_roots = expected.get("roots", [])
        if equation is None or variable is None or not expected_roots:
            return self._heuristic_score(response)

        candidates = self._extract_numeric_candidates(response)
        valid_roots = []
        for candidate in candidates:
            try:
                residual = float(abs(equation.subs(variable, candidate).evalf()))
                if residual <= 1e-5:
                    valid_roots.append(candidate)
            except Exception:
                continue

        valid_roots = self._unique_numeric(valid_roots)
        if not valid_roots:
            return 0.0

        matched = 0
        unmatched_expected = expected_roots[:]
        for candidate in valid_roots:
            for idx, target in enumerate(unmatched_expected):
                if abs(candidate - target) <= 1e-4:
                    matched += 1
                    unmatched_expected.pop(idx)
                    break

        precision = matched / max(len(valid_roots), 1)
        recall = matched / max(len(expected_roots), 1)

        if matched == len(expected_roots) and len(valid_roots) == len(expected_roots):
            return 1.0

        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
        if f1 >= 0.9:
            return 0.9
        if f1 >= 0.7:
            return 0.75
        if f1 >= 0.5:
            return 0.5
        return 0.25
