"""
Prompt Generator Module
Generates multiple prompting strategies for comparative benchmarking.
"""

from typing import Any, List, Dict, Optional, Set, Tuple
import json
import os
import re

PROBLEM_KEYWORDS = {
    # Algebra - solving
    'solve', 'find', 'calculate', 'determine', 'evaluate', 'value of', 'equation', 'equals', '=',
    # Algebra - operations
    'factor', 'expand', 'simplify', 'system',
    # Algebra - compare/evaluate choices
    'compare', 'least', 'greatest', 'smallest', 'largest', 'minimum', 'maximum',
    'least value', 'greatest value', 'smallest value', 'largest value', 'which of the following',
    # Pre-calculus
    'derivative', 'differentiate', 'integrate', 'integral', 'limit', 'lim',
    'd/dx', 'dy/dx', '∫', '∂',
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
    'arcsin', 'arccos', 'arctan',
    'polar', 'rectangular', 'complex', 'vector',
    'sequence', 'series', 'domain', 'range', 'asymptote',
    # Counting & Probability - central tendency
    'mean', 'median', 'mode', 'average',
    # Counting & Probability - spread
    'variance', 'standard deviation', 'range', 'deviation',
    # Counting & Probability - general
    'probability', 'p(', 'chance', 'odds', 'likely',
    # Counting & Probability - conditional
    'given that', 'given', '|', 'conditional',
    # Counting & Probability - objects
    'coin', 'coins', 'dice', 'die', 'card', 'ball', 'balls', 'bag',
    # Counting & Probability - concepts
    'flip', 'flipped', 'roll', 'rolled', 'draw', 'drawn',
    'heads', 'tails', 'outcome', 'outcomes', 'event', 'favorable',
    # Combinatorics (Counting)
    'combinations', 'permutations', 'c(', 'factorial',
    'how many', 'number of ways', 'arrangement', 'arrangements', 'choose', 'select',
    'without replacement', 'with replacement',
    # Variation / proportional reasoning
    'variation', 'varies', 'direct variation', 'inverse variation',
    'directly proportional', 'inversely proportional',
    'rate', 'unit rate', 'at this rate', 'at the same rate',
    'miles per hour', 'per minute', 'per hour',
    # Distributions (Counting & Probability)
    'random', 'distribution', 'expected value', 'expectation', 'sample'
}

OPERATION_WORDS = {
    'solve', 'find', 'calculate', 'evaluate', 'value', 'factor', 'expand', 'simplify',
    'derivative', 'integrate', 'limit',
    'mean', 'median', 'mode', 'probability', 'variance',
    'roll', 'flip', 'draw', 'combinations', 'permutations',
    'arrange', 'arrangement', 'choose', 'select',
    'expected value', 'expectation',
    'sin', 'cos', 'tan', 'domain', 'range', 'sequence', 'series',
    'variation', 'proportion', 'rate',
    'compare', 'least', 'greatest', 'smallest', 'largest', 'minimum', 'maximum',
    'which of the following'
}

SPECIFIC_OBJECTS = ('dice', 'die', 'coin', 'ball', 'card', 'bag')

STRATEGY_PAIRS = (
    ("given that", "given that"),
    ("probability", "probability"),
    ("derivative", "derivative"),
    ("integral", "integral"),
    ("limit", "limit"),
    ("factor", "factor"),
    ("system", "system"),
    ("mean", "mean"),
    ("variance", "variance"),
    ("how many", "how many"),
    ("number of ways", "number of ways"),
    ("arrange", "arrange"),
    ("choose", "choose"),
    ("at this rate", "at this rate"),
    ("per hour", "per hour"),
    ("per minute", "per minute"),
    ("expected value", "expected value"),
    ("sin", "sin"),
    ("cos", "cos"),
    ("tan", "tan"),
    ("sequence", "sequence"),
    ("series", "series"),
    ("polar", "polar"),
    ("least value", "least value"),
    ("greatest value", "greatest value"),
    ("smallest value", "smallest value"),
    ("largest value", "largest value"),
    ("which of the following", "which of the following"),
)

RELATED_INTENT_PAIRS = {
    ("solve_equation", "real_solutions"),
    ("real_solutions", "solve_equation"),
    ("probability", "conditional_probability"),
    ("conditional_probability", "probability"),
    ("derivative", "integral"),
    ("integral", "derivative"),
    ("mean", "median"),
    ("median", "mean"),
    ("mean", "mode"),
    ("mode", "mean"),
    ("counting_arrangements", "probability"),
    ("probability", "counting_arrangements"),
    ("ratio_proportion", "percent"),
    ("percent", "ratio_proportion"),
    ("evaluate_substitution", "solve_equation"),
    ("solve_equation", "evaluate_substitution"),
    ("variation", "ratio_proportion"),
    ("ratio_proportion", "variation"),
    ("compare_values", "evaluate_substitution"),
    ("evaluate_substitution", "compare_values"),
    ("compare_values", "solve_equation"),
    ("solve_equation", "compare_values"),
}

CRITICAL_MATH_FEATURES = {
    "conditional_probability",
    "composition",
    "derivative",
    "integral",
    "limit",
    "trigonometric",
    "logarithm",
    "root",
    "system",
    "substitution",
}

TEMPLATE_FEWSHOT_EXAMPLES = {
    "compare_values": [
        {
            "problem": "Which of the following has the least value? A = 3/2, B = 5/4, C = 7/8.",
            "solution": "Convert to comparable values: A = 1.5, B = 1.25, C = 0.875. The least value is C.",
            "type": "compare_values",
        },
        {
            "problem": "Which option has the greatest value? A = 2^3, B = 3^2, C = 10 - 1.",
            "solution": "Evaluate each option: A = 8, B = 9, C = 9. The greatest value is B and C (tie).",
            "type": "compare_values",
        },
        {
            "problem": "Among the choices, which has the smallest value? A = -2, B = -5/2, C = -2.1.",
            "solution": "Compare negatives: A = -2, B = -2.5, C = -2.1. The smallest value is B.",
            "type": "compare_values",
        },
    ]
}

DIFFICULTY_ALIASES = {
    "beginner": "basic",
    "easy": "basic",
    "medium": "intermediate",
    "moderate": "intermediate",
    "hard": "advanced",
    "expert": "advanced",
}

TYPE_ALIASES = {
    "equation": "solve_equation",
    "equation_solving": "solve_equation",
    "substitution": "evaluate_substitution",
    "substitute": "evaluate_substitution",
    "evaluation": "evaluate_substitution",
    "real_solution": "real_solutions",
    "real_root": "real_solutions",
    "conditional_prob": "conditional_probability",
    "counting": "counting_arrangements",
    "comparison": "compare_values",
    "value_comparison": "compare_values",
}

CONSTRAINT_ALIASES = {
    "real_solution": "real_solutions",
    "real_root": "real_solutions",
    "real_value": "real_values",
    "positive_solution": "positive_solutions",
    "positive_root": "positive_solutions",
    "positive_value": "positive_values",
    "integer_solution": "integer_solutions",
    "integer_value": "integer_values",
    "nonnegative": "non_negative",
}

class PromptGenerator:
    """
    Generates prompts using multiple techniques for research benchmarking.
    
    Implements two prompting strategies:
    1. Zero-shot: Direct question without examples
    2. Few-shot: Includes examples before the question
    """
    
    # Subject aliases for consistent mapping
    _SUBJECT_ALIASES = {
        "calculus": "pre-calculus",
        "precalculus": "pre-calculus",
        "statistics": "counting-probability",
        "stat": "counting-probability",
        "probability": "counting-probability",
    }
    
    def __init__(self):
        """Initialize the prompt generator and load example dataset from JSON."""
        self.few_shot_min_examples = int(os.getenv("FEW_SHOT_MIN_EXAMPLES", "1"))
        # Keep few-shot concise by default to reduce unrelated support examples.
        self.few_shot_max_examples = int(os.getenv("FEW_SHOT_MAX_EXAMPLES", "1"))
        self.few_shot_medium_examples = int(os.getenv("FEW_SHOT_MEDIUM_EXAMPLES", "1"))
        self.few_shot_hard_examples = int(os.getenv("FEW_SHOT_HARD_EXAMPLES", "1"))
        self.few_shot_diversity_lambda = float(os.getenv("FEW_SHOT_DIVERSITY_LAMBDA", "0.15"))
        self.few_shot_min_relevance = float(os.getenv("FEW_SHOT_MIN_RELEVANCE", "0.35"))

        # Get the path to the JSON file (in the same directory as this module)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "example_problems.json")
        
        # Load examples from JSON file
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                self.example_dataset = self._normalize_example_dataset(json.load(f))
            print(f"Loaded example dataset from {json_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find {json_path}, using minimal fallback examples")
            # Fallback to minimal examples if JSON file not found
            self.example_dataset = self._normalize_example_dataset({
                "general": [
                    {"problem": "What is 12 + 8?", "solution": "12 + 8 = 20"},
                    {"problem": "Calculate 3 × 7", "solution": "3 × 7 = 21"}
                ],
                "algebra": [
                    {"problem": "Solve for x: 3x + 7 = 22", "solution": "3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 15/3\nx = 5"}
                ],
                "counting-probability": [
                    {"problem": "Find the mean of: 4, 8, 12, 16, 20", "solution": "Mean = (4 + 8 + 12 + 16 + 20)/5\n= 60/5\n= 12"}
                ],
                "pre-calculus": [
                    {"problem": "Find the derivative: f(x) = x³", "solution": "f(x) = x³\nf'(x) = 3x⁽³⁻¹⁾\nf'(x) = 3x²"}
                ]
            })
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing JSON file: {e}")
            # Use minimal fallback if JSON is malformed
            self.example_dataset = self._normalize_example_dataset({
                "general": [
                    {"problem": "What is 12 + 8?", "solution": "12 + 8 = 20"}
                ]
            })

    def _normalize_metadata_label(self, value: str) -> str:
        """Normalize metadata labels to stable snake_case tokens."""
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    def _normalize_label_list(self, value: object) -> List[str]:
        """Normalize metadata values into de-duplicated token lists."""
        if value is None:
            return []

        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = [str(item) for item in value if item is not None]
        else:
            raw_items = [str(value)]

        normalized: List[str] = []
        seen = set()
        for raw in raw_items:
            label = self._normalize_metadata_label(raw)
            if label and label not in seen:
                seen.add(label)
                normalized.append(label)

        return normalized

    def _normalize_type_label(self, value: str) -> str:
        """Normalize and map type aliases to canonical intent-like labels."""
        normalized = self._normalize_metadata_label(value)
        return TYPE_ALIASES.get(normalized, normalized)

    def _normalize_constraints(self, value: object) -> List[str]:
        """Normalize constraint labels and map aliases to canonical forms."""
        labels = self._normalize_label_list(value)
        normalized: List[str] = []
        seen = set()
        for label in labels:
            canonical = CONSTRAINT_ALIASES.get(label, label)
            if canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)
        return normalized

    def _normalize_format_metadata(self, value: object) -> Optional[object]:
        """Normalize optional format metadata while supporting string/list/dict inputs."""
        if value is None:
            return None

        if isinstance(value, dict):
            normalized: Dict[str, object] = {}
            for raw_key, raw_value in value.items():
                key = self._normalize_metadata_label(str(raw_key))
                if not key:
                    continue

                if isinstance(raw_value, bool):
                    normalized[key] = raw_value
                elif isinstance(raw_value, (int, float)):
                    normalized[key] = raw_value
                elif raw_value is not None:
                    labels = self._normalize_label_list(raw_value)
                    if labels:
                        normalized[key] = labels[0] if len(labels) == 1 else labels

            return normalized or None

        labels = self._normalize_label_list(value)
        if not labels:
            return None
        return labels[0] if len(labels) == 1 else labels

    def _normalize_difficulty_label(self, value: object) -> Optional[str]:
        """Normalize difficulty into compact labels while keeping numeric scales valid."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric <= 0:
                return None
            if numeric.is_integer():
                return str(int(numeric))
            return str(numeric)

        normalized = self._normalize_metadata_label(str(value))
        if not normalized:
            return None
        return DIFFICULTY_ALIASES.get(normalized, normalized)

    def _coerce_anchor_priority(self, value: object) -> Optional[float]:
        """Coerce anchor priority to a normalized [0, 1] score."""
        if value is None:
            return None

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        if numeric < 0:
            return 0.0
        if numeric > 1:
            # Accept 0-10 or 0-100 style priorities and normalize them.
            if numeric <= 10:
                numeric = numeric / 10.0
            else:
                numeric = numeric / 100.0

        return max(0.0, min(1.0, numeric))

    def _add_optional_metadata(
        self,
        normalized_example: Dict[str, Any],
        source_example: Dict[str, Any],
        fallback_difficulty: Optional[object],
    ) -> None:
        """Attach optional metadata fields without requiring them in legacy data."""
        difficulty_value = source_example.get("difficulty")
        if difficulty_value is None:
            difficulty_value = fallback_difficulty

        normalized_difficulty = self._normalize_difficulty_label(difficulty_value)
        if normalized_difficulty is not None:
            normalized_example["difficulty"] = normalized_difficulty

        type_values = self._normalize_label_list(source_example.get("type"))
        if type_values:
            normalized_example["type"] = self._normalize_type_label(type_values[0])

        concept_values = self._normalize_label_list(source_example.get("concept"))
        if concept_values:
            normalized_example["concept"] = concept_values[0] if len(concept_values) == 1 else concept_values

        skill_values = self._normalize_label_list(source_example.get("skills"))
        if skill_values:
            normalized_example["skills"] = skill_values

        format_value = self._normalize_format_metadata(source_example.get("format"))
        if format_value is not None:
            normalized_example["format"] = format_value

        tag_values = self._normalize_label_list(source_example.get("tags"))
        if tag_values:
            normalized_example["tags"] = tag_values

        constraint_values = self._normalize_constraints(source_example.get("constraints"))
        if constraint_values:
            normalized_example["constraints"] = constraint_values

        anchor_priority = self._coerce_anchor_priority(source_example.get("anchor_priority"))
        if anchor_priority is not None:
            normalized_example["anchor_priority"] = anchor_priority

    def _normalize_example_dataset(self, raw_data: object) -> Dict[str, List[Dict[str, Any]]]:
        """Flatten subject->difficulty->examples JSON into subject->examples with optional metadata."""
        normalized: Dict[str, List[Dict[str, Any]]] = {}

        if not isinstance(raw_data, dict):
            return normalized

        for subject, subject_value in raw_data.items():
            subject_examples: List[Dict[str, Any]] = []

            if isinstance(subject_value, list):
                source_groups = [(None, subject_value)]
            elif isinstance(subject_value, dict):
                source_groups = list(subject_value.items())
            else:
                continue

            for difficulty, examples in source_groups:
                if not isinstance(examples, list):
                    continue

                for example in examples:
                    if not isinstance(example, dict):
                        continue

                    problem = example.get("problem")
                    solution = example.get("solution") or example.get("answer")
                    if not problem or not solution:
                        continue

                    normalized_example: Dict[str, Any] = {
                        "problem": self._normalize_example_entry_text(str(problem), is_solution=False),
                        "solution": self._normalize_example_entry_text(str(solution), is_solution=True),
                    }

                    self._add_optional_metadata(normalized_example, example, difficulty)

                    subject_examples.append(normalized_example)

            normalized[str(subject)] = subject_examples

        return normalized
    
    def generate_zero_shot(self, problem: str, subject: str = "general") -> str:
        """
        Generate a deterministic zero-shot prompt with no worked examples.
        
        Args:
            problem: The math problem
            subject: Subject category (reserved for API compatibility)
            
        Returns:
            Zero-shot prompt
        """
        _ = subject
        target_problem_text = str(problem)
        return (
        "Solve the following math problem and end with a concise final answer. "
        "Do NOT show steps or explanations.\n\n"
        f"Q: {target_problem_text}\n"
        "A:"
    )

    def _normalize_problem_text(self, problem: str) -> str:
        """Normalize user input to avoid duplicated Q:/A: wrappers in prompts."""
        normalized = problem.strip()
        if normalized.lower().startswith("q:"):
            normalized = normalized[2:].lstrip()

        lines = normalized.splitlines()
        while lines and lines[-1].strip().lower() == "a:":
            lines.pop()

        normalized = "\n".join(lines).strip()
        
        # Convert caret notation to Unicode superscripts
        normalized = self._convert_caret_to_superscripts(normalized)
        
        return normalized

    def _is_fragmented_line_layout(self, lines: List[str]) -> bool:
        """Detect OCR-like token-per-line formatting that should be flattened."""
        if len(lines) < 6:
            return False

        compact = [re.sub(r"\s+", "", line) for line in lines if line]
        if not compact:
            return False

        short_tokens = sum(1 for token in compact if len(token) <= 3)
        symbol_only = sum(
            1 for token in compact
            if re.fullmatch(r"[0-9a-zA-Z+\-*/=^().,%]+", token) is not None
        )

        short_ratio = short_tokens / len(compact)
        symbol_ratio = symbol_only / len(compact)
        return short_ratio >= 0.55 or (len(compact) >= 8 and symbol_ratio >= 0.75)

    def _normalize_example_entry_text(self, text: str, *, is_solution: bool) -> str:
        """Normalize stored example text while preserving mathematical content."""
        value = str(text).replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", value).strip()

        if is_solution:
            value = re.sub(r"^\s*(?:a|answer)\s*:\s*", "", value, flags=re.IGNORECASE)
        else:
            value = re.sub(r"^\s*(?:q|question)\s*:\s*", "", value, flags=re.IGNORECASE)

        normalized_lines: List[str] = []
        for line in value.splitlines():
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            if not cleaned_line:
                continue
            if normalized_lines and normalized_lines[-1] == cleaned_line:
                continue
            normalized_lines.append(cleaned_line)

        if not normalized_lines:
            return value

        if self._is_fragmented_line_layout(normalized_lines):
            value = " ".join(normalized_lines)
        else:
            value = "\n".join(normalized_lines)

        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    def _convert_latex_fractions(self, text: str) -> str:
        """Convert common LaTeX fraction forms into plain-text divisions."""
        value = text
        brace_fraction = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
        while True:
            value, count = brace_fraction.subn(r"(\1)/(\2)", value)
            if count == 0:
                break

        # Support shorthand forms like '\frac 12' while avoiding greedy token capture.
        return re.sub(r"\\frac\s+([0-9A-Za-z])\s*([0-9A-Za-z])", r"(\1)/(\2)", value)

    def _simplify_latex_for_prompt(self, text: str) -> str:
        """Simplify high-noise LaTeX markup for readable few-shot prompt text."""
        value = text
        value = self._convert_latex_fractions(value)
        value = value.replace("\\displaystyle", "")
        value = value.replace("\\left", "")
        value = value.replace("\\right", "")
        value = value.replace("\\times", " x ")
        value = value.replace("\\cdot", " * ")
        value = value.replace("\\div", " / ")
        value = value.replace("\\quad", " ")
        value = value.replace("\\qquad", " ")
        value = value.replace("\\,", " ")
        value = value.replace("\\!", " ")
        value = re.sub(r"\\begin\{align\*?\}", "", value)
        value = re.sub(r"\\end\{align\*?\}", "", value)
        value = value.replace("\\\\", " ; ")
        value = value.replace("\\[", " ")
        value = value.replace("\\]", " ")
        value = value.replace("\\(", " ")
        value = value.replace("\\)", " ")
        value = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", value)
        value = re.sub(r"\^\{([^{}]+)\}", r"^\1", value)
        value = self._convert_latex_fractions(value)
        return value

    def _prepare_example_text_for_prompt(self, text: str, *, is_solution: bool) -> str:
        """Render example text into a concise, readable few-shot format."""
        value = self._normalize_example_entry_text(text, is_solution=is_solution)
        value = re.sub(r"\[asy\].*?\[/asy\]", " ", value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"\\begin\{asy\}.*?\\end\{asy\}", " ", value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"```(?:[a-zA-Z0-9_+\-]+)?\s*[\s\S]*?```", " ", value)
        value = self._simplify_latex_for_prompt(value)

        cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
        cleaned_lines = [line for line in cleaned_lines if line]
        if not cleaned_lines:
            return ""

        if self._is_fragmented_line_layout(cleaned_lines):
            value = " ".join(cleaned_lines)
        else:
            value = "\n".join(cleaned_lines)

        value = re.sub(r"\n{3,}", "\n\n", value)
        value = re.sub(r"[ \t]{2,}", " ", value)
        value = re.sub(r"\s+([,.;:!?])", r"\1", value)
        return value.strip()
    
    def _convert_caret_to_superscripts(self, text: str) -> str:
        """Convert x^n notation to Unicode superscripts (x³, x², etc.)."""
        # Mapping of digits and common characters to superscript Unicode
        superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
        }
        
        result = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and text[i] == '^':
                # Check if next character is a digit or should be superscript
                j = i + 1
                superscript_part = ''
                
                # Handle parentheses: x^(3+2) -> x⁽³⁺²⁾
                if j < len(text) and text[j] == '(':
                    superscript_part += superscript_map.get('(', '(')
                    j += 1
                    while j < len(text) and text[j] != ')':
                        if text[j] in superscript_map:
                            superscript_part += superscript_map[text[j]]
                        else:
                            superscript_part += text[j]
                        j += 1
                    if j < len(text) and text[j] == ')':
                        superscript_part += superscript_map.get(')', ')')
                        j += 1
                else:
                    # Handle single character: x^3 -> x³
                    while j < len(text) and (text[j].isdigit() or text[j] in superscript_map):
                        if text[j] in superscript_map:
                            superscript_part += superscript_map[text[j]]
                        else:
                            superscript_part += text[j]
                        j += 1
                
                if superscript_part:
                    result.append(superscript_part)
                    i = j
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject with case/whitespace handling and aliases."""
        if not subject:
            return "general"
        key = subject.strip().lower()
        return self._SUBJECT_ALIASES.get(key, key)

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
        has_probability = any(marker in example_text for marker in ["probability", " p(", "p("])
        has_conditional = "given that" in example_text or ("|" in example_text and "p(" in example_text)
        return has_probability and has_conditional
    
    def _detect_problem_keywords(self, problem: str) -> set:
        """
        Extract keywords from a problem to help match relevant examples.
        
        Args:
            problem: The math problem text
            
        Returns:
            Set of relevant keywords found
        """
        problem_lower = problem.lower()
        token_set = self._tokenize_text(problem_lower)

        # Find matching keywords with token-aware matching where possible.
        found_keywords = set()
        for keyword in PROBLEM_KEYWORDS:
            if keyword == "=":
                if "=" in problem_lower:
                    found_keywords.add(keyword)
                continue

            # Keep substring behavior for multi-word/punctuation-heavy markers.
            if any(ch in keyword for ch in (" ", "(", ")", "/", "|", "%", "-")):
                if keyword in problem_lower:
                    found_keywords.add(keyword)
                continue

            if keyword in token_set:
                found_keywords.add(keyword)

        return found_keywords

    def _extract_operation_markers(self, text: str) -> Set[str]:
        """Extract operation words present in a text using token-aware matching."""
        value = text.lower()
        token_set = self._tokenize_text(value)
        found: Set[str] = set()

        for marker in OPERATION_WORDS:
            if " " in marker:
                if marker in value:
                    found.add(marker)
            elif marker in token_set:
                found.add(marker)

        return found

    def _extract_assigned_variables(self, text: str) -> Set[str]:
        """Extract variable names explicitly assigned with '=' in text (e.g., x = 2)."""
        value = text.lower()
        return {match.group(1) for match in re.finditer(r"\b([a-z])\s*=", value)}

    def _extract_math_features(self, text: str) -> Set[str]:
        """Extract structural math features used for stronger few-shot matching."""
        value = text.lower()
        features: Set[str] = set()
        assignment_count = len(
            re.findall(r"\b[a-z]\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", value)
        )

        if "=" in value:
            features.add("equation")
        if re.search(r"[<>]=?|\\le|\\ge|≤|≥", value):
            features.add("inequality")
        if "\\frac" in value or "/" in value:
            features.add("fraction")
        if "^" in value or any(char in value for char in "²³⁴⁵⁶⁷⁸⁹⁰¹"):
            features.add("exponent")
        if "\\sqrt" in value or "√" in value or "root" in value:
            features.add("root")
        if "\\log" in value or re.search(r"\blog\b", value):
            features.add("logarithm")
        if re.search(r"\b(sin|cos|tan|sec|csc|cot|arcsin|arccos|arctan)\b", value):
            features.add("trigonometric")
        if re.search(r"\b(derivative|differentiate|d/dx|dy/dx)\b", value) or "∂" in value:
            features.add("derivative")
        if re.search(r"\b(integral|integrate)\b", value) or "∫" in value:
            features.add("integral")
        if re.search(r"\b(limit|lim)\b", value):
            features.add("limit")

        has_probability = any(marker in value for marker in ["probability", " p(", "p(", "chance", "odds"])
        if has_probability:
            features.add("probability")
        if has_probability and ("given that" in value or "|" in value or "conditional" in value):
            features.add("conditional_probability")

        if re.search(r"\b[a-z]\s*\(", value):
            features.add("function_notation")
        if re.search(r"\b[a-z]\s*\(\s*[a-z]\s*\(", value):
            features.add("composition")
        if "%" in value or "\\%" in value:
            features.add("percent")
        if any(marker in value for marker in ["ratio", "proportion", "directly proportional", "inversely proportional"]):
            features.add("proportion")
        if self._is_rate_proportion_problem(value):
            features.add("proportion")
        if any(marker in value for marker in ["varies directly", "varies inversely", "direct variation", "inverse variation"]):
            features.add("variation")
        if assignment_count > 0:
            features.add("assignment")
        if assignment_count >= 2:
            features.add("multi_assignment")
        if self._has_labeled_value_options(value):
            features.add("labeled_options")
        if self._is_compare_values_problem(value):
            features.add("value_comparison")
        if (
            assignment_count > 0
            and any(marker in value for marker in ["evaluate", "value of", "expression", " when ", "if ", "given"])
        ):
            features.add("substitution")
        if (
            "system" in value
            or "\\begin{align" in value
            or (
                value.count("=") >= 2
                and assignment_count == 0
                and (" and " in value or "," in value)
            )
        ):
            features.add("system")

        return features

    def _intent_similarity(self, problem_intent: str, example_intent: str) -> float:
        """Score semantic closeness between intents."""
        if problem_intent == example_intent:
            return 1.0
        if (problem_intent, example_intent) in RELATED_INTENT_PAIRS:
            return 0.55
        return 0.0

    def _tokenize_text(self, text: str) -> set:
        """Tokenize text into normalized lexical units for similarity scoring."""
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    def _lexical_jaccard(self, text_a: str, text_b: str) -> float:
        """Compute lexical Jaccard similarity between two texts."""
        tokens_a = self._tokenize_text(text_a)
        tokens_b = self._tokenize_text(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return intersection / union if union else 0.0

    def _estimate_problem_complexity(self, problem: str, subject: str) -> int:
        """Estimate problem complexity (1=low, 2=medium, 3=high)."""
        text = problem.lower()
        complexity = 1

        if len(problem.split()) >= 18:
            complexity += 1

        advanced_markers = [
            "given that", "conditional", "system", "variance", "standard deviation",
            "integral", "derivative", "limit", "chain rule", "product rule", "prove"
        ]
        # Count how many advanced markers are present
        marker_count = sum(1 for marker in advanced_markers if marker in text)
        if marker_count >= 1:
            complexity += 1
        if marker_count >= 2:
            complexity += 1

        if subject in {"counting-probability", "pre-calculus"}:
            complexity += 1

        return max(1, min(3, complexity))

    def _extract_equation_signature(self, text: str) -> Dict[str, bool]:
        """Extract structural markers that help match equation-solving strategy."""
        value = text.lower().replace("⁴", "^4").replace("³", "^3").replace("²", "^2")
        has_equation = "=" in value
        assignment_count = len(
            re.findall(r"\b[a-z]\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", value)
        )
        has_assignment = assignment_count > 0
        has_system = (
            "system" in value
            or "\\begin{align" in value
            or (
                value.count("=") >= 2
                and not has_assignment
                and (" and " in value or "," in value)
            )
        )

        return {
            "has_equation": has_equation,
            "has_x4": "x^4" in value or "x4" in value,
            "has_x3": "x^3" in value or "x3" in value,
            "has_x2": "x^2" in value or "x2" in value,
            "has_abs": "|" in value or "abs(" in value,
            "has_system": has_system,
            "has_fractional": "/" in value,
            "has_assignment": has_assignment,
        }

    def _has_labeled_value_options(self, text: str) -> bool:
        """Detect labeled choice structures such as A=..., B=..., C=...."""
        value = text.lower()
        labeled_assignments = re.findall(r"\b[a-e]\s*=\s*[^,;\n]+", value)
        if len(labeled_assignments) >= 2:
            return True

        # Supports (A)/(B)/(C) style multiple-choice prompts.
        option_labels = re.findall(r"\([a-e]\)", value)
        return len(option_labels) >= 3

    def _is_compare_values_problem(self, text: str) -> bool:
        """Detect prompts that compare labeled option values (least/greatest)."""
        value = text.lower()

        comparison_markers = [
            "least", "greatest", "smallest", "largest", "minimum", "maximum",
            "least value", "greatest value", "smallest value", "largest value",
        ]
        choice_markers = [
            "which of the following",
            "which option",
            "which choice",
            "which has",
            "which is",
            "among",
        ]

        has_comparison = any(marker in value for marker in comparison_markers)
        has_choice_prompt = any(marker in value for marker in choice_markers)
        has_labeled_options = self._has_labeled_value_options(value)

        if not (has_comparison and has_choice_prompt and has_labeled_options):
            return False

        # Guard against coordinate/geometry descriptions that use A=, B= labels.
        geometry_markers = ["point", "points", "triangle", "vertex", "vertices", "coordinate", "vector"]
        if any(marker in value for marker in geometry_markers) and "which of the following" not in value:
            return False

        return True

    def _is_rate_proportion_problem(self, text: str) -> bool:
        """Detect word problems that are fundamentally unit-rate/proportion algebra."""
        value = text.lower()

        travel_terms = [
            "travel", "travels", "distance", "speed", "train", "car", "bike", "walk", "drive", "rate",
        ]
        distance_units = [
            "mile", "miles", "km", "kilometer", "kilometers", "meter", "meters", "foot", "feet", "yard", "yards",
        ]
        time_units = ["second", "seconds", "minute", "minutes", "hour", "hours", "day", "days"]

        has_rate_phrase = any(marker in value for marker in ["at this rate", "at the same rate", "unit rate"])
        has_travel_context = any(term in value for term in travel_terms)
        has_distance_unit = any(unit in value for unit in distance_units)
        has_time_unit = any(unit in value for unit in time_units)

        has_unit_over_time_form = bool(
            re.search(
                r"\b(?:mile|miles|km|kilometers?|meters?|foot|feet|yards?)\s+per\s+(?:second|seconds|minute|minutes|hour|hours|day|days)\b",
                value,
            )
        )
        has_distance_in_time_form = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*(?:mile|miles|km|kilometers?|meters?|foot|feet|yards?)\s+in\s+\d+(?:\.\d+)?\s*(?:second|seconds|minute|minutes|hour|hours|day|days)\b",
                value,
            )
        )

        return (
            has_unit_over_time_form
            or has_distance_in_time_form
            or (has_rate_phrase and has_distance_unit and has_time_unit)
            or (has_travel_context and has_distance_unit and has_time_unit and "how many" in value)
        )

    def _normalize_detection_text(self, text: str) -> str:
        """Normalize unicode math glyphs so intent and subject rules stay stable."""
        value = str(text or "")
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
        for source, target in replacements.items():
            value = value.replace(source, target)

        return re.sub(r"\s+", " ", value).strip().lower()

    def _looks_like_algebraic_equation_problem(self, text: str) -> bool:
        """Detect equation-solving prompts that should map to algebra by default."""
        value = self._normalize_detection_text(text)

        if "=" not in value:
            return False

        has_variable = bool(re.search(r"\b[a-z]\b|\d+[a-z]|[a-z]\d", value))
        if not has_variable:
            return False

        if re.search(
            r"\bd/dx\b|\bdy/dx\b|∫|\\int\b|\bderivative\b|\bdifferentiate\b|\bintegral\b|\bintegrate\b|\blimit\b|\blim\b|\bsin\b|\bcos\b|\btan\b|\bsec\b|\bcsc\b|\bcot\b|\barcsin\b|\barccos\b|\barctan\b",
            value,
        ):
            return False

        solve_markers = [
            "solve",
            "solution",
            "solutions",
            "root",
            "roots",
            "equation",
            "polynomial",
            "factor",
            "zero",
            "zeros",
            "find x",
            "solve for",
        ]
        if any(marker in value for marker in solve_markers):
            return True

        if re.search(r"\b[a-z]\^?\d", value):
            return True
        if re.search(r"[-+]?\d+[a-z]", value):
            return True
        if re.search(r"\b[a-z]\s*[+\-*/]\s*[a-z0-9]", value):
            return True

        return False

    def _detect_primary_intent(self, text: str) -> str:
        """Detect primary solve intent from problem text."""
        value = self._normalize_detection_text(text)
        has_equation = "=" in value and bool(re.search(r"\b[a-z]\b|\d+[a-z]|[a-z]\d+", value))
        assignment_count = len(
            re.findall(r"\b[a-z]\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", value)
        )

        real_solution_markers = [
            "real solution",
            "real solutions",
            "real root",
            "real roots",
            "all real values",
            "real values",
        ]
        if any(marker in value for marker in real_solution_markers):
            return "real_solutions"

        if self._is_compare_values_problem(value):
            return "compare_values"

        if self._is_rate_proportion_problem(value):
            return "ratio_proportion"

        if any(marker in value for marker in ["how many", "number of ways", "arrange", "arrangement", "permutation", "combination", "choose", "select"]):
            return "counting_arrangements"
        if "expected value" in value or "expectation" in value:
            return "expected_value"

        if (
            assignment_count >= 1
            and any(marker in value for marker in ["evaluate", "value of", "expression", " when ", "if ", "given"])
            and all(marker not in value for marker in ["solve", "satisfy", "satisfies", "roots", "solutions"])
        ):
            return "evaluate_substitution"

        if re.search(r"\b[a-z]\s*\(\s*[a-z]\s*\(", value):
            return "function_composition"

        has_probability = any(marker in value for marker in ["probability", " p(", "p(", "chance", "odds"])
        if has_probability and any(marker in value for marker in ["conditional", "given that", "|"]):
            return "conditional_probability"
        if has_probability:
            return "probability"

        trig_markers = ["sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan", "radian", "degrees"]
        if any(re.search(rf"\b{re.escape(marker)}\b", value) for marker in trig_markers):
            return "trigonometric"

        if "derivative" in value or "differentiate" in value:
            return "derivative"
        if "integral" in value or "integrate" in value or "∫" in value:
            return "integral"
        if "limit" in value or "lim(" in value or "lim" in value:
            return "limit"
        if any(marker in value for marker in ["sequence", "series", "nth term", "geometric sequence", "arithmetic sequence"]):
            return "sequence_series"
        if any(marker in value for marker in ["domain", "range", "asymptote", "vertex", "turning point", "increasing", "decreasing"]):
            return "function_analysis"
        if any(marker in value for marker in ["polar", "rectangular", "cartesian", "complex plane"]):
            return "coordinate_conversion"
        if "variance" in value:
            return "variance"
        if "mean" in value:
            return "mean"
        if "median" in value:
            return "median"
        if "mode" in value:
            return "mode"
        if "factor" in value:
            return "factor"
        if "expand" in value:
            return "expand"
        if "simplify" in value:
            return "simplify"
        if "system" in value:
            return "system"
        if "%" in value or "\\%" in value:
            return "percent"
        if any(marker in value for marker in ["varies directly", "varies inversely", "direct variation", "inverse variation"]):
            return "variation"
        if any(marker in value for marker in ["ratio", "proportion", "directly proportional", "inversely proportional"]):
            return "ratio_proportion"
        if has_equation and any(marker in value for marker in ["calculate", "find", "determine", "what is"]):
            return "solve_equation"
        if "solve" in value or "find x" in value or "solve for x" in value:
            return "solve_equation"
        if self._looks_like_algebraic_equation_problem(value):
            return "solve_equation"
        return "general"

    def _extract_constraints_from_text(self, text: str) -> Set[str]:
        """Extract requirement constraints such as real/positive/integer conditions."""
        value = text.lower()
        constraints: Set[str] = set()
        assignment_count = len(
            re.findall(r"\b[a-z]\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", value)
        )
        substitution_expression_query = (
            assignment_count >= 1
            and "expression" in value
            and any(marker in value for marker in ["value of", "evaluate", "compute"])
        )

        if any(marker in value for marker in ["real solution", "real solutions", "real root", "real roots"]):
            constraints.add("real_solutions")
        if any(marker in value for marker in ["real value", "real values"]):
            constraints.add("real_values")
        if any(marker in value for marker in ["positive solution", "positive solutions", "positive root", "positive roots", "positive difference"]):
            constraints.add("positive_values")
        elif "positive value" in value and not substitution_expression_query:
            constraints.add("positive_values")
        if any(marker in value for marker in ["integer solution", "integer solutions"]):
            constraints.add("integer_solutions")
        if any(marker in value for marker in ["integer value", "integer values", "integers"]):
            constraints.add("integer_values")
        if any(marker in value for marker in ["nonnegative", "non-negative"]):
            constraints.add("non_negative")

        return constraints

    def _extract_problem_format_labels(self, text: str) -> Set[str]:
        """Extract structural labels that can be matched against optional format metadata."""
        value = text.lower()
        features = self._extract_math_features(value)
        labels: Set[str] = set()

        feature_to_label = {
            "equation": "equation",
            "inequality": "inequality",
            "fraction": "fractional_form",
            "exponent": "exponent_expression",
            "root": "root_expression",
            "function_notation": "function_notation",
            "composition": "function_composition",
            "assignment": "assigned_values",
            "multi_assignment": "multi_assignment",
            "substitution": "expression_substitution",
            "system": "system_of_equations",
            "conditional_probability": "conditional_probability_form",
            "labeled_options": "labeled_options",
            "value_comparison": "value_comparison",
        }
        for feature, label in feature_to_label.items():
            if feature in features:
                labels.add(label)

        if "solve" in value:
            labels.add("solve_for_variable")
        if any(marker in value for marker in ["evaluate", "value of", "compute"]):
            labels.add("expression_evaluation")

        signature = self._extract_equation_signature(value)
        if signature["has_x4"]:
            labels.add("quartic")
        if signature["has_x3"]:
            labels.add("cubic")
        if signature["has_x2"]:
            labels.add("quadratic")
        if signature["has_system"]:
            labels.add("system_of_equations")

        return labels

    def _format_metadata_labels(self, value: object) -> Set[str]:
        """Normalize optional format metadata into a comparable label set."""
        labels: Set[str] = set()
        if value is None:
            return labels

        if isinstance(value, dict):
            for raw_key, raw_value in value.items():
                key_label = self._normalize_metadata_label(str(raw_key))
                if key_label:
                    labels.add(key_label)
                if isinstance(raw_value, bool):
                    if raw_value and key_label:
                        labels.add(f"{key_label}_true")
                elif raw_value is not None:
                    labels.update(self._normalize_label_list(raw_value))
            return labels

        labels.update(self._normalize_label_list(value))
        return labels

    def _difficulty_to_level(self, difficulty: object) -> Optional[float]:
        """Map difficulty labels/scales to a comparable 1-3 level."""
        if difficulty is None:
            return None

        if isinstance(difficulty, (int, float)):
            numeric = float(difficulty)
            if numeric <= 0:
                return None
            if numeric <= 3:
                return numeric
            return max(1.0, min(3.0, 1.0 + ((numeric - 1.0) / 2.0)))

        label = self._normalize_difficulty_label(difficulty)
        if label is None:
            return None

        if label == "basic":
            return 1.0
        if label == "intermediate":
            return 2.0
        if label == "advanced":
            return 3.0

        try:
            return self._difficulty_to_level(float(label))
        except ValueError:
            return None

    def _example_matches_problem_type(self, example: dict, problem_type: str) -> bool:
        """Check whether an example should be considered type-compatible for a problem."""
        if not isinstance(example, dict):
            return False

        type_values = self._normalize_label_list(example.get("type"))
        if type_values:
            example_type = self._normalize_type_label(type_values[0])
            # Keep type matching strict so few-shot examples mirror target intent.
            if example_type == problem_type:
                return True

        # The migrated bank has many speed/rate algebra word problems typed as
        # counting_arrangements/general. Keep these eligible for proportion queries.
        if problem_type == "ratio_proportion":
            example_problem = str(example.get("problem", ""))
            if example_problem and self._is_rate_proportion_problem(example_problem):
                return True

        return False

    def _detect_equation_family(self, text: str) -> Optional[str]:
        """Detect coarse equation family for strict few-shot alignment."""
        value = self._normalize_detection_text(text)
        if "=" not in value:
            return None

        signature = self._extract_equation_signature(value)
        if signature["has_system"]:
            return "system"

        # Detect polynomial degree by any variable symbol, not just x.
        if re.search(r"\b[a-z]\s*\^\s*4\b", value):
            return "quartic"
        if re.search(r"\b[a-z]\s*\^\s*3\b", value):
            return "cubic"
        if re.search(r"\b[a-z]\s*\^\s*2\b", value):
            return "quadratic"
        if signature["has_abs"]:
            return "absolute"

        has_variable = bool(re.search(r"\b[a-z]\b|\d+[a-z]|[a-z]\d+", value))
        if has_variable:
            return "linear"
        return None

    def _problem_pattern_signature(self, text: str) -> str:
        """Build a canonical signature so structurally identical problems align."""
        value = self._normalize_detection_text(text)
        # Normalize numbers and symbolic variable names while keeping operators/keywords.
        value = re.sub(r"(?<![a-z])(\d+(?:\.\d+)?)([a-z])(?![a-z])", r"<num><var>", value)
        value = re.sub(r"\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\b", "<num>", value)
        value = re.sub(r"(?<![a-z])[a-z](?![a-z])", "<var>", value)
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"[.,;:!?]+$", "", value)
        return value

    def _canonical_problem_text(self, text: str) -> str:
        """Canonicalize problem text for duplicate detection."""
        value = self._normalize_detection_text(text)
        value = value.replace("q:", " ")
        value = value.replace("a:", " ")
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"[^a-z0-9^=+\-*/()\[\]{}|<>.,]", "", value)
        value = re.sub(r"[.,;:!?]+$", "", value)
        return value

    def _requires_strict_type_matching(self, problem: str, subject: str = "general") -> bool:
        """Enable strict few-shot type matching for the three core math domains."""
        normalized_subject = (subject or "").strip().lower()
        if normalized_subject not in {"algebra", "counting-probability", "pre-calculus"}:
            return False

        problem_type = self._normalize_type_label(self._detect_primary_intent(problem))
        return problem_type != "general"

    def _score_metadata_alignment(
        self,
        example: dict,
        problem: str,
        problem_keywords: set,
        problem_features: Set[str],
    ) -> float:
        """Score explicit metadata alignment for intent-driven ranking quality."""
        metadata_score = 0.0
        problem_lower = problem.lower()
        problem_type = self._normalize_type_label(self._detect_primary_intent(problem_lower))

        example_type_values = self._normalize_label_list(example.get("type"))
        if example_type_values:
            example_type = self._normalize_type_label(example_type_values[0])
            intent_similarity = self._intent_similarity(problem_type, example_type)
            if intent_similarity > 0:
                metadata_score += intent_similarity * 0.55
            elif problem_type == "ratio_proportion" and self._is_rate_proportion_problem(str(example.get("problem", ""))):
                metadata_score += 0.24
            elif problem_type != "general":
                metadata_score -= 0.16

        example_concepts = set(self._normalize_label_list(example.get("concept")))
        if example_concepts:
            problem_concepts = set(problem_features)
            if problem_type != "general":
                problem_concepts.add(problem_type)
            concept_overlap = len(problem_concepts.intersection(example_concepts)) / len(example_concepts)
            metadata_score += concept_overlap * 0.24

        example_skills = set(self._normalize_label_list(example.get("skills")))
        if example_skills:
            problem_skills = set(problem_features).union(self._extract_operation_markers(problem_lower))
            skill_overlap = len(problem_skills.intersection(example_skills)) / len(example_skills)
            metadata_score += skill_overlap * 0.20

        example_format = self._format_metadata_labels(example.get("format"))
        if example_format:
            problem_format = self._extract_problem_format_labels(problem_lower)
            if problem_format:
                format_overlap = len(problem_format.intersection(example_format)) / len(problem_format)
                metadata_score += format_overlap * 0.28
                if format_overlap == 0:
                    metadata_score -= 0.08

        problem_constraints = self._extract_constraints_from_text(problem_lower)
        example_constraints = set(self._normalize_constraints(example.get("constraints")))
        if problem_constraints and example_constraints:
            overlap = len(problem_constraints.intersection(example_constraints)) / len(problem_constraints)
            metadata_score += overlap * 0.36
            missing_constraints = problem_constraints - example_constraints
            if missing_constraints:
                metadata_score -= 0.12 * len(missing_constraints)

        example_difficulty = self._difficulty_to_level(example.get("difficulty"))
        if example_difficulty is not None:
            problem_difficulty = float(self._estimate_problem_complexity(problem, "general"))
            difficulty_gap = abs(problem_difficulty - example_difficulty)
            metadata_score += max(0.0, 0.14 - (0.07 * difficulty_gap))

        example_tags = set(self._normalize_label_list(example.get("tags")))
        if example_tags:
            problem_tags = set(problem_features)
            problem_tags.add(problem_type)
            problem_tags.update(self._normalize_label_list(problem_keywords))
            tag_overlap = len(problem_tags.intersection(example_tags)) / len(example_tags)
            metadata_score += tag_overlap * 0.14

        anchor_priority = self._coerce_anchor_priority(example.get("anchor_priority"))
        if anchor_priority is not None:
            metadata_score += anchor_priority * 0.06

        return metadata_score

    def _filter_examples_by_metadata(self, available_examples: List[dict], problem: str, subject: str = "general") -> List[dict]:
        """Apply optional type/constraint filtering when metadata is available."""
        if not available_examples:
            return []

        filtered_examples = available_examples
        problem_intent = self._detect_primary_intent(problem)
        problem_type = self._normalize_type_label(problem_intent)
        strict_intent_matching = self._requires_strict_type_matching(problem, subject)

        typed_examples = [
            example for example in filtered_examples
            if isinstance(example, dict) and example.get("type") is not None
        ]

        if typed_examples and problem_type != "general":
            type_matches: List[dict] = []
            for example in typed_examples:
                if self._example_matches_problem_type(example, problem_type):
                    type_matches.append(example)

            if type_matches:
                filtered_examples = type_matches
            elif strict_intent_matching:
                # If typed metadata exists but none match, enforce strictness.
                filtered_examples = []

        if strict_intent_matching and problem_type != "general":
            intent_matches = [
                example for example in filtered_examples
                if isinstance(example, dict)
                and self._detect_primary_intent(str(example.get("problem", ""))) == problem_intent
            ]
            if intent_matches:
                filtered_examples = intent_matches
            elif filtered_examples:
                # Enforce strict intent matching even when type metadata is absent.
                filtered_examples = []

        equation_family_intents = {"solve_equation", "real_solutions", "system"}
        if strict_intent_matching and problem_intent in equation_family_intents and filtered_examples:
            problem_family = self._detect_equation_family(problem)
            if problem_family is not None:
                family_matches = [
                    example for example in filtered_examples
                    if isinstance(example, dict)
                    and self._detect_equation_family(str(example.get("problem", ""))) == problem_family
                ]
                if family_matches:
                    filtered_examples = family_matches
                else:
                    filtered_examples = []

        if strict_intent_matching and filtered_examples:
            target_signature = self._problem_pattern_signature(problem)
            target_canonical = self._canonical_problem_text(problem)
            signature_matches = [
                example for example in filtered_examples
                if isinstance(example, dict)
                and self._problem_pattern_signature(str(example.get("problem", ""))) == target_signature
            ]
            # In strict mode, require same structure from the example bank.
            if signature_matches:
                non_identical_signature_matches = [
                    example for example in signature_matches
                    if self._canonical_problem_text(str(example.get("problem", ""))) != target_canonical
                ]
                filtered_examples = non_identical_signature_matches
            else:
                filtered_examples = []

        problem_constraints = self._extract_constraints_from_text(problem)
        effective_constraints = set(problem_constraints)
        if problem_type in {"evaluate_substitution", "function_composition"}:
            # "positive value" in plug-in evaluation prompts is often descriptive,
            # not a domain restriction for selecting examples.
            effective_constraints.discard("positive_values")

        constrained_examples = [
            example for example in filtered_examples
            if isinstance(example, dict) and example.get("constraints") is not None
        ]
        if effective_constraints and constrained_examples:
            constraint_matches: List[dict] = []
            for example in constrained_examples:
                example_constraints = set(self._normalize_constraints(example.get("constraints")))
                if effective_constraints.intersection(example_constraints):
                    constraint_matches.append(example)

            if constraint_matches:
                filtered_examples = constraint_matches

        return filtered_examples
    
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
            keyword_coverage = keyword_matches / len(problem_keywords)
            score += keyword_coverage * 0.45
        
        # Bonus for similar operation words in problem statement
        example_lower = example['problem'].lower()

        problem_operations = self._extract_operation_markers(problem_lower)
        example_operations = self._extract_operation_markers(example_lower)
        if problem_operations:
            operation_overlap = len(problem_operations.intersection(example_operations)) / len(problem_operations)
            score += operation_overlap * 0.30
        
        # Extra bonus for matching specific objects (dice, coins, balls, etc.)
        for obj in SPECIFIC_OBJECTS:
            if obj in problem_lower and obj in example_lower:
                score += 0.2
                break

        # Lexical similarity between target problem and example problem
        lexical_similarity = self._lexical_jaccard(problem_lower, example_lower)
        score += lexical_similarity * 0.20

        problem_features = self._extract_math_features(problem_lower)
        example_features = self._extract_math_features(example_text)
        if problem_features:
            feature_overlap = len(problem_features.intersection(example_features)) / len(problem_features)
            score += feature_overlap * 0.40

            if "value_comparison" in problem_features and "value_comparison" in example_features:
                score += 0.26
            elif "value_comparison" in problem_features and "value_comparison" not in example_features:
                score -= 0.18

            if "labeled_options" in problem_features and "labeled_options" in example_features:
                score += 0.10
            elif "labeled_options" in problem_features and "labeled_options" not in example_features:
                score -= 0.06

            if "assignment" in problem_features and "assignment" in example_features:
                score += 0.22
            elif "assignment" in problem_features and "assignment" not in example_features:
                score -= 0.18

            if "multi_assignment" in problem_features and "multi_assignment" in example_features:
                score += 0.12

            if "substitution" in problem_features and "substitution" in example_features:
                score += 0.28
            elif "substitution" in problem_features and "substitution" not in example_features:
                score -= 0.22

            missing_critical = (problem_features.intersection(CRITICAL_MATH_FEATURES)) - example_features
            if missing_critical:
                score -= 0.08 * len(missing_critical)

        problem_assigned_vars = self._extract_assigned_variables(problem_lower)
        example_assigned_vars = self._extract_assigned_variables(example_lower)
        if problem_assigned_vars:
            assignment_var_overlap = len(problem_assigned_vars.intersection(example_assigned_vars)) / len(problem_assigned_vars)
            score += assignment_var_overlap * 0.40

            if len(example_assigned_vars) < len(problem_assigned_vars):
                score -= 0.14

            if len(problem_assigned_vars) >= 2 and len(example_assigned_vars) >= 2:
                score += 0.10

        problem_intent = self._detect_primary_intent(problem_lower)
        example_intent = self._detect_primary_intent(example_lower)
        intent_alignment = self._intent_similarity(problem_intent, example_intent)
        if intent_alignment > 0:
            score += intent_alignment * 0.45
        elif problem_intent != "general" and example_intent != "general":
            score -= 0.15

        if problem_intent == "compare_values":
            if example_intent in {"compare_values", "evaluate_substitution", "solve_equation"}:
                score += 0.12
            else:
                score -= 0.12

        # Equation-structure alignment (critical for solve-equation few-shot quality)
        problem_sig = self._extract_equation_signature(problem_lower)
        example_sig = self._extract_equation_signature(example_lower)

        if problem_sig["has_equation"] and example_sig["has_equation"]:
            score += 0.08

            if problem_sig["has_x4"]:
                if example_sig["has_x4"]:
                    score += 0.35
                else:
                    score -= 0.25

            if problem_sig["has_x3"] and example_sig["has_x3"]:
                score += 0.2

            if problem_sig["has_x2"] and example_sig["has_x2"]:
                score += 0.12

            if problem_sig["has_abs"] != example_sig["has_abs"]:
                score -= 0.1

            if problem_sig["has_system"] != example_sig["has_system"]:
                score -= 0.08

            if problem_sig["has_fractional"] and example_sig["has_fractional"]:
                score += 0.08

            if problem_sig["has_assignment"] and example_sig["has_assignment"]:
                score += 0.18
            elif problem_sig["has_assignment"] != example_sig["has_assignment"]:
                score -= 0.16

        solution_length = len(example.get("solution", "").split())
        if solution_length >= 25:
            score += 0.06
        elif solution_length <= 4:
            score -= 0.05

        # Strategy-pattern matching (boost examples that use same solving style)
        for problem_marker, example_marker in STRATEGY_PAIRS:
            if problem_marker in problem_lower and example_marker in example_text:
                score += 0.08

        score += self._score_metadata_alignment(example, problem, problem_keywords, problem_features)
        
        return max(score, 0.0)

    def _rank_examples_by_relevance(
        self,
        examples: List[dict],
        problem: str,
        problem_keywords: set,
    ) -> List[Tuple[float, dict]]:
        """Return examples ranked by relevance to the target problem."""
        ranked: List[Tuple[float, dict]] = []
        for example in examples:
            if not isinstance(example, dict):
                continue
            if not example.get("problem") or not example.get("solution"):
                continue
            relevance = self._score_example_relevance(example, problem, problem_keywords)
            ranked.append((relevance, example))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _select_anchor_then_diverse(
        self,
        candidates: List[dict],
        problem: str,
        num_examples: int,
        problem_keywords: Optional[set] = None,
    ) -> List[dict]:
        """Select the best anchor example first, then diversify supporting examples."""
        if num_examples <= 0 or not candidates:
            return []

        problem_keywords = problem_keywords or self._detect_problem_keywords(problem)
        ranked = self._rank_examples_by_relevance(candidates, problem, problem_keywords)
        if not ranked:
            return []

        candidate_pool = [example for _, example in ranked[:max(num_examples * 6, num_examples)]]
        anchor = candidate_pool[0]

        if num_examples == 1 or len(candidate_pool) == 1:
            return [anchor]

        support_candidates = candidate_pool[1:]
        support_count = min(num_examples - 1, len(support_candidates))
        support_examples = self._select_diverse_examples(
            support_candidates,
            problem,
            support_count,
            problem_keywords=problem_keywords,
        )

        return [anchor, *support_examples]

    def _top_relevance_score(self, examples: List[dict], problem: str, problem_keywords: Optional[set] = None) -> float:
        """Return the top relevance score for a set of examples."""
        if not examples:
            return 0.0

        problem_keywords = problem_keywords or self._detect_problem_keywords(problem)
        ranked = self._rank_examples_by_relevance(examples, problem, problem_keywords)
        if not ranked:
            return 0.0
        return ranked[0][0]

    def _gather_all_examples(self) -> List[dict]:
        """Gather unique examples from all subjects for last-resort reranking."""
        pooled: List[dict] = []
        seen_pairs = set()

        for subject_examples in self.example_dataset.values():
            if not isinstance(subject_examples, list):
                continue
            for example in subject_examples:
                if not isinstance(example, dict):
                    continue
                problem_text = str(example.get("problem", ""))
                solution_text = str(example.get("solution", ""))
                if not problem_text or not solution_text:
                    continue

                key = (problem_text, solution_text)
                if key in seen_pairs:
                    continue

                seen_pairs.add(key)
                pooled.append(example)

        return pooled

    def _resolve_subject_for_problem(self, problem: str, requested_subject: str) -> str:
        """Resolve the most appropriate subject with fallback to automatic classification."""
        explicit_subject = (requested_subject or "").strip().lower()
        if (
            explicit_subject in self.example_dataset
            and self.example_dataset.get(explicit_subject)
            and explicit_subject != "general"
        ):
            return explicit_subject

        requested = self._normalize_subject(requested_subject)

        if requested in self.example_dataset:
            if requested == "general":
                detected = self._normalize_subject(self.classify_subject(problem))
                if detected in self.example_dataset and self.example_dataset.get(detected):
                    return detected
            return requested

        detected = self._normalize_subject(self.classify_subject(problem))
        if detected in self.example_dataset and self.example_dataset.get(detected):
            return detected

        if "general" in self.example_dataset:
            return "general"

        return next(iter(self.example_dataset), "general")
    
    def _select_relevant_examples(
        self,
        available_examples: list,
        problem: str,
        num_examples: int,
        subject: str = "general",
        exclude_identical_target: bool = False,
    ) -> list:
        """
        Select the most relevant examples for the given problem.
        
        Args:
            available_examples: List of available examples
            problem: The user's problem
            num_examples: Number of examples to select
            
        Returns:
            List of selected examples
        """
        if num_examples <= 0:
            return []

        if exclude_identical_target:
            normalized_problem = self._normalize_problem_text(problem)
            # Avoid direct answer leakage: never use an example whose problem
            # text is identical to the current target problem.
            available_examples = [
                ex for ex in available_examples
                if isinstance(ex, dict)
                and self._normalize_problem_text(str(ex.get("problem", ""))) != normalized_problem
            ]

            if not available_examples:
                return []

        problem_keywords = self._detect_problem_keywords(problem)
        available_examples = self._filter_examples_by_metadata(available_examples, problem, subject=subject)
        problem_intent = self._detect_primary_intent(problem)

        if problem_intent == "compare_values":
            compare_like_count = sum(
                1
                for ex in available_examples
                if isinstance(ex, dict) and self._is_compare_values_problem(str(ex.get("problem", "")))
            )
            if compare_like_count < max(1, num_examples):
                template_examples = [dict(example) for example in TEMPLATE_FEWSHOT_EXAMPLES.get("compare_values", [])]
                available_examples = [*available_examples, *template_examples]

        # Explicitly prioritize conditional probability examples when detected.
        # We still rank inside this filtered set so the best teaching example is selected.
        if self._is_conditional_probability_problem(problem):
            conditional_examples = [
                ex for ex in available_examples
                if isinstance(ex, dict) and self._is_conditional_probability_example(ex)
            ]
            if conditional_examples:
                return self._select_anchor_then_diverse(
                    conditional_examples,
                    problem,
                    min(num_examples, len(conditional_examples)),
                    problem_keywords=problem_keywords,
                )

        if problem_intent == "real_solutions":
            real_solution_examples = [
                ex for ex in available_examples
                if isinstance(ex, dict) and self._detect_primary_intent(ex.get("problem", "")) == "real_solutions"
            ]
            if real_solution_examples:
                return self._select_anchor_then_diverse(
                    real_solution_examples,
                    problem,
                    min(num_examples, len(real_solution_examples)),
                    problem_keywords=problem_keywords,
                )
        
        # Score all examples
        scored_examples = self._rank_examples_by_relevance(available_examples, problem, problem_keywords)
        if not scored_examples:
            return []
        
        # If we have highly relevant examples (score > 0.4), prioritize them
        highly_relevant = [ex for score, ex in scored_examples if score > 0.4]
        
        if len(highly_relevant) >= num_examples:
            candidate_pool = highly_relevant[:min(len(highly_relevant), num_examples * 4)]
            return self._select_anchor_then_diverse(
                candidate_pool,
                problem,
                num_examples,
                problem_keywords=problem_keywords,
            )
        else:
            # Fall back to top-ranked examples by relevance
            top_candidates = [ex for _, ex in scored_examples[:max(num_examples * 3, num_examples)]]
            return self._select_anchor_then_diverse(
                top_candidates,
                problem,
                num_examples,
                problem_keywords=problem_keywords,
            )

    def _select_diverse_examples(
        self,
        candidates: list,
        problem: str,
        num_examples: int,
        problem_keywords: Optional[set] = None,
    ) -> list:
        """
        Select examples using relevance-diversity tradeoff.

        Greedy MMR-like selection: maximize relevance while reducing redundancy.
        """
        if num_examples >= len(candidates):
            return candidates

        problem_keywords = problem_keywords or self._detect_problem_keywords(problem)
        relevance_cache = {
            id(candidate): self._score_example_relevance(candidate, problem, problem_keywords)
            for candidate in candidates
        }

        selected = []
        remaining = candidates[:]

        while remaining and len(selected) < num_examples:
            best_example = None
            best_score = float("-inf")

            for candidate in remaining:
                base_relevance = relevance_cache[id(candidate)]
                if not selected:
                    mmr_score = base_relevance
                else:
                    redundancy = max(
                        self._lexical_jaccard(candidate.get("problem", ""), chosen.get("problem", ""))
                        for chosen in selected
                    )
                    mmr_score = base_relevance - (self.few_shot_diversity_lambda * redundancy)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_example = candidate

            selected.append(best_example)
            remaining = [item for item in remaining if item is not best_example]

        return selected
    
    def classify_subject(self, problem: str) -> str:
        """
        Classify and detect the subject of a math problem automatically.
        
        Args:
            problem: The math problem text
            
        Returns:
            Subject category: 'pre-calculus', 'counting-probability', 'algebra', or 'general'
        """
        text = self._normalize_detection_text(problem)

        intent = self._detect_primary_intent(text)
        if intent in {
            "derivative",
            "integral",
            "limit",
            "trigonometric",
            "sequence_series",
            "function_analysis",
            "coordinate_conversion",
            "function_composition",
        }:
            return "pre-calculus"
        if intent in {
            "probability",
            "conditional_probability",
            "counting_arrangements",
            "expected_value",
            "variance",
            "mean",
            "median",
            "mode",
        }:
            return "counting-probability"
        if intent in {
            "evaluate_substitution",
            "compare_values",
            "solve_equation",
            "real_solutions",
            "factor",
            "expand",
            "simplify",
            "system",
            "percent",
            "ratio_proportion",
            "variation",
        }:
            return "algebra"

        if self._looks_like_algebraic_equation_problem(text):
            return "algebra"

        scores = {'pre-calculus': 0, 'counting-probability': 0, 'algebra': 0}
        
        # Pre-calculus indicators
        precalculus_keywords = {
            'derivative', 'differentiate', 'integral', 'integrate', 'limit', 'lim',
            'd/dx', 'dy/dx', '∫', '∂', 'rate of change', 'optimization', 'concave',
            'inflection', 'slope', 'curve', 'velocity', 'acceleration', 'extrema',
            'maximum', 'minimum', 'gradient', 'second derivative', 'critical point',
            'antiderivative', 'riemann', 'area under', 'taylor', 'maclaurin',
            'convergence', 'divergence', 'function', 'exponential', 'logarithmic',
            'sequence', 'series', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
            'arcsin', 'arccos', 'arctan', 'radian', 'degrees',
            'polar', 'rectangular', 'complex', 'vector', 'matrix', 'determinant',
            'domain', 'range', 'asymptote'
        }
        
        # Counting & Probability keywords
        counting_probability_keywords = {
            'probability', 'mean', 'median', 'mode', 'variance', 'standard deviation',
            'distribution', 'expected value', 'random', 'coin', 'dice', 'die', 'sample',
            'permutation', 'combination', 'factorial', 'count', 'counting', 'arrange',
            'arrangement', 'selection', 'flip', 'flipped', 'roll', 'rolled', 'choose',
            'outcome', 'outcomes', 'event', 'favorable', 'nCr', 'nPr', 'odds', 'chance',
            'bell curve', 'normal distribution', 'bayes', 'conditional', 'dependent', 'independent',
            'without replacement', 'with replacement', 'how many', 'number of ways', 'at least', 'at most'
        }
        
        # Algebra keywords
        algebra_keywords = {
            'solve', 'factor', 'factorize', 'simplify', 'expand', 'quadratic',
            'equation', 'polynomial', 'inequality', 'linear', 'matrix', 'system',
            'system of equations', 'roots', 'zero', 'parabola', 'binomial', 'trinomial',
            'monomial', 'rational', 'radical', 'algebraic', 'expression', 'substitute',
            'evaluate', 'variable', 'coefficient'
        }
        
        def _keyword_hits(keywords: set) -> int:
            return sum(1 for kw in keywords if kw in text)

        # Check for keyword pattern density
        scores['pre-calculus'] += _keyword_hits(precalculus_keywords)
        scores['counting-probability'] += _keyword_hits(counting_probability_keywords)
        scores['algebra'] += _keyword_hits(algebra_keywords)
        
        # Regex pattern checking
        if re.search(r'\bd/dx\b|\bdy/dx\b|∫|∂|\blimit\b|\blim\b|derivative|integral|optimization|\bsin\b|\bcos\b|\btan\b|\barcsin\b|\barccos\b|\barctan\b|\bpolar\b|\brectangular\b', text):
            scores['pre-calculus'] += 3
        if re.search(r'\bp\(|probability|permutation|combination|C\(|nCr|counting|factorial|how many|number of ways|without replacement|with replacement|expected value', text):
            scores['counting-probability'] += 3
        if re.search(r'\bsolve\s+for\b|factor|simplify|expand|quadratic|equation', text):
            scores['algebra'] += 2
        
        # Find the subject with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return 'general'

        # Break algebra/pre-calculus ties toward algebra for equation-solving prompts
        # unless explicit calculus/trig markers are present.
        if (
            scores['algebra'] == max_score
            and scores['pre-calculus'] == max_score
            and '=' in text
            and re.search(r'\breal\s+solutions?\b|\breal\s+roots?\b|\bsolve\b|\bquadratic\b|\bequation\b|\bpolynomial\b', text)
            and not re.search(r'\bd/dx\b|\bdy/dx\b|∫|∂|\blimit\b|\bderivative\b|\bintegral\b|\bsin\b|\bcos\b|\btan\b|\bsequence\b|\bseries\b', text)
        ):
            return 'algebra'
        
        # Return the subject with the highest score
        if scores['pre-calculus'] == max_score:
            return 'pre-calculus'
        elif scores['counting-probability'] == max_score:
            return 'counting-probability'
        else:
            return 'algebra'
    
    def generate_few_shot(self, problem: str, subject: str = "general", num_examples: int = None) -> str:
        """
        Generate few-shot prompt: includes examples from the specified subject.
        Uses semantic matching to select the most relevant examples.
        
        Args:
            problem: The math problem
            subject: Subject category (algebra, counting-probability (Counting & Probability), pre-calculus (Pre-calculus), general)
            num_examples: Number of examples to include (auto-determined by subject if None)
            
        Returns:
            Few-shot prompt with subject-specific, relevant examples
        """
        normalized_problem = self._normalize_problem_text(problem)
        # Keep target prompt text identical to user input; use normalized text only for retrieval.
        target_problem_text = str(problem)
        
        # Track if num_examples determination is automatic (used for smart relevance fallback)
        auto_mode = num_examples is None

        # Resolve subject, with auto-detection fallback when needed.
        self.example_dataset = self._normalize_example_dataset(self.example_dataset)
        subject = self._resolve_subject_for_problem(normalized_problem, subject)

        # Auto-determine number of examples based on subject if not specified
        if num_examples is None:
            complexity = self._estimate_problem_complexity(normalized_problem, subject)
            target_examples = self.few_shot_min_examples
            if complexity >= 3:
                target_examples = self.few_shot_hard_examples
            elif complexity == 2:
                target_examples = self.few_shot_medium_examples

            num_examples = max(
                self.few_shot_min_examples,
                min(target_examples, self.few_shot_max_examples),
            )

        num_examples = max(self.few_shot_min_examples, min(num_examples, self.few_shot_max_examples))

        available_examples = self.example_dataset.get(subject, [])
        
        if not available_examples:
            print(f"Warning: No examples found for subject '{subject}'")
            # Return zero-shot if no examples
            return self.generate_zero_shot(target_problem_text, subject=subject)
        
        # Select relevant examples (smart selection based on problem content)
        num_to_select = min(num_examples, len(available_examples))
        problem_keywords = self._detect_problem_keywords(normalized_problem)
        selected_examples = self._select_relevant_examples(
            available_examples,
            normalized_problem,
            num_to_select,
            subject=subject,
        )
        strict_matching = self._requires_strict_type_matching(normalized_problem, subject)
        if strict_matching and not selected_examples:
            return self.generate_zero_shot(target_problem_text, subject=subject)

        best_relevance = self._top_relevance_score(selected_examples, normalized_problem, problem_keywords)

        # If relevance is weak, try a detected-subject rerank and then a global rerank.
        if not strict_matching and best_relevance < self.few_shot_min_relevance:
            detected_subject = self._normalize_subject(self.classify_subject(normalized_problem))
            if detected_subject != subject and detected_subject in self.example_dataset:
                detected_examples = self.example_dataset.get(detected_subject, [])
                if detected_examples:
                    detected_selected = self._select_relevant_examples(
                        detected_examples,
                        normalized_problem,
                        min(num_examples, len(detected_examples)),
                        subject=detected_subject,
                    )
                    detected_best = self._top_relevance_score(detected_selected, normalized_problem, problem_keywords)
                    if detected_best > best_relevance + 0.05:
                        subject = detected_subject
                        selected_examples = detected_selected
                        best_relevance = detected_best

        if not strict_matching and best_relevance < self.few_shot_min_relevance:
            pooled_examples = self._gather_all_examples()
            if pooled_examples:
                pooled_selected = self._select_relevant_examples(
                    pooled_examples,
                    normalized_problem,
                    min(num_examples, len(pooled_examples)),
                    subject=subject,
                )
                pooled_best = self._top_relevance_score(pooled_selected, normalized_problem, problem_keywords)
                if pooled_best > best_relevance + 0.05:
                    selected_examples = pooled_selected
                    best_relevance = pooled_best

        # If selected examples are not sufficiently related, fall back to bank-anchored zero-shot.
        # Only apply this for auto-mode with large example banks (to preserve test coverage).
        if auto_mode and len(available_examples) >= 8 and best_relevance < self.few_shot_min_relevance:
            return self.generate_zero_shot(target_problem_text, subject=subject)

        # Never include an example that is the same as the target problem text.
        target_canonical = self._canonical_problem_text(target_problem_text)
        selected_examples = [
            ex for ex in selected_examples
            if self._canonical_problem_text(str(ex.get("problem", ""))) != target_canonical
        ]
        if not selected_examples:
            return self.generate_zero_shot(target_problem_text, subject=subject)
        
        # Format examples (concise format for speed)
        examples_text = "\n\n".join([
            (
                f"Q: {self._prepare_example_text_for_prompt(ex.get('problem', ''), is_solution=False)}\n"
                f"A: {self._prepare_example_text_for_prompt(ex.get('solution', ''), is_solution=True)}"
            )
            for ex in selected_examples
        ])
        return (
            "Solve the following math problems and give the final answer. "
            "Use the following examples only as style references. "
            "Do NOT repeat or copy any example answer. "
            "You must solve ONLY the target problem shown after 'TARGET PROBLEM'. "
            "Think carefully and use the examples only for internal reasoning. "
            "Output ONLY the final answer for the TARGET PROBLEM. Do NOT include steps, explanations, or extra text.\n\n"
            f"{examples_text}\n\n"
            "TARGET PROBLEM\n"
            f"Q: {target_problem_text}\n"
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
            "zero_shot": self.generate_zero_shot(problem, subject=subject),
            "few_shot": self.generate_few_shot(problem, subject=subject)
        }
    
    def get_technique_names(self) -> List[str]:
        """Get list of all available prompting techniques."""
        return ["zero_shot", "few_shot"]
    
    def get_available_subjects(self) -> List[str]:
        """Get list of available subject categories."""
        return list(self.example_dataset.keys())
