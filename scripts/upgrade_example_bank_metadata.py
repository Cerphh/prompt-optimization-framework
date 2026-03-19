"""Upgrade the entire example bank with retrieval metadata.

This script is backward-compatible with existing entries. It preserves
`problem` and `solution`/`answer` while adding optional metadata fields:
- type
- concept
- skills
- format
- difficulty
- tags
- constraints
- anchor_priority
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.prompt_generator import PromptGenerator


TARGET_PATH = Path("framework/example_problems.json")


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _difficulty_label_from_level(level: float) -> str:
    if level <= 1.5:
        return "basic"
    if level <= 2.5:
        return "intermediate"
    return "advanced"


def _infer_template(
    intent: str,
    features: Set[str],
    signature: Dict[str, bool],
    assigned_count: int,
    problem_word_count: int,
) -> str:
    if intent == "conditional_probability":
        return "conditional_probability_statement"
    if intent in {"counting_arrangements", "expected_value", "probability"} and problem_word_count >= 10:
        return "counting_probability_word_problem"
    if signature.get("has_system"):
        return "system_of_equations"
    if "substitution" in features and assigned_count >= 2:
        return "expression_with_assignments"
    if "substitution" in features:
        return "single_assignment_expression"
    if signature.get("has_equation"):
        return "single_equation"
    if problem_word_count >= 14:
        return "word_problem"
    return "direct_expression"


def _infer_concept(
    intent: str,
    features: Set[str],
    signature: Dict[str, bool],
    constraints: Set[str],
    problem_word_count: int,
) -> str:
    if intent == "evaluate_substitution":
        if "multi_assignment" in features:
            return "multi_variable_expression_evaluation"
        return "single_variable_expression_evaluation"

    if intent == "real_solutions":
        if signature.get("has_x4"):
            return "quartic_real_roots"
        if signature.get("has_x3"):
            return "cubic_real_roots"
        if signature.get("has_x2"):
            return "quadratic_real_roots"
        return "real_solution_equation"

    if intent == "solve_equation":
        if signature.get("has_system"):
            return "system_of_equations_solving"
        if signature.get("has_x4"):
            return "quartic_equation_solving"
        if signature.get("has_x3"):
            return "cubic_equation_solving"
        if signature.get("has_x2"):
            return "quadratic_equation_solving"
        if "fraction" in features:
            return "rational_equation_solving"
        return "linear_equation_solving"

    if intent == "conditional_probability":
        return "conditional_probability_application"
    if intent == "probability":
        return "probability_computation"
    if intent == "counting_arrangements":
        return "combinatorics_counting"
    if intent == "expected_value":
        return "expected_value_computation"
    if intent == "derivative":
        return "derivative_computation"
    if intent == "integral":
        return "integral_computation"
    if intent == "limit":
        return "limit_evaluation"
    if intent == "function_composition":
        return "function_composition_evaluation"
    if intent == "sequence_series":
        return "sequence_series_analysis"
    if intent == "trigonometric":
        return "trigonometric_evaluation"
    if intent == "variation":
        return "direct_inverse_variation"
    if intent == "ratio_proportion":
        return "ratio_proportion_reasoning"
    if intent == "percent":
        if "positive_values" in constraints:
            return "percent_difference_comparison"
        return "percent_application"
    if intent in {"mean", "median", "mode", "variance"}:
        return f"{intent}_calculation"
    if intent == "factor":
        return "polynomial_factoring"
    if intent == "expand":
        return "algebraic_expansion"
    if intent == "simplify":
        return "algebraic_simplification"
    if intent == "system":
        return "simultaneous_equations"

    if problem_word_count >= 14:
        return "word_problem_reasoning"
    if "exponent" in features:
        return "exponent_expression_evaluation"
    if "fraction" in features:
        return "fraction_expression_evaluation"
    return "algebraic_expression_evaluation"


def _infer_skills(intent: str, features: Set[str], constraints: Set[str]) -> List[str]:
    skills: List[str] = []

    intent_skill_map = {
        "evaluate_substitution": ["substitution", "expression_evaluation"],
        "solve_equation": ["equation_solving"],
        "real_solutions": ["equation_solving", "root_filtering"],
        "conditional_probability": ["conditional_probability_rule", "probability_rules"],
        "probability": ["probability_rules"],
        "counting_arrangements": ["counting_principles"],
        "expected_value": ["expected_value"],
        "derivative": ["differentiation"],
        "integral": ["integration"],
        "limit": ["limit_laws"],
        "function_composition": ["function_composition"],
        "sequence_series": ["sequence_analysis"],
        "trigonometric": ["trigonometric_evaluation"],
        "ratio_proportion": ["proportional_reasoning"],
        "variation": ["proportional_reasoning"],
        "percent": ["percent_conversion"],
        "factor": ["factoring"],
        "expand": ["expansion"],
        "simplify": ["algebraic_simplification"],
        "system": ["simultaneous_equations"],
        "mean": ["statistical_computation"],
        "median": ["statistical_computation"],
        "mode": ["statistical_computation"],
        "variance": ["statistical_computation"],
    }
    skills.extend(intent_skill_map.get(intent, []))

    if "fraction" in features:
        skills.append("fraction_simplification")
    if "exponent" in features:
        skills.append("exponent_rules")
    if "root" in features:
        skills.append("root_simplification")
    if "assignment" in features:
        skills.append("value_substitution")
    if "system" in features:
        skills.append("simultaneous_equations")

    if constraints:
        skills.append("constraint_filtering")

    normalized = _dedupe(skills)
    if not normalized:
        normalized = ["algebraic_reasoning"]

    return normalized[:4]


def _infer_tags(
    subject: str,
    intent: str,
    features: Set[str],
    signature: Dict[str, bool],
    constraints: Set[str],
) -> List[str]:
    tags: List[str] = [subject.replace("-", "_"), intent]

    for feature in [
        "substitution",
        "system",
        "fraction",
        "exponent",
        "probability",
        "conditional_probability",
        "derivative",
        "integral",
        "limit",
        "trigonometric",
        "percent",
        "proportion",
        "variation",
    ]:
        if feature in features:
            tags.append(feature)

    if signature.get("has_x4"):
        tags.append("quartic")
    if signature.get("has_x3"):
        tags.append("cubic")
    if signature.get("has_x2"):
        tags.append("quadratic")

    tags.extend(sorted(constraints))

    return _dedupe(tags)[:8]


def _infer_anchor_priority(
    intent: str,
    features: Set[str],
    signature: Dict[str, bool],
    constraints: Set[str],
    difficulty: str,
    solution_word_count: int,
    assigned_count: int,
) -> float:
    score = 0.58

    if intent != "general":
        score += 0.08
    if constraints:
        score += 0.08
    if signature.get("has_system") or signature.get("has_x4") or "conditional_probability" in features:
        score += 0.08
    if assigned_count >= 2:
        score += 0.06
    if len(features.intersection({"substitution", "fraction", "exponent", "probability", "derivative", "integral"})) >= 2:
        score += 0.04
    if 15 <= solution_word_count <= 180:
        score += 0.04
    if difficulty == "advanced":
        score += 0.05
    elif difficulty == "basic":
        score -= 0.02

    score = max(0.50, min(0.97, score))
    return round(score, 2)


def _normalize_existing_format(pg: PromptGenerator, value: Any) -> Dict[str, Any]:
    normalized = pg._normalize_format_metadata(value)
    if isinstance(normalized, dict):
        return dict(normalized)

    if isinstance(normalized, list):
        if normalized:
            return {"template": normalized[0]}
        return {}

    if isinstance(normalized, str):
        return {"template": normalized}

    return {}


def _iter_subject_groups(raw_data: Dict[str, Any]) -> Iterable[Tuple[str, Optional[str], List[Dict[str, Any]]]]:
    for subject, subject_value in raw_data.items():
        if isinstance(subject_value, list):
            yield str(subject), None, subject_value
            continue

        if not isinstance(subject_value, dict):
            continue

        for difficulty, examples in subject_value.items():
            if isinstance(examples, list):
                yield str(subject), str(difficulty), examples


def main() -> None:
    data = json.loads(TARGET_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Expected top-level object in example bank JSON")

    pg = PromptGenerator()

    total = 0
    changed = 0

    for subject, fallback_difficulty, examples in _iter_subject_groups(data):
        for entry in examples:
            if not isinstance(entry, dict):
                continue

            problem_raw = entry.get("problem")
            if not problem_raw:
                continue

            solution_raw = entry.get("solution") or entry.get("answer")
            if not solution_raw:
                continue

            total += 1
            before = json.dumps(entry, sort_keys=True, ensure_ascii=False)

            problem = str(problem_raw)
            solution = str(solution_raw)
            entry["problem"] = problem
            if not entry.get("solution"):
                entry["solution"] = solution

            problem_lower = problem.lower()
            text_for_features = f"{problem} {solution}".lower()

            features = pg._extract_math_features(text_for_features)
            signature = pg._extract_equation_signature(problem_lower)
            assigned_count = len(pg._extract_assigned_variables(problem_lower))
            solution_word_count = len(solution.split())
            problem_word_count = len(problem.split())

            detected_intent = pg._detect_primary_intent(problem_lower)
            intent = pg._normalize_type_label(detected_intent)
            existing_type = entry.get("type")
            if existing_type:
                existing_label = pg._normalize_label_list(existing_type)
                if existing_label:
                    intent = pg._normalize_type_label(existing_label[0])
            entry["type"] = intent

            existing_constraints = set(pg._normalize_constraints(entry.get("constraints")))
            inferred_constraints = pg._extract_constraints_from_text(problem_lower)
            constraints = set(existing_constraints).union(inferred_constraints)
            entry["constraints"] = sorted(constraints)

            existing_concept = pg._normalize_label_list(entry.get("concept"))
            if existing_concept:
                entry["concept"] = existing_concept[0]
            else:
                entry["concept"] = _infer_concept(
                    intent=intent,
                    features=features,
                    signature=signature,
                    constraints=constraints,
                    problem_word_count=problem_word_count,
                )

            existing_skills = pg._normalize_label_list(entry.get("skills"))
            inferred_skills = _infer_skills(intent, features, constraints)
            skills = _dedupe([*existing_skills, *inferred_skills])
            entry["skills"] = skills[:4]

            format_obj = _normalize_existing_format(pg, entry.get("format"))
            if "template" not in format_obj:
                format_obj["template"] = _infer_template(
                    intent=intent,
                    features=features,
                    signature=signature,
                    assigned_count=assigned_count,
                    problem_word_count=problem_word_count,
                )
            format_obj.setdefault("equation_count", problem.count("="))
            format_obj.setdefault("has_assignment", bool(assigned_count))
            format_obj.setdefault("assigned_variable_count", assigned_count)
            format_obj.setdefault("has_fraction", "fraction" in features)
            format_obj.setdefault("has_exponent", "exponent" in features)
            format_obj.setdefault("has_system", bool(signature.get("has_system")))
            entry["format"] = format_obj

            existing_difficulty = entry.get("difficulty")
            normalized_existing = pg._normalize_difficulty_label(existing_difficulty)
            normalized_fallback = pg._normalize_difficulty_label(fallback_difficulty)
            if normalized_existing is not None:
                level = pg._difficulty_to_level(normalized_existing)
            elif normalized_fallback is not None:
                level = pg._difficulty_to_level(normalized_fallback)
            else:
                level = float(pg._estimate_problem_complexity(problem, subject))

            difficulty = _difficulty_label_from_level(level or 2.0)
            entry["difficulty"] = difficulty

            existing_tags = pg._normalize_label_list(entry.get("tags"))
            inferred_tags = _infer_tags(
                subject=subject,
                intent=intent,
                features=features,
                signature=signature,
                constraints=constraints,
            )
            tags = _dedupe([*existing_tags, *inferred_tags])
            entry["tags"] = tags[:8]

            existing_anchor = pg._coerce_anchor_priority(entry.get("anchor_priority"))
            if existing_anchor is not None:
                anchor = existing_anchor
            else:
                anchor = _infer_anchor_priority(
                    intent=intent,
                    features=features,
                    signature=signature,
                    constraints=constraints,
                    difficulty=difficulty,
                    solution_word_count=solution_word_count,
                    assigned_count=assigned_count,
                )
            entry["anchor_priority"] = anchor

            after = json.dumps(entry, sort_keys=True, ensure_ascii=False)
            if before != after:
                changed += 1

    TARGET_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Processed entries: {total}")
    print(f"Entries updated: {changed}")
    print(f"Output: {TARGET_PATH}")


if __name__ == "__main__":
    main()
