"""
Consistency Scorer Module
Evaluates output consistency across repeated runs of the same technique.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

from sympy import nsimplify, simplify, sympify
from sympy.core.sympify import SympifyError


class ConsistencyScorer:
    """Compute intra-technique consistency with robust answer normalization."""

    def normalize_output(self, response: str) -> str:
        """Normalize a model response into a canonical comparison key."""
        candidate = self._extract_candidate_text(response)
        if not candidate:
            return "text:__empty__"

        candidate = self._strip_verbal_wrappers(candidate)
        candidate = self._normalize_math_symbols(candidate)
        candidate = candidate.strip().rstrip(".,;:!?")

        if "=" in candidate:
            rhs = candidate.split("=")[-1].strip()
            if rhs:
                candidate = rhs

        math_key = self._canonicalize_math(candidate)
        if math_key is not None:
            return math_key

        # Text fallback: case-insensitive, punctuation-insensitive, space-insensitive.
        cleaned = candidate.lower()
        cleaned = re.sub(r"[\s_]+", "", cleaned)
        cleaned = cleaned.rstrip(".,;:!?")

        # Normalize plain numeric tokens like 002 -> 2 in fallback mode.
        cleaned = re.sub(r"\b0+(\d)\b", r"\1", cleaned)

        return f"text:{cleaned or '__empty__'}"

    def compute_consistency(self, normalized_outputs: List[str]) -> Dict[str, object]:
        """Compute consistency from normalized outputs collected so far."""
        runs_used = len(normalized_outputs)
        if runs_used == 0:
            return {
                "value": None,
                "runs_used": 0,
                "matching_runs": None,
                "is_provisional": True,
                "canonical_output": None,
                "output_counts": {},
            }

        counts = Counter(normalized_outputs)
        canonical_output, matching_runs = counts.most_common(1)[0]
        is_provisional = runs_used < 2

        value = None
        if not is_provisional:
            value = matching_runs / runs_used

        return {
            "value": value,
            "runs_used": runs_used,
            "matching_runs": matching_runs if not is_provisional else None,
            "is_provisional": is_provisional,
            "canonical_output": canonical_output,
            "output_counts": dict(sorted(counts.items(), key=lambda item: item[0])),
        }

    def _extract_candidate_text(self, response: str) -> str:
        """Extract the most likely final answer text from a response."""
        text = str(response or "").strip()
        if not text:
            return ""

        explicit_matches = re.findall(
            r"(?im)(?:final\s+answer|answer|ans(?:wer)?|result|solution)\s*[:=\-]\s*([^\n]+)",
            text,
        )
        if explicit_matches:
            return explicit_matches[-1].strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text

        candidate = lines[-1]
        if re.fullmatch(
            r"(?i)(?:final\s+answer|answer|ans(?:wer)?|result|solution)\s*[:=\-]?\s*",
            candidate,
        ) and len(lines) > 1:
            candidate = lines[-2]

        return candidate

    def _strip_verbal_wrappers(self, text: str) -> str:
        """Remove common leading wrapper phrases from answers."""
        value = str(text or "").strip()
        if not value:
            return value

        prefixes = [
            r"(?i)^\s*(?:therefore|thus|hence|so)\b\s*[,:\-]?\s*",
            r"(?i)^\s*(?:the\s+)?(?:final\s+)?(?:answer|ans(?:wer)?|result|solution)\b\s*(?:is|equals)?\s*[:=\-]?\s*",
            r"(?i)^\s*(?:it\s+is|we\s+get|we\s+have|this\s+gives)\b\s*[:=\-]?\s*",
        ]

        changed = True
        while changed:
            changed = False
            for pattern in prefixes:
                updated = re.sub(pattern, "", value).strip()
                if updated != value:
                    value = updated
                    changed = True

        return value

    def _normalize_math_symbols(self, text: str) -> str:
        replacements = {
            "−": "-",
            "–": "-",
            "×": "*",
            "÷": "/",
        }
        normalized = str(text or "")
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _canonicalize_math(self, text: str) -> Optional[str]:
        """Return canonical key for numeric/symbolic expressions when possible."""
        value = str(text or "").strip()
        if not value:
            return None

        candidate = self._prepare_expression(value)
        if not candidate:
            return None

        try:
            expression = sympify(candidate)
            simplified = simplify(expression)

            if simplified.is_number:
                numeric = nsimplify(simplified, rational=True)
                numeric = simplify(numeric)
                return f"num:{str(numeric).replace(' ', '').lower()}"

            return f"expr:{str(simplified).replace(' ', '').lower()}"
        except (SympifyError, TypeError, ValueError, AttributeError):
            return None

    def _prepare_expression(self, text: str) -> str:
        """Normalize an expression for symbolic parsing."""
        candidate = str(text or "").strip()
        if not candidate:
            return ""

        candidate = re.sub(r"(?<=\d),(?=\d)", "", candidate)
        candidate = candidate.replace("^", "**")

        # Normalize implicit multiplication patterns: 2x, x(, )x, and mn.
        candidate = re.sub(r"(?<=\d)\s*(?=[A-Za-z(])", "*", candidate)
        candidate = re.sub(r"(?<=[A-Za-z)])\s*(?=\d|\()", "*", candidate)
        candidate = re.sub(r"(?<=[A-Za-z])\s*(?=[A-Za-z])", "*", candidate)

        candidate = re.sub(r"[^A-Za-z0-9+\-*/()._=\s*]", " ", candidate)
        candidate = " ".join(candidate.split())

        if "=" in candidate:
            rhs = candidate.split("=")[-1].strip()
            if rhs:
                candidate = rhs

        return candidate.strip()
