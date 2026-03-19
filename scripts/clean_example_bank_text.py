"""Clean example bank text formatting without changing problem intent/content.

This script rewrites `framework/example_problems.json` with normalized, readable
`problem` and `solution` text while preserving structure and metadata.

Usage:
  python scripts/clean_example_bank_text.py            # dry run (no write)
  python scripts/clean_example_bank_text.py --apply    # write updates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

TARGET_PATH = Path("framework/example_problems.json")


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


def _is_fragmented_line_layout(lines: List[str]) -> bool:
    """Detect OCR-like token-per-line artifacts and flatten them."""
    if len(lines) < 6:
        return False

    compact = [re.sub(r"\s+", "", line) for line in lines if line]
    if not compact:
        return False

    short_tokens = sum(1 for token in compact if len(token) <= 3)
    symbol_only = sum(
        1
        for token in compact
        if re.fullmatch(r"[0-9a-zA-Z+\-*/=^().,%]+", token) is not None
    )

    short_ratio = short_tokens / len(compact)
    symbol_ratio = symbol_only / len(compact)
    return short_ratio >= 0.55 or (len(compact) >= 8 and symbol_ratio >= 0.75)


def _normalize_text(text: Any, *, is_solution: bool) -> str:
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
        normalized_lines.append(cleaned_line)

    if not normalized_lines:
        return value

    has_block_context = any(
        marker in value
        for marker in (
            "[asy]",
            "[/asy]",
            "\\begin{",
            "\\end{",
        )
    )

    if _is_fragmented_line_layout(normalized_lines) and not has_block_context:
        value = " ".join(normalized_lines)
    else:
        value = "\n".join(normalized_lines)

    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _first_diff_line(old: str, new: str) -> int:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    for idx, (o, n) in enumerate(zip(old_lines, new_lines), start=1):
        if o != n:
            return idx

    if len(old_lines) != len(new_lines):
        return min(len(old_lines), len(new_lines)) + 1

    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean formatting in framework/example_problems.json")
    parser.add_argument("--apply", action="store_true", help="Write updates to framework/example_problems.json")
    parser.add_argument("--show", type=int, default=20, help="How many sample edits to display")
    args = parser.parse_args()

    raw = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected top-level object in example bank JSON")

    total_entries = 0
    problem_updates = 0
    solution_updates = 0
    sample_changes: List[Tuple[str, Optional[str], int, str, int, str, str]] = []

    for subject, difficulty, examples in _iter_subject_groups(raw):
        for idx, entry in enumerate(examples):
            if not isinstance(entry, dict):
                continue

            problem = entry.get("problem")
            solution = entry.get("solution")
            if problem is None and solution is None:
                continue

            total_entries += 1

            if problem is not None:
                old_problem = str(problem)
                new_problem = _normalize_text(old_problem, is_solution=False)
                if new_problem != old_problem:
                    problem_updates += 1
                    if len(sample_changes) < max(args.show, 0):
                        line_no = _first_diff_line(old_problem, new_problem)
                        sample_changes.append(
                            (subject, difficulty, idx, "problem", line_no, old_problem[:160], new_problem[:160])
                        )
                    if args.apply:
                        entry["problem"] = new_problem

            if solution is not None:
                old_solution = str(solution)
                new_solution = _normalize_text(old_solution, is_solution=True)
                if new_solution != old_solution:
                    solution_updates += 1
                    if len(sample_changes) < max(args.show, 0):
                        line_no = _first_diff_line(old_solution, new_solution)
                        sample_changes.append(
                            (subject, difficulty, idx, "solution", line_no, old_solution[:160], new_solution[:160])
                        )
                    if args.apply:
                        entry["solution"] = new_solution

    total_updates = problem_updates + solution_updates

    print(f"Processed entries: {total_entries}")
    print(f"Updated problem fields: {problem_updates}")
    print(f"Updated solution fields: {solution_updates}")
    print(f"Total updated fields: {total_updates}")

    if sample_changes:
        print("\nSample edits:")
        for subject, difficulty, idx, field, line_no, old_preview, new_preview in sample_changes:
            print(
                f"  [{subject}/{difficulty}] entry #{idx} {field} line {line_no}:\n"
                f"    old: {old_preview}\n"
                f"    new: {new_preview}"
            )

    if args.apply:
        TARGET_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote cleaned example bank to {TARGET_PATH}")
    else:
        print("\nDry run only. Re-run with --apply to persist updates.")


if __name__ == "__main__":
    main()
