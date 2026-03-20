"""Retag and regroup example-bank difficulty labels.

This script normalizes per-entry `difficulty` labels and re-sorts each subject
into `basic`/`intermediate`/`advanced` buckets.

Rules:
- Prefer normalized entry-level difficulty when present.
- Otherwise, use the source bucket label (if available).
- Otherwise, infer from PromptGenerator complexity heuristic.

Usage:
  python scripts/retag_example_bank_difficulties.py            # dry run
  python scripts/retag_example_bank_difficulties.py --apply    # write updates
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.prompt_generator import PromptGenerator


TARGET_PATH = Path("framework/example_problems.json")
DIFFICULTY_ORDER = ("basic", "intermediate", "advanced")

# Additional aliases seen in manually edited banks.
DIFFICULTY_ALIASES_EXTRA = {
    "interm": "intermediate",
    "int": "intermediate",
    "adv": "advanced",
    "beg": "basic",
}


def _difficulty_from_level(level: float) -> str:
    if level <= 1.5:
        return "basic"
    if level <= 2.5:
        return "intermediate"
    return "advanced"


def _normalize_difficulty_label(pg: PromptGenerator, value: Any) -> Optional[str]:
    if value is None:
        return None

    # Normalize textual aliases first.
    if isinstance(value, str):
        base = pg._normalize_metadata_label(value)
        if base in DIFFICULTY_ALIASES_EXTRA:
            value = DIFFICULTY_ALIASES_EXTRA[base]

    normalized = pg._normalize_difficulty_label(value)
    if normalized is None:
        return None

    if normalized in DIFFICULTY_ORDER:
        return normalized

    level = pg._difficulty_to_level(normalized)
    if level is None:
        return None
    return _difficulty_from_level(level)


def _iter_subject_groups(raw_data: Dict[str, Any]) -> Iterable[Tuple[str, Optional[str], List[Any]]]:
    for subject, subject_value in raw_data.items():
        if isinstance(subject_value, list):
            yield str(subject), None, subject_value
            continue

        if not isinstance(subject_value, dict):
            continue

        for difficulty_key, examples in subject_value.items():
            if isinstance(examples, list):
                yield str(subject), str(difficulty_key), examples


def _entry_problem_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("problem", "") or "")


def _resolve_target_difficulty(
    pg: PromptGenerator,
    *,
    subject: str,
    problem_text: str,
    entry_difficulty: Optional[str],
    source_bucket: Optional[str],
) -> str:
    if entry_difficulty in DIFFICULTY_ORDER:
        return entry_difficulty

    if source_bucket in DIFFICULTY_ORDER:
        return source_bucket

    if problem_text:
        complexity = pg._estimate_problem_complexity(problem_text, subject)
        return _difficulty_from_level(float(complexity))

    return "intermediate"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize and regroup example-bank difficulties"
    )
    parser.add_argument("--apply", action="store_true", help="Write updates to framework/example_problems.json")
    parser.add_argument("--show", type=int, default=30, help="How many sample moved entries to print")
    args = parser.parse_args()

    data = json.loads(TARGET_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Expected top-level object in example bank JSON")

    pg = PromptGenerator()

    processed_subjects = 0
    processed_entries = 0
    relabeled_entries = 0
    moved_entries = 0

    source_bucket_counter: Counter[str] = Counter()
    target_bucket_counter: Counter[str] = Counter()
    transition_counter: Counter[Tuple[str, str]] = Counter()
    sample_moves: List[Tuple[str, str, str, str]] = []

    regrouped_data: Dict[str, Dict[str, List[Any]]] = {}

    # Build per-subject staging areas first so dry-runs are side-effect free.
    subject_to_groups: Dict[str, List[Tuple[Optional[str], List[Any]]]] = {}
    for subject, difficulty_key, examples in _iter_subject_groups(data):
        subject_to_groups.setdefault(subject, []).append((difficulty_key, examples))

    for subject, groups in subject_to_groups.items():
        processed_subjects += 1
        buckets: Dict[str, List[Any]] = {name: [] for name in DIFFICULTY_ORDER}

        for difficulty_key, examples in groups:
            source_bucket = _normalize_difficulty_label(pg, difficulty_key)

            for raw_entry in examples:
                # Preserve non-dict entries by routing them to source/heuristic bucket.
                if not isinstance(raw_entry, dict):
                    target = source_bucket or "intermediate"
                    buckets[target].append(raw_entry)
                    source_bucket_counter[source_bucket or "(none)"] += 1
                    target_bucket_counter[target] += 1
                    if source_bucket and source_bucket != target:
                        moved_entries += 1
                        transition_counter[(source_bucket, target)] += 1
                    continue

                entry = raw_entry
                processed_entries += 1

                problem_text = _entry_problem_text(entry)
                entry_difficulty = _normalize_difficulty_label(pg, entry.get("difficulty"))
                target = _resolve_target_difficulty(
                    pg,
                    subject=subject,
                    problem_text=problem_text,
                    entry_difficulty=entry_difficulty,
                    source_bucket=source_bucket,
                )

                source_bucket_label = source_bucket or "(list)"
                source_bucket_counter[source_bucket_label] += 1
                target_bucket_counter[target] += 1

                previous_stored = entry.get("difficulty")
                if previous_stored != target:
                    relabeled_entries += 1

                if source_bucket is not None and source_bucket != target:
                    moved_entries += 1
                    transition_counter[(source_bucket, target)] += 1
                    if len(sample_moves) < max(args.show, 0):
                        sample_moves.append(
                            (
                                subject,
                                source_bucket,
                                target,
                                problem_text[:140].replace("\n", " "),
                            )
                        )

                if args.apply:
                    entry["difficulty"] = target
                buckets[target].append(entry)

        regrouped_data[subject] = {name: buckets[name] for name in DIFFICULTY_ORDER}

    print(f"Processed subjects: {processed_subjects}")
    print(f"Processed entries: {processed_entries}")
    print(f"Entries with normalized/updated difficulty label: {relabeled_entries}")
    print(f"Entries moved across difficulty buckets: {moved_entries}")

    print("\nSource bucket distribution:")
    for bucket, count in source_bucket_counter.most_common():
        print(f"  {bucket}: {count}")

    print("\nTarget bucket distribution:")
    for bucket in DIFFICULTY_ORDER:
        print(f"  {bucket}: {target_bucket_counter.get(bucket, 0)}")

    if transition_counter:
        print("\nTop bucket transitions:")
        for (source, target), count in transition_counter.most_common(20):
            print(f"  {source} -> {target}: {count}")

    if sample_moves:
        print("\nSample moved entries:")
        for subject, source, target, snippet in sample_moves:
            print(f"  [{subject}] {source} -> {target} | {snippet}")

    if args.apply:
        for subject, regrouped_subject in regrouped_data.items():
            data[subject] = regrouped_subject

        TARGET_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nOutput: {TARGET_PATH}")
    else:
        print("\nDry run only. Re-run with --apply to persist updates.")


if __name__ == "__main__":
    main()
