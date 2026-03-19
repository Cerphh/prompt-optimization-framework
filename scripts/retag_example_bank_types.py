"""Retag misclassified example types in the example bank.

This script uses the current `PromptGenerator._detect_primary_intent` logic to
identify entries whose `type` metadata is out of sync with problem text, then
optionally updates `type` (and keeps tags aligned) across the full bank.

Usage:
  python scripts/retag_example_bank_types.py            # dry run
  python scripts/retag_example_bank_types.py --apply    # write updates
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


def _normalized_type(pg: PromptGenerator, value: Any) -> Optional[str]:
    labels = pg._normalize_label_list(value)
    if not labels:
        return None
    return pg._normalize_type_label(labels[0])


def _sync_tags(pg: PromptGenerator, entry: Dict[str, Any], old_type: Optional[str], new_type: str) -> None:
    """Keep tags consistent with retagged type while preserving user-defined tags."""
    raw_tags = entry.get("tags")
    if not isinstance(raw_tags, list):
        return

    normalized_tags = []
    seen = set()
    for raw in raw_tags:
        tag = pg._normalize_metadata_label(str(raw))
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized_tags.append(tag)

    if old_type and old_type in normalized_tags:
        normalized_tags = [tag for tag in normalized_tags if tag != old_type]

    if new_type not in normalized_tags:
        normalized_tags.append(new_type)

    entry["tags"] = normalized_tags[:8]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retag misclassified example types in example_problems.json")
    parser.add_argument("--apply", action="store_true", help="Write updates to framework/example_problems.json")
    parser.add_argument("--show", type=int, default=30, help="How many sample mismatches to print")
    args = parser.parse_args()

    data = json.loads(TARGET_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Expected top-level object in example bank JSON")

    pg = PromptGenerator()

    total_entries = 0
    mismatches = 0
    changed = 0
    transitions: Counter[Tuple[Optional[str], str]] = Counter()
    samples: List[Tuple[str, Optional[str], str, str]] = []

    for subject, _, examples in _iter_subject_groups(data):
        for entry in examples:
            if not isinstance(entry, dict):
                continue

            problem = entry.get("problem")
            if not problem:
                continue

            total_entries += 1

            old_type = _normalized_type(pg, entry.get("type"))
            detected = pg._normalize_type_label(pg._detect_primary_intent(str(problem)))

            if old_type == detected:
                continue

            mismatches += 1
            transitions[(old_type, detected)] += 1
            if len(samples) < max(args.show, 0):
                samples.append((subject, old_type, detected, str(problem)))

            if args.apply:
                entry["type"] = detected
                _sync_tags(pg, entry, old_type, detected)
                changed += 1

    print(f"Processed entries: {total_entries}")
    print(f"Mismatched type entries: {mismatches}")
    print("Top transitions:")
    for (old_type, new_type), count in transitions.most_common(20):
        print(f"  {old_type} -> {new_type}: {count}")

    if samples:
        print("\nSample mismatches:")
        for subject, old_type, new_type, problem in samples:
            print(f"  [{subject}] {old_type} -> {new_type} | {problem}")

    if args.apply:
        TARGET_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nEntries updated: {changed}")
        print(f"Output: {TARGET_PATH}")
    else:
        print("\nDry run only. Re-run with --apply to persist updates.")


if __name__ == "__main__":
    main()
