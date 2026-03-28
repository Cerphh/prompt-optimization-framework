"""
Firestore persistence module.
Stores benchmark results in Firebase Firestore.
"""

from __future__ import annotations

import json
import os
import hashlib
import re
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Set, Tuple

import firebase_admin
from firebase_admin import credentials, firestore


class FirestoreStore:
    """Handles benchmark result persistence to Firestore."""

    def __init__(
        self,
        collection_name: str = "benchmark_results",
        enabled: bool = True,
        required: bool = False,
    ):
        self.collection_name = collection_name
        self.enabled = enabled
        self.required = required
        self.db = None
        self.initialization_error: Optional[str] = None

        if self.enabled:
            self._initialize()

    def _load_credentials(self):
        """Load Firebase credentials from env vars."""
        service_account_key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

        if service_account_key_path:
            return credentials.Certificate(service_account_key_path)

        if service_account_json:
            parsed = json.loads(service_account_json)
            return credentials.Certificate(parsed)

        return None

    def _initialize(self):
        """Initialize Firebase app and Firestore client."""
        try:
            cred = self._load_credentials()
            if not firebase_admin._apps:
                options = {}
                project_id = os.getenv("FIREBASE_PROJECT_ID")
                if project_id:
                    options["projectId"] = project_id

                if cred is not None:
                    if options:
                        firebase_admin.initialize_app(cred, options)
                    else:
                        firebase_admin.initialize_app(cred)
                elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    if options:
                        firebase_admin.initialize_app(options=options)
                    else:
                        firebase_admin.initialize_app()
                else:
                    self.initialization_error = (
                        "Missing Firebase Admin credentials. Set FIREBASE_SERVICE_ACCOUNT_KEY, "
                        "FIREBASE_SERVICE_ACCOUNT_JSON, or GOOGLE_APPLICATION_CREDENTIALS. "
                        "Note: Web SDK config (apiKey/authDomain/projectId) is not sufficient "
                        "for server-side Firestore writes."
                    )
                    return

            self.db = firestore.client()
            self.initialization_error = None
        except Exception as exc:
            self.initialization_error = str(exc)

    def get_status(self) -> Dict[str, Any]:
        """Get Firestore integration status."""
        return {
            "enabled": self.enabled,
            "required": self.required,
            "ready": self.db is not None,
            "collection": self.collection_name,
            "error": self.initialization_error,
        }

    def save_benchmark_result(
        self,
        benchmark_result: Dict[str, Any],
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a benchmark result to a domain/difficulty document in Firestore.
        
        Schema: benchmark_results/{domain}/{difficulty}/{problem_id}
        - result_per_run: stores run1/run2/run3 payloads
        - 3_run_ave: stores aggregated metrics from those runs
        """
        if not self.enabled:
            return {
                "success": False,
                "reason": "disabled",
                "error": "Firestore persistence is disabled.",
            }

        if self.db is None:
            return {
                "success": False,
                "reason": "not_initialized",
                "error": self.initialization_error or "Firestore is not initialized.",
            }

        try:
            metadata = metadata or {}
            domain = self._resolve_domain(benchmark_result=benchmark_result, metadata=metadata)
            difficulty = self._resolve_difficulty(benchmark_result=benchmark_result, metadata=metadata)

            payload = self._build_storage_document(
                benchmark_result=benchmark_result,
                metadata=metadata,
                source=source,
            )
            problem_text = self._resolve_problem_text(benchmark_result=benchmark_result, metadata=metadata)
            problem_id = self._build_problem_document_id(problem_text)

            domain_doc_ref = self.db.collection(self.collection_name).document(domain)
            difficulty_collection_ref = domain_doc_ref.collection(difficulty)
            problem_doc_ref = difficulty_collection_ref.document(problem_id)

            # Build run1/run2/run3 directly from the current benchmark run history.
            result_per_run = self._build_run_entries_from_benchmark_result(
                benchmark_result=benchmark_result,
                fallback_payload=payload,
            )
            
            # Compute 3_run_ave from last 3 results
            three_run_ave = self._compute_3_run_average(result_per_run)

            problem_doc_ref.set(
                {
                    "problem_id": problem_id,
                    "problem": problem_text,
                    "has_ground_truth": payload.get("has_ground_truth"),
                    "evaluation_quality": payload.get("evaluation_quality"),
                    "result_per_run": self._encode_runs_as_named_map(result_per_run),
                    "3_run_ave": three_run_ave,
                },
                merge=True,
            )

            return {
                "success": True,
                "collection": self.collection_name,
                "domain": domain,
                "difficulty": difficulty,
                "problem_id": problem_id,
                "result_id": payload.get("result_id"),
                "result_per_run_count": len(result_per_run),
            }
        except Exception as exc:
            return {
                "success": False,
                "reason": "write_failed",
                "error": str(exc),
            }

    def _build_run_entries_from_benchmark_result(
        self,
        benchmark_result: Dict[str, Any],
        fallback_payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build up to 3 run entries from best_result.run_history with metric-level fields."""
        best_result = benchmark_result.get("best_result", {}) if isinstance(benchmark_result, dict) else {}
        run_history = best_result.get("run_history", []) if isinstance(best_result, dict) else []

        entries: List[Dict[str, Any]] = []
        if isinstance(run_history, list):
            for run in run_history:
                if not isinstance(run, dict):
                    continue
                run_scores = run.get("scores", {}) if isinstance(run.get("scores"), dict) else {}
                overall_value = self._to_float(run_scores.get("overall"))

                entries.append(
                    {
                        "result_id": fallback_payload.get("result_id"),
                        "source": fallback_payload.get("source"),
                        "domain": fallback_payload.get("domain"),
                        "difficulty": fallback_payload.get("difficulty"),
                        "run_mode": fallback_payload.get("run_mode"),
                        "has_ground_truth": fallback_payload.get("has_ground_truth"),
                        "evaluation_quality": fallback_payload.get("evaluation_quality"),
                        "problem": fallback_payload.get("problem"),
                        "run_index": run.get("run_index"),
                        "prompt_used": run.get("prompt") or fallback_payload.get("prompt_used"),
                        "model_response": run.get("response") or fallback_payload.get("model_response"),
                        "performance_score": overall_value,
                        "overall": overall_value,
                        "metric_result": {
                            "accuracy": self._to_float(run_scores.get("accuracy")),
                            "consistency": self._to_float(run_scores.get("consistency")),
                            "efficiency": self._to_float(run_scores.get("efficiency")),
                            "overall": overall_value,
                        },
                        "technique_comparison": fallback_payload.get("technique_comparison", []),
                    }
                )

        # Fallback when run_history is missing.
        if not entries:
            scores = (
                best_result.get("scores", {})
                if isinstance(best_result, dict) and isinstance(best_result.get("scores"), dict)
                else {}
            )
            fallback_entry = dict(fallback_payload)
            fallback_entry["run_index"] = 1
            fallback_entry["overall"] = self._to_float(scores.get("overall")) or self._to_float(
                fallback_payload.get("performance_score")
            )
            fallback_entry["metric_result"] = {
                "accuracy": self._to_float(scores.get("accuracy")),
                "consistency": self._to_float(scores.get("consistency")),
                "efficiency": self._to_float(scores.get("efficiency")),
                "overall": fallback_entry["overall"],
            }
            entries = [fallback_entry]

        return entries[-3:]

    def _encode_runs_as_named_map(self, runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Encode latest runs as explicit run1/run2/run3 slots."""
        named_runs: Dict[str, Dict[str, Any]] = {}
        latest_runs = runs[-3:]
        for index in range(1, 4):
            if index <= len(latest_runs):
                named_runs[f"run{index}"] = latest_runs[index - 1]
            else:
                named_runs[f"run{index}"] = {
                    "run_index": index,
                    "overall": None,
                    "metric_result": {
                        "accuracy": None,
                        "consistency": None,
                        "efficiency": None,
                        "overall": None,
                    },
                }
        return named_runs

    def _decode_runs_from_result_per_run_doc(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decode run payloads from either named slots or legacy list shape."""
        raw = (doc_data or {}).get("result_per_run")
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)][-3:]

        if isinstance(raw, dict):
            runs: List[Dict[str, Any]] = []
            for key in ("run1", "run2", "run3"):
                value = raw.get(key)
                if isinstance(value, dict):
                    runs.append(value)
            return runs[-3:]

        return []

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _compute_3_run_average(self, result_per_run: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated metrics from the last 3 results in result_per_run."""
        if not result_per_run:
            return {}
        
        # Get last 3 results
        last_3 = result_per_run[-3:] if len(result_per_run) >= 3 else result_per_run
        
        # Aggregate top-level performance scores.
        performance_scores = [r.get("performance_score") for r in last_3 if r.get("performance_score") is not None]
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else None

        metric_totals = {"accuracy": 0.0, "consistency": 0.0, "efficiency": 0.0, "overall": 0.0}
        metric_counts = {"accuracy": 0, "consistency": 0, "efficiency": 0, "overall": 0}

        # Aggregate run-level metric_result values (run1/run2/run3).
        for result in last_3:
            metric_result = result.get("metric_result", {})
            if not isinstance(metric_result, dict):
                continue
            for metric_key in ("accuracy", "consistency", "efficiency", "overall"):
                metric_value = self._to_float(metric_result.get(metric_key))
                if metric_value is None:
                    continue
                metric_totals[metric_key] += metric_value
                metric_counts[metric_key] += 1
        
        # Aggregate per-technique metrics across runs.
        technique_totals: Dict[str, Dict[str, float]] = {}
        technique_counts: Dict[str, Dict[str, int]] = {}
        
        for result in last_3:
            comparisons = result.get("technique_comparison", [])
            if isinstance(comparisons, list):
                for comparison in comparisons:
                    if isinstance(comparison, dict):
                        technique = comparison.get("technique")
                        if not technique:
                            continue

                        if technique not in technique_totals:
                            technique_totals[technique] = {
                                "accuracy": 0.0,
                                "consistency": 0.0,
                                "efficiency": 0.0,
                                "overall": 0.0,
                            }
                            technique_counts[technique] = {
                                "accuracy": 0,
                                "consistency": 0,
                                "efficiency": 0,
                                "overall": 0,
                            }

                        for metric_key in ("accuracy", "consistency", "efficiency", "overall"):
                            metric_value = self._to_float(comparison.get(metric_key))
                            if metric_value is None:
                                continue
                            technique_totals[technique][metric_key] += metric_value
                            technique_counts[technique][metric_key] += 1
        
        technique_averages: Dict[str, Dict[str, Optional[float]]] = {}
        for technique, totals in technique_totals.items():
            counts = technique_counts[technique]
            technique_averages[technique] = {}
            for metric_key in ("accuracy", "consistency", "efficiency", "overall"):
                count = counts[metric_key]
                technique_averages[technique][metric_key] = round(totals[metric_key] / count, 4) if count else None

        metric_averages: Dict[str, Optional[float]] = {}
        for metric_key in ("accuracy", "consistency", "efficiency", "overall"):
            count = metric_counts[metric_key]
            metric_averages[metric_key] = round(metric_totals[metric_key] / count, 4) if count else None
        
        return {
            "avg_performance_score": round(avg_performance, 4) if avg_performance is not None else None,
            "metric_result": metric_averages,
            "metric_averages": metric_averages,
            "technique_averages": technique_averages,
            "sample_count": len(last_3),
        }

    def _resolve_problem_text(self, benchmark_result: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        problem = metadata.get("problem") or benchmark_result.get("problem") or ""
        text = str(problem).strip()
        return text or "unknown_problem"

    def _normalize_problem_text(self, problem: str) -> str:
        return " ".join(str(problem or "").strip().lower().split())

    def _build_problem_document_id(self, problem: str) -> str:
        normalized_problem = self._normalize_problem_text(problem)
        digest = hashlib.sha1(normalized_problem.encode("utf-8")).hexdigest()
        return f"problem_{digest}"

    def _build_storage_document(
        self,
        benchmark_result: Dict[str, Any],
        metadata: Dict[str, Any],
        source: str,
    ) -> Dict[str, Any]:
        """Build compact result payload for a problem-level results list."""
        best_result = benchmark_result.get("best_result", {})
        scores = best_result.get("scores", {})

        domain = self._resolve_domain(benchmark_result=benchmark_result, metadata=metadata)
        difficulty = self._resolve_difficulty(benchmark_result=benchmark_result, metadata=metadata)
        problem_text = self._resolve_problem_text(benchmark_result=benchmark_result, metadata=metadata)

        raw_profile = metadata.get("problem_profile")
        if raw_profile is None:
            raw_profile = benchmark_result.get("problem_profile")
        normalized_profile = self._normalize_problem_profile(raw_profile)

        run_mode_raw = metadata.get("run_mode") or benchmark_result.get("run_mode") or "normal"
        run_mode = self._normalize_key(run_mode_raw, default="normal")

        has_ground_truth: Optional[bool] = None
        raw_ground_truth_flag = metadata.get("has_ground_truth")
        if isinstance(raw_ground_truth_flag, bool):
            has_ground_truth = raw_ground_truth_flag
        elif isinstance(raw_ground_truth_flag, (int, float)):
            has_ground_truth = bool(raw_ground_truth_flag)
        elif isinstance(raw_ground_truth_flag, str):
            normalized_flag = raw_ground_truth_flag.strip().lower()
            if normalized_flag in {"1", "true", "yes", "on"}:
                has_ground_truth = True
            elif normalized_flag in {"0", "false", "no", "off"}:
                has_ground_truth = False

        if has_ground_truth is None:
            ground_truth_value = metadata.get("ground_truth", benchmark_result.get("ground_truth"))
            has_ground_truth = bool(str(ground_truth_value).strip()) if ground_truth_value is not None else False

        evaluation_quality = "ground_truth" if has_ground_truth else "no_ground_truth"

        document = {
            "result_id": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "domain": domain,
            "difficulty": difficulty,
            "run_mode": run_mode,
            "has_ground_truth": has_ground_truth,
            "evaluation_quality": evaluation_quality,
            "problem": problem_text,
            "prompt_used": best_result.get("prompt"),
            "model_response": best_result.get("response"),
            "performance_score": scores.get("overall"),
            "technique_comparison": benchmark_result.get("comparison", []),
        }

        if normalized_profile:
            document["problem_profile"] = normalized_profile

        return document

    def _extract_result_entries(self, doc_ref: Any, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract result entries from result_per_run document schema."""
        return self._decode_runs_from_result_per_run_doc(data)

    @staticmethod
    def _bool_from_value(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    def _entry_has_ground_truth(self, entry: Dict[str, Any], data: Dict[str, Any]) -> Optional[bool]:
        for payload in (entry, data):
            parsed = self._bool_from_value((payload or {}).get("has_ground_truth"))
            if parsed is not None:
                return parsed

        for payload in (entry, data):
            quality = str((payload or {}).get("evaluation_quality", "") or "").strip().lower()
            if quality == "ground_truth":
                return True
            if quality in {"no_ground_truth", "heuristic"}:
                return False

        return None

    def _doc_has_ground_truth(self, data: Dict[str, Any]) -> Optional[bool]:
        """Check ground truth at the document level, falling back to run entries."""
        # Check doc-level field first.
        result = self._entry_has_ground_truth(data, data)
        if result is not None:
            return result
        # Fallback: check inside result_per_run entries (legacy data).
        for entry in self._extract_result_entries(None, data):
            result = self._entry_has_ground_truth(entry, data)
            if result is not None:
                return result
        return None

    def _normalize_profile_labels(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = [str(item) for item in value if item is not None]
        else:
            raw_items = [str(value)]

        labels: List[str] = []
        seen = set()
        for raw in raw_items:
            normalized = self._normalize_key(raw, default="")
            if normalized and normalized not in seen:
                seen.add(normalized)
                labels.append(normalized)

        labels.sort()
        return labels

    def _normalize_problem_profile(self, profile: Any) -> Dict[str, Any]:
        if not isinstance(profile, dict):
            return {}

        normalized: Dict[str, Any] = {}

        subject = self._normalize_key(profile.get("subject"), default="")
        if subject:
            normalized["subject"] = subject

        difficulty = self._normalize_key(profile.get("difficulty"), default="")
        if difficulty:
            normalized["difficulty"] = difficulty

        intent = self._normalize_key(profile.get("intent"), default="")
        if intent:
            normalized["intent"] = intent

        features = self._normalize_profile_labels(profile.get("features"))
        if features:
            normalized["features"] = features

        format_labels = self._normalize_profile_labels(profile.get("format_labels"))
        if format_labels:
            normalized["format_labels"] = format_labels

        constraints = self._normalize_profile_labels(profile.get("constraints"))
        if constraints:
            normalized["constraints"] = constraints

        return normalized

    @staticmethod
    def _overlap_ratio(target: Set[str], candidate: Set[str]) -> float:
        if not target:
            return 0.0
        return len(target.intersection(candidate)) / len(target)

    # Common English stop words to strip when refining problem text for matching.
    _STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must",
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
        "his", "she", "her", "it", "its", "they", "them", "their",
        "this", "that", "these", "those", "which", "who", "whom", "what",
        "and", "or", "but", "nor", "not", "no", "so", "if", "then", "else",
        "when", "where", "how", "why", "than", "because", "since", "while",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
        "into", "through", "during", "before", "after", "about", "between",
        "above", "below", "up", "down", "out", "off", "over", "under",
        "again", "further", "once", "here", "there", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "too", "very", "just", "also", "now",
    }

    @staticmethod
    def _tokenize_text(text: str) -> Set[str]:
        """Tokenize text into normalized lowercase word tokens."""
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    @classmethod
    def _refine_problem_text(cls, text: str) -> Set[str]:
        """Refine problem text for matching: tokenize, remove stop words, keep content words.

        For natural-language problems this strips noise words so the
        remaining tokens (math terms, numbers, variable names) drive
        the similarity comparison.
        """
        tokens = cls._tokenize_text(text)
        return tokens - cls._STOP_WORDS

    @classmethod
    def _lexical_similarity(cls, text_a: str, text_b: str) -> float:
        """Jaccard similarity between two problem texts based on refined word overlap."""
        tokens_a = cls._refine_problem_text(text_a)
        tokens_b = cls._refine_problem_text(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union else 0.0

    def _profile_similarity(
        self,
        target: Dict[str, Any],
        candidate: Dict[str, Any],
        target_text: str = "",
        candidate_text: str = "",
    ) -> float:
        """Compute similarity between two problems using refined word-based matching.

        The 4 structured components (intent/features/format/constraints) have
        been removed — similarity is now purely lexical.  Stop words are
        stripped so that content words (math terms, numbers, variables) drive
        the comparison.  This is simpler to tune and works equally well for
        natural-language and symbolic problems.
        """
        target_t = str(target_text or "").strip()
        candidate_t = str(candidate_text or "").strip()
        if not target_t or not candidate_t:
            return 0.0
        return self._lexical_similarity(target_t, candidate_t)

    @staticmethod
    def _normalize_key(value: Any, default: str) -> str:
        if not value:
            return default
        normalized = str(value).strip().lower().replace(" ", "_")
        return normalized or default

    def _problem_collection_ref(self, domain: str, difficulty: str) -> Any:
        """Return collection ref for domain/difficulty aggregated results (legacy hierarchy fallback).
        
        New schema (Option A): benchmark_results/{domain}/{difficulty}/aggregated
        Legacy schema: benchmark_results/{domain}/{difficulty}/3_runs/problems/{problem_id}
        """
        # New schema location
        return (
            self.db.collection(self.collection_name)
            .document(domain)
            .collection(difficulty)
        )

    def _stream_problem_docs(self, domain: str, difficulty: str):
        """Stream all problem documents under domain/difficulty."""
        try:
            docs = list(
                self.db.collection(self.collection_name)
                .document(domain)
                .collection(difficulty)
                .stream()
            )
            for doc in docs:
                yield doc
            return
        except Exception:
            # If query fails, no data available
            pass

    def _resolve_domain(self, benchmark_result: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        domain = (
            metadata.get("domain")
            or metadata.get("subject")
            or metadata.get("category")
            or benchmark_result.get("domain")
            or "general"
        )
        return self._normalize_key(domain, default="general")

    def _resolve_difficulty(self, benchmark_result: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        difficulty = (
            metadata.get("difficulty")
            or benchmark_result.get("difficulty")
            or "basic"
        )
        return self._normalize_key(difficulty, default="basic")

    def get_best_technique_by_domain(
        self,
        domain: str,
        difficulty: str = "basic",
        available_techniques: Optional[List[str]] = None,
        require_ground_truth: bool = False,
    ) -> Dict[str, Any]:
        """Get best technique for a domain based on historical Firestore results."""
        if not self.enabled:
            return {
                "success": False,
                "reason": "disabled",
                "error": "Firestore persistence is disabled.",
            }

        if self.db is None:
            return {
                "success": False,
                "reason": "not_initialized",
                "error": self.initialization_error or "Firestore is not initialized.",
            }

        try:
            totals: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            allowed = set(available_techniques or [])
            allow_unknown_quality = os.getenv("DB_HISTORY_ALLOW_UNKNOWN_QUALITY", "false").strip().lower() == "true"
            normalized_domain = self._normalize_key(domain, default="general")
            normalized_difficulty = self._normalize_key(difficulty, default="basic")

            query = (
                self._stream_problem_docs(
                    domain=normalized_domain,
                    difficulty=normalized_difficulty,
                )
            )
            for doc in query:
                data = doc.to_dict() or {}

                if require_ground_truth:
                    has_ground_truth = self._doc_has_ground_truth(data)
                    if has_ground_truth is False:
                        continue
                    if has_ground_truth is None and not allow_unknown_quality:
                        continue

                # Read per-problem averaged scores from 3_run_ave
                three_run_ave = data.get("3_run_ave") or {}
                technique_averages = three_run_ave.get("technique_averages") or {}

                for technique, metrics in technique_averages.items():
                    if not isinstance(metrics, dict):
                        continue
                    overall = metrics.get("overall")
                    if overall is None or not isinstance(overall, (int, float)):
                        continue
                    if allowed and technique not in allowed:
                        continue

                    totals[technique] = totals.get(technique, 0.0) + float(overall)
                    counts[technique] = counts.get(technique, 0) + 1

            if not counts:
                return {
                    "success": False,
                    "reason": "no_data",
                    "error": (
                        f"No historical technique data found for domain '{normalized_domain}' "
                        f"and difficulty '{normalized_difficulty}'."
                    ),
                }

            ranking = []
            for technique, count in counts.items():
                average = totals[technique] / count
                ranking.append({
                    "technique": technique,
                    "average_overall": round(average, 4),
                    "samples": count,
                })

            ranking.sort(key=lambda x: (-x["average_overall"], -x["samples"], x["technique"]))
            best = ranking[0]

            return {
                "success": True,
                "domain": normalized_domain,
                "difficulty": normalized_difficulty,
                "require_ground_truth": require_ground_truth,
                "best_technique": best["technique"],
                "ranking": ranking,
            }
        except Exception as exc:
            return {
                "success": False,
                "reason": "query_failed",
                "error": str(exc),
            }

    def get_best_technique_by_profile(
        self,
        domain: str,
        difficulty: str,
        problem_profile: Optional[Dict[str, Any]],
        available_techniques: Optional[List[str]] = None,
        min_similarity: float = 0.35,
        require_ground_truth: bool = False,
        problem_text: str = "",
    ) -> Dict[str, Any]:
        """Get best technique using word-based similarity matching over historical results.
        
        Similarity is computed purely from problem text (lexical/Jaccard after
        stop-word removal).  Structured profile components are no longer used
        for matching — problem_text is the primary input.
        
        Improvements:
        - Implements weighted matching (problems weighted by similarity)
        - Adaptive threshold based on data availability
        - Confidence/variance reporting for recommendations
        """
        if not self.enabled:
            return {
                "success": False,
                "reason": "disabled",
                "error": "Firestore persistence is disabled.",
            }

        if self.db is None:
            return {
                "success": False,
                "reason": "not_initialized",
                "error": self.initialization_error or "Firestore is not initialized.",
            }

        normalized_profile = self._normalize_problem_profile(problem_profile)
        normalized_text = str(problem_text or "").strip()
        if not normalized_text:
            return {
                "success": False,
                "reason": "missing_problem_text",
                "error": "Problem text is required for profile-based selection.",
            }

        try:
            min_similarity = max(0.0, min(1.0, float(min_similarity)))
        except (TypeError, ValueError):
            min_similarity = 0.35

        try:
            allowed = set(available_techniques or [])
            allow_unknown_quality = os.getenv("DB_HISTORY_ALLOW_UNKNOWN_QUALITY", "false").strip().lower() == "true"
            normalized_domain = self._normalize_key(domain, default="general")
            normalized_difficulty = self._normalize_key(difficulty, default="basic")

            query = (
                self._stream_problem_docs(
                    domain=normalized_domain,
                    difficulty=normalized_difficulty,
                )
            )

            # First pass: Count available matches to determine adaptive threshold
            # NOTE: Counts per-problem (not per-run) for honest sample counts.
            preliminary_count = 0
            for doc in query:
                data = doc.to_dict() or {}
                historical_text = str(data.get("problem") or "").strip()
                if not historical_text:
                    continue

                similarity = self._profile_similarity(
                    {},
                    {},
                    target_text=normalized_text,
                    candidate_text=historical_text,
                )
                if similarity >= 0.20:  # Low threshold for counting
                    preliminary_count += 1

            # Adapt threshold based on available data (relaxed for easier matching)
            adaptive_min_similarity = min_similarity  # Start with provided value
            if preliminary_count > 15:
                adaptive_min_similarity = 0.45  # Lots of data → moderately strict
            elif preliminary_count > 10:
                adaptive_min_similarity = 0.40
            elif preliminary_count > 5:
                adaptive_min_similarity = 0.30
            elif preliminary_count > 0:
                adaptive_min_similarity = min(0.20, min_similarity)  # Little data → very lenient

            # IMPROVEMENT #2 & #5 (Weighted problems + Confidence/Variance):
            # Second pass: Accumulate technique scores with similarity weighting.
            # Uses per-problem averages (3_run_ave) instead of individual runs
            # so that 1 problem = 1 sample regardless of how many runs it had.
            totals: Dict[str, float] = {}
            similarity_weighted_totals: Dict[str, float] = {}
            similarity_weights: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            scores_per_technique: Dict[str, List[float]] = {}  # For variance calculation

            matched_documents = 0
            similarities: List[float] = []

            query = (
                self._stream_problem_docs(
                    domain=normalized_domain,
                    difficulty=normalized_difficulty,
                )
            )

            for doc in query:
                data = doc.to_dict() or {}

                if require_ground_truth:
                    has_ground_truth = self._doc_has_ground_truth(data)
                    if has_ground_truth is False:
                        continue
                    if has_ground_truth is None and not allow_unknown_quality:
                        continue

                historical_text = str(data.get("problem") or "").strip()
                if not historical_text:
                    continue

                similarity = self._profile_similarity(
                    {},
                    {},
                    target_text=normalized_text,
                    candidate_text=historical_text,
                )
                if similarity < adaptive_min_similarity:
                    continue

                matched_documents += 1
                similarities.append(similarity)

                # Read per-problem averaged scores from 3_run_ave
                three_run_ave = data.get("3_run_ave") or {}
                technique_averages = three_run_ave.get("technique_averages") or {}

                for technique, metrics in technique_averages.items():
                    if not isinstance(metrics, dict):
                        continue
                    overall = metrics.get("overall")
                    if overall is None or not isinstance(overall, (int, float)):
                        continue
                    if allowed and technique not in allowed:
                        continue

                    overall_f = float(overall)

                    totals[technique] = totals.get(technique, 0.0) + overall_f
                    similarity_weighted_totals[technique] = similarity_weighted_totals.get(technique, 0.0) + (overall_f * similarity)
                    similarity_weights[technique] = similarity_weights.get(technique, 0.0) + similarity
                    counts[technique] = counts.get(technique, 0) + 1

                    if technique not in scores_per_technique:
                        scores_per_technique[technique] = []
                    scores_per_technique[technique].append(overall_f)

            if not counts:
                return {
                    "success": False,
                    "reason": "no_profile_match",
                    "error": (
                        f"No historical profile match found for domain '{normalized_domain}' "
                        f"and difficulty '{normalized_difficulty}'."
                    ),
                    "problem_profile": normalized_profile,
                    "min_similarity": adaptive_min_similarity,
                    "matched_documents": matched_documents,
                    "preliminary_count": preliminary_count,
                }

            # IMPROVEMENT #5: Calculate statistics including variance and confidence
            ranking = []
            for technique, count in counts.items():
                simple_avg = totals[technique] / count
                
                # Weighted average (more similar problems have more influence)
                weighted_avg = similarity_weighted_totals[technique] / similarity_weights[technique]
                
                # Calculate variance and standard deviation
                scores = scores_per_technique.get(technique, [])
                std_dev = 0.0
                variance = 0.0
                if len(scores) > 1:
                    try:
                        variance = statistics.variance(scores)
                        std_dev = statistics.stdev(scores)
                    except Exception:
                        pass
                
                # Determine confidence level based on std_dev
                if std_dev < 0.08:
                    confidence = "high"
                elif std_dev < 0.15:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                # Consistency score (1.0 = perfect consistency)
                consistency = max(0.0, 1.0 - std_dev)
                
                ranking.append({
                    "technique": technique,
                    "simple_average": round(simple_avg, 4),
                    "weighted_average": round(weighted_avg, 4),  # Primary metric now
                    "average_overall": round(weighted_avg, 4),  # For backwards compatibility
                    "samples": count,
                    "unweighted_samples": count,
                    "effective_samples": round(similarity_weights[technique], 2),
                    "std_dev": round(std_dev, 4),
                    "variance": round(variance, 4),
                    "confidence": confidence,
                    "consistency": round(consistency, 4),
                    "individual_scores": [round(s, 4) for s in scores],
                })

            ranking.sort(key=lambda x: (-x["weighted_average"], -x["effective_samples"], x["technique"]))
            best = ranking[0]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            return {
                "success": True,
                "domain": normalized_domain,
                "difficulty": normalized_difficulty,
                "require_ground_truth": require_ground_truth,
                "best_technique": best["technique"],
                "ranking": ranking,
                "problem_profile": normalized_profile,
                "match_type": "profile_rule",
                "min_similarity_requested": min_similarity,
                "min_similarity_adaptive": round(adaptive_min_similarity, 4),
                "preliminary_count": preliminary_count,
                "average_similarity": round(avg_similarity, 4),
                "matched_documents": matched_documents,
                "recommendation_confidence": best.get("confidence"),
                "recommendation_consistency": best.get("consistency"),
            }
        except Exception as exc:
            return {
                "success": False,
                "reason": "query_failed",
                "error": str(exc),
            }
