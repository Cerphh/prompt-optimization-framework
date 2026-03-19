"""
Firestore persistence module.
Stores benchmark results in Firebase Firestore.
"""

from __future__ import annotations

import json
import os
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
        """Persist a benchmark result document to Firestore."""
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
            )

            doc_ref = (
                self.db.collection(self.collection_name)
                .document(domain)
                .collection(difficulty)
                .document()
            )
            doc_ref.set(payload)

            return {
                "success": True,
                "collection": self.collection_name,
                "domain": domain,
                "difficulty": difficulty,
                "document_id": doc_ref.id,
            }
        except Exception as exc:
            return {
                "success": False,
                "reason": "write_failed",
                "error": str(exc),
            }

    def _build_storage_document(
        self,
        benchmark_result: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build compact Firestore document with only required fields."""
        best_result = benchmark_result.get("best_result", {})
        scores = best_result.get("scores", {})

        domain = self._resolve_domain(benchmark_result=benchmark_result, metadata=metadata)
        difficulty = self._resolve_difficulty(benchmark_result=benchmark_result, metadata=metadata)

        raw_profile = metadata.get("problem_profile")
        if raw_profile is None:
            raw_profile = benchmark_result.get("problem_profile")
        normalized_profile = self._normalize_problem_profile(raw_profile)

        document = {
            "domain": domain,
            "difficulty": difficulty,
            "prompt_used": best_result.get("prompt"),
            "model_response": best_result.get("response"),
            "performance_score": scores.get("overall"),
            "technique_comparison": benchmark_result.get("comparison", []),
        }

        if normalized_profile:
            document["problem_profile"] = normalized_profile

        return document

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

    def _profile_similarity(self, target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Compute weighted profile similarity for IF-THEN pattern matching."""
        weighted_sum = 0.0
        total_weight = 0.0

        target_intent = str(target.get("intent", "") or "")
        candidate_intent = str(candidate.get("intent", "") or "")
        if target_intent and candidate_intent:
            total_weight += 0.45
            weighted_sum += 0.45 if target_intent == candidate_intent else 0.0

        target_features = set(self._normalize_profile_labels(target.get("features")))
        candidate_features = set(self._normalize_profile_labels(candidate.get("features")))
        if target_features and candidate_features:
            feature_score = self._overlap_ratio(target_features, candidate_features)
            total_weight += 0.30
            weighted_sum += 0.30 * feature_score

        target_format = set(self._normalize_profile_labels(target.get("format_labels")))
        candidate_format = set(self._normalize_profile_labels(candidate.get("format_labels")))
        if target_format and candidate_format:
            format_score = self._overlap_ratio(target_format, candidate_format)
            total_weight += 0.20
            weighted_sum += 0.20 * format_score

        target_constraints = set(self._normalize_profile_labels(target.get("constraints")))
        candidate_constraints = set(self._normalize_profile_labels(candidate.get("constraints")))
        if target_constraints and candidate_constraints:
            constraints_score = self._overlap_ratio(target_constraints, candidate_constraints)
            total_weight += 0.05
            weighted_sum += 0.05 * constraints_score

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    @staticmethod
    def _normalize_key(value: Any, default: str) -> str:
        if not value:
            return default
        normalized = str(value).strip().lower().replace(" ", "_")
        return normalized or default

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
            normalized_domain = self._normalize_key(domain, default="general")
            normalized_difficulty = self._normalize_key(difficulty, default="basic")

            query = (
                self.db.collection(self.collection_name)
                .document(normalized_domain)
                .collection(normalized_difficulty)
            )
            for doc in query.stream():
                data = doc.to_dict() or {}
                comparisons = data.get("technique_comparison", [])
                if not isinstance(comparisons, list):
                    continue

                for comparison in comparisons:
                    if not isinstance(comparison, dict):
                        continue

                    technique = comparison.get("technique")
                    overall = comparison.get("overall")

                    if not technique or not isinstance(overall, (int, float)):
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
    ) -> Dict[str, Any]:
        """Get best technique using profile-level IF-THEN matching over historical results."""
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
        if not normalized_profile:
            return {
                "success": False,
                "reason": "missing_problem_profile",
                "error": "Problem profile is required for profile-based selection.",
            }

        try:
            min_similarity = max(0.0, min(1.0, float(min_similarity)))
        except (TypeError, ValueError):
            min_similarity = 0.35

        try:
            totals: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            allowed = set(available_techniques or [])
            normalized_domain = self._normalize_key(domain, default="general")
            normalized_difficulty = self._normalize_key(difficulty, default="basic")

            query = (
                self.db.collection(self.collection_name)
                .document(normalized_domain)
                .collection(normalized_difficulty)
            )

            matched_documents = 0
            similarities: List[float] = []

            for doc in query.stream():
                data = doc.to_dict() or {}
                historical_profile = self._normalize_problem_profile(data.get("problem_profile"))
                if not historical_profile:
                    continue

                similarity = self._profile_similarity(normalized_profile, historical_profile)
                if similarity < min_similarity:
                    continue

                matched_documents += 1
                similarities.append(similarity)

                comparisons = data.get("technique_comparison", [])
                if not isinstance(comparisons, list):
                    continue

                for comparison in comparisons:
                    if not isinstance(comparison, dict):
                        continue

                    technique = comparison.get("technique")
                    overall = comparison.get("overall")

                    if not technique or not isinstance(overall, (int, float)):
                        continue
                    if allowed and technique not in allowed:
                        continue

                    totals[technique] = totals.get(technique, 0.0) + float(overall)
                    counts[technique] = counts.get(technique, 0) + 1

            if not counts:
                return {
                    "success": False,
                    "reason": "no_profile_match",
                    "error": (
                        f"No historical profile match found for domain '{normalized_domain}' "
                        f"and difficulty '{normalized_difficulty}'."
                    ),
                    "problem_profile": normalized_profile,
                    "min_similarity": min_similarity,
                    "matched_documents": matched_documents,
                }

            ranking = []
            for technique, count in counts.items():
                average = totals[technique] / count
                ranking.append(
                    {
                        "technique": technique,
                        "average_overall": round(average, 4),
                        "samples": count,
                    }
                )

            ranking.sort(key=lambda x: (-x["average_overall"], -x["samples"], x["technique"]))
            best = ranking[0]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            return {
                "success": True,
                "domain": normalized_domain,
                "difficulty": normalized_difficulty,
                "best_technique": best["technique"],
                "ranking": ranking,
                "problem_profile": normalized_profile,
                "match_type": "profile_rule",
                "min_similarity": min_similarity,
                "average_similarity": round(avg_similarity, 4),
                "matched_documents": matched_documents,
            }
        except Exception as exc:
            return {
                "success": False,
                "reason": "query_failed",
                "error": str(exc),
            }
