"""
Firestore persistence module.
Stores benchmark results in Firebase Firestore.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

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
            payload = self._build_storage_document(
                benchmark_result=benchmark_result,
                metadata=metadata or {},
            )

            doc_ref = self.db.collection(self.collection_name).document()
            doc_ref.set(payload)

            return {
                "success": True,
                "collection": self.collection_name,
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

        domain = (
            metadata.get("domain")
            or metadata.get("subject")
            or metadata.get("category")
            or benchmark_result.get("domain")
            or "general"
        )

        return {
            "domain": domain,
            "prompt_used": best_result.get("prompt"),
            "model_response": best_result.get("response"),
            "performance_score": scores.get("overall"),
            "technique_comparison": benchmark_result.get("comparison", []),
        }

    def get_best_technique_by_domain(
        self,
        domain: str,
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

            query = self.db.collection(self.collection_name).where("domain", "==", domain)
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
                    "error": f"No historical technique data found for domain '{domain}'.",
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
                "domain": domain,
                "best_technique": best["technique"],
                "ranking": ranking,
            }
        except Exception as exc:
            return {
                "success": False,
                "reason": "query_failed",
                "error": str(exc),
            }
