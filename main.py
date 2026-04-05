"""
FastAPI application for the Prompt Optimization Framework.
Research Benchmarking API for comparative prompt evaluation.
"""

import os
import json
import asyncio
import shutil
import tempfile
import hashlib
import re
import jsonschema
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict, Tuple
from framework.pipeline import BenchmarkPipeline
from framework.dataset import get_sample_dataset
from framework.firestore_store import FirestoreStore

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

app = FastAPI(
    title="Prompt Optimization Research Framework",
    description="API for benchmarking and comparing prompting techniques on LLMs",
    version="1.0.0"
)


RUN_MODE_NORMAL = "normal"
RUN_MODE_BENCHMARK = "benchmark"
SUPPORTED_RUN_MODES = {RUN_MODE_NORMAL, RUN_MODE_BENCHMARK}

SUPPORTED_NORMAL_MODE_DOMAINS = {
    "algebra",
    "pre-calculus",
    "counting-probability",
}

_NORMAL_MODE_DOMAIN_ALIASES = {
    "algebra": "algebra",
    "pre-calculus": "pre-calculus",
    "precalculus": "pre-calculus",
    "precal": "pre-calculus",
    "calculus": "pre-calculus",
    "counting-probability": "counting-probability",
    "counting_probability": "counting-probability",
    "counting-and-probability": "counting-probability",
    "counting_and_probability": "counting-probability",
    "counting&probability": "counting-probability",
    "counting_and_prob": "counting-probability",
    "statistics": "counting-probability",
    "stat": "counting-probability",
}

_MATH_SCOPE_KEYWORDS = {
    "solve",
    "equation",
    "factor",
    "polynomial",
    "simplify",
    "expand",
    "evaluate",
    "derivative",
    "integral",
    "limit",
    "function",
    "sequence",
    "series",
    "probability",
    "permutation",
    "combination",
    "ratio",
    "proportion",
    "mean",
    "median",
    "variance",
    "percent",
    "matrix",
    "vector",
    "trigonometric",
    "log",
}

_OUT_OF_DOMAIN_MATH_KEYWORDS = {
    "geometry",
    "triangle",
    "triangles",
    "rectangle",
    "rectangles",
    "square",
    "squares",
    "circle",
    "circles",
    "polygon",
    "polygons",
    "perimeter",
    "area",
    "volume",
    "surface",
    "hypotenuse",
    "radius",
    "diameter",
    "circumference",
    "chord",
    "tangent",
    "secant",
    "parallel",
    "perpendicular",
    "angles",
    "angle",
}

_DOMAIN_SIGNAL_KEYWORDS = {
    "algebra": {
        "equation",
        "solve",
        "factor",
        "simplify",
        "expand",
        "polynomial",
        "quadratic",
        "linear",
        "system",
        "inequality",
        "substitute",
        "evaluate",
        "expression",
        "variable",
        "root",
        "roots",
        "real",
    },
    "pre-calculus": {
        "trigonometric",
        "sin",
        "cos",
        "tan",
        "sec",
        "csc",
        "cot",
        "arcsin",
        "arccos",
        "arctan",
        "function",
        "domain",
        "range",
        "limit",
        "derivative",
        "integral",
        "sequence",
        "series",
        "log",
        "logarithm",
        "exponential",
        "matrix",
        "vector",
    },
    "counting-probability": {
        "probability",
        "permutation",
        "combination",
        "arrangement",
        "arrangements",
        "choose",
        "select",
        "random",
        "event",
        "outcome",
        "outcomes",
        "odds",
        "chance",
        "expected",
        "variance",
        "mean",
        "median",
        "mode",
        "coin",
        "dice",
        "die",
        "card",
        "bag",
    },
}

# Topics intentionally shared between algebra and pre-calculus.
_ALGEBRA_PRECALC_OVERLAP_KEYWORDS = {
    "function",
    "functions",
    "graph",
    "graphs",
    "sequence",
    "sequences",
    "series",
    "exponent",
    "exponents",
    "log",
    "logarithm",
    "coordinate",
    "slope",
    "line",
    "parabola",
    "trigonometric",
    "sin",
    "cos",
    "tan",
}

_PURE_GEOMETRY_SHAPE_KEYWORDS = {
    "triangle",
    "triangles",
    "rectangle",
    "rectangles",
    "square",
    "squares",
    "circle",
    "circles",
    "polygon",
    "polygons",
    "sphere",
    "cube",
    "cylinder",
    "cone",
    "prism",
}

_PURE_GEOMETRY_MEASURE_KEYWORDS = {
    "area",
    "perimeter",
    "volume",
    "surface",
    "circumference",
    "radius",
    "diameter",
    "hypotenuse",
    "height",
    "width",
    "length",
}

_GEOMETRIC_WORD_PROBLEM_CUES = {
    "ladder",
    "wall",
    "building",
    "shadow",
    "distance",
    "height",
    "angle",
    "angles",
}


def _normalize_domain_label(raw_value: Optional[str]) -> str:
    """Normalize domain labels and map common aliases to canonical domain keys."""
    value = (raw_value or "").strip().lower()
    if not value:
        return ""

    collapsed = value.replace(" ", "-")
    compact = value.replace(" ", "").replace("_", "").replace("-", "")
    compact = compact.replace("&", "and")

    if collapsed in _NORMAL_MODE_DOMAIN_ALIASES:
        return _NORMAL_MODE_DOMAIN_ALIASES[collapsed]
    if value in _NORMAL_MODE_DOMAIN_ALIASES:
        return _NORMAL_MODE_DOMAIN_ALIASES[value]
    if compact in _NORMAL_MODE_DOMAIN_ALIASES:
        return _NORMAL_MODE_DOMAIN_ALIASES[compact]
    return value


def _has_math_scope_signal(problem: str) -> bool:
    """Return True when prompt text shows clear math-task surface signals."""
    value = str(problem or "").strip().lower()
    if not value:
        return False

    # Numeric/operator cues cover direct arithmetic prompts such as "2 + 2".
    if re.search(r"\d", value) and re.search(r"[=+\-*/^%]|\bwhat\s+is\b|\bfind\b|\bcompute\b", value):
        return True

    token_set = set(re.findall(r"[a-zA-Z]+", value))
    return any(keyword in token_set for keyword in _MATH_SCOPE_KEYWORDS)


def _detect_supported_intent(problem: str) -> Optional[str]:
    """Best-effort detect of supported intent labels from prompt_generator."""
    try:
        prompt_generator = pipeline.prompt_generator
        normalized = prompt_generator._normalize_detection_text(problem)
        intent = str(prompt_generator._detect_primary_intent(normalized) or "").strip().lower()
        return intent or None
    except Exception:
        return None


def _is_supported_special_case(problem: str) -> bool:
    """Allow in-scope edge cases that lightweight keyword guard can miss."""
    value = str(problem or "").strip().lower()
    if not value:
        return False

    supported_intents = {
        "solve_equation",
        "real_solutions",
        "evaluate_substitution",
        "rewrite_expression",
        "find_asymptotes",
        "find_maximum",
        "vietas_formulas",
        "expand",
        "simplify",
        "ratio_proportion",
        "variation",
        "system",
        "function_evaluation",
        "slope_formula",
        "angle_conversion",
        "trigonometric_values",
        "trigonometric_identity",
        "trigonometric_equation",
        "harmonic_form",
        "dot_product_orthogonality",
        "law_of_sines",
        "parametric_vector_function",
        "vector_magnitude",
        "matrix_determinant",
        "analytic_geometry",
        "probability",
        "conditional_probability",
        "counting_arrangements",
        "counting_principle",
        "counting_with_restrictions",
        "expected_value",
        "factorial_number_theory",
        "palindrome_number_theory",
    }

    detected_intent = _detect_supported_intent(problem)
    if detected_intent in supported_intents:
        return True

    # Algebraic perfect-square relation prompts.
    if re.search(r"\bperfect\s+squares?\b", value):
        return True

    # Analytic-geometry phrasing used by in-scope set.
    if "circle" in value and any(marker in value for marker in ["diameter", "endpoint", "radius"]):
        return True

    return False


def _is_problem_in_framework_scope(problem: str) -> bool:
    """Return True only when problem text shows clear in-scope math intent."""
    value = str(problem or "").strip()
    if not value:
        return False

    if _has_math_scope_signal(value):
        return True

    try:
        prompt_generator = pipeline.prompt_generator
        normalized = prompt_generator._normalize_detection_text(value)
        intent = prompt_generator._detect_primary_intent(normalized)
        if intent and intent != "general":
            return True

        features = prompt_generator._extract_math_features(normalized)
        if features:
            return True
    except Exception:
        return False

    return False


def _looks_like_out_of_domain_math(problem: str) -> bool:
    """Detect common math topics that are currently outside supported domains."""
    value = str(problem or "").strip().lower()
    if not value:
        return False

    if _is_supported_special_case(problem):
        return False

    # Keep algebraic number-theory prompts in scope, e.g.
    # "Two perfect squares differ by 45" or "find n^2 - m^2".
    if re.search(r"\bperfect\s+squares?\b", value):
        return False
    if re.search(r"\b(?:difference\s+of|differ(?:ence)?\s+by)\b", value) and re.search(r"\bsquares?\b", value):
        return False
    if len(re.findall(r"\b[a-z]\s*(?:\^2|²)\b", value)) >= 2:
        return False

    tokens = set(re.findall(r"[a-zA-Z]+", value))
    keyword_hits = sum(1 for keyword in _OUT_OF_DOMAIN_MATH_KEYWORDS if keyword in tokens)
    if keyword_hits >= 2:
        return True

    # Geometry-like formulas/phrasing without clear supported-domain intent.
    if re.search(r"\b(area|perimeter|circumference|volume)\b", value) and re.search(r"\b(circle|triangle|rectangle|square|sphere|cube|cylinder|cone)\b", value):
        return True

    return False


def _get_domain_signal_scores(problem: str) -> Dict[str, int]:
    """Score lightweight domain evidence using deterministic keyword signals."""
    value = str(problem or "").strip().lower()
    tokens = set(re.findall(r"[a-zA-Z]+", value))

    scores = {domain: 0 for domain in SUPPORTED_NORMAL_MODE_DOMAINS}
    for domain, keywords in _DOMAIN_SIGNAL_KEYWORDS.items():
        scores[domain] += sum(1 for keyword in keywords if keyword in tokens)

    if any(keyword in tokens for keyword in _ALGEBRA_PRECALC_OVERLAP_KEYWORDS):
        scores["algebra"] += 1
        scores["pre-calculus"] += 1

    if re.search(r"\b(sin|cos|tan|sec|csc|cot)\b", value):
        scores["pre-calculus"] += 2
    if re.search(r"\b(p\(|probability|permutation|combination|how\s+many)\b", value):
        scores["counting-probability"] += 2
    if re.search(r"\b(solve\s+for|factor|simplify|quadratic|polynomial)\b", value):
        scores["algebra"] += 2

    # Algebraic perfect-square relationships (difference of squares style).
    if re.search(r"\bperfect\s+squares?\b", value):
        scores["algebra"] += 3
    if re.search(r"\b(?:difference\s+of|differ(?:ence)?\s+by)\b", value) and re.search(r"\bsquares?\b", value):
        scores["algebra"] += 2
    if len(re.findall(r"\b[a-z]\s*(?:\^2|²)\b", value)) >= 2:
        scores["algebra"] += 2

    # Coordinate/analytic geometry phrasing handled under algebra scope.
    if "circle" in value and any(marker in value for marker in ["diameter", "endpoint", "radius"]):
        scores["algebra"] += 2
    if re.search(r"\(\s*[-+]?\d+\s*,\s*[-+]?\d+\s*\)", value):
        scores["algebra"] += 1

    # Counting & probability number-theory style prompts.
    if "!" in value or "factorial" in value:
        scores["counting-probability"] += 2
    if any(marker in value for marker in ["tens digit", "ones digit", "units digit", "palindrome"]):
        scores["counting-probability"] += 2

    return scores


def _infer_domain_from_signals(problem: str) -> Tuple[Optional[str], int]:
    """Infer a likely domain and confidence gap from deterministic signal scores."""
    scores = _get_domain_signal_scores(problem)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_domain, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    gap = top_score - second_score

    if top_score <= 0:
        return None, 0
    if top_score >= 2 and gap >= 1:
        return top_domain, gap
    return None, gap


def _looks_like_pure_geometry_problem(problem: str) -> bool:
    """Detect plane/solid geometry style prompts outside the supported 3-domain boundary."""
    value = str(problem or "").strip().lower()
    if not value:
        return False

    # Allow supported analytic/trigonometric geometry subtypes to pass.
    if _is_supported_special_case(problem):
        return False

    tokens = set(re.findall(r"[a-zA-Z]+", value))

    has_shape = any(keyword in tokens for keyword in _PURE_GEOMETRY_SHAPE_KEYWORDS)
    has_measure = any(keyword in tokens for keyword in _PURE_GEOMETRY_MEASURE_KEYWORDS)
    if has_shape and has_measure:
        return True

    # Block common real-world right-triangle geometry word problems.
    if "ladder" in tokens and any(keyword in tokens for keyword in {"wall", "ground", "angle", "height"}):
        return True

    cue_hits = sum(1 for cue in _GEOMETRIC_WORD_PROBLEM_CUES if cue in tokens)
    return cue_hits >= 3 and "probability" not in tokens


def _resolve_domain_for_normal_mode(problem: str, requested_subject: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Resolve effective domain for normal mode and detect unsupported-domain requests."""
    requested_raw = (requested_subject or "").strip()
    normalized_requested = _normalize_domain_label(requested_raw)

    details: Dict[str, Any] = {
        "requested_subject": requested_raw or None,
        "resolved_domain": None,
        "detection_source": None,
        "detected_subject": None,
        "signal_inferred_subject": None,
        "signal_confidence_gap": 0,
        "math_scope_signal": False,
    }

    # If the caller explicitly requested a non-general unsupported subject,
    # fail fast with a clear scope message.
    if requested_raw and normalized_requested not in SUPPORTED_NORMAL_MODE_DOMAINS and normalized_requested != "general":
        details["detection_source"] = "explicit_subject_unsupported"
        return None, details

    in_scope = _is_problem_in_framework_scope(problem)
    has_math_signal = _has_math_scope_signal(problem)
    details["math_scope_signal"] = has_math_signal
    details["in_framework_scope"] = in_scope

    if not in_scope:
        details["detection_source"] = "out_of_scope_problem"
        return None, details

    detected_subject = _normalize_domain_label(pipeline.prompt_generator.classify_subject(problem))
    signal_subject, signal_gap = _infer_domain_from_signals(problem)
    details["detected_subject"] = detected_subject or None
    details["signal_inferred_subject"] = signal_subject
    details["signal_confidence_gap"] = signal_gap

    if _looks_like_pure_geometry_problem(problem):
        details["detection_source"] = "pure_geometry_out_of_scope"
        return None, details

    if normalized_requested in SUPPORTED_NORMAL_MODE_DOMAINS:
        if signal_subject and signal_subject != normalized_requested:
            details["detection_source"] = "subject_mismatch_signal"
            return None, details

        if detected_subject in SUPPORTED_NORMAL_MODE_DOMAINS and detected_subject != normalized_requested:
            details["detection_source"] = "subject_mismatch"
            return None, details

        if _looks_like_out_of_domain_math(problem):
            details["detection_source"] = "out_of_domain_math"
            return None, details

        # Fail closed when subject is explicitly selected but both detectors are uncertain.
        if detected_subject == "general" and not signal_subject:
            details["detection_source"] = "ambiguous_requested_subject"
            return None, details

        details["resolved_domain"] = normalized_requested
        details["detection_source"] = "request_subject"
        return normalized_requested, details

    if signal_subject in SUPPORTED_NORMAL_MODE_DOMAINS:
        details["resolved_domain"] = signal_subject
        details["detection_source"] = "auto_detected_signal"
        return signal_subject, details

    if detected_subject in SUPPORTED_NORMAL_MODE_DOMAINS:
        details["resolved_domain"] = detected_subject
        details["detection_source"] = "auto_detected"
        return detected_subject, details

    details["detection_source"] = "unsupported_detected"
    return None, details


def _get_cors_origins() -> List[str]:
    """Resolve allowed CORS origins from env or sensible local defaults."""
    configured = os.getenv("CORS_ALLOW_ORIGINS", "")
    if configured.strip():
        origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
        if origins:
            return origins

    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


def _get_env_int(name: str, default: int, min_value: Optional[int] = None) -> int:
    """Parse integer environment variable with fallback and optional lower bound."""
    raw_value = os.getenv(name)
    try:
        value = int(raw_value) if raw_value is not None else default
    except (TypeError, ValueError):
        value = default

    if min_value is not None and value < min_value:
        return min_value
    return value


def _get_env_bool(name: str, default: bool) -> bool:
    """Parse boolean environment variables with fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_env_float(
    name: str,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Parse float environment variable with fallback and optional bounds."""
    raw_value = os.getenv(name)
    try:
        value = float(raw_value) if raw_value is not None else default
    except (TypeError, ValueError):
        value = default

    if min_value is not None and value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


def _resolve_speed_profile(raw_value: Optional[str]) -> str:
    """Normalize speed profile to a supported value."""
    normalized = (raw_value or "balanced").strip().lower()
    if normalized in {"balanced", "fast"}:
        return normalized
    return "balanced"


def _resolve_run_mode(raw_value: Optional[str]) -> str:
    """Normalize run mode to supported values."""
    normalized = (raw_value or RUN_MODE_NORMAL).strip().lower()
    if normalized in SUPPORTED_RUN_MODES:
        return normalized
    return RUN_MODE_NORMAL


def _normalize_ground_truth(raw_value: Optional[str]) -> Optional[str]:
    """Normalize ground truth string and return None when empty."""
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    return normalized or None


def _build_fast_pipeline_from_defaults() -> BenchmarkPipeline:
    """Create a per-request fast pipeline without mutating shared global state."""
    request_pipeline = BenchmarkPipeline(
        model_name=pipeline.model_runner.model_name,
        base_url=pipeline.model_runner.base_url,
        accuracy_weight=pipeline.weights.get("accuracy", 1.0),
        consistency_weight=pipeline.weights.get("consistency", 1.0),
        efficiency_weight=pipeline.weights.get("efficiency", 1.0),
        runs_per_technique=pipeline.default_runs_per_technique,
    )

    runner = request_pipeline.model_runner
    runner.verifier_retry_enabled = False
    runner.verifier_retry_attempts = 0
    runner.max_continue_rounds = _get_env_int("FAST_MODEL_MAX_CONTINUE_ROUNDS", default=1, min_value=0)

    fast_num_predict = _get_env_int("FAST_MODEL_NUM_PREDICT", default=768, min_value=64)
    fast_num_ctx = _get_env_int("FAST_MODEL_NUM_CTX", default=4096, min_value=512)

    runner.generation_options["num_predict"] = min(
        int(runner.generation_options.get("num_predict", fast_num_predict)),
        fast_num_predict,
    )
    runner.generation_options["num_ctx"] = min(
        int(runner.generation_options.get("num_ctx", fast_num_ctx)),
        fast_num_ctx,
    )

    return request_pipeline

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = BenchmarkPipeline(
    model_name=os.getenv("MODEL_NAME", "llama3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    accuracy_weight=1.0,
    consistency_weight=1.0,
    efficiency_weight=1.0,
    runs_per_technique=3,
)

# Initialize dataset
dataset = get_sample_dataset()

# Initialize Firestore persistence
firestore_store = FirestoreStore(
    collection_name=os.getenv("FIRESTORE_COLLECTION", "benchmark_results"),
    enabled=os.getenv("ENABLE_FIRESTORE", "true").lower() == "true",
    required=os.getenv("FIRESTORE_REQUIRED", "false").lower() == "true"
)

model_startup_check: Dict[str, Any] = {
    "checked_at": None,
    "ready": None,
    "message": None,
    "active_model": pipeline.model_runner.model_name,
}


def _refresh_model_startup_check() -> Dict[str, Any]:
    ready, detail = pipeline.model_runner.validate_model_ready()
    checked_at = datetime.now(timezone.utc).isoformat()

    model_startup_check.update(
        {
            "checked_at": checked_at,
            "ready": ready,
            "message": detail,
            "active_model": pipeline.model_runner.model_name,
        }
    )
    return dict(model_startup_check)


@app.on_event("startup")
async def startup_model_readiness_check():
    info = _refresh_model_startup_check()
    if info.get("ready"):
        print(f"[startup] model ready: {info.get('active_model')}")
    else:
        print(
            "[startup] model not ready: "
            f"{info.get('active_model')} - {info.get('message')}"
        )


def _normalize_key(value: Any, default: str = "") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower().replace(" ", "_")
    return normalized or default


def _build_problem_profile(problem: str, domain: str, difficulty: str) -> Dict[str, Any]:
    """Build normalized problem profile for rule-mined IF-THEN selection."""
    profile: Dict[str, Any] = {
        "subject": _normalize_key(domain, default="general"),
        "difficulty": _normalize_key(difficulty, default="basic"),
    }

    text = str(problem or "").strip()
    if not text:
        return profile

    prompt_generator = pipeline.prompt_generator
    try:
        normalized_text = prompt_generator._normalize_problem_text(text)
        lowered = normalized_text.lower()

        intent = prompt_generator._normalize_type_label(prompt_generator._detect_primary_intent(lowered))
        if intent:
            profile["intent"] = intent

        features = sorted(prompt_generator._extract_math_features(lowered))
        if features:
            profile["features"] = features

        format_labels = sorted(prompt_generator._extract_problem_format_labels(lowered))
        if format_labels:
            profile["format_labels"] = format_labels

        constraints = sorted(prompt_generator._extract_constraints_from_text(lowered))
        if constraints:
            profile["constraints"] = constraints
    except Exception:
        # Keep profile generation best-effort so benchmarking never fails on metadata extraction.
        pass

    return profile


def _attach_problem_profile(result: Dict[str, Any], domain: str, difficulty: str) -> Dict[str, Any]:
    """Attach profile metadata used by historical pattern-rule mining."""
    if not isinstance(result, dict):
        return result

    problem = str(result.get("problem", "") or "")
    if not problem:
        return result

    result["problem_profile"] = _build_problem_profile(problem=problem, domain=domain, difficulty=difficulty)
    return result


def _evaluate_selection_confidence(
    selection: Dict[str, Any],
    min_samples: int,
    min_gap: float,
    top_only_samples: bool = False,
) -> Tuple[bool, str]:
    """Apply confidence gates to historical selection ranking.

    Args:
        top_only_samples: When True, only the top technique needs min_samples
            (the second technique is not required to meet the threshold).
            Useful for profile-based selection where data is scarcer.
    """
    if not isinstance(selection, dict):
        return False, "invalid_selection_payload"

    if not selection.get("success", False):
        return False, str(selection.get("reason") or "selection_failed")

    ranking = selection.get("ranking", [])
    if not isinstance(ranking, list) or not ranking:
        return False, "no_ranking"

    def _extract_samples(row: Dict[str, Any]) -> int:
        raw_samples = row.get("samples")
        if raw_samples is None:
            raw_samples = row.get("unweighted_samples")
        if raw_samples is None:
            raw_samples = row.get("effective_samples")
        try:
            return max(0, int(float(raw_samples or 0)))
        except (TypeError, ValueError):
            return 0

    top = ranking[0] if isinstance(ranking[0], dict) else {}
    try:
        top_samples = _extract_samples(top)
        top_average = float(top.get("average_overall", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False, "invalid_ranking_data"

    if top_samples < min_samples:
        return False, "insufficient_samples"

    if len(ranking) < 2:
        return True, "ok_single_technique"

    second = ranking[1] if isinstance(ranking[1], dict) else {}
    try:
        second_samples = _extract_samples(second)
        second_average = float(second.get("average_overall", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False, "invalid_ranking_data"

    if not top_only_samples and min(top_samples, second_samples) < min_samples:
        return False, "insufficient_samples"

    # ── Adaptive thresholds based on data volume ──
    # With more samples, averages are more stable, so:
    #   1) Smaller gaps become statistically meaningful → lower effective_min_gap
    #   2) Consistent equality is a stronger signal → wider indistinguishable zone
    data_samples = max(top_samples, second_samples)
    if data_samples >= 10:
        effective_min_gap = min_gap * 0.5
        indistinguishable_threshold = 0.008
    elif data_samples >= 5:
        effective_min_gap = min_gap * 0.75
        indistinguishable_threshold = 0.004
    else:
        effective_min_gap = min_gap
        indistinguishable_threshold = 0.003

    actual_gap = top_average - second_average

    if actual_gap < effective_min_gap:
        # When both techniques score virtually identically, they are
        # indistinguishable — either choice is equally valid.
        # With more data the threshold widens: consistent equality
        # across many samples is a strong signal both are truly equal.
        if actual_gap < indistinguishable_threshold and top_samples >= min_samples:
            top_confidence = str(top.get("confidence", "") or "").lower()
            if top_confidence != "low":
                return True, "ok_indistinguishable"
        return False, "low_confidence_gap"

    # Reject if top technique has erratic/inconsistent scores (std_dev ≥ 0.15).
    top_confidence = str(top.get("confidence", "") or "").lower()
    if top_confidence == "low":
        return False, "low_technique_confidence"

    return True, "ok"


def _resolve_pre_execution_techniques(
    *,
    request_pipeline: BenchmarkPipeline,
    run_mode: str,
    problem: str,
    domain: str,
    difficulty: str,
) -> Tuple[Optional[List[str]], Dict[str, Any]]:
    """Resolve whether to run all techniques or preselect from historical data."""
    technique_names = request_pipeline.prompt_generator.get_technique_names()
    details: Dict[str, Any] = {
        "run_mode": run_mode,
        "enabled": False,
        "reason": "disabled",
        "selected_techniques": technique_names,
        "best_technique": None,
        "skipped_technique": None,
        "history_source": None,
        "profile_selection": None,
        "domain_selection": None,
        "min_samples_per_technique": None,
        "min_average_gap": None,
        "require_ground_truth_history": None,
        "profile_min_similarity": None,
    }

    if len(technique_names) < 2:
        details["reason"] = "single_technique"
        return None, details

    # Benchmark mode always runs all techniques (runtime selection only).
    if run_mode == RUN_MODE_BENCHMARK:
        details["reason"] = "benchmark_mode_runs_all"
        return None, details

    enabled = _get_env_bool("NORMAL_MODE_HISTORY_PRESELECTION_ENABLED", True)
    details["enabled"] = enabled
    if not enabled:
        return None, details

    min_samples = _get_env_int(
        "NORMAL_MODE_MIN_SAMPLES_PER_TECHNIQUE",
        default=3,
        min_value=1,
    )
    min_gap = _get_env_float(
        "NORMAL_MODE_MIN_AVG_SCORE_GAP",
        default=0.03,
        min_value=0.0,
    )
    profile_min_similarity = _get_env_float(
        "DB_PROFILE_MIN_SIMILARITY",
        default=0.35,
        min_value=0.0,
        max_value=1.0,
    )
    require_ground_truth_history = _get_env_bool("DB_HISTORY_REQUIRE_GROUND_TRUTH", True)

    details["min_samples_per_technique"] = min_samples
    details["min_average_gap"] = min_gap
    details["require_ground_truth_history"] = require_ground_truth_history
    details["profile_min_similarity"] = profile_min_similarity
    details["db_confidence_rules"] = {
        "profile_min_samples_per_technique": min_samples,
        "profile_base_min_gap": min_gap + 0.01,
        "min_samples_per_technique": min_samples,
        "min_average_gap": min_gap,
        "profile_min_similarity": profile_min_similarity,
    }

    problem_profile = _build_problem_profile(problem=problem, domain=domain, difficulty=difficulty)

    profile_selection = firestore_store.get_best_technique_by_profile(
        domain=domain,
        difficulty=difficulty,
        problem_profile=problem_profile,
        available_techniques=technique_names,
        min_similarity=profile_min_similarity,
        require_ground_truth=require_ground_truth_history,
        problem_text=problem,
    )
    details["profile_selection"] = profile_selection

    # Adaptive profile thresholds: Tier 1 matched specific similar problems,
    # so its data is more trustworthy than Tier 2's broad domain averages.
    # Base gap is LOWER than Tier 2 (more trust in specific data).
    # With more matches, averages are more stable → gap decreases further.
    # min_samples still scales UP with more data to require broader evidence.
    _matched = int(profile_selection.get("matched_documents", 0) or 0)
    tier1_base_gap = max(0.005, min_gap * 0.5)  # half of Tier 2's gap
    if _matched >= 20:
        adaptive_profile_min = min_samples + 3
        adaptive_profile_gap = tier1_base_gap * 0.3   # ~0.005 (very confident)
    elif _matched >= 10:
        adaptive_profile_min = min_samples + 2
        adaptive_profile_gap = tier1_base_gap * 0.5   # ~0.008
    elif _matched >= 5:
        adaptive_profile_min = min_samples + 1
        adaptive_profile_gap = tier1_base_gap * 0.75  # ~0.011
    else:
        adaptive_profile_min = min_samples
        adaptive_profile_gap = tier1_base_gap          # ~0.015

    profile_ok, profile_reason = _evaluate_selection_confidence(
        profile_selection,
        min_samples=adaptive_profile_min,
        min_gap=adaptive_profile_gap,
        top_only_samples=True,
    )

    # Relaxed retry for Tier 1: mirror the Tier 2 / post-execution fallback.
    if not profile_ok and profile_reason == "low_confidence_gap":
        relaxed_profile_gap = max(0.005, adaptive_profile_gap / 4)
        profile_ok, profile_reason = _evaluate_selection_confidence(
            profile_selection,
            min_samples=adaptive_profile_min,
            min_gap=relaxed_profile_gap,
            top_only_samples=True,
        )
        if profile_ok:
            profile_reason = "ok_relaxed_gap"

    # Dead-zone fallback for Tier 1.
    if not profile_ok and profile_reason == "low_confidence_gap":
        p_ranking = profile_selection.get("ranking", [])
        if isinstance(p_ranking, list) and len(p_ranking) >= 1:
            p_top = p_ranking[0] if isinstance(p_ranking[0], dict) else {}
            p_top_samples_raw = p_top.get("samples") or p_top.get("unweighted_samples") or 0
            try:
                p_top_samples_val = int(float(p_top_samples_raw))
            except (TypeError, ValueError):
                p_top_samples_val = 0
            if p_top_samples_val >= adaptive_profile_min:
                profile_ok = True
                profile_reason = "ok_indistinguishable"

    details["profile_reason"] = profile_reason
    if profile_ok:
        best_technique = str(profile_selection.get("best_technique") or "")
        if best_technique in technique_names:
            selected_techniques = [best_technique]
            skipped_candidates = [name for name in technique_names if name != best_technique]
            details["selected_techniques"] = selected_techniques
            details["best_technique"] = best_technique
            details["history_source"] = "db_profile_rules"
            details["skipped_technique"] = skipped_candidates[0] if len(skipped_candidates) == 1 else None
            details["reason"] = "preselected_by_profile_history"
            return selected_techniques, details

    selection = firestore_store.get_best_technique_by_domain(
        domain=domain,
        difficulty=difficulty,
        available_techniques=technique_names,
        require_ground_truth=require_ground_truth_history,
    )
    details["domain_selection"] = selection

    if not isinstance(selection, dict):
        details["reason"] = "invalid_selection_payload"
        return None, details

    can_preselect, reason = _evaluate_selection_confidence(
        selection,
        min_samples=min_samples,
        min_gap=min_gap,
    )

    # Relaxed retry: if strict gap check fails, try with a smaller gap.
    # Pre-execution Tier 2 mirrors the post-execution fallback logic.
    if not can_preselect and reason == "low_confidence_gap":
        relaxed_gap = max(0.005, min_gap / 4)
        can_preselect, reason = _evaluate_selection_confidence(
            selection,
            min_samples=min_samples,
            min_gap=relaxed_gap,
        )
        if can_preselect:
            reason = "ok_relaxed_gap"

    # Dead-zone fallback: gap sits between indistinguishable threshold
    # and relaxed_gap.  Accept top technique if it has enough samples.
    if not can_preselect and reason == "low_confidence_gap":
        ranking = selection.get("ranking", [])
        if isinstance(ranking, list) and len(ranking) >= 1:
            top_row = ranking[0] if isinstance(ranking[0], dict) else {}
            top_samples_raw = top_row.get("samples") or top_row.get("unweighted_samples") or 0
            try:
                top_samples_val = int(float(top_samples_raw))
            except (TypeError, ValueError):
                top_samples_val = 0
            if top_samples_val >= min_samples:
                can_preselect = True
                reason = "ok_indistinguishable"

    details["reason"] = reason
    if not can_preselect:
        return None, details

    best_technique = str(selection.get("best_technique") or "")
    if not best_technique or best_technique not in technique_names:
        details["reason"] = "invalid_best_technique"
        return None, details

    selected_techniques = [best_technique]
    skipped_candidates = [name for name in technique_names if name != best_technique]
    details["selected_techniques"] = selected_techniques
    details["best_technique"] = best_technique
    details["history_source"] = "db_history"
    details["skipped_technique"] = skipped_candidates[0] if len(skipped_candidates) == 1 else None
    details["reason"] = "preselected_by_domain_history"
    return selected_techniques, details


def _apply_db_based_selection(
    result: Dict[str, Any],
    domain: str,
    difficulty: str,
    *,
    min_samples_override: Optional[int] = None,
    min_gap_override: Optional[float] = None,
    profile_min_samples_override: Optional[int] = None,
    profile_min_gap_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Apply profile-rule selection first, then domain-average fallback, else runtime greedy."""
    result = _attach_problem_profile(result=result, domain=domain, difficulty=difficulty)

    def _limit_comparison_to_technique(payload: Dict[str, Any], technique: str) -> None:
        comparison = payload.get("comparison")
        if not isinstance(comparison, list):
            return
        payload["comparison"] = [
            row for row in comparison
            if isinstance(row, dict) and row.get("technique") == technique
        ]

    successful_techniques = [
        technique
        for technique, technique_result in result.get("all_results", {}).items()
        if technique_result.get("success", False)
    ]

    min_samples = (
        _get_env_int("DB_MIN_SAMPLES_PER_TECHNIQUE", default=3, min_value=1)
        if min_samples_override is None
        else max(1, int(min_samples_override))
    )
    min_gap = (
        _get_env_float("DB_MIN_AVG_SCORE_GAP", default=0.05, min_value=0.0)
        if min_gap_override is None
        else max(0.0, float(min_gap_override))
    )
    profile_min_samples = (
        _get_env_int(
            "DB_PROFILE_MIN_SAMPLES_PER_TECHNIQUE",
            default=3,
            min_value=1,
        )
        if profile_min_samples_override is None
        else max(1, int(profile_min_samples_override))
    )
    profile_min_gap = (
        _get_env_float(
            "DB_PROFILE_MIN_AVG_SCORE_GAP",
            default=0.005,
            min_value=0.0,
        )
        if profile_min_gap_override is None
        else max(0.0, float(profile_min_gap_override))
    )
    profile_min_similarity = _get_env_float(
        "DB_PROFILE_MIN_SIMILARITY",
        default=0.25,
        min_value=0.0,
        max_value=1.0,
    )
    require_ground_truth_history = _get_env_bool("DB_HISTORY_REQUIRE_GROUND_TRUTH", True)
    exploration_rate = _get_env_float(
        "DB_EXPLORATION_RATE",
        default=0.0,
        min_value=0.0,
        max_value=1.0,
    )

    problem_profile = result.get("problem_profile") if isinstance(result.get("problem_profile"), dict) else None
    problem_text = str(result.get("problem", "") or "")

    profile_selection = firestore_store.get_best_technique_by_profile(
        domain=domain,
        difficulty=difficulty,
        problem_profile=problem_profile,
        available_techniques=successful_techniques,
        min_similarity=profile_min_similarity,
        require_ground_truth=require_ground_truth_history,
        problem_text=problem_text,
    )

    if not isinstance(profile_selection, dict):
        profile_selection = {
            "success": False,
            "reason": "invalid_selection_payload",
            "error": "Profile selection payload must be a dictionary.",
        }

    # Adaptive profile min_samples: require more evidence when more similar
    # problems are available in the DB.
    _matched = int(profile_selection.get("matched_documents", 0) or 0)
    if _matched >= 20:
        adaptive_profile_min = profile_min_samples + 3
    elif _matched >= 10:
        adaptive_profile_min = profile_min_samples + 2
    elif _matched >= 5:
        adaptive_profile_min = profile_min_samples + 1
    else:
        adaptive_profile_min = profile_min_samples

    can_use_profile, profile_decision_reason = _evaluate_selection_confidence(
        profile_selection,
        min_samples=adaptive_profile_min,
        min_gap=profile_min_gap,
        top_only_samples=True,
    )

    if can_use_profile and exploration_rate > 0:
        profile_intent = ""
        if isinstance(problem_profile, dict):
            profile_intent = str(problem_profile.get("intent", "") or "")
        exploration_key = f"profile|{domain}|{difficulty}|{profile_intent}|{result.get('problem', '')}"
        bucket = int(hashlib.sha256(exploration_key.encode("utf-8")).hexdigest()[:8], 16)
        ratio = bucket / 0xFFFFFFFF
        if ratio < exploration_rate:
            can_use_profile = False
            profile_decision_reason = "exploration_runtime"

    if can_use_profile:
        profile_best = profile_selection.get("best_technique")
        if profile_best in result.get("all_results", {}):
            result["best_technique"] = profile_best
            result["best_result"] = result["all_results"][profile_best]
            _limit_comparison_to_technique(result, str(profile_best))
            result["selection_source"] = "db_profile_rules"
            result["selection_details"] = {
                "profile_selection": profile_selection,
                "domain_selection": None,
                "profile_decision_reason": profile_decision_reason,
                "db_confidence_rules": {
                    "profile_min_samples_per_technique": profile_min_samples,
                    "adaptive_profile_min_samples": adaptive_profile_min,
                    "profile_matched_documents": _matched,
                    "profile_min_average_gap": profile_min_gap,
                    "profile_min_similarity": profile_min_similarity,
                    "min_samples_per_technique": min_samples,
                    "min_average_gap": min_gap,
                    "exploration_rate": exploration_rate,
                },
            }
            return result

        can_use_profile = False
        profile_decision_reason = "profile_best_missing_result"

    selection = firestore_store.get_best_technique_by_domain(
        domain=domain,
        difficulty=difficulty,
        available_techniques=successful_techniques,
        require_ground_truth=require_ground_truth_history,
    )

    if not isinstance(selection, dict):
        selection = {
            "success": False,
            "reason": "invalid_selection_payload",
            "error": "Selection payload must be a dictionary.",
        }

    can_use_db, db_decision_reason = _evaluate_selection_confidence(
        selection,
        min_samples=min_samples,
        min_gap=min_gap,
        top_only_samples=True,
    )

    # When the strict gap check fails but the domain has enough samples,
    # retry with a relaxed gap.  Tier 2 is already a fallback after Tier 1
    # failed, so a smaller gap is acceptable to avoid falling to Tier 3
    # unnecessarily (especially with small datasets).
    if not can_use_db and db_decision_reason == "low_confidence_gap":
        relaxed_gap = max(0.005, min_gap / 4)
        can_use_db, db_decision_reason = _evaluate_selection_confidence(
            selection,
            min_samples=min_samples,
            min_gap=relaxed_gap,
            top_only_samples=True,
        )
        if can_use_db:
            db_decision_reason = "ok_relaxed_gap"

    # Final Tier 2 fallback: if both strict and relaxed gap checks failed,
    # the gap sits in a "dead zone" between the indistinguishable threshold
    # and the relaxed min_gap.  Since Tier 2 is already a fallback (Tier 1
    # failed) and both techniques ran, accept the top technique as long as
    # it has enough samples — a small gap in domain-level data shouldn't
    # force an unnecessary Tier 3 runtime selection.
    if not can_use_db and db_decision_reason == "low_confidence_gap":
        ranking = selection.get("ranking", [])
        if isinstance(ranking, list) and len(ranking) >= 1:
            top_row = ranking[0] if isinstance(ranking[0], dict) else {}
            top_samples_t2 = top_row.get("samples", 0)
            if isinstance(top_samples_t2, (int, float)) and int(top_samples_t2) >= min_samples:
                can_use_db = True
                db_decision_reason = "ok_indistinguishable"

    if can_use_db and exploration_rate > 0:
        exploration_key = f"{domain}|{result.get('problem', '')}"
        bucket = int(hashlib.sha256(exploration_key.encode("utf-8")).hexdigest()[:8], 16)
        ratio = bucket / 0xFFFFFFFF
        if ratio < exploration_rate:
            can_use_db = False
            db_decision_reason = "exploration_runtime"

    if can_use_db:
        best_technique = selection["best_technique"]
        if best_technique in result.get("all_results", {}):
            result["best_technique"] = best_technique
            result["best_result"] = result["all_results"][best_technique]
            _limit_comparison_to_technique(result, str(best_technique))
            result["selection_source"] = "db_history"
            result["selection_details"] = {
                "profile_selection": profile_selection,
                "domain_selection": selection,
                "profile_decision_reason": profile_decision_reason,
                "db_decision_reason": db_decision_reason,
                "db_confidence_rules": {
                    "profile_min_samples_per_technique": profile_min_samples,
                    "adaptive_profile_min_samples": adaptive_profile_min,
                    "profile_matched_documents": _matched,
                    "profile_min_average_gap": profile_min_gap,
                    "profile_min_similarity": profile_min_similarity,
                    "min_samples_per_technique": min_samples,
                    "min_average_gap": min_gap,
                    "require_ground_truth_history": require_ground_truth_history,
                    "exploration_rate": exploration_rate,
                },
            }
            return result

    result["selection_source"] = "runtime_scores"
    result["selection_details"] = {
        "profile_selection": profile_selection,
        "domain_selection": selection,
        "profile_decision_reason": profile_decision_reason,
        "db_decision_reason": db_decision_reason,
        "db_confidence_rules": {
            "profile_min_samples_per_technique": profile_min_samples,
            "adaptive_profile_min_samples": adaptive_profile_min,
            "profile_matched_documents": _matched,
            "profile_min_average_gap": profile_min_gap,
            "profile_min_similarity": profile_min_similarity,
            "min_samples_per_technique": min_samples,
            "min_average_gap": min_gap,
            "require_ground_truth_history": require_ground_truth_history,
            "exploration_rate": exploration_rate,
        },
    }
    return result


def _finalize_benchmark_result(
    result: Dict[str, Any],
    *,
    domain: str,
    difficulty: str,
    run_mode: str,
) -> Dict[str, Any]:
    """Apply selection strategy and validate winner without persisting."""
    result = _attach_problem_profile(result=result, domain=domain, difficulty=difficulty)

    if run_mode == RUN_MODE_NORMAL:
        # Normal mode: the real Tier 1/2 decision already happened in
        # _resolve_pre_execution_techniques (pre-execution).  If a tier
        # was confident, only one technique ran and its result is the
        # winner.  If both tiers failed, both techniques ran — use
        # runtime scores (Tier 3) to pick the best.  No redundant
        # post-execution DB re-query needed.
        result["selection_source"] = "runtime_scores"
        result["selection_details"] = {
            "profile_selection": None,
            "domain_selection": None,
            "profile_decision_reason": "normal_mode_decided_pre_execution",
            "db_decision_reason": "normal_mode_decided_pre_execution",
        }
    else:
        # Benchmark mode: pure runtime greedy (Tier 3 only).
        # No DB-based override — all techniques' comparison data is preserved
        # so unbiased data accumulates for normal mode's Tier 1/2 queries.
        result["selection_source"] = "runtime_scores"
        result["selection_details"] = {
            "profile_selection": None,
            "domain_selection": None,
            "profile_decision_reason": "benchmark_mode_runtime_only",
            "db_decision_reason": "benchmark_mode_runtime_only",
        }

    all_results = result.get("all_results", {}) if isinstance(result, dict) else {}
    attempted_techniques = sorted(all_results.keys()) if isinstance(all_results, dict) else []
    successful_count = 0
    if isinstance(all_results, dict):
        successful_count = sum(
            1 for payload in all_results.values()
            if isinstance(payload, dict) and payload.get("success", False)
        )

    result["run_mode"] = run_mode
    result["model_name"] = pipeline.model_runner.model_name
    result["execution_summary"] = {
        "attempted_techniques": attempted_techniques,
        "attempted_count": len(attempted_techniques),
        "successful_count": successful_count,
    }

    best_result = result.get("best_result", {})
    if not best_result.get("success", False):
        raise HTTPException(
            status_code=500,
            detail=best_result.get("error", "Benchmark failed"),
        )

    return result


class BenchmarkRequest(BaseModel):
    """Request model for benchmarking."""
    problem: str
    ground_truth: Optional[str] = None
    subject: Optional[str] = "algebra"  # algebra, counting-probability, or pre-calculus
    difficulty: Optional[str] = "basic"  # basic, intermediate, advanced
    run_mode: Optional[str] = RUN_MODE_NORMAL  # normal or benchmark
    speed_profile: Optional[str] = "balanced"  # balanced, fast
    runs_per_technique: Optional[int] = Field(default=None, ge=1)


class WeightsUpdate(BaseModel):
    """Model for updating metric weights."""
    accuracy: Optional[float] = None
    consistency: Optional[float] = None
    efficiency: Optional[float] = None


class ProblemAdd(BaseModel):
    """Model for adding problems to dataset."""
    problem: str
    answer: str
    category: str = "general"


class SaveResultRequest(BaseModel):
    """Model for manually saving benchmark results to Firestore."""
    result: Dict[str, Any]
    source: Optional[str] = "manual_save"
    metadata: Optional[Dict[str, Any]] = None


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "message": "Prompt Optimization Research Framework",
        "version": "1.0.0",
        "description": "Comparative benchmarking of prompting techniques",
        "docs": "/docs"
    }


@app.get("/example-types", tags=["Info"])
async def example_types():
    """Return the available few-shot example types grouped by subject and difficulty."""
    raw = pipeline.prompt_generator.example_dataset
    # raw is already normalized: subject -> flat list of examples
    # Rebuild subject -> difficulty -> types from example metadata.
    import json as _json, os as _os
    json_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "framework", "example_problems.json")
    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            source = _json.load(f)
    except Exception:
        source = {}

    # Merge user examples (non-expired) into the source for concept listing
    user_data = _load_user_examples()
    user_data, _ = _purge_expired_entries(user_data)
    for subj, diffs in user_data.items():
        if subj not in source:
            source[subj] = {}
        if isinstance(diffs, dict):
            for diff, examples in diffs.items():
                if diff not in source[subj]:
                    source[subj][diff] = []
                if isinstance(examples, list):
                    source[subj][diff].extend(examples)

    result = {}
    for subject, subject_val in source.items():
        subject_out: dict = {}
        if isinstance(subject_val, dict):
            for diff, examples in subject_val.items():
                if not isinstance(examples, list):
                    continue
                concepts: dict = {}
                for ex in examples:
                    c = ex.get("concept", ex.get("type", "general"))
                    if c not in concepts:
                        concepts[c] = {"concept": c, "count": 0, "sample": ex.get("problem", "")}
                    concepts[c]["count"] += 1
                subject_out[diff] = list(concepts.values())
        result[subject] = subject_out
    return result


@app.get("/health", tags=["Health"])
async def health_check():
    """Check if the API and model are operational."""
    model_connected = pipeline.test_connection()
    startup_info = dict(model_startup_check)
    if startup_info.get("checked_at") is None:
        startup_info = _refresh_model_startup_check()

    return {
        "status": "healthy" if model_connected else "degraded",
        "model_connected": model_connected,
        "model_readiness": startup_info,
        "dataset_size": dataset.size(),
        "weights": pipeline.weights,
        "firestore": firestore_store.get_status()
    }


@app.post("/benchmark", tags=["Benchmarking"])
async def run_benchmark(request: BenchmarkRequest):
    """
    Run comprehensive benchmark on a problem.
    
    Subjects: algebra, statistics (Counting & Probability), calculus (Pre-calculus)
    
    Evaluates all prompting techniques:
    - Zero-shot
    - Few-shot
    
    Returns comparative results and optimal technique selection.
    """
    try:
        run_mode = _resolve_run_mode(request.run_mode)
        ground_truth = _normalize_ground_truth(request.ground_truth)
        resolved, domain_guard = _resolve_domain_for_normal_mode(
            problem=request.problem,
            requested_subject=request.subject,
        )
        if not resolved:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This framework currently only supports the following 3 domains "
                    "(algebra, pre-calculus, and counting-probability)."
                ),
            )
        resolved_domain = resolved

        if run_mode == RUN_MODE_BENCHMARK and not ground_truth:
            raise HTTPException(
                status_code=400,
                detail="Benchmark mode requires a non-empty ground_truth value.",
            )
        if run_mode == RUN_MODE_NORMAL:
            ground_truth = None

        speed_profile = _resolve_speed_profile(request.speed_profile)
        if run_mode == RUN_MODE_BENCHMARK:
            # Benchmark mode prioritizes reproducibility over speed shortcuts.
            speed_profile = "balanced"
        request_pipeline = pipeline if speed_profile == "balanced" else _build_fast_pipeline_from_defaults()

        techniques_to_run, pre_execution_policy = _resolve_pre_execution_techniques(
            request_pipeline=request_pipeline,
            run_mode=run_mode,
            problem=request.problem,
            domain=resolved_domain,
            difficulty=request.difficulty or "basic",
        )
        pre_execution_policy["domain_guard"] = domain_guard

        benchmark_kwargs: Dict[str, Any] = {
            "problem": request.problem,
            "ground_truth": ground_truth,
            "subject": resolved_domain,
        }
        if techniques_to_run:
            benchmark_kwargs["techniques_to_run"] = techniques_to_run
        if request.runs_per_technique is not None:
            benchmark_kwargs["runs_per_technique"] = request.runs_per_technique

        result = request_pipeline.benchmark(**benchmark_kwargs)

        result = _finalize_benchmark_result(
            result=result,
            domain=resolved_domain,
            difficulty=request.difficulty or "basic",
            run_mode=run_mode,
        )
        result["ground_truth_used"] = bool(ground_truth)
        result["pre_execution_policy"] = pre_execution_policy
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark/stream", tags=["Benchmarking"])
async def run_benchmark_stream(request: BenchmarkRequest):
    """
    Stream benchmark progress and model output in real time (NDJSON).

    Event types:
    - status
    - token
    - complete
    - error
    """
    run_mode = _resolve_run_mode(request.run_mode)
    ground_truth = _normalize_ground_truth(request.ground_truth)
    resolved, domain_guard = _resolve_domain_for_normal_mode(
        problem=request.problem,
        requested_subject=request.subject,
    )
    if not resolved:
        raise HTTPException(
            status_code=422,
            detail=(
                "This framework currently only supports the following 3 domains "
                "(algebra, pre-calculus, and counting-probability)."
            ),
        )
    resolved_domain = resolved

    if run_mode == RUN_MODE_BENCHMARK and not ground_truth:
        raise HTTPException(
            status_code=400,
            detail="Benchmark mode requires a non-empty ground_truth value.",
        )
    if run_mode == RUN_MODE_NORMAL:
        ground_truth = None

    speed_profile = _resolve_speed_profile(request.speed_profile)
    if run_mode == RUN_MODE_BENCHMARK:
        speed_profile = "balanced"
    request_pipeline = pipeline if speed_profile == "balanced" else _build_fast_pipeline_from_defaults()
    techniques_to_run, pre_execution_policy = _resolve_pre_execution_techniques(
        request_pipeline=request_pipeline,
        run_mode=run_mode,
        problem=request.problem,
        domain=resolved_domain,
        difficulty=request.difficulty or "basic",
    )
    pre_execution_policy["domain_guard"] = domain_guard

    def event_stream():
        try:
            stream_kwargs: Dict[str, Any] = {
                "problem": request.problem,
                "ground_truth": ground_truth,
                "subject": resolved_domain,
            }
            if techniques_to_run:
                stream_kwargs["techniques_to_run"] = techniques_to_run
            if request.runs_per_technique is not None:
                stream_kwargs["runs_per_technique"] = request.runs_per_technique

            for event in request_pipeline.benchmark_stream_events(**stream_kwargs):
                if event.get("type") == "complete":
                    result = event.get("result", {})
                    result = _finalize_benchmark_result(
                        result=result,
                        domain=resolved_domain,
                        difficulty=request.difficulty or "basic",
                        run_mode=run_mode,
                    )
                    result["ground_truth_used"] = bool(ground_truth)
                    result["pre_execution_policy"] = pre_execution_policy
                    event["result"] = result
                elif event.get("type") == "error":
                    # Ensure error messages are never empty
                    error_msg = event.get("error", "").strip()
                    if not error_msg:
                        error_msg = "Unknown error during benchmark streaming"
                    event["error"] = error_msg
                yield json.dumps(event) + "\n"
        except Exception as e:
            error_msg = str(e).strip() if str(e) else "Unknown error during benchmark"
            yield json.dumps({"type": "error", "error": error_msg}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/techniques", tags=["Info"])
async def get_techniques():
    """Get list of available prompting techniques."""
    return {
        "techniques": pipeline.prompt_generator.get_technique_names(),
        "descriptions": {
            "zero_shot": "Direct question without examples or context",
            "few_shot": "Includes example problems and solutions"
        }
    }


@app.get("/subjects", tags=["Info"])
async def get_subjects():
    """Get list of available subject categories for few-shot examples."""
    return {
        "subjects": pipeline.prompt_generator.get_available_subjects(),
        "descriptions": {
            "algebra": "Linear equations, quadratic equations, factoring, systems",
            "statistics": "Counting, combinations, permutations, probability distributions",
            "calculus": "Limits, functions, transformations, sequences",
            "general": "Basic arithmetic problems"
        }
    }


@app.get("/example-difficulties", tags=["Info"])
async def example_difficulties():
    """Return available difficulty levels per subject from example_problems.json + user_examples.json."""
    import json as _json, os as _os
    json_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "framework", "example_problems.json")
    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            source = _json.load(f)
    except Exception:
        source = {}

    # Include user examples
    user_data = _load_user_examples()
    user_data, _ = _purge_expired_entries(user_data)
    for subj, diffs in user_data.items():
        if subj not in source:
            source[subj] = {}
        if isinstance(diffs, dict):
            for diff in diffs.keys():
                if diff not in source[subj]:
                    source[subj][diff] = []

    result: dict[str, list[str]] = {}
    for subject, subject_val in source.items():
        if isinstance(subject_val, dict):
            result[subject] = sorted(subject_val.keys())
    return result


@app.get("/weights", tags=["Configuration"])
async def get_weights():
    """Get current metric weights."""
    return {
        "weights": pipeline.weights,
        "description": "Weights used for calculating overall scores"
    }


@app.post("/weights", tags=["Configuration"])
async def update_weights(weights: WeightsUpdate):
    """
    Update metric weights for scoring.
    
    Weights are automatically normalized to sum to 1.0.
    """
    try:
        pipeline.set_weights(
            accuracy=weights.accuracy,
            consistency=weights.consistency,
            efficiency=weights.efficiency
        )
        return {
            "message": "Weights updated successfully",
            "weights": pipeline.weights
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/dataset", tags=["Dataset"])
async def get_dataset():
    """Get all problems in the dataset."""
    return {
        "size": dataset.size(),
        "problems": dataset.get_problems()
    }


@app.get("/dataset/{problem_id}", tags=["Dataset"])
async def get_problem(problem_id: int):
    """Get a specific problem by ID."""
    problem = dataset.get_problem(problem_id)
    if problem is None:
        raise HTTPException(status_code=404, detail="Problem not found")
    return problem


@app.post("/dataset", tags=["Dataset"])
async def add_problem(problem: ProblemAdd):
    """Add a new problem to the dataset."""
    try:
        dataset.add_problem(
            problem=problem.problem,
            answer=problem.answer,
            category=problem.category
        )
        return {
            "message": "Problem added successfully",
            "dataset_size": dataset.size()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/benchmark/dataset/{problem_id}", tags=["Benchmarking"])
async def benchmark_dataset_problem(problem_id: int):
    """
    Run benchmark on a problem from the dataset.
    
    Uses the ground truth from the dataset for evaluation.
    """
    problem_data = dataset.get_problem(problem_id)
    if problem_data is None:
        raise HTTPException(status_code=404, detail="Problem not found")

    resolved_domain, _ = _resolve_domain_for_normal_mode(
        problem=str(problem_data.get("problem", "") or ""),
        requested_subject=str(problem_data.get("category", "") or ""),
    )
    if not resolved_domain:
        raise HTTPException(
            status_code=422,
            detail=(
                "This framework currently only supports the following 3 domains "
                "(algebra, pre-calculus, and counting-probability)."
            ),
        )
    
    try:
        techniques_to_run, pre_execution_policy = _resolve_pre_execution_techniques(
            request_pipeline=pipeline,
            run_mode=RUN_MODE_BENCHMARK,
            problem=problem_data["problem"],
            domain=resolved_domain,
            difficulty="basic",
        )

        benchmark_kwargs: Dict[str, Any] = {
            "problem": problem_data["problem"],
            "ground_truth": problem_data["answer"],
        }
        if techniques_to_run:
            benchmark_kwargs["techniques_to_run"] = techniques_to_run

        result = pipeline.benchmark(**benchmark_kwargs)

        result = _finalize_benchmark_result(
            result=result,
            domain=resolved_domain,
            difficulty="basic",
            run_mode=RUN_MODE_BENCHMARK,
        )
        result["ground_truth_used"] = True
        result["pre_execution_policy"] = pre_execution_policy
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/results/save", tags=["Benchmarking"])
async def save_result(request: SaveResultRequest):
    """Manually save an existing benchmark result to Firestore."""
    try:
        metadata = dict(request.metadata or {})

        # Only benchmark mode results are saved to DB.
        # Normal mode consumes DB data but never writes to it,
        # preventing biased / filtered data from polluting the store.
        save_run_mode = _resolve_run_mode(
            metadata.get("run_mode") or request.result.get("run_mode")
        )
        if save_run_mode != RUN_MODE_BENCHMARK:
            return {
                "message": "Skipped: only benchmark mode results are saved to the database.",
                "storage": {"success": False, "reason": "normal_mode_save_blocked"},
            }

        domain = metadata.get("domain") or metadata.get("subject") or metadata.get("category") or "general"
        difficulty = metadata.get("difficulty") or request.result.get("difficulty") or "basic"

        benchmark_result = _attach_problem_profile(
            result=dict(request.result),
            domain=str(domain),
            difficulty=str(difficulty),
        )

        resolved_domain, domain_guard = _resolve_domain_for_normal_mode(
            problem=str(benchmark_result.get("problem", "") or ""),
            requested_subject=str(domain),
        )
        if not resolved_domain:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This framework currently only supports the following 3 domains "
                    "(algebra, pre-calculus, and counting-probability)."
                ),
            )

        # Keep normalized domain metadata in persisted records.
        metadata["domain"] = resolved_domain
        metadata["domain_guard"] = domain_guard

        run_mode = _resolve_run_mode(metadata.get("run_mode") or benchmark_result.get("run_mode"))
        has_ground_truth = bool(_normalize_ground_truth(benchmark_result.get("ground_truth")))

        metadata["run_mode"] = run_mode
        metadata["has_ground_truth"] = has_ground_truth
        metadata["problem_profile"] = benchmark_result.get("problem_profile")

        storage = firestore_store.save_benchmark_result(
            benchmark_result=benchmark_result,
            source=request.source or "manual_save",
            metadata=metadata
        )

        if firestore_store.required and not storage.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Firestore write failed: {storage.get('error', 'Unknown error')}"
            )

        return {
            "message": "Result saved" if storage.get("success") else "Save failed",
            "storage": storage
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/coverage/{domain}/{difficulty}", tags=["Coverage"])
async def get_coverage(domain: str, difficulty: str):
    """Get benchmark coverage stats for a domain/difficulty."""
    result = firestore_store.get_coverage(domain=domain, difficulty=difficulty)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Coverage query failed"))
    return result


# ═══════════════════════════════════════════════════════════════════════
# Example Bank Management
# ═══════════════════════════════════════════════════════════════════════

_EXAMPLE_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework", "example_problems.json")
_USER_EXAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework", "user_examples.json")
_VALID_SUBJECTS = {"algebra", "counting-probability", "pre-calculus"}
_VALID_DIFFICULTIES = {"basic", "intermediate", "advanced"}
_example_lock = asyncio.Lock()

# Default TTL for user-submitted examples: 3 days (in seconds)
_USER_EXAMPLE_TTL_SECONDS = int(os.getenv("USER_EXAMPLE_TTL_SECONDS", str(3 * 24 * 60 * 60)))

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework", "example_bank.schema.json")
with open(_SCHEMA_PATH, "r", encoding="utf-8") as _sf:
    _EXAMPLE_SCHEMA = json.load(_sf)
# Build a standalone entry schema that carries the shared $defs for $ref resolution
_ENTRY_SCHEMA = {**_EXAMPLE_SCHEMA["$defs"]["exampleEntry"], "$defs": _EXAMPLE_SCHEMA["$defs"]}


def _load_user_examples() -> dict:
    """Load user_examples.json safely."""
    try:
        with open(_USER_EXAMPLES_PATH, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _purge_expired_entries(source: dict) -> tuple:
    """Remove expired entries from user examples. Returns (cleaned_source, removed_count)."""
    now = datetime.now(timezone.utc).timestamp()
    removed = 0
    for subject in list(source.keys()):
        if not isinstance(source[subject], dict):
            continue
        for difficulty in list(source[subject].keys()):
            if not isinstance(source[subject][difficulty], list):
                continue
            before = len(source[subject][difficulty])
            source[subject][difficulty] = [
                ex for ex in source[subject][difficulty]
                if ex.get("expires_at", float("inf")) > now
            ]
            removed += before - len(source[subject][difficulty])
            # Clean up empty lists
            if not source[subject][difficulty]:
                del source[subject][difficulty]
        # Clean up empty subjects
        if not source[subject]:
            del source[subject]
    return source, removed


def _atomic_write_json(path: str, data: dict):
    """Write JSON atomically via temp file + os.replace."""
    dir_name = os.path.dirname(path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _reload_prompt_generator_examples():
    """Reload prompt_generator's example_dataset by merging dev + valid user examples."""
    pg = pipeline.prompt_generator

    # Load dev bank
    try:
        with open(_EXAMPLE_JSON_PATH, "r", encoding="utf-8-sig") as f:
            dev_data = json.load(f)
    except Exception:
        dev_data = {}

    # Load user bank and purge expired
    user_data = _load_user_examples()
    user_data, _ = _purge_expired_entries(user_data)

    # Merge: user examples are appended after dev examples per subject/difficulty
    merged = json.loads(json.dumps(dev_data))  # deep copy
    for subject, diffs in user_data.items():
        if subject not in merged:
            merged[subject] = {}
        if isinstance(diffs, dict):
            for difficulty, examples in diffs.items():
                if difficulty not in merged[subject]:
                    merged[subject][difficulty] = []
                if isinstance(examples, list):
                    # Strip TTL metadata before passing to prompt_generator
                    for ex in examples:
                        clean_ex = {k: v for k, v in ex.items() if k not in ("created_at", "expires_at")}
                        merged[subject][difficulty].append(clean_ex)

    pg.example_dataset = pg._normalize_example_dataset(merged)
    pg._type_to_subject = pg._build_type_to_subject_map()


class ExampleAnalyzeRequest(BaseModel):
    problem: str
    solution: str = ""


class ExampleSaveRequest(BaseModel):
    problem: str = Field(..., min_length=1, max_length=5000)
    solution: str = Field(..., min_length=1, max_length=10000)
    subject: str = Field(..., min_length=1, max_length=50)
    difficulty: str = Field(..., min_length=1, max_length=20)
    type: str = Field(..., min_length=1, max_length=100)
    concept: str = Field(..., min_length=1, max_length=100)


@app.post("/examples/analyze", tags=["Examples"])
async def analyze_example(req: ExampleAnalyzeRequest):
    """Auto-detect subject, type, and difficulty for a problem using Ollama LLM (with rule-based fallback)."""
    pg = pipeline.prompt_generator

    # Build existing types from dev bank + user bank
    try:
        with open(_EXAMPLE_JSON_PATH, "r", encoding="utf-8-sig") as f:
            source = json.load(f)
    except Exception:
        source = {}

    # Also include user types
    user_data = _load_user_examples()
    user_data, _ = _purge_expired_entries(user_data)
    for subj, diffs in user_data.items():
        if subj not in source:
            source[subj] = {}
        if isinstance(diffs, dict):
            for diff, examples in diffs.items():
                if diff not in source[subj]:
                    source[subj][diff] = []
                if isinstance(examples, list):
                    source[subj][diff].extend(examples)

    existing_types: dict[str, list[str]] = {}
    for subj, diffs in source.items():
        types_set: set[str] = set()
        if isinstance(diffs, dict):
            for diff_examples in diffs.values():
                if isinstance(diff_examples, list):
                    for ex in diff_examples:
                        types_set.add(ex.get("type", "general"))
        existing_types[subj] = sorted(types_set)

    # Try Ollama LLM classification first
    llm_result = None
    detection_method = "rule-based"
    try:
        llm_result = pipeline.model_runner.classify_problem(
            problem=req.problem,
            solution=req.solution,
            existing_types=existing_types,
        )
    except Exception:
        pass

    if llm_result:
        detection_method = "ollama-llm"
        detected_subject = llm_result["subject"]
        detected_type = llm_result["type"]
        detected_difficulty = llm_result["difficulty"]
        detected_concept = llm_result.get("concept", detected_type)
    else:
        # Fallback to rule-based detection
        detected_subject = pg.classify_subject(req.problem)
        detected_type = pg._detect_primary_intent(pg._normalize_detection_text(req.problem))
        detected_concept = detected_type

        word_count = len(req.problem.split())
        if word_count <= 15:
            detected_difficulty = "basic"
        elif word_count <= 30:
            detected_difficulty = "intermediate"
        else:
            detected_difficulty = "advanced"

    return {
        "detected_subject": detected_subject if detected_subject in _VALID_SUBJECTS else "algebra",
        "detected_type": detected_type,
        "detected_difficulty": detected_difficulty,
        "detected_concept": detected_concept,
        "detection_method": detection_method,
        "existing_types": existing_types,
    }


@app.post("/examples", tags=["Examples"])
async def save_example(req: ExampleSaveRequest):
    """Save a new user-submitted example to user_examples.json (with TTL expiry)."""
    if req.subject not in _VALID_SUBJECTS:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {req.subject}. Must be one of {_VALID_SUBJECTS}")
    if req.difficulty not in _VALID_DIFFICULTIES:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {req.difficulty}. Must be one of {_VALID_DIFFICULTIES}")
    if not req.problem.strip() or not req.solution.strip():
        raise HTTPException(status_code=400, detail="Problem and solution are required")

    # Build example entry
    clean_type = req.type.strip().lower().replace(" ", "_")
    clean_concept = req.concept.strip().lower().replace(" ", "_")

    now = datetime.now(timezone.utc)
    new_example = {
        "problem": req.problem.strip(),
        "solution": req.solution.strip(),
        "type": clean_type,
        "constraints": [],
        "concept": clean_concept,
        "skills": [clean_type],
        "format": {
            "template": "single_equation",
            "equation_count": 1,
            "has_assignment": False,
            "assigned_variable_count": 0,
            "has_fraction": False,
            "has_exponent": False,
            "has_system": False,
        },
        "difficulty": req.difficulty,
        "tags": [req.subject, clean_type],
        "anchor_priority": 0.8,
        "created_at": now.isoformat(),
        "expires_at": now.timestamp() + _USER_EXAMPLE_TTL_SECONDS,
    }

    # Validate entry against JSON schema (strip TTL fields for validation)
    entry_for_validation = {k: v for k, v in new_example.items() if k not in ("created_at", "expires_at")}
    try:
        jsonschema.validate(instance=entry_for_validation, schema=_ENTRY_SCHEMA)
    except jsonschema.ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Entry failed schema validation: {e.message}")

    async with _example_lock:
        # Load user examples
        source = _load_user_examples()

        # Purge expired entries first
        source, purged_count = _purge_expired_entries(source)

        # Ensure structure exists
        if req.subject not in source:
            source[req.subject] = {}
        if req.difficulty not in source[req.subject]:
            source[req.subject][req.difficulty] = []

        # Duplicate check against user bank
        existing = source[req.subject][req.difficulty]
        for ex in existing:
            if ex.get("problem", "").strip().lower() == req.problem.strip().lower():
                raise HTTPException(status_code=409, detail="This exact problem already exists in the user example bank")

        # Also check against dev bank (don't allow users to duplicate dev examples)
        try:
            with open(_EXAMPLE_JSON_PATH, "r", encoding="utf-8-sig") as f:
                dev_source = json.load(f)
            dev_examples = dev_source.get(req.subject, {}).get(req.difficulty, [])
            for ex in dev_examples:
                if ex.get("problem", "").strip().lower() == req.problem.strip().lower():
                    raise HTTPException(status_code=409, detail="This problem already exists in the curated example bank")
        except HTTPException:
            raise
        except Exception:
            pass

        existing.append(new_example)

        # Backup before write
        if os.path.exists(_USER_EXAMPLES_PATH):
            shutil.copy2(_USER_EXAMPLES_PATH, _USER_EXAMPLES_PATH + ".bak")

        # Atomic write
        _atomic_write_json(_USER_EXAMPLES_PATH, source)

        # Reload prompt_generator with merged examples
        _reload_prompt_generator_examples()

    expires_at_str = datetime.fromtimestamp(new_example["expires_at"], tz=timezone.utc).isoformat()
    return {
        "success": True,
        "message": f"Example added to {req.subject}/{req.difficulty} as type '{clean_type}'",
        "bank": "user",
        "expires_at": expires_at_str,
        "ttl_seconds": _USER_EXAMPLE_TTL_SECONDS,
        "purged_expired": purged_count,
    }


@app.get("/examples/user", tags=["Examples"])
async def list_user_examples():
    """List all user-submitted examples with their expiry status."""
    async with _example_lock:
        source = _load_user_examples()
        source, purged_count = _purge_expired_entries(source)
        if purged_count > 0:
            _atomic_write_json(_USER_EXAMPLES_PATH, source)

    now = datetime.now(timezone.utc).timestamp()
    result = {}
    total = 0
    for subject, diffs in source.items():
        if not isinstance(diffs, dict):
            continue
        result[subject] = {}
        for difficulty, examples in diffs.items():
            if not isinstance(examples, list):
                continue
            result[subject][difficulty] = []
            for ex in examples:
                remaining = ex.get("expires_at", float("inf")) - now
                result[subject][difficulty].append({
                    "problem": ex.get("problem", ""),
                    "type": ex.get("type", ""),
                    "difficulty": ex.get("difficulty", difficulty),
                    "created_at": ex.get("created_at", ""),
                    "expires_at": datetime.fromtimestamp(ex.get("expires_at", 0), tz=timezone.utc).isoformat() if ex.get("expires_at") else None,
                    "ttl_remaining_hours": round(remaining / 3600, 1) if remaining != float("inf") else None,
                })
                total += 1

    return {
        "total": total,
        "ttl_seconds": _USER_EXAMPLE_TTL_SECONDS,
        "ttl_human": f"{_USER_EXAMPLE_TTL_SECONDS // 86400} days",
        "purged_expired": purged_count,
        "examples": result,
    }


@app.delete("/examples/user/expired", tags=["Examples"])
async def purge_expired_user_examples():
    """Manually purge all expired user examples."""
    async with _example_lock:
        source = _load_user_examples()
        source, purged_count = _purge_expired_entries(source)
        if purged_count > 0:
            if os.path.exists(_USER_EXAMPLES_PATH):
                shutil.copy2(_USER_EXAMPLES_PATH, _USER_EXAMPLES_PATH + ".bak")
            _atomic_write_json(_USER_EXAMPLES_PATH, source)
            _reload_prompt_generator_examples()

    return {"purged": purged_count, "message": f"Removed {purged_count} expired example(s)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
