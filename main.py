"""
FastAPI application for the Prompt Optimization Framework.
Research Benchmarking API for comparative prompt evaluation.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
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


def _build_fast_pipeline_from_defaults() -> BenchmarkPipeline:
    """Create a per-request fast pipeline without mutating shared global state."""
    request_pipeline = BenchmarkPipeline(
        model_name=pipeline.model_runner.model_name,
        base_url=pipeline.model_runner.base_url,
        accuracy_weight=pipeline.weights.get("accuracy", 0.5),
        completeness_weight=pipeline.weights.get("completeness", 0.3),
        efficiency_weight=pipeline.weights.get("efficiency", 0.2),
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
    accuracy_weight=0.5,
    completeness_weight=0.3,
    efficiency_weight=0.2
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


def _persist_and_attach_storage(
    result: Dict[str, Any],
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist benchmark result to Firestore and attach storage status to result payload."""
    storage = firestore_store.save_benchmark_result(
        benchmark_result=result,
        source=source,
        metadata=metadata or {},
    )
    result["storage"] = storage

    if firestore_store.required and not storage.get("success", False):
        raise HTTPException(
            status_code=500,
            detail=f"Firestore write failed: {storage.get('error', 'Unknown error')}",
        )

    return result


def _apply_db_based_selection(result: Dict[str, Any], domain: str, difficulty: str) -> Dict[str, Any]:
    """Apply DB-based greedy selection from historical Firestore results."""
    successful_techniques = [
        technique
        for technique, technique_result in result.get("all_results", {}).items()
        if technique_result.get("success", False)
    ]

    selection = firestore_store.get_best_technique_by_domain(
        domain=domain,
        difficulty=difficulty,
        available_techniques=successful_techniques,
    )

    if not isinstance(selection, dict):
        selection = {
            "success": False,
            "reason": "invalid_selection_payload",
            "error": "Selection payload must be a dictionary.",
        }

    min_samples = _get_env_int("DB_MIN_SAMPLES_PER_TECHNIQUE", default=5, min_value=1)
    min_gap = _get_env_float("DB_MIN_AVG_SCORE_GAP", default=0.03, min_value=0.0)
    exploration_rate = _get_env_float(
        "DB_EXPLORATION_RATE",
        default=0.15,
        min_value=0.0,
        max_value=1.0,
    )

    ranking = selection.get("ranking", [])
    can_use_db = bool(selection.get("success", False))
    db_decision_reason = "ok"

    if can_use_db and len(ranking) >= 2:
        top = ranking[0] if isinstance(ranking[0], dict) else {}
        second = ranking[1] if isinstance(ranking[1], dict) else {}

        try:
            top_samples = max(0, int(top.get("samples", 0) or 0))
            second_samples = max(0, int(second.get("samples", 0) or 0))
            top_average = float(top.get("average_overall", 0.0) or 0.0)
            second_average = float(second.get("average_overall", 0.0) or 0.0)
            score_gap = top_average - second_average
        except (TypeError, ValueError):
            can_use_db = False
            db_decision_reason = "invalid_ranking_data"
        else:
            if min(top_samples, second_samples) < min_samples:
                can_use_db = False
                db_decision_reason = "insufficient_samples"
            elif score_gap < min_gap:
                can_use_db = False
                db_decision_reason = "low_confidence_gap"

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
            result["selection_source"] = "db_history"
            result["selection_details"] = {
                **selection,
                "db_decision_reason": db_decision_reason,
                "db_confidence_rules": {
                    "min_samples_per_technique": min_samples,
                    "min_average_gap": min_gap,
                    "exploration_rate": exploration_rate,
                },
            }
            return result

    result["selection_source"] = "runtime_scores"
    result["selection_details"] = {
        **selection,
        "db_decision_reason": db_decision_reason,
        "db_confidence_rules": {
            "min_samples_per_technique": min_samples,
            "min_average_gap": min_gap,
            "exploration_rate": exploration_rate,
        },
    }
    return result


def _finalize_benchmark_result(
    result: Dict[str, Any],
    *,
    domain: str,
    difficulty: str,
    source: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply selection strategy, validate winner, and persist storage status."""
    result = _apply_db_based_selection(
        result=result,
        domain=domain,
        difficulty=difficulty,
    )

    best_result = result.get("best_result", {})
    if not best_result.get("success", False):
        raise HTTPException(
            status_code=500,
            detail=best_result.get("error", "Benchmark failed"),
        )

    return _persist_and_attach_storage(
        result=result,
        source=source,
        metadata=metadata,
    )


class BenchmarkRequest(BaseModel):
    """Request model for benchmarking."""
    problem: str
    ground_truth: Optional[str] = None
    subject: Optional[str] = "algebra"  # algebra, counting-probability, or pre-calculus
    difficulty: Optional[str] = "basic"  # basic, intermediate, advanced
    speed_profile: Optional[str] = "balanced"  # balanced, fast


class WeightsUpdate(BaseModel):
    """Model for updating metric weights."""
    accuracy: Optional[float] = None
    completeness: Optional[float] = None
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
        speed_profile = _resolve_speed_profile(request.speed_profile)
        request_pipeline = pipeline if speed_profile == "balanced" else _build_fast_pipeline_from_defaults()

        result = request_pipeline.benchmark(
            problem=request.problem,
            ground_truth=request.ground_truth,
            subject=request.subject
        )

        result = _finalize_benchmark_result(
            result=result,
            domain=request.subject or "general",
            difficulty=request.difficulty or "basic",
            source="benchmark_api",
            metadata={
                "subject": request.subject,
                "difficulty": request.difficulty,
                "speed_profile": speed_profile,
            },
        )
        
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
    speed_profile = _resolve_speed_profile(request.speed_profile)
    request_pipeline = pipeline if speed_profile == "balanced" else _build_fast_pipeline_from_defaults()

    def event_stream():
        try:
            for event in request_pipeline.benchmark_stream_events(
                problem=request.problem,
                ground_truth=request.ground_truth,
                subject=request.subject,
            ):
                if event.get("type") == "complete":
                    result = event.get("result", {})
                    result = _finalize_benchmark_result(
                        result=result,
                        domain=request.subject or "general",
                        difficulty=request.difficulty or "basic",
                        source="benchmark_stream_api",
                        metadata={
                            "subject": request.subject,
                            "difficulty": request.difficulty,
                            "speed_profile": speed_profile,
                        },
                    )
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
            completeness=weights.completeness,
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
    
    try:
        result = pipeline.benchmark(
            problem=problem_data["problem"],
            ground_truth=problem_data["answer"]
        )

        result = _finalize_benchmark_result(
            result=result,
            domain=problem_data.get("category", "general"),
            difficulty="basic",
            source="benchmark_dataset_api",
            metadata={
                "category": problem_data.get("category", "general"),
                "difficulty": "basic",
            },
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/results/save", tags=["Benchmarking"])
async def save_result(request: SaveResultRequest):
    """Manually save an existing benchmark result to Firestore."""
    try:
        storage = firestore_store.save_benchmark_result(
            benchmark_result=request.result,
            source=request.source or "manual_save",
            metadata=request.metadata or {}
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
