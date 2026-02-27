"""
FastAPI application for the Prompt Optimization Framework.
Research Benchmarking API for comparative prompt evaluation.
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from framework.pipeline import BenchmarkPipeline
from framework.dataset import MathDataset, get_sample_dataset
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

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = BenchmarkPipeline(
    model_name=os.getenv("MODEL_NAME", "llama3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
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


def _apply_db_based_selection(result: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Apply DB-based greedy selection from historical Firestore results."""
    successful_techniques = [
        technique
        for technique, technique_result in result.get("all_results", {}).items()
        if technique_result.get("success", False)
    ]

    selection = firestore_store.get_best_technique_by_domain(
        domain=domain,
        available_techniques=successful_techniques,
    )

    if selection.get("success"):
        best_technique = selection["best_technique"]
        if best_technique in result.get("all_results", {}):
            result["best_technique"] = best_technique
            result["best_result"] = result["all_results"][best_technique]
            result["selection_source"] = "db_history"
            result["selection_details"] = selection
            return result

    result["selection_source"] = "runtime_scores"
    result["selection_details"] = selection
    return result


class BenchmarkRequest(BaseModel):
    """Request model for benchmarking."""
    problem: str
    ground_truth: Optional[str] = None
    subject: Optional[str] = "algebra"  # algebra, statistics, or calculus


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
    return {
        "status": "healthy" if model_connected else "degraded",
        "model_connected": model_connected,
        "dataset_size": dataset.size(),
        "weights": pipeline.weights,
        "firestore": firestore_store.get_status()
    }


@app.post("/benchmark", tags=["Benchmarking"])
async def run_benchmark(request: BenchmarkRequest):
    """
    Run comprehensive benchmark on a problem.
    
    Evaluates all prompting techniques:
    - Zero-shot
    - Few-shot
    
    Returns comparative results and optimal technique selection.
    """
    try:
        result = pipeline.benchmark(
            problem=request.problem,
            ground_truth=request.ground_truth,
            subject=request.subject
        )

        result = _apply_db_based_selection(
            result=result,
            domain=request.subject or "general",
        )
        
        if not result["best_result"]["success"]:
            raise HTTPException(
                status_code=500,
                detail=result["best_result"].get("error", "Benchmark failed")
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            "statistics": "Mean, median, mode, variance, probability, combinations",
            "calculus": "Derivatives, integrals, limits",
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

        result = _apply_db_based_selection(
            result=result,
            domain=problem_data.get("category", "general"),
        )
        
        if not result["best_result"]["success"]:
            raise HTTPException(
                status_code=500,
                detail=result["best_result"].get("error", "Benchmark failed")
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
