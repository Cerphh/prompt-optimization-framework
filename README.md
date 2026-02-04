# Prompt Optimization Research Framework

A Python-based research benchmarking framework for evaluating and optimizing prompting strategies on large language models (LLMs).

## Overview

This framework implements a **comparative experimental design** to evaluate multiple prompting techniques on math problem-solving tasks. It uses a **greedy selection algorithm** to automatically identify the optimal prompting strategy based on weighted metrics.

## Features

### 🎯 Four Prompting Techniques

1. **Zero-shot**: Direct question without examples or context
2. **Few-shot**: Includes example problems and solutions
3. **Chain-of-thought**: Encourages step-by-step reasoning
4. **Instruction-based**: Detailed instructions with constraints

### 📊 Three Evaluation Metrics

1. **Accuracy** (default weight: 0.5)
   - Exact string matching
   - Numeric comparison
   - Symbolic math evaluation using SymPy
   - Supports fractions and algebraic expressions

2. **Completeness** (default weight: 0.3)
   - Step-by-step reasoning detection
   - Explanation quality
   - Structure and organization
   - Detail sufficiency

3. **Efficiency** (default weight: 0.2)
   - Response latency
   - Token usage (prompt + completion)
   - Response conciseness

### 🔍 Greedy Selection Algorithm

Automatically selects the best-performing prompting technique based on:
- Highest overall weighted score
- Tie-breaking: accuracy → completeness → efficiency

## Project Structure

```
prompt-optimization-framework/
│
├── framework/
│   ├── __init__.py
│   ├── dataset.py              # Math problem dataset management
│   ├── prompt_generator.py     # Four prompting strategies
│   ├── model_runner.py         # Ollama LLM interface
│   ├── accuracy_scorer.py      # Accuracy evaluation
│   ├── completeness_scorer.py  # Completeness evaluation
│   ├── efficiency_scorer.py    # Efficiency evaluation
│   └── pipeline.py             # Main benchmark pipeline
│
├── main.py                      # FastAPI application
├── test_pipeline.py             # Test script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

### 1. Prerequisites

- Python 3.8+
- Ollama installed and running
- llama3 model downloaded

### 2. Setup Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama & Model

```bash
# Download from https://ollama.ai
ollama pull llama3
```

## Usage

### Command Line Testing

```bash
python test_pipeline.py
```

This runs a comprehensive benchmark on a sample problem, evaluating all four prompting techniques and displaying comparative results.

### API Server

```bash
uvicorn main:app --reload
```

Visit: http://127.0.0.1:8000/docs

### Programmatic Usage

```python
from framework.pipeline import BenchmarkPipeline
from framework.dataset import get_sample_dataset

# Initialize pipeline
pipeline = BenchmarkPipeline(
    model_name="llama3",
    accuracy_weight=0.5,
    completeness_weight=0.3,
    efficiency_weight=0.2
)

# Run benchmark on a problem
result = pipeline.benchmark(
    problem="What is 15 + 27?",
    ground_truth="42"
)

# Access results
print(f"Best technique: {result['best_technique']}")
print(f"Overall score: {result['best_result']['scores']['overall']}")

# View comparison
for comp in result['comparison']:
    print(f"{comp['technique']}: {comp['overall']:.3f}")
```

## API Endpoints

### Benchmarking

- `POST /benchmark` - Run comprehensive benchmark on a problem
- `POST /benchmark/dataset/{problem_id}` - Benchmark a dataset problem

### Dataset Management

- `GET /dataset` - Get all problems
- `GET /dataset/{problem_id}` - Get specific problem
- `POST /dataset` - Add new problem

### Configuration

- `GET /weights` - Get current metric weights
- `POST /weights` - Update metric weights
- `GET /techniques` - List available techniques

### Health

- `GET /health` - Check system status

## Example Results

```
======================================================================
RESULTS COMPARISON
======================================================================
Technique            Accuracy   Complete   Efficiency Overall
----------------------------------------------------------------------
chain_of_thought     1.000      0.900      0.755      0.921
instruction_based    1.000      0.750      0.835      0.892
few_shot             1.000      0.250      0.810      0.737
zero_shot            1.000      0.150      0.670      0.679

======================================================================
OPTIMAL TECHNIQUE SELECTED (GREEDY ALGORITHM)
======================================================================
Best Technique: chain_of_thought
Overall Score: 0.921
```

## Research Design

### Methodology

1. **Comparative Experimental Design**
   - Each problem evaluated with all four techniques
   - Independent execution per prompt
   - Controlled variables (model, temperature, dataset)

2. **Dataset**
   - Math problems with ground truth answers
   - Categories: arithmetic, algebra, word problems, geometry
   - Small initial size (10 problems), easily scalable

3. **Evaluation Pipeline**
   ```
   Input Problem → Generate 4 Prompts → Execute Each → Score → Greedy Select → Return Best
   ```

4. **Scoring Formula**
   ```
   Overall = (Accuracy × 0.5) + (Completeness × 0.3) + (Efficiency × 0.2)
   ```

### Key Design Decisions

- **Modular Architecture**: Each scorer is independent and reusable
- **Symbolic Math**: Uses SymPy for intelligent answer matching
- **Configurable Weights**: Researchers can adjust metric importance
- **Research Clarity**: Clear, documented code over premature optimization

## Extending the Framework

### Add Custom Prompting Technique

Edit `framework/prompt_generator.py`:

```python
def generate_custom_technique(self, problem: str) -> str:
    return f"Your custom prompt template: {problem}"
```

### Add New Evaluation Metric

Create `framework/custom_scorer.py`:

```python
class CustomScorer:
    def score(self, response: str, **kwargs) -> float:
        # Your evaluation logic
        return score_value  # 0.0 to 1.0
```

### Add Problems to Dataset

```python
from framework.dataset import MathDataset

dataset = MathDataset()
dataset.add_problem(
    problem="Your problem here",
    answer="Expected answer",
    category="category_name"
)
```

## Technical Specifications

- **Language**: Python 3.13
- **LLM Backend**: Ollama (local)
- **Model**: Llama 3
- **API Framework**: FastAPI
- **Math Processing**: SymPy
- **Response Time**: 2-20 seconds per technique

## Dependencies

- `fastapi` - API framework
- `uvicorn` - ASGI server
- `requests` - HTTP client for Ollama
- `sympy` - Symbolic mathematics
- `pandas` - Data manipulation (optional)
- `numpy` - Numerical computing (optional)
- `scipy` - Scientific computing (optional)

## Future Enhancements

- [ ] Support for multiple models (GPT, Claude, etc.)
- [ ] Advanced metrics (hallucination detection, citation accuracy)
- [ ] Batch benchmarking across entire dataset
- [ ] Result visualization and reporting
- [ ] Automated prompt optimization using feedback loops
- [ ] Support for non-math domains (code, text generation, etc.)

## License

MIT License - Feel free to use for research and education.

## Citation

If you use this framework in your research, please cite:

```
Prompt Optimization Research Framework
A comparative benchmarking system for LLM prompting strategies
https://github.com/yourusername/prompt-optimization-framework
```

## Contributing

Contributions welcome! Please focus on:
- Additional prompting techniques
- New evaluation metrics
- Dataset expansion
- Documentation improvements

## Contact

For questions or collaboration: [Your contact information]

---

**Built for research clarity, not production complexity.**
