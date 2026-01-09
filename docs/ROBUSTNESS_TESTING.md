# Robustness Testing for Mellea M-Programs

Evaluate m-program consistency by testing against semantic variations of a baseline problem and measuring how reliably your m-program answers them.

## Setup & Installation

### Step 1: Install BenchDrift
Install BenchDrift from source (required for robustness testing pipeline):
```bash
git clone https://github.ibm.com/Granite-debug/BenchDrift.git BenchDrift-Pipeline
cd BenchDrift-Pipeline
pip install -e .
cd ..
```

**Important:** BenchDrift is currently hosted on IBM's internal GitHub. You need access to IBM's internal repositories to run these tests. The codebase is in the process of being approved for public placement. If you don't have access, please contact your IBM administrator or the BenchDrift maintainers.

### Step 2: Install mellea-contribs
Install mellea-contribs in editable mode:
```bash
git clone https://github.com/generative-computing/mellea-contribs.git
cd mellea-contribs
pip install -e .
```

### Step 3: Set RITS API Key
Set the RITS API key environment variable for model access:
```bash
export RITS_API_KEY="your-api-key-here"
```

### Prerequisites
- Python 3.10+
- BenchDrift (installed from source above)
- Mellea (installed as dependency of mellea-contribs)
- RITS API key for model access via BenchDrift

## Overview

Generate and execute robustness test suites for your m-program by creating semantic variations of a problem and measuring how consistently your m-program answers them. This produces comprehensive test datasets that reveal m-program reliability patterns.

## How It Works

1. **Generate test variations**: Create semantic variations of your problem (different phrasings, same meaning)
2. **Test m-program**: Execute m-program on original problem + all variations to collect answers
3. **Measure consistency**: Compare m-program's correctness across all test cases
4. **Analyze robustness**: Get pass rates, drift metrics, and stability analysis

## Test Suite Architecture

```
Robustness Test Suite Generation Process
════════════════════════════════════════

Test Stage 1: Generate Variations
    │
    ├─ Original problem (baseline test case)
    │
    └─ Semantic variations
       (same meaning, different wording)

    ▼

Test Stage 2: Execute M-Program on All Cases
    │
    ├─ Run m-program on original
    │
    ├─ Run m-program on each variation
    │
    └─ Collect all m-program answers

    ▼

Test Stage 3: Evaluate M-Program Performance
    │
    ├─ Compare m-program answers to ground truth
    │  ├─ Does m-program answer baseline correctly?
    │  └─ Does m-program answer each variation correctly?
    │
    └─ Measure m-program behavior change
       ├─ Positive drift: m-program improved on variant
       ├─ Negative drift: m-program worsened on variant
       └─ No drift: m-program consistent across variants

    ▼

Test Results & Metrics
    │
    ├─ Pass rate: What % of test cases does m-program pass?
    │
    ├─ Consistency: How stable is m-program across variations?
    │
    └─ Stability metrics: How often does m-program produce consistent results?
```

## Core Tools

### 1. `benchdrift_runner.py`

**Primary toolkit for generating robustness test suites.**

Uses [BenchDrift](https://github.ibm.com/Granite-debug/BenchDrift) for variation generation and evaluation orchestration.

- `run_benchdrift_pipeline()`: Generate and execute complete test suite
  - Input: baseline problem + ground truth answer
  - Output: Complete test dataset with variations + m-program answers
  - Returns: All test cases with m-program responses and consistency metrics
  - Supports `config_overrides` parameter to customize variation types, models, and all pipeline settings

- `analyze_robustness_from_probes()`: Compute robustness metrics from test results
  - Measures m-program pass rate across all test variations
  - Reports consistency metrics (how stable is m-program?)
  - Identifies failure patterns (where does m-program break?)

### 2. `benchdrift_model_client_adapter.py`

**Enables m-program to work within the BenchDrift test suite framework.**

- `MelleaModelClientAdapter`: Connects m-program to BenchDrift test generation
  - Takes m-program callable + Mellea session
  - Executes m-program on each test variation (BenchDrift's test stage 2)
  - Provides batch (`get_model_response()`) and single (`get_single_response()`) methods
  - Parallel test execution via ThreadPoolExecutor
  - Configurable answer extraction

## Test Execution Flow

```
Input: Baseline problem + Ground truth answer
  │
  ├─→ Initialize m-program: MelleaModelClientAdapter(m_program, m_session)
  │
  ├─→ run_benchdrift_pipeline(..., config_overrides={...})
  │
  ├─→ Test Stage 1: Generate variations
  │   └─→ result.json: [baseline, variant1, variant2, ...]
  │
  ├─→ Test Stage 2: Execute m-program on each test case
  │   └─→ Adapter calls m_program for each variation
  │   └─→ Collect m-program answers
  │   └─→ result.json updated with m-program responses
  │
  ├─→ Test Stage 3: Evaluate m-program performance
  │   └─→ LLM judge compares m-program answers vs ground truth
  │   └─→ Flag consistency patterns
  │   └─→ result.json with drift metrics
  │
  └─→ Output: Complete test dataset
      └─→ analyze_robustness_from_probes(test_results)
          └─→ pass_rate, drift metrics, stability analysis
```

## Test Suite Usage

```python
from mellea import start_session
from mellea_contribs.tools.benchdrift_runner import run_benchdrift_pipeline, analyze_robustness_from_probes

# 1. Initialize m-program
m = start_session(backend_name="ollama", model_id="granite3.3:8b")

# 2. Define m-program
# Note: m_program should take a single string argument (the question/problem)
# The type hint is Callable[[str, Dict[str, Any]], Any] to allow optional
# future extensions, but current usage only requires the string parameter
def m_program(question: str):
    response = m.instruct(description=question, grounding_context={...})
    return response.value if hasattr(response, 'value') else response

# 3. Configure variation types and other settings
# Use config_overrides to customize which semantic variations to generate
# and control all aspects of the pipeline
config = {
    'use_generic': True,              # Generic semantic variations
    'use_cluster_variations': True,   # Cluster-based variations
    'use_persona': False,             # Persona-based variations
    'use_long_context': False,        # Long context variations
    'max_workers': 4,                 # Parallel processing
    'semantic_threshold': 0.35,       # Semantic similarity threshold
}

# 4. Generate robustness test suite
test_suite = run_benchdrift_pipeline(
    baseline_problem="Your problem here",
    ground_truth_answer="Expected answer",
    m_program_callable=m_program,
    mellea_session=m,
    config_overrides=config  # Apply custom configuration
)

# 5. Analyze test results
report = analyze_robustness_from_probes(test_suite)
print(f"M-program pass rate: {report['overall_pass_rate']:.1%}")
print(f"Consistency: {report['drift_analysis']}")
```

## Configuration Options

Customize the robustness testing pipeline using `config_overrides`. All available parameters are documented in `config/benchdrift_config.yaml`:

**Variation Types:**
```python
config = {
    'use_generic': True,              # Generic semantic variations
    'use_cluster_variations': True,   # Cluster-based variations
    'use_persona': False,             # Persona-based variations
    'use_long_context': False,        # Long context variations
}
```

**Models:**
```python
config = {
    'generation_model': 'phi-4',           # Model for generating variations
    'judge_model': 'llama_3_3_70b',        # Model for evaluating answers
    'response_model': 'granite-3-3-8b',    # Default response model
}
```

**Processing:**
```python
config = {
    'max_workers': 8,                 # Parallel workers
    'batch_size': 4,                  # Batch processing size
}
```

See `config/benchdrift_config.yaml` for complete documentation of all parameters.

## Test Example

See `test/test_mprogram_robustness.py` for a complete robustness testing example.

**Why this test file name?** The test uses BenchDrift to evaluate m-program robustness, not to test BenchDrift itself. The name clarifies that the m-program is under test.

Run: `python test/test_mprogram_robustness.py` (requires `RITS_API_KEY` and Ollama server running)

## Advanced Configuration

For complete control over the testing pipeline, see `config/benchdrift_config.yaml` which documents all available parameters including:
- Variation types (generic, cluster, persona, long-context)
- Model selection (generation, judge, response models)
- Processing parameters (workers, batch size, thresholds)
- Evaluation settings (LLM judge, semantic thresholds)

All parameters can be overridden via `config_overrides` in `run_benchdrift_pipeline()`.
