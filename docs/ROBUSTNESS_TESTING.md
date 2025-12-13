# Robustness Testing for Mellea M-Programs

Evaluate m-program consistency by testing against semantic variations of a baseline problem and measuring how reliably your m-program answers them.

## Setup & Installation

### Step 1: Install BenchDrift
Install BenchDrift from source (required for robustness testing pipeline):
```bash
git clone https://github.com/ritterinvest/BenchDrift.git
cd BenchDrift
pip install -e .
cd ..
```

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

Uses [BenchDrift](https://github.com/ritterinvest/BenchDrift) for variation generation and evaluation orchestration.

- `run_benchdrift_pipeline()`: Generate and execute complete test suite
  - Input: baseline problem + ground truth answer
  - Output: Complete test dataset with variations + m-program answers
  - Returns: All test cases with m-program responses and consistency metrics
  - **New feature**: `variation_types` parameter to customize which variation types to use

- `analyze_robustness_from_probes()`: Compute robustness metrics from test results
  - Measures m-program pass rate across all test variations
  - Reports consistency metrics (how stable is m-program?)
  - Identifies failure patterns (where does m-program break?)

### 2. `mellea_model_client_adapter.py`

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
  ├─→ run_benchdrift_pipeline(..., variation_types={...})
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
def m_program(question: str):
    response = m.instruct(description=question, grounding_context={...})
    return response.value if hasattr(response, 'value') else response

# 3. Configure variation types (NEW FEATURE)
variation_types = {
    'generic': True,              # Generic semantic variations
    'cluster_variations': True,   # Cluster-based variations
    'persona': False,             # Persona-based variations
    'long_context': False         # Long context variations
}

# 4. Generate robustness test suite
test_suite = run_benchdrift_pipeline(
    baseline_problem="Your problem here",
    ground_truth_answer="Expected answer",
    m_program_callable=m_program,
    mellea_session=m,
    max_workers=4,
    variation_types=variation_types
)

# 5. Analyze test results
report = analyze_robustness_from_probes(test_suite)
print(f"M-program pass rate: {report['overall_pass_rate']:.1%}")
print(f"Consistency: {report['drift_analysis']}")
```

## Variation Types Configuration

The new `variation_types` parameter allows you to customize which semantic variations to generate:

```python
variation_types = {
    'generic': True,              # Enable generic semantic variations
    'cluster_variations': True,   # Enable cluster-based variations
    'persona': False,             # Disable persona-based variations
    'long_context': False         # Disable long context variations
}
```

You can enable/disable each variation type independently to focus your robustness testing on specific aspects.

## Test Example

See `test/1_test_robustness_testing.py` for a complete robustness testing example.

Run: `python test/1_test_robustness_testing.py` (requires `RITS_API_KEY`)

## Test Suite Configuration

Customize test generation via `config_overrides` in `run_benchdrift_pipeline()`:
- Test models: `generation_model`, `response_model`, `judge_model`
- Evaluation: `semantic_threshold`, `use_llm_judge`
- Parallelization: `max_workers`

Example:
```python
config = {
    'semantic_threshold': 0.4,
    'max_workers': 8
}
test_suite = run_benchdrift_pipeline(..., config_overrides=config)
```
