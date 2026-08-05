# Instruction Replacement for Mellea M-Programs

Discover validated alternative instructions by generating semantic variations and identifying which alternative phrasings work reliably with your m-program.

## Setup & Installation

### Step 1: Install BenchDrift
Install BenchDrift from source (required for instruction replacement pipeline):
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

Find validated alternative instructions by running your m-program against semantic variations of your baseline problem. This produces a dataset of alternative phrasings that are semantically similar to the original but have been confirmed by the pipeline to yield the correct answer.

## How It Works

1. **Generate test variations**: Create semantic variations of your problem (different phrasings, same meaning)
2. **Test m-program**: Execute m-program on original problem + all variations to collect answers
3. **Evaluate answers**: Compare m-program's answers to ground truth across all variants
4. **Extract alternatives**: Collect all variations where m-program answered correctly

## Test Suite Architecture

```
Instruction Replacement Discovery Process
═══════════════════════════════════════════

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
    └─ Compare m-program answers to ground truth
       ├─ Which variations did m-program answer correctly?
       └─ Mark as validated alternatives

    ▼

Extract Validated Alternatives
    │
    └─ All variations where m-program produced correct answers
       (Ready to use as alternative instructions)
```

## Core Tools

### 1. `benchdrift_runner.py`

**Primary toolkit for extracting alternative instructions.**

Uses [BenchDrift](https://github.com/ritterinvest/BenchDrift) for variation generation and evaluation orchestration.

- `run_benchdrift_pipeline()`: Generate and execute complete test suite
  - Input: baseline problem + ground truth answer
  - Output: Complete test dataset with variations + m-program answers
  - Returns: All test cases with m-program responses and correctness flags
  - **New feature**: `variation_types` parameter to customize which variation types to use

- `extract_replacement_instructions()`: Extract validated alternative instructions
  - Returns test cases where m-program consistently succeeds
  - Dataset: Which alternative phrasings work reliably?
  - Only includes variations where m-program answered correctly

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
  │   └─→ Mark which variations were answered correctly
  │   └─→ result.json with correctness flags
  │
  └─→ Extract Validated Alternatives
      └─→ extract_replacement_instructions(test_results)
          └─→ List of alternative instructions that work
```

## Test Suite Usage

```python
from mellea import start_session
from mellea_contribs.tools.benchdrift_runner import (
    run_benchdrift_pipeline,
    extract_replacement_instructions
)

# 1. Initialize m-program
m = start_session(backend_name="ollama", model_id="granite3.3:8b")

# 2. Define m-program
def m_program(question: str):
    response = m.instruct(
        description=question,
        grounding_context={"rules": context}
    )
    return response.value if hasattr(response, 'value') else response

# 3. Configure variation types (NEW FEATURE)
variation_types = {
    'generic': True,              # Generic semantic variations
    'cluster_variations': True,   # Cluster-based variations
    'persona': False,             # Persona-based variations
    'long_context': False         # Long context variations
}

# 4. Generate test suite
test_suite = run_benchdrift_pipeline(
    baseline_problem="Your problem here",
    ground_truth_answer="Expected answer",
    m_program_callable=m_program,
    mellea_session=m,
    max_workers=4,
    variation_types=variation_types
)

# 5. Extract validated alternative instructions
alternatives = extract_replacement_instructions(test_suite)

print(f"Found {len(alternatives)} validated alternative instructions:")
for i, alt in enumerate(alternatives, 1):
    print(f"  {i}. {alt[:100]}...")
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

You can enable/disable each variation type independently to focus on finding specific types of alternative instructions.

## Test Example

See `test/2_test_instruction_replacement.py` for a complete instruction replacement example.

Run: `python test/2_test_instruction_replacement.py` (requires `RITS_API_KEY`)

## Use Cases

- **Prompt engineering**: Discover which alternative phrasings work best with your m-program
- **Robustness validation**: Verify your instruction works across semantically similar variations
- **Instruction library**: Build a collection of validated alternative instructions for your m-program
- **Language variants**: Find instructions in different styles/tones that achieve the same results

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
