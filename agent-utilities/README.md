# mellea-contribs-agent-utilities

Agent-side utilities for Mellea: selectors, evaluators, and robustness tools.

This subpackage provides drop-in helpers that an agent can use during
generation, evaluation, or testing of m-programs.

## Modules

| Module | Purpose |
|--------|---------|
| `top_k` | Generic Top-K LLM-as-judge selector. Pick the best K of N candidate items using a comparison prompt. |
| `double_round_robin` | Pairwise tournament selector. Runs A-vs-B and B-vs-A across all pairs and ranks by accumulated wins. |
| `benchdrift_runner` | BenchDrift integration for robustness testing of Mellea m-programs against semantic problem variations. |
| `simbauq` | SIMBA-UQ confidence-aware sampling strategy. Generates samples across temperatures and selects the most confident one via similarity-based uncertainty quantification. |

## Install

```bash
pip install mellea-contribs-agent-utilities

# With BenchDrift robustness extras
pip install "mellea-contribs-agent-utilities[robustness]"

# With SIMBA-UQ sampling extras (scikit-learn, sentence-transformers, tqdm, datasets)
pip install "mellea-contribs-agent-utilities[simbauq]"
```

## Usage

### Top-K selector

```python
from mellea import start_session
from mellea_contribs.agent_utilities.core.top_k import top_k

m = start_session()
items = [
    {"name": "flagd-config", "latency": "500", "severity": "3"},
    {"name": "adService", "latency": "50", "severity": "1"},
    {"name": "payment-svc", "latency": "900", "severity": "4"},
]
results = top_k(
    items=items,
    comparison_prompt="Select the most severe issues. Higher severity and latency are worse.",
    m=m,
    k=2,
)
```

### Double Round Robin

```python
from mellea import start_session
from mellea_contribs.agent_utilities.core.double_round_robin import double_round_robin

m = start_session()
ranked = double_round_robin(
    items=items,
    comparison_prompt="Pick the more likely root cause based on severity and signals.",
    m=m,
)
for item, score in ranked:
    print(item["name"], score)
```

### SIMBA-UQ sampling

Confidence-aware sample selection. Requires the `simbauq` extra for the
`sbert` metric and the `classifier` confidence method:

```python
from mellea import start_session
from mellea_contribs.agent_utilities.core.simbauq import SIMBAUQSamplingStrategy

m = start_session()
result = m.instruct(
    "What is the capital of France?",
    strategy=SIMBAUQSamplingStrategy(
        temperatures=[0.3, 0.5, 0.7, 1.0],
        n_per_temp=3,
        similarity_metric="rouge",
        confidence_method="aggregation",
        aggregation="mean",
    ),
    return_sampling_results=True,
)

best = result.result
print(best._meta["simba_uq"]["confidence"], str(best))
```

See `docs/simbauq.mdx` and `examples/simbauq/` for the full guide.

### BenchDrift robustness

Install with the `robustness` extra and have an Ollama server running:

```bash
pip install "mellea-contribs-agent-utilities[robustness]"
ollama serve
ollama pull qwen3:8b
ollama pull granite3.3:8b
```

```python
from mellea_contribs.agent_utilities.core.benchdrift_runner import (
    run_benchdrift_pipeline,
    analyze_robustness,
)

probes = run_benchdrift_pipeline(
    baseline_problem="...",
    ground_truth_answer="$351",
    m_program_callable=my_m_program,
    mellea_session=m,
    config_overrides={"top_k": 5, "no_enrich": True},
)
report = analyze_robustness(probes)
print(report["pass_rate"])
```

See `docs/ROBUSTNESS_TESTING.md` for the full robustness testing guide.

## Development

```bash
uv sync
uv run pytest
```

## License

Apache-2.0.
