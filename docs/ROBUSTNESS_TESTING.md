# Robustness Testing for Mellea M-Programs

Test how consistently your m-program answers semantic variations of a problem. Uses [BenchDrift](https://github.com/IBM/BenchDrift) (`demo-ui` branch) for variation generation. All models run locally via Ollama.

## Setup

```bash
# 1. Install BenchDrift (demo-ui branch)
git clone -b demo-ui https://github.com/IBM/BenchDrift.git BenchDrift-Pipeline
cd BenchDrift-Pipeline && pip install -e . && cd ..

# 2. Install Mellea + Mellea-Contribs
pip install -e path/to/mellea
pip install -e path/to/mellea-contribs

# 3. Ollama
ollama serve
ollama pull qwen3:8b         # variation generation
ollama pull qwen2.5:3b       # m-program backend (or any model you want to test)
```

## Run

```bash
cd mellea-contribs

python test/test_mprogram_robustness.py --backend-model qwen2.5:3b --top-k 5 --no-enrich
```

## CLI Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--backend-model` | No | `granite3.3:8b` | Ollama model for the Mellea m-program (model under test) |
| `--gen-model` | No | `qwen3:8b` | Ollama model for generating prompt variations |
| `--top-k` | No | `10` | Number of ranked transformations to generate |
| `--use-axes` | No | 5 core axes | Taxonomy axes (comma-separated, or `all`) |
| `--no-enrich` | No | off | Skip LLM feature enrichment (recommended for speed) |

**Mandatory prerequisites:** Ollama running (`ollama serve`) with the required models pulled.

**Recommended for quick testing:** `--top-k 5 --no-enrich`

## Expected Output

Running `python test/test_mprogram_robustness.py --backend-model qwen2.5:3b --top-k 5 --no-enrich`:

```
BenchDrift — M-Program Robustness Test
──────────────────────────────────────────────────────────────────────
  Variation model  : qwen3:8b  (generates prompt variations)
  Backend model    : qwen2.5:3b  (Mellea m-program, model under test)
  Ground truth     : $351
──────────────────────────────────────────────────────────────────────
Results
──────────────────────────────────────────────────────────────────────
  Baseline                        $330                  FAIL
  interrogative_expansion         $1044                 FAIL
  politeness_variation            $554.80               FAIL
  scale_extremes                  000                   FAIL
  causal_framing                  $351                  PASS
  emotional_bias_injection        $446                  FAIL
──────────────────────────────────────────────────────────────────────
  Pass rate: 20%  (1/5)  |  baseline: FAIL  |  calls: 6
  Results saved: logs/robustness_20260305_122542.json
──────────────────────────────────────────────────────────────────────
```

Results are saved as JSON in `logs/` after each run, including all variation texts, m-program answers, and pass/fail status. Use these to inspect which prompt phrasings the m-program handles correctly and which it doesn't.

## Programmatic Usage

```python
from mellea import start_session
from mellea_contribs.tools.benchdrift_runner import run_benchdrift_pipeline, analyze_robustness

m = start_session(backend_name="ollama", model_id="qwen2.5:3b")

def m_program(question):
    response = m.instruct(question)
    return response.value if hasattr(response, 'value') else response

probes = run_benchdrift_pipeline(
    baseline_problem="Your problem here",
    ground_truth_answer="Expected answer",
    m_program_callable=m_program,
    mellea_session=m,
    config_overrides={'gen_model': 'qwen3:8b', 'top_k': 5},
)

report = analyze_robustness(probes)
print(f"Pass rate: {report['pass_rate']:.0%}")
```
