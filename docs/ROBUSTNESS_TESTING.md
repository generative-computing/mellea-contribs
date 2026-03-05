# Robustness Testing for Mellea M-Programs

Test how consistently your m-program answers semantic variations of a problem. Uses [BenchDrift](https://github.com/IBM/BenchDrift) (`demo-ui` branch) for variation generation. All models run locally via Ollama.

## Setup

```bash
# 1. Install BenchDrift (demo-ui branch)
git clone -b demo-ui https://github.com/IBM/BenchDrift.git BenchDrift-Pipeline
cd BenchDrift-Pipeline && pip install -e . && cd ..

# 2. Install Mellea + Mellea-Contribs
pip install -e path/to/mellea
pip install -e "path/to/mellea-contribs[robustness]"

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
  Model under test : qwen2.5:3b
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

## Note on the Variation Model

The variation generation model (`--gen-model`, default: `qwen3:8b`) is internal to BenchDrift and generates the semantic variations of your problem. Ideally, this should be a more capable model than the model under test — it needs to understand the problem well enough to rephrase it while preserving the answer. You can change it with `--gen-model`, but the default is a good choice for most cases.
