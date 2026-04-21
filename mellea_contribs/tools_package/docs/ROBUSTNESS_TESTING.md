# Robustness Testing for Mellea M-Programs

Test how consistently your m-program answers semantic variations of a problem. Uses [BenchDrift](https://github.com/IBM/BenchDrift) (`demo-ui` branch) for variation generation.

## Setup

```bash
# 1. Install BenchDrift (demo-ui branch)
git clone -b demo-ui https://github.com/IBM/BenchDrift.git BenchDrift-Pipeline
cd BenchDrift-Pipeline && pip install -e . && cd ..

# 2. Install Mellea + Mellea-Contribs
pip install -e path/to/mellea
pip install -e "path/to/mellea-contribs[robustness]"

# 3. Ollama (for m-program backend)
ollama serve
ollama pull qwen2.5:3b       # m-program backend (or any model you want to test)

# 4. (Optional) For fast variation generation via Groq
export GROQ_API_KEY=gsk_your_key_here    # get from https://console.groq.com/keys
```

## Run

```bash
cd mellea-contribs

# Using Groq for fast variation generation (recommended)
python test/test_mprogram_robustness.py --backend-model qwen2.5:3b --gen-model groq/llama-3.3-70b-versatile --top-k 5 --no-enrich

# Using Ollama only (slower, no API key needed)
python test/test_mprogram_robustness.py --backend-model qwen2.5:3b --gen-model qwen3:8b --top-k 5 --no-enrich
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend-model` | `granite3.3:8b` | Ollama model for the Mellea m-program (model under test) |
| `--gen-model` | `qwen3:8b` | Model for generating variations. Supports `client/model` format |
| `--judge-model` | same as gen-model | Model for validation + LLM judge evaluation |
| `--top-k` | `10` | Number of ranked transformations to generate |
| `--use-axes` | 5 core axes | Taxonomy axes (comma-separated, or `all`) |
| `--no-enrich` | off | Skip LLM feature enrichment (recommended for speed) |
| `--skip-validation` | off | Skip semantic validation of variations |
| `--use-llm-judge` | off | Use LLM judge for answer evaluation (slower, more accurate) |

**Model format:** `--gen-model`, `--judge-model`, and `--backend-model` accept `client/model` format. Supported clients: `ollama`, `groq`, `rits`, `vllm`, `openai`.
- `qwen3:8b` → Ollama local (default)
- `glm-5:cloud` → Ollama cloud
- `groq/llama-3.3-70b-versatile` → Groq cloud

You can mix clients — e.g., Groq for fast variation generation + Ollama for m-program testing:
```bash
--gen-model groq/llama-3.3-70b-versatile --backend-model qwen2.5:3b
```

**Recommended Groq models for variation generation:**
- `groq/llama-3.3-70b-versatile` — fast, good quality
- `groq/llama-3.1-8b-instant` — fastest
- `groq/qwen-qwq-32b` — strong reasoning
- `groq/gemma2-9b-it` — compact, good quality

Requires `GROQ_API_KEY` env var. Free at https://console.groq.com/keys

## Expected Output

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

Results are saved as JSON in `logs/` after each run, including all variation texts, m-program answers, and pass/fail status.

## Note on the Variation Model

The variation model (`--gen-model`) generates semantic variations of your problem. It should ideally be more capable than the model under test — it needs to understand the problem well enough to rephrase it while preserving the answer. Using Groq with a 70B model is recommended for both speed and quality.
