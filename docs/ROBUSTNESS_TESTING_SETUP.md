# Robustness Testing Installation Guide

Quick setup guide for running robustness tests on Mellea m-programs.

## Prerequisites

- **Python 3.10+**
- **Git**
- **Ollama** - Download from https://ollama.com/download
- **IBM GitHub Access** - Required for BenchDrift repository

## Installation Steps

### 1. Clone and Install BenchDrift

```bash
git clone https://github.ibm.com/Granite-debug/BenchDrift.git BenchDrift-Pipeline
cd BenchDrift-Pipeline
pip install -e .
cd ..
```

**Note:** BenchDrift is currently IBM-internal. Contact your administrator if you don't have access.

### 2. Clone and Install Mellea

```bash
git clone https://github.com/generative-computing/mellea.git
cd mellea
pip install -e .
cd ..
```

### 3. Install Mellea-Contribs

```bash
git clone https://github.com/generative-computing/mellea-contribs.git
cd mellea-contribs
pip install -e .
```

### 4. Set Environment Variable

```bash
export RITS_API_KEY="your-api-key-here"
```

Make it permanent by adding to your shell profile (`~/.bashrc` or `~/.zshrc`).

### 5. Setup Ollama

```bash
# Start Ollama server (in separate terminal or background)
ollama serve

# Pull required model
ollama pull granite3.3:8b
```

## Verify Installation

```bash
# Test imports
python -c "from benchdrift.pipeline.unified_batched_pipeline_semantic import UnifiedBatchedPipeline; print('BenchDrift: OK')"
python -c "from mellea import start_session; print('Mellea: OK')"
python -c "from mellea_contribs.tools.benchdrift_runner import run_benchdrift_pipeline; print('Mellea-Contribs: OK')"
```

## Run Tests

```bash
cd mellea-contribs
python test/test_mprogram_robustness.py
```

## Troubleshooting

**BenchDrift import fails:**
- Ensure you have access to IBM's internal GitHub
- Try reinstalling: `cd BenchDrift-Pipeline && pip install -e .`

**RITS_API_KEY not set:**
```bash
export RITS_API_KEY="your-key"
```

**Ollama not running:**
```bash
ollama serve
```

**Model not available:**
```bash
ollama pull granite3.3:8b
```

For detailed documentation, see:
- `docs/ROBUSTNESS_TEST_SUITES.md` - Full robustness testing guide
- `config/benchdrift_config.yaml` - Configuration parameters
