#!/usr/bin/env bash
# Run the systemone integration tests against real GLiNER2 via bsub.
#
# Usage:
#   # Submit as a bsub job with GPU:
#   bsub -G grp_data -R "select[ngpus>0] rusage[ngpus_physical=1]" \
#        -o integration_%J.log -e integration_%J.err \
#        bash scripts/run_integration.sh
#
#   # Or run directly on a GPU node:
#   bash scripts/run_integration.sh
#
# Environment variables:
#   CHECKPOINT  — GLiNER2 checkpoint (default: fastino/gliner2.5-small-v1)
#   SKIP_EXAMPLES — set to 1 to skip running the example scripts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "systemone integration test"
echo "=========================================="
echo "Date:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Host:    $(hostname)"
echo "Project: $PROJECT_DIR"

# GPU info
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found — running on CPU (expect slower inference)"
fi

cd "$PROJECT_DIR"

# Use project-local cache to avoid home-dir quota issues
export UV_CACHE_DIR="${PROJECT_DIR}/../.uv_cache"
export UV_HTTP_TIMEOUT=300
mkdir -p "$UV_CACHE_DIR"

# Create or reuse venv
if [ ! -d .venv ]; then
    echo ""
    echo "Creating virtual environment..."
    uv venv --python 3.12 .venv
fi

echo ""
echo "Installing dependencies..."
uv pip install -e ".[local,all]" --group dev 2>&1

# transformers 4.57.x has a tokenizer bug with DeBERTa-v2 checkpoints used by
# GLiNER2: extra_special_tokens passes a list to .keys(). Pin to 4.52.x which
# predates this breakage. Also install protobuf + sentencepiece for the
# sentencepiece tokenizer path that DeBERTa-v2 needs.
echo ""
echo "Applying compatibility pins..."
uv pip install "transformers>=4.50,<4.53" "protobuf>=3.20" "sentencepiece>=0.1.99" 2>&1

echo ""
echo "=========================================="
echo "Running integration tests"
echo "=========================================="

EXTRA_ARGS=""
if [ "${SKIP_EXAMPLES:-0}" = "1" ]; then
    EXTRA_ARGS="--skip-examples"
fi

.venv/bin/python examples/run_all_real.py --provider gliner2 \
    --checkpoint "${CHECKPOINT:-fastino/gliner2.5-small-v1}" \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Also running pytest integration tier"
echo "=========================================="
.venv/bin/python -m pytest -m integration -v --tb=short 2>&1 || true

echo ""
echo "Done. Exit code: $EXIT_CODE"
exit $EXIT_CODE
