#!/bin/bash
# run.sh — Orchestrates the five-stage KG-RAG pipeline for the movie domain.
#
# Before running, configure your LLM and graph database:
#   cp ../env_template ../.env && editor ../.env
#
# Requires: Python env with mellea-contribs[kg] installed, and a running
# graph database (or pass --mock for local testing without any database).
set -e  # Exit the script (not the terminal) on any command failure

# Change to the script's directory to ensure correct module paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Get the parent directory (kgrag root)
KGRAG_ROOT="$(cd .. && pwd)"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $0 [--tiny|--full] [STEPS...]

Run the full KG-RAG pipeline or individual steps.

Dataset mode (default: --tiny):
  --tiny   Use the tiny dataset (10 docs); step 0 creates it if missing.
  --full   Use the full dataset (crag_movie_dev.jsonl.bz2); skips step 0.

STEPS (space-separated, default: all for the chosen mode):
  0   Create tiny dataset       (tiny mode only)
  1   Load movie database into the graph database
  2   Compute entity embeddings
  3   Update KG with documents
  4   Run QA
  5   Evaluate QA results (LLM judge)

Examples:
  $0                  # tiny mode, run all steps
  $0 --full           # full dataset, run steps 1-5
  $0 --tiny 3 4 5     # tiny mode, run only steps 3-5
  $0 --full 4 5       # full dataset, run only steps 4-5
  $0 1 2              # tiny mode, run only steps 1 and 2
EOF
  exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
USE_TINY=true
STEPS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage ;;
    --tiny) USE_TINY=true ;;
    --full) USE_TINY=false ;;
    [0-5]) STEPS+=("$arg") ;;
    *) echo "Unknown argument: $arg"; usage ;;
  esac
done

# Default steps based on mode
if [ ${#STEPS[@]} -eq 0 ]; then
  if $USE_TINY; then
    STEPS=(0 1 2 3 4 5)
  else
    STEPS=(1 2 3 4 5)
  fi
fi

# Helper: check if a step is requested
run_step() {
  local step="$1"
  for s in "${STEPS[@]}"; do
    [ "$s" = "$step" ] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
mkdir -p "$KGRAG_ROOT/data"
mkdir -p "$KGRAG_ROOT/output"

export PYTHONPATH="${PYTHONPATH}:${KGRAG_ROOT}"
export KG_BASE_DIRECTORY="$KGRAG_ROOT"

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

export OTEL_SDK_DISABLED=true

export MOVIE_DATASET="$KGRAG_ROOT/dataset/crag_movie_dev.jsonl.bz2"
export TINY_DATASET="$KGRAG_ROOT/dataset/crag_movie_tiny.jsonl.bz2"

# Set the active dataset based on mode
if $USE_TINY; then
  ACTIVE_DATASET="$TINY_DATASET"
  DATASET_LABEL="tiny"
else
  ACTIVE_DATASET="$MOVIE_DATASET"
  DATASET_LABEL="full"
fi

# LLM endpoint: set API_BASE / API_KEY / MODEL_NAME in ../.env (or export them
# before running).  Do NOT export these as empty strings — empty exports shadow
# the .env values.  The scripts fall back to 'gpt-4o-mini' when MODEL_NAME is
# unset.  Optional: EVAL_API_BASE/EVAL_MODEL_NAME, EMB_API_BASE/EMB_MODEL_NAME.

if [ -z "${API_BASE:-}" ]; then
  echo "⚠  WARNING: API_BASE is not set in the environment."
  echo "   Steps 3 and 4 (KG update + QA) require an LLM endpoint."
  echo "   Either export API_BASE / API_KEY / MODEL_NAME before running,"
  echo "   or create $KGRAG_ROOT/.env with those values."
  echo ""
fi

echo "=================================================="
echo "KG-RAG Pipeline Execution"
echo "=================================================="
echo "Working directory: $(pwd)"
echo "KG Base directory: $KG_BASE_DIRECTORY"
echo "Dataset mode:      $DATASET_LABEL ($ACTIVE_DATASET)"
echo "Steps to run:      ${STEPS[*]}"
echo "=================================================="

# ---------------------------------------------------------------------------
# Step 0: Create tiny dataset (tiny mode only)
# ---------------------------------------------------------------------------
if run_step 0; then
  if ! $USE_TINY; then
    echo ""
    echo "Step 0: Skipped — running in full dataset mode"
  elif [ ! -f "$TINY_DATASET" ]; then
    if [ ! -f "$MOVIE_DATASET" ]; then
      echo "⚠ No dataset found for KG update"
      echo "  To enable: place crag_movie_dev.jsonl.bz2 or crag_movie_tiny.jsonl.bz2 in dataset/"
      exit 1
    else
      echo ""
      echo "Step 0: Creating tiny dataset for testing..."
      python create_tiny_dataset.py \
          --num-docs 10 \
          --input "$MOVIE_DATASET" \
          --output "$TINY_DATASET"
      echo "✓ Tiny dataset created"
    fi
  else
    echo ""
    echo "Step 0: Skipped — tiny dataset already exists"
  fi
fi

# ---------------------------------------------------------------------------
# Step 1: Load predefined movie data into the graph database
# ---------------------------------------------------------------------------
if run_step 1; then
  echo ""
  echo "Step 1: Loading predefined movie database into KG..."
  uv run --with mellea-contribs[kg] run_kg_preprocess.py \
    --data-dir ../dataset/movie \
    --db-uri "$NEO4J_URI" \
    --db-user "$NEO4J_USER" \
    --db-password "$NEO4J_PASSWORD" \
    --batch-size 500 > "$KGRAG_ROOT/output/preprocess_stats.json"
  echo "✓ Movie database loaded"
fi

# ---------------------------------------------------------------------------
# Step 2: Compute entity embeddings
# ---------------------------------------------------------------------------
if run_step 2; then
  echo ""
  echo "Step 2: Running KG embedding on loaded entities..."
  uv run --with mellea-contribs[kg] run_kg_embed.py \
    --db-uri "$NEO4J_URI" \
    --db-user "$NEO4J_USER" \
    --db-password "$NEO4J_PASSWORD" \
    --batch-size 100 > "$KGRAG_ROOT/output/embedding_stats.json"
  echo "✓ Entity embeddings computed"
fi

# ---------------------------------------------------------------------------
# Step 3: Update KG with documents
# ---------------------------------------------------------------------------
if run_step 3; then
  echo ""
  echo "Step 3: Updating Knowledge Graph with documents ($DATASET_LABEL)..."
  uv run --with mellea-contribs[kg] run_kg_update.py \
    --dataset "$ACTIVE_DATASET" \
    --domain movie \
    --db-uri "$NEO4J_URI" \
    --db-user "$NEO4J_USER" \
    --db-password "$NEO4J_PASSWORD" \
    --num-workers 10 \
    --verbose > "$KGRAG_ROOT/output/update_stats.json"
  echo "✓ Knowledge Graph updated with documents"
fi

# ---------------------------------------------------------------------------
# Step 4: Run QA
# ---------------------------------------------------------------------------
if run_step 4; then
  echo ""
  echo "Step 4: Running QA ($DATASET_LABEL)..."
  uv run --with mellea-contribs[kg] run_qa.py \
    --dataset "$ACTIVE_DATASET" \
    --output "$KGRAG_ROOT/output/qa_results.jsonl" \
    --progress "$KGRAG_ROOT/output/qa_progress.json" \
    --reset-progress \
    --domain movie \
    --routes 3 \
    --width 30 \
    --depth 3 \
    --db-uri "$NEO4J_URI" \
    --db-user "$NEO4J_USER" \
    --db-password "$NEO4J_PASSWORD"
  echo "✓ QA completed"
fi

# ---------------------------------------------------------------------------
# Step 5: Evaluate QA results
# ---------------------------------------------------------------------------
if run_step 5; then
  echo ""
  echo "Step 5: Evaluating QA results with LLM judge..."
  uv run --with mellea-contribs[kg] run_eval.py \
    --input "$KGRAG_ROOT/output/qa_results.jsonl" \
    --output "$KGRAG_ROOT/output/eval_results.json" \
    --metrics "$KGRAG_ROOT/output/eval_metrics.json"
  echo "✓ Evaluation completed"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "✅ KG-RAG Pipeline Execution Completed!"
echo "=================================================="
echo "Steps run: ${STEPS[*]}"
echo "Dataset:   $DATASET_LABEL ($ACTIVE_DATASET)"
echo ""
echo "Graph DB is running at: $NEO4J_URI"
echo "Logs saved to: $KGRAG_ROOT/output/"
echo "  - preprocess_stats.json  (step 1)"
echo "  - embedding_stats.json   (step 2)"
echo "  - update_stats.json      (step 3)"
echo "  - qa_results.jsonl       (step 4)"
echo "  - qa_progress.json       (step 4)"
echo "  - eval_results.json      (step 5)"
echo "  - eval_metrics.json      (step 5)"
echo "=================================================="
