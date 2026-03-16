#!/bin/bash
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
Usage: $0 [STEPS...]

Run the full KG-RAG pipeline or individual steps.

STEPS (space-separated, default: all):
  0   Create tiny dataset
  1   Load movie database into Neo4j
  2   Compute entity embeddings
  3   Update KG with documents
  4   Run QA on tiny dataset

Examples:
  $0              # run all steps
  $0 1 2          # run only steps 1 and 2
  $0 4            # run only the QA step
EOF
  exit 0
}

# Parse arguments — collect requested steps (or default to all)
STEPS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage ;;
    [0-4]) STEPS+=("$arg") ;;
    *) echo "Unknown argument: $arg"; usage ;;
  esac
done

# If no steps given, run all
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(0 1 2 3 4)
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

# LLM session — set API_BASE / API_KEY / MODEL_NAME before running this script:
#   export API_BASE=https://your-rits-endpoint/v1
#   export API_KEY=your-api-key
#   export MODEL_NAME=meta-llama/llama-3-70b-instruct
# Or place them in docs/examples/kgrag/.env (loaded automatically by the scripts).
#
# NOTE: Do NOT export API_BASE/API_KEY/MODEL_NAME as empty strings here.
# Exporting a value (even an empty one) would prevent the scripts' dotenv
# loader from filling them in from the .env file (load_dotenv uses
# override=False by default).  Leave these unset so that the .env file
# (at docs/examples/kgrag/.env) is the authoritative source.
#
# The Python scripts fall back to 'gpt-4o-mini' when MODEL_NAME is unset.
#
# Optional separate eval model: EVAL_API_BASE / EVAL_API_KEY / EVAL_MODEL_NAME
# Optional embeddings:          EMB_API_BASE  / EMB_API_KEY  / EMB_MODEL_NAME

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
echo "Dataset directory: $KGRAG_ROOT/dataset"
echo "Steps to run:      ${STEPS[*]}"
echo "=================================================="

# ---------------------------------------------------------------------------
# Step 0: Create tiny dataset
# ---------------------------------------------------------------------------
if run_step 0; then
  if [ ! -f "$TINY_DATASET" ]; then
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
# Step 1: Load predefined movie data into Neo4j
# ---------------------------------------------------------------------------
if run_step 1; then
  echo ""
  echo "Step 1: Loading predefined movie database into KG..."
  uv run --with mellea-contribs[kg] run_kg_preprocess.py \
    --data-dir ../dataset/movie \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-user "$NEO4J_USER" \
    --neo4j-password "$NEO4J_PASSWORD" \
    --batch-size 500 > "$KGRAG_ROOT/output/preprocess_stats.json"
  echo "✓ Movie database loaded into Neo4j"
fi

# ---------------------------------------------------------------------------
# Step 2: Compute entity embeddings
# ---------------------------------------------------------------------------
if run_step 2; then
  echo ""
  echo "Step 2: Running KG embedding on loaded entities..."
  uv run --with mellea-contribs[kg] run_kg_embed.py \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-user "$NEO4J_USER" \
    --neo4j-password "$NEO4J_PASSWORD" \
    --batch-size 100 > "$KGRAG_ROOT/output/embedding_stats.json"
  echo "✓ Entity embeddings computed"
fi

# ---------------------------------------------------------------------------
# Step 3: Update KG with documents
# ---------------------------------------------------------------------------
if run_step 3; then
  echo ""
  echo "Step 3: Updating Knowledge Graph with documents..."
  uv run --with mellea-contribs[kg] run_kg_update.py \
    --dataset "$TINY_DATASET" \
    --domain movie \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-user "$NEO4J_USER" \
    --neo4j-password "$NEO4J_PASSWORD" \
    --num-workers 10 \
    --verbose > "$KGRAG_ROOT/output/update_stats.json"
  echo "✓ Knowledge Graph updated with documents"
fi

# ---------------------------------------------------------------------------
# Step 4: Run QA on tiny dataset
# ---------------------------------------------------------------------------
if run_step 4; then
  echo ""
  echo "Step 4: Running QA on tiny dataset..."
  uv run --with mellea-contribs[kg] run_qa.py \
    --dataset "$TINY_DATASET" \
    --output "$KGRAG_ROOT/output/qa_results.jsonl" \
    --progress "$KGRAG_ROOT/output/qa_progress.json" \
    --reset-progress \
    --domain movie \
    --routes 3 \
    --width 30 \
    --depth 3 \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-user "$NEO4J_USER" \
    --neo4j-password "$NEO4J_PASSWORD"
  echo "✓ QA completed"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "✅ KG-RAG Pipeline Execution Completed!"
echo "=================================================="
echo "Steps run: ${STEPS[*]}"
echo ""
echo "Neo4j is running at: $NEO4J_URI"
echo "Logs saved to: $KGRAG_ROOT/output/"
echo "  - preprocess_stats.json  (step 1)"
echo "  - embedding_stats.json   (step 2)"
echo "  - update_stats.json      (step 3)"
echo "  - qa_results.jsonl       (step 4)"
echo "  - qa_progress.json       (step 4)"
echo "=================================================="
