#!/bin/bash
# Change to the script's directory to ensure correct module paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Get the parent directory (kgrag root)
KGRAG_ROOT="$(cd .. && pwd)"

# Create output directories
mkdir -p "$KGRAG_ROOT/data"
mkdir -p "$KGRAG_ROOT/output"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${KGRAG_ROOT}"
export KG_BASE_DIRECTORY="$KGRAG_ROOT"

# Neo4j connection settings (customize if different from defaults)
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

# Disable OpenTelemetry if OTEL collector is not available
export OTEL_SDK_DISABLED=true
echo "=================================================="
echo "KG-RAG Pipeline Execution"
echo "=================================================="
echo "Working directory: $(pwd)"
echo "KG Base directory: $KG_BASE_DIRECTORY"
echo "Dataset directory: $KGRAG_ROOT/dataset"
echo "=================================================="

# Step 1: Create tiny dataset from full dataset
TINY_DATASET="$KGRAG_ROOT/dataset/crag_movie_tiny.jsonl.bz2"
echo ""
echo "Step 1: Creating tiny dataset from full CRAG dataset..."
echo "Creating tiny dataset (first 10 documents)..."
python create_tiny_dataset.py --num-docs 10 --output "$TINY_DATASET"
echo "✓ Tiny dataset created"

# Step 3: Run preprocessing
echo ""
echo "Step 3: Running KG preprocessing..."
uv run --with mellea-contribs run_kg_preprocess.py \
  --input "$TINY_DATASET" \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD" > "$KGRAG_ROOT/output/preprocess_stats.json"

# Step 4: Run KG embedding
echo ""
echo "Step 4: Running KG embedding..."
uv run --with mellea-contribs run_kg_embed.py \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"

# Step 5: Run KG update with tiny dataset
echo ""
echo "Step 5: Running KG update with tiny dataset..."
uv run --with mellea run_kg_update.py --dataset "$TINY_DATASET" \
  --num-workers 32 --queue-size 32 \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"

# Step 6: Run QA with tiny dataset
echo ""
echo "Step 6: Running QA..."
uv run --with mellea-contribs run_qa.py \
  --input "$TINY_DATASET" \
  --output "$KGRAG_ROOT/output/qa_results.jsonl" \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"


# Step 7: Run evaluation
echo ""
echo "Step 7: Running evaluation..."
QA_RESULTS_FILE="$KGRAG_ROOT/output/qa_results.jsonl"
EVAL_OUTPUT_FILE="$KGRAG_ROOT/output/eval_metrics.json"
if [ -f "$QA_RESULTS_FILE" ]; then
    uv run --with mellea-contribs run_eval.py \
      --input "$QA_RESULTS_FILE" \
      --output "$EVAL_OUTPUT_FILE"
else
    echo "Warning: QA results file not found at $QA_RESULTS_FILE"
fi

echo ""
echo "=================================================="
echo "Pipeline execution completed successfully!"
echo "=================================================="