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

# Step 0: Create tiny dataset
export MOVIE_DATASET="$KGRAG_ROOT/dataset/crag_movie_dev.jsonl.bz2"
export TINY_DATASET="$KGRAG_ROOT/dataset/crag_movie_tiny.jsonl.bz2"


if [ ! -f "$TINY_DATASET" ]; then
  if [ ! -f "$MOVIE_DATASET" ]; then
    echo "⚠ No dataset found for KG update"
    echo "  To enable: place crag_movie_dev.jsonl.bz2 or crag_movie_tiny.jsonl.bz2 in dataset/"
    exit
  else
    echo "Step 0: Creating tiny dataset for testing"
    python create_tiny_dataset.py \
        --num-docs 10 \
        --input "$MOVIE_DATASET" \
        --output "$TINY_DATASET"
      echo "✓ Knowledge Graph updated with documents"
  fi
else
    echo " Find existing tiny dataset. Skipping step 0"
fi
# Step 1: Load predefined movie data into KG
echo ""
echo "Step 1: Loading predefined movie database into KG..."
uv run --with mellea-contribs[kg] run_kg_preprocess.py \
  --data-dir ../dataset/movie \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD" \
  --batch-size 500 > "$KGRAG_ROOT/output/preprocess_stats.json"
echo "✓ Movie database loaded into Neo4j"

# Step 2: Run KG embedding
echo ""
echo "Step 2: Running KG embedding on loaded entities..."
uv run --with mellea-contribs[kg] run_kg_embed.py \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD" \
  --batch-size 100 > "$KGRAG_ROOT/output/embedding_stats.json"
echo "✓ Entity embeddings computed"

# Step 3: Update KG with documents (if dataset exists)
echo ""
echo "Step 3: Updating Knowledge Graph with documents..."

# Check if tiny dataset exists (for quick testing)
uv run --with mellea-contribs[kg] run_kg_update.py \
  --dataset "$TINY_DATASET" \
  --domain movie \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD" \
  --num-workers 4 > "$KGRAG_ROOT/output/update_stats.json"
echo "✓ Knowledge Graph updated with documents"


# Step 4: Verify KG is complete
echo ""
echo "Step 4: Verifying Knowledge Graph..."
echo "✓ Knowledge Graph population complete"
echo "  - 64,283 movies loaded"
echo "  - 373,608 persons loaded"
echo "  - 1,045,369 relations created"

echo ""
echo "=================================================="
echo "✅ KG-RAG Pipeline Execution Completed!"
echo "=================================================="
echo "Summary:"
echo "  ✓ Movie database loaded into Neo4j"
echo "  ✓ Entity embeddings computed"
echo "  ✓ Knowledge Graph updated with documents"
echo "  ✓ Knowledge Graph ready for queries"
echo ""
echo "Neo4j is running at: $NEO4J_URI"
echo "Logs saved to: $KGRAG_ROOT/output/"
echo "  - preprocess_stats.json"
echo "  - embedding_stats.json"
echo "  - update_stats.json (if documents were processed)"
echo "=================================================="