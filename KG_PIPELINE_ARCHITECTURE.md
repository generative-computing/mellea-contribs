# KG-RAG Pipeline Architecture

## Overview
The KG-RAG pipeline consists of three main scripts that work together to build, embed, and update a knowledge graph.

## Three-Stage Pipeline

### Stage 1: Preprocessing (`run_kg_preprocess.py`)
**Purpose**: Load predefined data into Neo4j knowledge graph

**Input**:
- movie_db.json (64,283 movies)
- person_db.json (373,608 persons)
- year_db.json (year data)

**Processing**:
1. Load JSON files
2. Batch insert entities (movies, persons, genres)
3. Batch insert relations (ACTED_IN, DIRECTED, BELONGS_TO_GENRE)

**Output**:
- Neo4j database with 437,891 entities, 1,045,369 relations
- Statistics JSON: preprocess_stats.json

**Architecture**:
```
PredefinedDataPreprocessor
├─ load_json_file()
├─ _insert_movies() → Batch insert via Cypher UNWIND
├─ _insert_persons() → Batch insert via Cypher UNWIND
└─ _insert_movie_relations() → Batch insert via Cypher UNWIND
```

**Configuration**:
- `--data-dir`: Dataset directory (required)
- `--batch-size`: Batch size for inserts (default: 50 entities, 500 relations)
- `--neo4j-uri`, `--neo4j-user`, `--neo4j-password`: Connection settings
- `--mock`: Use mock backend for testing

### Stage 2: Embedding (`run_kg_embed.py`)
**Purpose**: Compute embeddings for entities and relations, store in Neo4j with vector indices

**Input**:
- Neo4j KG from preprocessing
- Embedding model (default: text-embedding-3-small, 1536 dims)

**Processing**:
1. Fetch all entities from Neo4j (Cypher query)
2. Fetch all relations from Neo4j (Cypher query)
3. Embed entities using KGEmbedder
4. Embed relations
5. Store embeddings back to Neo4j
6. Create vector indices for similarity search

**Output**:
- Stored embeddings in Neo4j (entity_embedding, relation_embedding properties)
- Vector indices for cosine similarity search
- Statistics JSON: embedding_stats.json

**Architecture**:
```
ComprehensiveKGEmbedder
├─ _fetch_entities_from_neo4j() → Cypher query
├─ _fetch_relations_from_neo4j() → Cypher query
├─ _embed_batch() → KGEmbedder.embed_entity()
├─ _store_entity_embeddings() → Cypher UNWIND batch insert
├─ _store_relation_embeddings() → Cypher UNWIND batch insert
└─ _create_vector_indices() → Create Neo4j indices
```

**Configuration**:
- `--model`: Embedding model (default: text-embedding-3-small)
- `--dimension`: Embedding dimension (default: 1536)
- `--batch-size`: Batch size for embedding (default: 100)
- `--neo4j-uri`, `--neo4j-user`, `--neo4j-password`: Connection settings
- `--mock`: Use mock backend

### Stage 3: Update (`run_kg_update.py`)
**Purpose**: Process documents, extract entities/relations, and update KG

**Input**:
- Documents in JSONL/JSONL.bz2 format (e.g., crag_movie_dev.jsonl.bz2)
- KG from preprocessing (for alignment)

**Processing**:
1. Load documents from JSONL
2. Extract entities and relations from each document (LLM-based)
3. Align extracted entities with KG entities
4. Decide entity merges
5. Align extracted relations with KG relations
6. Decide relation merges
7. Track progress and statistics

**Output**:
- Updated Neo4j KG with new entities/relations
- Progress file: update_kg_progress.json
- Statistics JSON: aggregated results

**Architecture**:
```
KG Update Pipeline
├─ SessionConfig: LLM settings
├─ UpdaterConfig: num_workers, loop budgets, alignment options
├─ DatasetConfig: dataset path, domain, progress path
│
└─ Async Processing
   ├─ orchestrate_kg_update()
   │  ├─ extract_entities_and_relations() (Layer 3 generative)
   │  ├─ align_entity_with_kg() (Layer 3 generative)
   │  ├─ decide_entity_merge() (Layer 3 generative)
   │  ├─ align_relation_with_kg() (Layer 3 generative)
   │  └─ decide_relation_merge() (Layer 3 generative)
   │
   └─ Process Multiple Documents
      ├─ Semaphore (num_workers) concurrency control
      ├─ Per-document LLM calls
      ├─ Progress tracking
      └─ Error handling with per-item tracking
```

**Configuration**:
```
SessionConfig:
  - model: LLM model to use (default: gpt-4o-mini)

UpdaterConfig:
  - num_workers: Concurrent document processors (default: 64)
  - queue_size: Data loading queue size (default: 64)
  - extraction_loop_budget: LLM refinement iterations (default: 3)
  - alignment_loop_budget: Alignment refinement iterations (default: 2)
  - align_topk: Top-K candidates for alignment (default: 10)
  - align_entity, merge_entity, align_relation, merge_relation: Behavior flags

DatasetConfig:
  - dataset_path: Path to JSONL/JSONL.bz2
  - domain: Knowledge domain (e.g., "movie")
  - progress_path: Progress file location
```

## End-to-End Pipeline Execution

### Commands
```bash
# Stage 1: Preprocess (load predefined data)
python run_kg_preprocess.py \
  --data-dir ../dataset/movie \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password

# Stage 2: Embed (compute and store embeddings)
python run_kg_embed.py \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password \
  --model text-embedding-3-small

# Stage 3: Update (process documents and update KG)
python run_kg_update.py \
  --dataset data/corpus.jsonl.bz2 \
  --domain movie \
  --model gpt-4o-mini \
  --num-workers 64 \
  --extraction-loop-budget 3 \
  --alignment-loop-budget 2 \
  --align-topk 10
```

### Full Script (run.sh)
```bash
#!/bin/bash
# Step 1: Load predefined movie data
python run_kg_preprocess.py \
  --data-dir ../dataset/movie \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"

# Step 2: Embed entities and relations
python run_kg_embed.py \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"

# Step 3: (Optional) Update with new documents
python run_kg_update.py \
  --dataset data/corpus.jsonl.bz2 \
  --domain movie
```

## Common Patterns Across Scripts

### 1. Configuration Management
All scripts follow mellea's pattern:

**Preprocessing**:
- Simple flat config (data loading specific)

**Embedding**:
- Model and batch configuration
- Backend specification

**Updating**:
- SessionConfig (LLM settings)
- UpdaterConfig (processing settings)
- DatasetConfig (data settings)

### 2. Backend Abstraction
```python
# Create backend (supports neo4j and mock)
backend = create_backend(
    backend_type="neo4j" if not args.mock else "mock",
    neo4j_uri=args.neo4j_uri,
    neo4j_user=args.neo4j_user,
    neo4j_password=args.neo4j_password,
)

# Create session (mellea-based)
session = create_session(model_id=args.model)
```

### 3. Batch Processing
All three scripts use batch processing for efficiency:
- **Preprocessing**: Cypher UNWIND batches
- **Embedding**: Batch vector computation
- **Updating**: Async semaphore-based concurrency

### 4. Progress Tracking
- **Preprocessing**: Statistics JSON output
- **Embedding**: Statistics with entity/relation counts
- **Updating**: Progress file with per-document tracking

### 5. Error Handling
- **Preprocessing**: Batch errors tracked, graceful continuation
- **Embedding**: Per-item error tracking, batch resilience
- **Updating**: Per-document error tracking with result accumulation

## Data Flow

```
Raw Data (JSON files, JSONL documents)
    │
    ├─→ Preprocessing (run_kg_preprocess.py)
    │       ↓
    │   Neo4j KG (entities + relations)
    │       │
    │       ├─→ Embedding (run_kg_embed.py)
    │       │       ↓
    │       │   Neo4j KG (+ embeddings, + vector indices)
    │       │
    │       └─→ Updating (run_kg_update.py)
    │               ↓
    │           Updated Neo4j KG (+ new entities/relations)
    │
    └─→ QA System
            (uses embedded entities, vector indices)
            (performs Cypher queries on updated KG)
```

## Performance Characteristics

### Preprocessing
- **64,283 movies**: ~71.8 seconds
- **373,608 persons**: Included in above
- **1,045,369 relations**: Included in above
- **Throughput**: ~6,000 entities/sec, ~14,500 relations/sec

### Embedding
- **Model**: text-embedding-3-small (1536 dims)
- **Rate**: LiteLLM API dependent (typically 100+ embeddings/min)
- **Storage**: ~1.5KB per embedding (1536 float32 = 6144 bytes compressed)

### Updating
- **Throughput**: Document-dependent (typical CRAG dataset: 10-30 docs/min)
- **Concurrency**: 64 workers (configurable)
- **LLM Calls**: Multiple calls per document (extraction + alignment + merge)

## Supported Use Cases

### 1. Build Knowledge Graph from Scratch
```bash
# Use preprocessing to load structured data
python run_kg_preprocess.py --data-dir ./dataset/movie
```

### 2. Add Embeddings for Search
```bash
# Preprocess first, then embed
python run_kg_preprocess.py --data-dir ./dataset/movie
python run_kg_embed.py --model text-embedding-3-small
# Now KG supports vector similarity search
```

### 3. Live Updates from Documents
```bash
# Preprocess and embed, then keep updating
python run_kg_preprocess.py --data-dir ./dataset/movie
python run_kg_embed.py
python run_kg_update.py --dataset new_documents.jsonl.bz2
# Repeat last command as documents arrive
```

### 4. Development/Testing with Mock
```bash
# All scripts support --mock for testing without database
python run_kg_preprocess.py --mock --data-dir ./dataset/movie
python run_kg_embed.py --mock
python run_kg_update.py --mock --dataset small_test.jsonl
```

## Architecture Decisions

### Why This Design?

1. **Stage Separation**
   - Each stage can run independently
   - Easier to test and debug
   - Modular pipeline

2. **Batch Processing**
   - Efficiency: group operations reduce overhead
   - Reliability: easier to track and recover errors
   - Scalability: batch sizes tunable per stage

3. **Backend Abstraction**
   - Development with mock backend
   - Production with Neo4j
   - Extensible to other backends

4. **Configuration Classes**
   - Clear separation of concerns
   - Easy to document options
   - Follows mellea's established patterns

5. **JSON I/O**
   - Pipeline-friendly format
   - Easy integration with downstream systems
   - Natural fit for JSONL datasets

## Future Extensions

1. **Stage 4: RAG/QA** - Query system that uses embedded entities and vector indices
2. **Incremental Updates** - Only process new/changed documents
3. **Distributed Processing** - Scale to larger datasets with task queues
4. **Advanced Alignment** - ML-based entity matching
5. **Vector Database** - Pinecone/Milvus for large-scale similarity search
