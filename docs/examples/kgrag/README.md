# KG-RAG: Knowledge Graph-Enhanced Retrieval-Augmented Generation

A Mellea-based implementation of the [Bidirection](https://github.com/junhongmit/Bidirection) project, which synergizes LLMs and Knowledge Graphs. The original Bidirection project demonstrates a two-stage framework: a KG Update stage where a LLM extracts knowledge from personal documents to enrich a general-purpose KG, and an LLM Inference stage where the enriched KG enhances LLM performance on downstream tasks.

This example ports the core pipeline into the Mellea framework, so that the infrastructure concerns — LLM session management, graph backend abstraction, entity alignment, generative functions — are handled by the library. Users can focus on domain-specific logic rather than reimplementing the common components from scratch.

## Overview

This example demonstrates a five-stage KG-RAG pipeline:

1. **Preprocessing**: Load predefined structured data into a Neo4j knowledge graph
2. **Embedding**: Generate and store vector embeddings for entities and relations
3. **Updating**: Process documents to extract and merge new entities/relations into the KG
4. **QA**: Answer questions using multi-hop Think-on-Graph reasoning over the KG
5. **Evaluation**: Score predictions with an LLM judge and compute CRAG metrics

**Tech Stack**:
- **Neo4j**: Graph database (localhost:7687)
- **RITS**: Cloud LLM service (llama-3-3-70b-instruct model)
- **Mellea**: LLM orchestration framework with `OpenAIBackend`

**Domain Example:** Movie & Entertainment domain with 64K+ movies, 373K+ persons, and 1M+ relations.

## Directory Structure

```
docs/examples/kgrag/
├── README.md                          (this file)
├── .env_template                      # Copy to .env and fill in credentials
├── dataset/
│   └── README.md                      # Dataset acquisition instructions
├── models/
│   └── movie_domain_models.py         # Movie entity classes (MovieEntity, PersonEntity, AwardEntity)
├── preprocessor/
│   └── movie_preprocessor.py          # Domain-specific preprocessing
├── rep/
│   └── movie_rep.py                   # Movie-specific representations for LLM prompts
└── scripts/
    ├── run.sh                         # Pipeline orchestration (all 5 steps)
    ├── create_tiny_dataset.py         # Slice a small dataset for testing
    ├── run_kg_preprocess.py           # Step 1: Load predefined data into Neo4j
    ├── run_kg_embed.py                # Step 2: Generate embeddings for entities
    ├── run_kg_update.py               # Step 3: Update KG with new documents
    ├── run_qa.py                      # Step 4: Run QA retrieval over questions
    └── run_eval.py                    # Step 5: Evaluate QA results (LLM judge + CRAG metrics)
```

## Quick Start

### Prerequisites

1. **Start Neo4j Server**
   ```bash
   # Neo4j should be running on localhost:7687
   docker ps | grep neo4j  # If using Docker
   ```

2. **Configure credentials**
   ```bash
   cd docs/examples/kgrag
   cp .env_template .env
   # Edit .env and set: API_BASE, RITS_API_KEY, MODEL_NAME
   ```

3. **Place dataset files** in `dataset/` (see [Dataset Files](#dataset-files) below)

### Running the Pipeline

Use `run.sh` to run the full pipeline or individual steps.

#### Dataset mode

| Flag | Description |
|------|-------------|
| `--tiny` (default) | Uses the tiny test dataset (~10 docs). Step 0 creates it if missing. |
| `--full` | Uses the full dataset (`crag_movie_dev.jsonl.bz2`). Skips step 0. |

```bash
cd scripts

# Run all steps on the tiny dataset (default)
bash run.sh

# Run all steps on the full dataset
bash run.sh --full

# Run specific steps only
bash run.sh --tiny 3 4 5     # update + QA + eval on tiny set
bash run.sh --full 4 5       # QA + eval on full set
bash run.sh 1 2              # reload Neo4j + recompute embeddings
```

#### Step reference

| Step | Script | Description |
|------|--------|-------------|
| 0 | `create_tiny_dataset.py` | Create tiny dataset from full set (tiny mode only) |
| 1 | `run_kg_preprocess.py` | Load 64K movies + 373K persons into Neo4j |
| 2 | `run_kg_embed.py` | Compute and store entity/relation embeddings |
| 3 | `run_kg_update.py` | Extract entities from documents and merge into KG |
| 4 | `run_qa.py` | Answer questions via Think-on-Graph retrieval |
| 5 | `run_eval.py` | Score predictions with LLM judge; compute CRAG metrics |

Output files (written to `output/`):

| File | Produced by |
|------|-------------|
| `preprocess_stats.json` | Step 1 |
| `embedding_stats.json` | Step 2 |
| `update_stats.json` | Step 3 |
| `qa_results.jsonl` | Step 4 |
| `qa_progress.json` | Step 4 (resumption state) |
| `eval_results.json` | Step 5 (annotated per-item results) |
| `eval_metrics.json` | Step 5 (aggregate CRAG metrics) |

## Individual Scripts

### Step 1: Preprocessing

Load predefined movie/person data into Neo4j:

```bash
python run_kg_preprocess.py \
  --data-dir ../dataset/movie \
  --db-uri bolt://localhost:7687 \
  --db-user neo4j \
  --db-password password \
  --batch-size 500
```

Output (`preprocess_stats.json`):
```json
{
  "total_documents": 1,
  "entities_loaded": 437891,
  "entities_inserted": 437891,
  "relations_inserted": 1045369
}
```

### Step 2: Embedding

Compute embeddings for entities and relations:

```bash
python run_kg_embed.py \
  --db-uri bolt://localhost:7687 \
  --db-user neo4j \
  --db-password password \
  --batch-size 100
```

Set `EMB_API_BASE` / `EMB_MODEL_NAME` in `.env` to use a custom embedding endpoint.

### Step 3: KG Update

Extract entities/relations from documents and merge them into the KG:

```bash
# Tiny dataset (testing)
python run_kg_update.py \
  --dataset ../dataset/crag_movie_tiny.jsonl.bz2 \
  --domain movie \
  --num-workers 10

# Full dataset
python run_kg_update.py \
  --dataset ../dataset/crag_movie_dev.jsonl.bz2 \
  --domain movie \
  --num-workers 64
```

LLM configuration is read from `.env` (`API_BASE`, `MODEL_NAME`, `RITS_API_KEY`).

### Step 4: QA

Answer questions using Think-on-Graph multi-hop retrieval:

```bash
python run_qa.py \
  --dataset ../dataset/crag_movie_tiny.jsonl.bz2 \
  --output ../output/qa_results.jsonl \
  --progress ../output/qa_progress.json \
  --domain movie \
  --routes 3 \
  --width 30 \
  --depth 3
```

Key options:
- `--routes N` — number of independent solving routes (default: 3)
- `--width N` — max candidate relations per traversal step (default: 30)
- `--depth N` — max traversal depth (default: 3)
- `--reset-progress` — ignore previous progress and reprocess all questions
- `--workers N` — parallel async workers (default: 1)

Output JSONL format:
```json
{
  "id": "q_0",
  "query": "Who directed Inception?",
  "predicted": "Christopher Nolan",
  "answer": "Christopher Nolan",
  "answer_aliases": ["Christopher Nolan", "Nolan"],
  "elapsed_ms": 1234.5
}
```

### Step 5: Evaluation

Evaluate predictions using LLM judge + CRAG-style scoring:

```bash
python run_eval.py \
  --input ../output/qa_results.jsonl \
  --output ../output/eval_results.json \
  --metrics ../output/eval_metrics.json

# Skip LLM calls (fuzzy match only, for testing)
python run_eval.py \
  --input ../output/qa_results.jsonl \
  --metrics ../output/eval_metrics.json \
  --mock
```

The evaluator runs each prediction through:
1. Exact match against `answer_aliases`
2. Fuzzy match (rapidfuzz token_set_ratio ≥ 0.8)
3. LLM judge (for cases not resolved by string matching)

Output (`eval_metrics.json`):
```json
{
  "total": 100,
  "n_correct": 72,
  "n_miss": 5,
  "n_hallucination": 23,
  "accuracy": 72.0,
  "score": 49.0,
  "hallucination": 23.0,
  "missing": 5.0,
  "eval_model": "meta-llama/llama-3-3-70b-instruct"
}
```

**CRAG score formula**: `((2 × correct + missing) / total − 1) × 100`
— penalises hallucination more than unanswered questions.

Use a separate eval model by setting `EVAL_API_BASE` / `EVAL_MODEL_NAME` / `EVAL_RITS_API_KEY`
in `.env`; the script falls back to the main session if these are not set.

## Configuration

### .env file

```bash
cp .env_template .env
```

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Primary LLM (RITS or any OpenAI-compatible endpoint)
API_BASE=https://your-rits-endpoint/v1
MODEL_NAME=meta-llama/llama-3-3-70b-instruct
API_KEY=dummy
RITS_API_KEY=your_rits_api_key

# Optional: separate eval model
EVAL_API_BASE=https://your-eval-endpoint/v1
EVAL_MODEL_NAME=meta-llama/llama-3-3-70b-instruct
# EVAL_RITS_API_KEY=  # falls back to RITS_API_KEY if unset

# Optional: embedding model
EMB_API_BASE=https://your-embedding-endpoint/v1
EMB_MODEL_NAME=text-embedding-3-small

# Misc
OTEL_SDK_DISABLED=true
```

The Python scripts load this file automatically via `python-dotenv` (`override=False`),
so values already exported in the shell take precedence.

### Session architecture

All scripts create sessions using `create_session_from_env` from
`mellea_contribs.kg.utils`, which wires up `MelleaSession(backend=OpenAIBackend(...))`
directly — no LiteLLM proxy needed. The `RITS_API_KEY` is forwarded as a custom HTTP
header that RITS requires for authentication.

## Domain-Specific Components

### Movie Domain Models (`models/movie_domain_models.py`)

Defines domain-specific entity classes extending the core `Entity`/`Relation` models:
- `MovieEntity`: genre, release_year, budget, box_office
- `PersonEntity`: birth_year, nationality
- `AwardEntity`: category, year, ceremony

### Movie Domain Preprocessor (`preprocessor/movie_preprocessor.py`)

Extends `KGPreprocessor` with movie-specific extraction hints and post-processing
(entity type standardisation, relation normalisation).

### Movie Domain Representation (`rep/movie_rep.py`)

Formatting utilities for LLM prompts: `movie_entity_to_text`, `person_entity_to_text`,
`format_movie_context`, `movie_relation_to_text`.

## Creating a Custom Domain

1. **Models** — create `models/[domain]_models.py` extending `Entity`/`Relation`
2. **Preprocessor** — create `preprocessor/[domain]_preprocessor.py` extending `KGPreprocessor`;
   implement `get_hints()` and optionally `post_process_extraction()`
3. **Representation** — create `rep/[domain]_rep.py` with domain-specific text formatters
4. **Run** — pass `--domain [domain]` to `run_kg_update.py` and `run_qa.py`

## Testing

```bash
# Run all KG utility tests from the project root
pytest test/kg/ -v
pytest test/kg/utils/ -v

# Test scripts without Neo4j or LLM (mock mode)
cd scripts
python run_kg_update.py --dataset ../dataset/crag_movie_tiny.jsonl.bz2 --mock
python run_qa.py --dataset ../dataset/crag_movie_tiny.jsonl.bz2 --mock
python run_eval.py --input ../output/qa_results.jsonl --mock
```

## Troubleshooting

### Neo4j connection

```bash
# Verify Neo4j is reachable
nc -zv localhost 7687
```

### LLM / RITS authentication

- Ensure `.env` exists and `RITS_API_KEY` is set (not just the template placeholder)
- Do **not** `export API_BASE` / `RITS_API_KEY` as empty strings in the shell before
  running `run.sh` — empty exports prevent `load_dotenv` from filling them in
- The scripts log `create_session_from_env(prefix=...): api_base=set/MISSING, rits_api_key=set/MISSING`
  at `INFO` level to help diagnose missing credentials

### Resuming an interrupted QA run

Step 4 writes a progress file (`qa_progress.json`). Re-running without
`--reset-progress` picks up where it left off. Use `--reset-progress` to start fresh.

### Dataset not found

```bash
ls -lh ../dataset/crag_movie_tiny.jsonl.bz2
# If missing, run step 0 first:
bash run.sh 0
```

## Architecture

```
Step 1: Preprocessing       Step 2: Embedding        Step 3: Updating
run_kg_preprocess.py        run_kg_embed.py          run_kg_update.py
        │                          │                         │
        ├─ Load predefined data    ├─ Fetch entities    ├─ Load documents
        ├─ Batch insert Neo4j      ├─ Compute embeddings├─ Extract entities/relations
        └─ Output stats            └─ Store + index     └─ Align & merge with KG

                         Neo4j Knowledge Graph
                        (bolt://localhost:7687)

Step 4: QA                            Step 5: Evaluation
run_qa.py                             run_eval.py
        │                                     │
        ├─ Decompose question            ├─ Exact / fuzzy match
        ├─ Align entities (embed+fuzzy)  ├─ LLM judge
        ├─ Think-on-Graph traversal      └─ CRAG metrics (accuracy, score,
        ├─ Prune + synthesise answer          hallucination, missing)
        └─ Output JSONL results
```

## Dataset Files

Large data files are **not tracked in git** to keep repository size manageable.

| File | Size | Used by |
|------|------|---------|
| `dataset/crag_movie_dev.jsonl.bz2` | ~140 MB | Step 3 (full mode) |
| `dataset/movie/movie_db.json` | ~181 MB | Step 1 |
| `dataset/movie/person_db.json` | ~44 MB | Step 1 |

### Generating a tiny test dataset

```bash
cd scripts
python create_tiny_dataset.py --output ../dataset/crag_movie_tiny.jsonl.bz2
```

### Acquiring the full dataset

- Full CRAG dataset: contact project maintainers for access to `crag_movie_dev.jsonl.bz2`
- Movie/person databases: sourced from the TMDB dataset

### Testing without any data files

Use `--mock` to test all scripts without a database or dataset:

```bash
cd scripts
python run_kg_update.py --dataset ../dataset/crag_movie_tiny.jsonl.bz2 --mock
python run_qa.py --dataset ../dataset/crag_movie_tiny.jsonl.bz2 --mock
python run_eval.py --input ../output/qa_results.jsonl --mock
```

## See Also

- **Bidirection** (original project): https://github.com/junhongmit/Bidirection
- **Core Library**: [mellea_contribs/kg/README.md](../../../mellea_contribs/kg/README.md)
- **Mellea Framework**: https://github.com/generative-computing/mellea

## License

Apache License 2.0 (same as mellea-contribs)
