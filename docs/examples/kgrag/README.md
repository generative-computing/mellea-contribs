# KG-RAG: Knowledge Graph-Enhanced Retrieval-Augmented Generation

A complete example system demonstrating how to build intelligent retrieval-augmented generation (RAG) using knowledge graphs with the Mellea framework and RITS cloud LLM service.

## Overview

This example demonstrates a three-stage KG-RAG pipeline:

1. **Preprocessing**: Load predefined structured data into Neo4j knowledge graph
2. **Embedding**: Generate and store vector embeddings for entities and relations
3. **Updating**: Process documents to extract and merge new entities/relations into the KG
4. **QA** (optional): Answer questions using multi-hop reasoning over the KG

**Tech Stack**:
- **Neo4j**: Graph database (localhost:7687)
- **RITS**: Cloud LLM service (llama-3-3-70b-instruct model)
- **LiteLLM**: LLM routing and API abstraction
- **Mellea**: LLM orchestration framework

**Domain Example:** Movie & Entertainment domain with 64K+ movies, 373K+ persons, and 1M+ relations.

## Directory Structure

```
docs/examples/kgrag/
├── README.md (this file)
├── models/
│   └── movie_domain_models.py     # Movie entity classes (MovieEntity, PersonEntity, AwardEntity)
├── preprocessor/
│   └── movie_preprocessor.py      # Domain-specific preprocessing (extract entities/relations from movie text)
├── rep/
│   └── movie_rep.py               # Movie-specific representations (formatting for LLM prompts)
└── scripts/
    ├── create_demo_dataset.py     # Generate 20 synthetic movie Q&A pairs
    ├── create_tiny_dataset.py     # Generate 5 Q&A pairs (quick testing)
    ├── create_truncated_dataset.py # Truncate existing dataset to N examples
    ├── run_kg_preprocess.py       # Extract entities/relations from documents
    ├── run_kg_embed.py            # Generate embeddings for entities
    ├── run_kg_update.py           # Update KG with new documents
    ├── run_qa.py                  # Run QA retrieval over questions
    └── run_eval.py                # Evaluate QA results with metrics
```

## Quick Start

### Prerequisites

1. **Start Neo4j Server**
   ```bash
   # Neo4j should be running on localhost:7687
   # Verify connection:
   docker ps | grep neo4j  # If using Docker
   # OR check neo4j process
   ```

2. **Configure RITS LLM Service**
   ```bash
   # Copy environment template
   cp .env_template .env

   # Edit .env and add your RITS API key:
   # RITS_API_KEY=your_actual_rits_api_key
   ```

3. **Verify Setup**
   ```bash
   cd scripts
   # Test Neo4j connection
   python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); print('✓ Neo4j connected')"
   ```

### Complete Pipeline (3-Stage)

Run the full pipeline with `run.sh`:

```bash
cd scripts
bash run.sh
```

This orchestrates:
1. **Step 1**: Preprocess - Load 64K movies + 373K persons into Neo4j
2. **Step 2**: Embed - Generate and store vector embeddings
3. **Step 3**: Update - Process documents and merge entities (if dataset available)

Output files:
- `../output/preprocess_stats.json` - Preprocessing statistics
- `../output/embedding_stats.json` - Embedding statistics
- `../output/update_stats.json` - Update statistics (if documents processed)

### Individual Pipeline Steps

#### Step 1: Preprocessing (Load Predefined Data)

Load movie and person data into Neo4j:

```bash
# Load predefined movie database (64K movies, 373K persons)
python run_kg_preprocess.py \
  --data-dir ../dataset/movie \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password \
  --batch-size 500

# Or test with mock backend (no database required)
python run_kg_preprocess.py \
  --data-dir ../dataset/movie \
  --mock
```

Output: `../output/preprocess_stats.json` with statistics:
```json
{
  "total_documents": 1,
  "entities_loaded": 437891,
  "entities_inserted": 437891,
  "relations_inserted": 1045369
}
```

#### Step 2: Embedding (Generate Vector Embeddings)

Compute embeddings for entities and relations:

```bash
# Generate embeddings using default model (text-embedding-3-small)
python run_kg_embed.py \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password \
  --batch-size 100

# With custom embedding model
python run_kg_embed.py \
  --model text-embedding-3-large \
  --dimension 3072 \
  --batch-size 50
```

Output: `../output/embedding_stats.json` with embedding statistics

#### Step 3: Update (Process Documents & Extract Entities)

Update KG by processing new documents with RITS LLM:

```bash
# Update with tiny dataset (for testing)
python run_kg_update.py \
  --dataset ../dataset/crag_movie_tiny.jsonl.bz2 \
  --domain movie \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password \
  --num-workers 1 \
  --extraction-loop-budget 3 \
  --alignment-loop-budget 2

# Or with full dataset
python run_kg_update.py \
  --dataset ../dataset/crag_movie_dev.jsonl.bz2 \
  --domain movie \
  --num-workers 64
```

Configuration via `.env`:
- `API_BASE`: RITS endpoint
- `MODEL_NAME`: Model to use (default: llama-3-3-70b-instruct)
- `RITS_API_KEY`: RITS authentication

Output: `../output/update_stats.json` with update results

## Domain-Specific Components

### Movie Domain Models

File: `models/movie_domain_models.py`

Defines domain-specific entity classes:
- `MovieEntity`: Movies with properties like genre, release_year, budget, box_office
- `PersonEntity`: People (directors, actors) with birth_year, nationality
- `AwardEntity`: Awards with category, year, ceremony

Example usage:
```python
from models.movie_domain_models import MovieEntity, PersonEntity

avatar = MovieEntity(
    id="movie-1",
    name="Avatar",
    description="Science fiction epic film",
    genre=["Science Fiction", "Epic"],
    release_year=2009,
    budget=237000000,
    box_office=2923706026
)

cameron = PersonEntity(
    id="person-1",
    name="James Cameron",
    birth_year=1954,
    nationality="Canada"
)
```

### Movie Domain Preprocessor

File: `preprocessor/movie_preprocessor.py`

Extends `KGPreprocessor` for movie domain:
- Extracts movie, person, and award entities
- Recognizes relations: DIRECTED, STARRED_IN, WON, NOMINATED_FOR
- Movie-specific entity type hints for LLM
- Post-processing to standardize entity types

Example usage:
```python
from preprocessor.movie_preprocessor import MovieKGPreprocessor
from mellea import start_session

async def preprocess_movie_document():
    session = start_session(backend_name="litellm", model_id="gpt-4o-mini")
    backend = MockGraphBackend()

    preprocessor = MovieKGPreprocessor(session=session, backend=backend)

    result = await preprocessor.process_document(
        doc_id="doc-1",
        doc_text="""
        Avatar was directed by James Cameron and released in 2009.
        It stars Sam Worthington and Zoe Saldana.
        The film won Best Cinematography and Best Art Direction at the Academy Awards.
        """
    )

    print(f"Extracted {len(result.entities_extracted)} entities")
    print(f"Extracted {len(result.relations_extracted)} relations")
```

### Movie Domain Representation

File: `rep/movie_rep.py`

Movie-specific utilities for formatting data for LLM consumption:
- `movie_entity_to_text()`: Format movie entity as natural language
- `person_entity_to_text()`: Format person entity
- `format_movie_context()`: Format context for QA prompts
- `movie_relation_to_text()`: Format relations for LLM

Example usage:
```python
from rep.movie_rep import format_movie_context, movie_entity_to_text

# Format entity
movie_text = movie_entity_to_text(avatar_entity)
# Output: "Avatar (2009) is a Science Fiction, Epic film directed by..."

# Format context for QA
context = format_movie_context(
    entities=[avatar_entity, cameron_entity],
    relations=[directed_relation],
    question="Who directed Avatar?"
)
# Used to build LLM prompt with formatted context
```

## Complete Workflow Example

Here's a complete end-to-end example with real Neo4j and RITS:

```bash
#!/bin/bash
set -e

cd scripts

# Prerequisite: .env configured with RITS credentials
if [ ! -f ../.env ]; then
    echo "Error: .env not found. Copy .env_template and configure RITS_API_KEY"
    exit 1
fi

echo "========================================================"
echo "KG-RAG Pipeline - Complete Workflow"
echo "========================================================"

# Step 1: Preprocess - Load predefined movie data
echo ""
echo "Step 1: Preprocessing - Load 64K movies + 373K persons into Neo4j"
python run_kg_preprocess.py \
    --data-dir ../dataset/movie \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password \
    --batch-size 500 \
    > ../output/preprocess_stats.json
echo "✓ Entities loaded and Neo4j database populated"

# Step 2: Embedding - Generate embeddings
echo ""
echo "Step 2: Embedding - Generate entity/relation embeddings"
python run_kg_embed.py \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password \
    --batch-size 100 \
    > ../output/embedding_stats.json
echo "✓ Embeddings computed and stored in Neo4j"

# Step 3: Update - Process documents with RITS LLM
echo ""
echo "Step 3: Updating - Extract entities from documents using RITS"
python run_kg_update.py \
    --dataset ../dataset/crag_movie_tiny.jsonl.bz2 \
    --domain movie \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password \
    --num-workers 2 \
    > ../output/update_stats.json
echo "✓ Documents processed and KG updated"

echo ""
echo "========================================================"
echo "✅ Pipeline Complete!"
echo "========================================================"
echo "Output files:"
echo "  - ../output/preprocess_stats.json"
echo "  - ../output/embedding_stats.json"
echo "  - ../output/update_stats.json"
echo "========================================================"
```

Or run the simplified `run.sh`:
```bash
cd scripts
bash run.sh
```

## Using the Validation Suite

The `sun.sh` script at the project root validates all implementations end-to-end:

```bash
# Full validation (Phase 0-4)
./sun.sh

# Quick validation (skip slower tests)
./sun.sh --quick

# Unit tests only
./sun.sh --unit-only
```

This validates:
- Phase 0: Environment setup
- Phase 1: Core KG modules
- Phase 2: All run scripts
- Phase 3: Utility modules (95 tests)
- Phase 4: Configuration

## Creating a Custom Domain

To create a custom domain (e.g., sports, finance, healthcare):

### 1. Create Domain Models

Create `models/[domain]_models.py`:
```python
from mellea_contribs.kg.models import Entity, Relation

class TeamEntity(Entity):
    """Sports team entity."""
    league: str
    founded_year: int
    # ... other domain-specific fields

class AthleteEntity(Entity):
    """Sports athlete entity."""
    position: str
    jersey_number: int
    # ... other domain-specific fields
```

### 2. Create Domain Preprocessor

Create `preprocessor/[domain]_preprocessor.py`:
```python
from mellea_contribs.kg.preprocessor import KGPreprocessor

class SportsKGPreprocessor(KGPreprocessor):
    """Domain-specific preprocessor for sports."""

    def get_hints(self) -> str:
        """Return domain-specific extraction hints."""
        return """
        Extract sports entities:
        - Teams (league, founded year, stadium)
        - Athletes (position, jersey number, achievements)
        - Competitions (tournament, year, winner)

        Relations:
        - PLAYS_FOR: athlete plays for team
        - WON: team won competition
        - DEFEATED: team defeated another team
        """

    def post_process_extraction(self, result):
        """Domain-specific post-processing."""
        # Standardize team names, normalize dates, etc.
        return result
```

### 3. Create Domain Representation

Create `rep/[domain]_rep.py`:
```python
def team_entity_to_text(entity) -> str:
    """Format team as natural language."""
    return f"{entity.name} (founded {entity.founded_year}) plays in {entity.league}"
```

### 4. Run Scripts for Custom Domain

```bash
python scripts/run_kg_preprocess.py \
    --input sports_documents.jsonl \
    --mock \
    --domain sports  # Use custom domain

python scripts/run_qa.py \
    --input sports_questions.jsonl \
    --mock \
    --domain sports
```

## Configuration

### Environment Setup (.env)

Create `.env` file from template:

```bash
cp .env_template .env
```

Configure your `.env` with:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# RITS Cloud LLM Configuration (Primary)
API_BASE=https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1
MODEL_NAME=openai/llama-3-3-70b-instruct
API_KEY=dummy
RITS_API_KEY=your_rits_api_key_here

# Optional: Embedding Configuration
EMB_API_BASE=https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2/v1
EMB_API_KEY=dummy
EMB_MODEL_NAME=ibm/slate-125m-english-rtrvr-v2

# Request Configuration
MAX_RETRIES=3
TIME_OUT=1800
OTEL_SDK_DISABLED=true
```

### Command-Line Arguments

All scripts support configuration via command-line arguments:

```bash
# Neo4j configuration
python run_kg_update.py \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password

# Model configuration
python run_kg_update.py \
    --model openai/llama-3-3-70b-instruct \
    --extraction-loop-budget 3 \
    --alignment-loop-budget 2

# Processing configuration
python run_kg_update.py \
    --dataset data.jsonl \
    --num-workers 64 \
    --batch-size 100
```

See `--help` on each script for all available options:
```bash
python run_kg_preprocess.py --help
python run_kg_embed.py --help
python run_kg_update.py --help
```

## Testing

Run the validation suite to test everything:

```bash
# From project root
./sun.sh

# Or run specific tests
pytest test/kg/ -v
pytest test/kg/utils/ -v
```

## API Reference

### Dataset Creation Scripts

All return JSONL files with questions and optional answers:

```python
# Format: {"question": "...", "answer": "...", "domain": "..."}
```

### Preprocessing Script

Input: JSONL with `{"id": "...", "text": "..."}`
Output: JSON with `UpdateStats`

### QA Script

Input: JSONL with questions
Output: JSONL with `QAResult` per question

### Evaluation Script

Input: JSONL with `QAResult` items
Output: JSON with `QAStats`

## Troubleshooting

### Environment & Installation

**ImportError: No module named 'mellea_contribs'**
```bash
# Install in development mode
cd /path/to/mellea-contribs
pip install -e .
```

**Missing python-dotenv**
```bash
pip install python-dotenv
```

### Neo4j Issues

**Neo4j Connection Failed**
```bash
# Verify Neo4j is running
docker ps | grep neo4j
# OR check service
nc -zv localhost 7687  # Should say 'succeeded'
```

**OTEL Connection Errors**
Already disabled in `.env_template`:
```bash
OTEL_SDK_DISABLED=true
```

### RITS/LLM Configuration

**LLM API calls failing with litellm_chat_response error**
- Verify `.env` file exists: `ls -la .env`
- Check RITS_API_KEY is configured (not just template value)
- Verify API_BASE endpoint is accessible
- Check API_KEY and RITS_API_KEY values in `.env`

**Using wrong model**
- Verify `MODEL_NAME` in `.env` matches your intent
- Session output should show: `Starting Mellea session: model=openai/llama-3-3-70b-instruct`
- JSON output should show: `"model_used": "openai/llama-3-3-70b-instruct"`

**Switch to local vLLM (Alternative)**
```bash
# Edit .env - uncomment local vLLM section:
# API_BASE=http://localhost:7878/v1
# MODEL_NAME=/path/to/local/model

# Or use mock for testing (no LLM calls):
python run_kg_update.py --dataset data.jsonl --mock
```

### Script Issues

**Dataset file not found**
```bash
# Check file exists
ls -lh ../dataset/crag_movie_tiny.jsonl.bz2

# Use correct path
python run_kg_update.py --dataset ../dataset/crag_movie_tiny.jsonl.bz2
```

**Run all tests to diagnose**
```bash
cd /path/to/mellea-contribs
pytest test/kg/utils/ -v  # Run all KG tests
```

## Architecture

The KG-RAG pipeline is organized into three independent stages:

```
Stage 1: Preprocessing          Stage 2: Embedding           Stage 3: Updating
(run_kg_preprocess.py)          (run_kg_embed.py)            (run_kg_update.py)
         │                               │                            │
         ├─ Load predefined data    ├─ Fetch entities          ├─ Load documents
         ├─ Batch insert into Neo4j ├─ Compute embeddings      ├─ Extract entities/relations
         └─ Track statistics        └─ Store in Neo4j          └─ Align & merge with KG

                          Neo4j Knowledge Graph
                         (bolt://localhost:7687)
                    437,890 entities + 1,045,369 relations
```

**Backend Architecture**:
- **MockGraphBackend**: For testing without database
- **Neo4jBackend**: Production graph database
- **LiteLLM**: Abstracts LLM routing to RITS cloud service

## See Also

- **Core Library**: [mellea_contribs/kg/README.md](../../mellea_contribs/kg/README.md)
- **Utility Modules**: [mellea_contribs/kg/utils/README.md](../../mellea_contribs/kg/utils/README.md)
- **Pipeline Architecture**: [KG_PIPELINE_ARCHITECTURE.md](../../KG_PIPELINE_ARCHITECTURE.md)
- **KG Update Architecture**: [KG_UPDATE_IMPROVEMENT_SUMMARY.md](../../KG_UPDATE_IMPROVEMENT_SUMMARY.md)
- **Preprocessing Summary**: [PREPROCESSING_REWRITE_SUMMARY.md](../../PREPROCESSING_REWRITE_SUMMARY.md)
- **Configuration Guide**: [PHASE4_CONFIGURATION.md](../../PHASE4_CONFIGURATION.md)
- **Mellea Framework**: https://github.com/generative-computing/mellea
- **Original PR#3**: https://github.com/ydzhu98/mellea/pull/3

## Key Concepts

### Multi-Route QA

Break complex questions into multiple solving paths and reach consensus:
1. Route 1: Direct entity matching
2. Route 2: Relation traversal
3. Route 3: Alternative reasoning
→ Validate consensus across routes

### Entity/Relation Alignment

When processing new documents:
1. Extract entities and relations
2. Align with existing KG entities
3. Decide merge strategy (merge or create new)
4. Update KG with merged data

### Domain-Specific Processing

Each domain can customize:
- Entity types and properties
- Relation types
- Extraction hints
- Post-processing rules
- LLM prompt formatting

## Implementation Details

- **Backend**: MockGraphBackend (testing) or Neo4jBackend (production)
- **Models**: Pydantic for validation and serialization
- **Async**: Full async/await throughout for scalability
- **LLM**: Mellea @generative decorators for all decisions
- **JSONL I/O**: Pipeline-friendly line-delimited JSON format

## License

Apache License 2.0 (same as mellea-contribs)
