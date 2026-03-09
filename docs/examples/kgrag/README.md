# KG-RAG: Knowledge Graph-Enhanced Retrieval-Augmented Generation

A complete example system demonstrating how to build intelligent retrieval-augmented generation (RAG) using knowledge graphs with the Mellea framework.

## Overview

This example shows how to:
- Extract entities and relations from documents using domain-specific preprocessors
- Build and maintain a knowledge graph from unstructured text
- Answer complex questions by reasoning over the knowledge graph
- Handle multi-hop queries with consensus validation
- Evaluate QA system performance with standard metrics

**Domain Example:** Movie & Entertainment domain with entity types (Movie, Person, Award) and relation types (DIRECTED, STARRED_IN, WON).

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

### 1. Create a Demo Dataset

```bash
# Generate 20 synthetic movie questions
python scripts/create_demo_dataset.py --output /tmp/demo.jsonl

# Generate 5 questions for quick testing
python scripts/create_tiny_dataset.py --output /tmp/tiny.jsonl

# Truncate a large dataset to N examples
python scripts/create_truncated_dataset.py --input /tmp/demo.jsonl --output /tmp/truncated.jsonl --max-examples 5
```

Output format (JSONL):
```json
{"question": "Who directed Avatar?", "answer": "James Cameron", "domain": "movies"}
{"question": "What awards did Oppenheimer win?", "answer": "Best Picture, Best Director, Best Actor", "domain": "movies"}
```

### 2. Preprocess Documents

Extract entities and relations from documents using the movie domain preprocessor:

```bash
# Create a sample documents file
cat > /tmp/documents.jsonl << 'EOF'
{"id": "doc-1", "text": "Avatar was directed by James Cameron. It stars Sam Worthington and Zoe Saldana."}
{"id": "doc-2", "text": "Oppenheimer is a biographical film directed by Christopher Nolan. It won Best Picture at the 2024 Academy Awards."}
EOF

# Preprocess with mock backend (no Neo4j needed)
python scripts/run_kg_preprocess.py --input /tmp/documents.jsonl --mock --output-stats /tmp/preprocess_stats.json

# With Neo4j backend (requires running Neo4j)
python scripts/run_kg_preprocess.py --input /tmp/documents.jsonl --output-stats /tmp/preprocess_stats.json
```

Output: `preprocess_stats.json` with `UpdateStats`:
```json
{
  "total_documents": 2,
  "successful_documents": 2,
  "failed_documents": 0,
  "entities_extracted": 8,
  "relations_extracted": 5,
  "average_processing_time_ms": 250.5
}
```

### 3. Generate Embeddings

Create vector embeddings for knowledge graph entities:

```bash
# Generate embeddings (requires embedding service via LiteLLM)
python scripts/run_kg_embed.py --mock --output-stats /tmp/embed_stats.json

# With custom embedding model
python scripts/run_kg_embed.py --mock --model text-embedding-3-small --dimension 1536
```

### 4. Update Knowledge Graph

Add new documents to the KG with entity/relation merging:

```bash
# Update KG with new documents
python scripts/run_kg_update.py --input /tmp/documents.jsonl --mock --output /tmp/update_results.jsonl

# With merge conflict reporting
python scripts/run_kg_update.py --input /tmp/documents.jsonl --output /tmp/update_results.jsonl \
  --similarity-threshold 0.8
```

Output: `update_results.jsonl` with update results per document.

### 5. Run QA Retrieval

Answer questions using multi-route reasoning over the knowledge graph:

```bash
# Run QA with mock backend
python scripts/run_qa.py --input /tmp/tiny.jsonl --mock --output /tmp/qa_results.jsonl

# With custom parameters
python scripts/run_qa.py --input /tmp/tiny.jsonl --mock --output /tmp/qa_results.jsonl \
  --num-routes 3 \
  --depth 2 \
  --domain movies

# With ground truth for evaluation
python scripts/run_qa.py --input /tmp/demo.jsonl --mock --output /tmp/qa_results.jsonl
```

Output: `qa_results.jsonl` with `QAResult` per question:
```json
{
  "question": "Who directed Avatar?",
  "answer": "James Cameron",
  "confidence": 0.95,
  "reasoning": {"routes": [...], "entities": [...], "relations": [...]},
  "processing_time_ms": 523.4
}
```

### 6. Evaluate Results

Compute evaluation metrics (exact match, fuzzy match, MRR):

```bash
# Evaluate QA results
python scripts/run_eval.py --input /tmp/qa_results.jsonl --output /tmp/eval_results.json

# With fuzzy matching threshold
python scripts/run_eval.py --input /tmp/qa_results.jsonl --output /tmp/eval_results.json \
  --fuzzy-threshold 0.8
```

Output: `eval_results.json` with `QAStats`:
```json
{
  "total_questions": 20,
  "successful_answers": 18,
  "failed_answers": 2,
  "average_confidence": 0.87,
  "exact_match_accuracy": 0.90,
  "mean_reciprocal_rank": 0.92
}
```

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

Here's a complete end-to-end example:

```bash
#!/bin/bash
set -e

SCRIPTS_DIR=scripts
OUTPUT_DIR=/tmp/kgrag_output
mkdir -p $OUTPUT_DIR

echo "Step 1: Create demo dataset"
python $SCRIPTS_DIR/create_demo_dataset.py --output $OUTPUT_DIR/demo.jsonl

echo "Step 2: Preprocess documents"
cat > $OUTPUT_DIR/documents.jsonl << 'EOF'
{"id": "doc-1", "text": "Avatar (2009) directed by James Cameron features Sam Worthington"}
{"id": "doc-2", "text": "Oppenheimer directed by Christopher Nolan won Best Picture"}
EOF

python $SCRIPTS_DIR/run_kg_preprocess.py \
    --input $OUTPUT_DIR/documents.jsonl \
    --mock \
    --output-stats $OUTPUT_DIR/preprocess_stats.json

echo "Step 3: Update KG with documents"
python $SCRIPTS_DIR/run_kg_update.py \
    --input $OUTPUT_DIR/documents.jsonl \
    --mock \
    --output $OUTPUT_DIR/update_results.jsonl

echo "Step 4: Run QA on demo dataset"
python $SCRIPTS_DIR/run_qa.py \
    --input $OUTPUT_DIR/demo.jsonl \
    --mock \
    --output $OUTPUT_DIR/qa_results.jsonl

echo "Step 5: Evaluate results"
python $SCRIPTS_DIR/run_eval.py \
    --input $OUTPUT_DIR/qa_results.jsonl \
    --output $OUTPUT_DIR/eval_results.json

echo "Results saved to $OUTPUT_DIR/"
ls -lh $OUTPUT_DIR/
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

All scripts support configuration via environment variables or arguments:

```bash
# Via environment variables
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
export LLM_MODEL=gpt-4o-mini

# Via arguments
python scripts/run_qa.py \
    --input questions.jsonl \
    --neo4j-uri bolt://localhost:7687 \
    --model gpt-4o-mini
```

See `.env_template` in the project root for all available configuration variables.

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

### ImportError: No module named 'movie_preprocessor'

Make sure you're running scripts from the project root and have installed the package:
```bash
cd /path/to/mellea-contribs
pip install -e .
```

### Neo4j Connection Error

Use `--mock` flag for testing without Neo4j:
```bash
python scripts/run_kg_preprocess.py --input data.jsonl --mock
```

Or configure Neo4j connection:
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

### Low Confidence Scores

Increase the number of routes and depth:
```bash
python scripts/run_qa.py --input questions.jsonl --num-routes 5 --depth 3
```

## See Also

- **Core Library**: [mellea_contribs/kg/README.md](../../mellea_contribs/kg/README.md)
- **Utility Modules**: [mellea_contribs/kg/utils/README.md](../../mellea_contribs/kg/utils/README.md)
- **Configuration Guide**: [PHASE4_CONFIGURATION.md](../../PHASE4_CONFIGURATION.md)
- **Validation Suite**: [sun.sh](../../sun.sh)
- **Implementation Status**: [missing_for_run_sh.txt](../../missing_for_run_sh.txt)
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
