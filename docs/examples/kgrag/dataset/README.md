# KG-RAG Dataset Directory

This directory contains large data files that are **not tracked in git** to keep repository size manageable.

## Files

### Required for Full Pipeline

- **`crag_movie_dev.jsonl.bz2`** (140 MB)
  - Full CRAG dataset with movie documents
  - Used by `run_kg_update.py` for document processing
  - Source: CRAG benchmark dataset

- **`crag_movie_tiny.jsonl.bz2`**
  - Tiny dataset (10 documents) for quick testing
  - Recommended for development and testing

### Movie Database (Predefined Structured Data)

- **`movie/movie_db.json`** (181 MB)
  - Predefined movie database with 64,283 movies
  - Used by `run_kg_preprocess.py`
  - Required for full pipeline execution

- **`movie/person_db.json`**
  - Predefined person database with 373,608 persons
  - Used by `run_kg_preprocess.py`

- **`movie/year_db.json`**
  - Year/temporal data
  - Used by `run_kg_preprocess.py`

## Getting Started

### Option 1: Generate Tiny Dataset (Recommended for Testing)

```bash
cd ../scripts

# Generate 5 synthetic Q&A pairs for quick testing
python create_tiny_dataset.py --output ../dataset/crag_movie_tiny.jsonl

# Or create custom size
python create_truncated_dataset.py --max-examples 100
```

### Option 2: Use Existing Files

If you have access to the full datasets:
1. Ensure Neo4j is running: `bolt://localhost:7687`
2. Place data files in this directory
3. Run the pipeline:
   ```bash
   cd ../scripts
   bash run.sh
   ```

### Option 3: Acquire Original Data

For the full movie database:
- Movie database files come from the TMDB dataset
- Contact project maintainers for access

## Testing Without Data Files

Use `--mock` flag to test without data:

```bash
# Test preprocessing with mock backend
python run_kg_preprocess.py --data-dir ./movie --mock

# Test embedding with mock backend
python run_kg_embed.py --mock

# Test updating with mock backend
python run_kg_update.py --dataset crag_movie_tiny.jsonl.bz2 --mock
```

## File Sizes

These files are not tracked in git due to size:

| File | Size | Note |
|------|------|------|
| crag_movie_dev.jsonl.bz2 | 140 MB | Full CRAG dataset |
| movie_db.json | 181 MB | Predefined movies |
| person_db.json | ~44 MB | Predefined persons |
| year_db.json | ~1.1 MB | Year data |

**Total**: ~370 MB (too large for efficient git tracking)

## Local Development

Files in this directory are:
- ✓ Kept in your local checkout
- ✓ Not pushed to GitHub
- ✓ Ignored by git (see `.gitignore`)
- ✓ Safe for local experimentation

## For Docker/Cloud Deployment

If deploying with Docker, you can:
1. Mount this directory as a volume
2. Generate data files at container startup
3. Download from cloud storage

## Questions?

See the main [README.md](../README.md) for troubleshooting and configuration details.
