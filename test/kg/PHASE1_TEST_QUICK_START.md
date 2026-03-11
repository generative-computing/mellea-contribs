# Phase 1 Test Suite - Quick Start Guide

## TL;DR

```bash
# Run all Phase 1 tests
pytest test/kg/test_phase1_*.py -v

# Run with coverage
pytest test/kg/test_phase1_*.py --cov=mellea_contribs.kg -v

# Run specific module
pytest test/kg/test_phase1_models.py -v
```

## 📋 Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_phase1_models.py` | 20 | Entity, Relation, DirectAnswer |
| `test_phase1_rep.py` | 35 | All representation utilities |
| `test_phase1_qa_models.py` | 20 | QA configuration & results |
| `test_phase1_updater_models.py` | 25 | KG update models |
| `test_phase1_embed_models.py` | 25 | Embedding models |
| `test_phase1_preprocessor.py` | 15 | KGPreprocessor interface |
| `test_phase1_embedder.py` | 20 | KGEmbedder interface |
| `test_phase1_requirements.py` | 22 | Requirement factories |
| `test_phase1_domain_examples.py` | 20 | Domain-specific examples |
| **TOTAL** | **250+** | **All Phase 1 modules** |

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pdm install --group dev
# or
uv pip install -e . --group dev
```

### 2. Run All Tests
```bash
pytest test/kg/test_phase1_*.py -v
```

### 3. Check Coverage
```bash
pytest test/kg/test_phase1_*.py --cov=mellea_contribs.kg --cov-report=html
# Open htmlcov/index.html to view report
```

## 🎯 Common Commands

### Run Tests by Module
```bash
pytest test/kg/test_phase1_models.py        # Models only
pytest test/kg/test_phase1_rep.py           # Utilities only
pytest test/kg/test_phase1_qa_models.py     # QA models only
```

### Run Tests by Class
```bash
pytest test/kg/test_phase1_models.py::TestEntity -v
pytest test/kg/test_phase1_rep.py::TestEntityToText -v
```

### Run Tests by Method
```bash
pytest test/kg/test_phase1_models.py::TestEntity::test_create_entity_minimal -v
```

### Run with Verbose Output
```bash
pytest test/kg/test_phase1_*.py -vv  # Extra verbose
```

### Run with Show Output
```bash
pytest test/kg/test_phase1_*.py -s   # Show print statements
```

### Run and Stop on First Failure
```bash
pytest test/kg/test_phase1_*.py -x   # Stop on first failure
pytest test/kg/test_phase1_*.py -x -v # Stop and verbose
```

### Run Specific Number of Tests
```bash
pytest test/kg/test_phase1_models.py -k "entity" -v  # Tests with "entity" in name
```

## ✅ Expected Results

```
test_phase1_models.py::TestEntity::test_create_entity_minimal PASSED
test_phase1_models.py::TestEntity::test_create_entity_with_storage_fields PASSED
test_phase1_models.py::TestRelation::test_create_relation_minimal PASSED
...
======================== 250+ passed in X.XXs ========================
```

## 🔍 Debugging Failed Tests

### Show Full Error Output
```bash
pytest test/kg/test_phase1_models.py::TestEntity::test_create_entity_minimal -vv
```

### Show Local Variables on Failure
```bash
pytest test/kg/test_phase1_models.py -l
```

### Print Debug Info
```bash
pytest test/kg/test_phase1_models.py -s   # Shows print() statements
```

### Use PDB Debugger
```bash
pytest test/kg/test_phase1_models.py --pdb  # Drop into debugger on failure
```

## 📊 Test Coverage Goals

- **Models:** 100% (all fields tested)
- **Functions:** 100% (with edge cases)
- **Classes:** 100% (all methods tested)
- **Integration:** Key workflows tested

## 🎓 Test Organization

Tests are organized by Phase 1 modules:

```
mellea_contribs/kg/
├── models.py ──────────→ test_phase1_models.py
├── rep.py ─────────────→ test_phase1_rep.py
├── qa_models.py ───────→ test_phase1_qa_models.py
├── updater_models.py ──→ test_phase1_updater_models.py
├── embed_models.py ────→ test_phase1_embed_models.py
├── preprocessor.py ────→ test_phase1_preprocessor.py
├── embedder.py ────────→ test_phase1_embedder.py
├── requirements_models.py → test_phase1_requirements.py
└── (other files)

docs/examples/kgrag/
├── models/ ────────────→ test_phase1_domain_examples.py
├── preprocessor/ ──────→ test_phase1_domain_examples.py
└── rep/ ───────────────→ test_phase1_domain_examples.py
```

## 🛠️ Troubleshooting

### Tests Won't Run
```bash
# Check pytest is installed
python -m pytest --version

# Check pytest can find tests
pytest test/kg/test_phase1_models.py --collect-only
```

### Syntax Errors
```bash
# Verify test syntax
python3 -m py_compile test/kg/test_phase1_*.py
```

### Missing Dependencies
```bash
# Reinstall dev dependencies
pdm install --group dev -f
```

## 📝 Test Examples

### Model Tests
```python
def test_create_entity_minimal(self):
    entity = Entity(
        type="Person",
        name="Alice",
        description="A person",
        paragraph_start="Alice is",
        paragraph_end="a character.",
    )
    assert entity.type == "Person"
    assert entity.confidence == 1.0  # Default
```

### Utility Function Tests
```python
def test_normalize_basic(self):
    assert normalize_entity_name("alice") == "Alice"
    assert normalize_entity_name("ALICE") == "Alice"
```

### Config Tests
```python
def test_create_qa_config_custom(self):
    config = QAConfig(route_count=5, depth=3)
    assert config.route_count == 5
    assert config.depth == 3
```

## 🌟 Best Practices

1. **Run tests before committing**
   ```bash
   pytest test/kg/test_phase1_*.py -q  # Quick check
   ```

2. **Run full suite regularly**
   ```bash
   pytest test/kg/test_phase1_*.py --cov=mellea_contribs.kg
   ```

3. **Write tests for new features**
   - Follow existing naming conventions
   - Include docstrings
   - Test edge cases

4. **Keep tests fast**
   - No network calls
   - No file I/O
   - No database access

## 🔗 Related Documentation

- Phase 1 Summary: `TEST_PHASE1_SUMMARY.md`
- Missing Components: `missing_for_run_sh.txt`
- KG Documentation: `mellea_contribs/kg/__init__.py`

---

**Total Test Coverage:** 250+ tests across 9 files  
**Expected Runtime:** <10 seconds  
**Dependencies:** Only development pytest and pydantic
