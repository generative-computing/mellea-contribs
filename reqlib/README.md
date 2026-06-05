# mellea-contribs-reqlib

Domain-specific `Requirement` recipes for Mellea's Instruct-Validate-Repair
patterns. Recipes under `mellea_contribs.reqlib.stdlib.reqlib` are
self-contained validators tailored to a particular problem space — legal
citations, Python imports, grounded context formatting — and can be mixed
and matched with mellea's standard requirements.

## Install

```bash
pip install mellea-contribs[reqlib]
```

This pulls in the full reqlib (all recipes). The transitive footprint is
small enough that we don't bother with per-recipe extras.

## Recipes

- **Legal** — citation existence checks, appellate-case classification,
  statute lookup
  - `citation_exists`, `is_appellate_case`, `statute_data`
- **Python imports** — import repair, import resolution, common-alias
  detection
  - `import_repair`, `import_resolution`, `common_aliases`
- **Grounding context** — grounding-context formatter for retrieval-aware
  prompts
  - `grounding_context_formatter`

Import a recipe directly:

```python
from mellea_contribs.reqlib.stdlib.reqlib.citation_exists import CaseNameExistsInDatabase
from mellea_contribs.reqlib.stdlib.reqlib.import_repair import PythonImportRepair
from mellea_contribs.reqlib.stdlib.reqlib.grounding_context_formatter import GroundingContextFormatter
```

The top-level `mellea_contribs.reqlib` module does not eagerly import any
recipes, so importing the package itself is cheap.
