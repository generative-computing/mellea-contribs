# mellea-contribs-systemone

Calibrated decision models — [GLiNER2](https://github.com/fastino-ai/GLiNER2) and
[Jev](https://docs.typesafe.ai/) and systemone models served Jev api on ollama — as typed Mellea decision functions.

## What this package provides

Two non-generative decision models made usable inside Mellea through
`@decisive`, a sibling decorator to Mellea's `@generative` for closed
decisions. Every answer comes back with a confidence attached.

It rests on one small `Judge` protocol, so the same task can be run against
any provider and compared.

## Installation

```bash
# Core only — no model deps
pip install mellea-contribs-systemone

# With GLiNER2 (torch-free inference engine)
pip install "mellea-contribs-systemone[gliner2]"

# With GLiNER2 + torch (local model execution)
pip install "mellea-contribs-systemone[local]"

# With Jev (TypeSafe hosted API)
pip install "mellea-contribs-systemone[jev]"

# With Ollama (local tev1/Clef)
pip install "mellea-contribs-systemone[ollama]"

# Everything
pip install "mellea-contribs-systemone[all]"
```

## Quick start


### `@decisive`

Runs locally against Ollama's Clef-Flash model — no API key needed (see
[Ollama / tev1](#ollama--tev1-local-no-api-key) for setup):

```python
from typing import Literal

from mellea_contribs.systemone import OllamaJudge, decisive

judge = OllamaJudge(model="clef-flash")  # localhost:11434, Clef-Flash


@decisive
def triage(ticket: str) -> Literal["billing", "bug", "feature"]:
    """Route a support ticket to the owning team."""


team = triage(judge=judge, ticket=text)  # -> "billing"
v = triage.verdict(
    judge=judge, ticket=text
)  # -> Verdict(value="billing", confidence=0.93, ...)
```

Unsupported return annotations raise `UnsupportedReturnType` **at decoration
time**, so a mis-typed stub fails at import rather than in production.

## Providers

| Provider | Class | Implements | Extra |
|---|---|---|---|
| GLiNER2 | `Gliner2Judge` | `Judge` | `gliner2` / `local` |
| Jev (TypeSafe) | `JevJudge` | `Judge` only | `jev` |
| Ollama tev1/Clef | `OllamaJudge` | `Judge` only (+ images for Clef) | `ollama` |
| Fake (tests) | `FakeJudge` | `Judge` | none |

### Ollama / tev1 (local, no API key)

`OllamaJudge` wraps the TypeSafe SDK pointed at Ollama's `/v1/systemone`
endpoint — no env-var hacking needed:

```bash
ollama pull tev1              # 4.4 GB; or tev1:0.8b for 800 MB
ollama pull clef-flash        # smaller multimodal; used in the Quick start
ollama serve                  # default: localhost:11434
```

```python
from mellea_contribs.systemone import OllamaJudge

judge = OllamaJudge()                              # localhost:11434, tev1
judge = OllamaJudge(host="http://gpu:11434")        # remote machine
judge = OllamaJudge(model="clef-flash")             # Clef-Flash multimodal

v = judge.truth(state="The sky is blue.", instructions="Is the sky blue?")
```

`@decisive` and `Judge.batch` work the same as with hosted Jev.

#### Image support (Clef models)

Clef (27B) and Clef-Flash (smaller) are multimodal models that accept images alongside
text questions. Annotate `@decisive` parameters with `Image()` to route
image bytes through the Judge protocol:

```python
from typing import Annotated, Literal
from mellea_contribs.systemone import OllamaJudge, Image, decisive

@decisive
def classify_photo(
    caption: str,
    photo: Annotated[bytes, Image()],
) -> Literal["indoor", "outdoor", "diagram"]:
    """Classify a photo by scene type."""

judge = OllamaJudge(model="clef-flash")
result = classify_photo(judge=judge, caption="office shot", photo=image_bytes)
```

`Image()`-annotated parameters are excluded from the state dict and base64-
encoded into the `images` field of the API call. Multiple image parameters
are collected into a single list. Non-Clef models (tev1) ignore the images
field.

See `examples/ollama_tev1.py` for a full walkthrough.

## Running the tests

### Unit tests (hermetic, no network, no torch)

```bash
cd systemone
uv venv --python 3.12 .venv
uv pip install -e ".[all]" --group dev
.venv/bin/python -m pytest -m unit
```

Only the `unit` tier runs in CI. `Gliner2Judge` accepts an injected `model`, so
its unit tests use a stub and download nothing.

### Integration tests (real GLiNER2 checkpoint)

These tests exercise the seam between systemone and a real GLiNER2 model — the
`truth` inversion, the batch composition, and the full decorator stack.  They require `torch` and a GPU for reasonable performance.

#### Quick start (GPU node with bsub)

```bash
cd systemone

# Submit to a GPU node:
bsub -G grp_data \
     -J "systemone-integ" \
     -R "select[ngpus>0] rusage[ngpus_physical=1]" \
     -o "integration_%J.log" -e "integration_%J.err" \
     bash scripts/run_integration.sh

# Watch progress:
bjobs -J systemone-integ
# When done, check the log:
cat integration_*.log
```

The script handles venv creation, dependency installation, and runs both the
comprehensive smoke test (`examples/run_all_real.py`) and the pytest
integration tier.

> **Note:** The `[local]` extra pulls in `torch` (~530 MB).  If your home
> directory has a quota, the script sets `UV_CACHE_DIR` to the project area
> automatically.  You can also set it yourself:
>
> ```bash
> export UV_CACHE_DIR=/path/with/space/.uv_cache
> ```

#### Running directly on a GPU machine

```bash
cd systemone
uv venv --python 3.12 .venv
uv pip install -e ".[local,all]" --group dev

# Compatibility: transformers 4.57.x has a tokenizer bug with DeBERTa-v2
# checkpoints used by GLiNER2. Pin to 4.52.x and add protobuf/sentencepiece:
uv pip install "transformers>=4.50,<4.53" "protobuf>=3.20" "sentencepiece>=0.1.99"

# Comprehensive smoke test (every feature against real model):
.venv/bin/python examples/run_all_real.py

# Or with a specific checkpoint:
.venv/bin/python examples/run_all_real.py --checkpoint fastino/gliner2.5-small-v1

# Pytest integration tier:
.venv/bin/python -m pytest -m integration -v
```

#### Running the example with real GLiNER2

`examples/decision_functions.py` supports `--real` to swap the scripted
`FakeJudge` for a real `Gliner2Judge`:

```bash
.venv/bin/python examples/decision_functions.py --real
```

#### What the integration tests validate

| Test | What it catches |
|---|---|
| Protocol conformance | `Gliner2Judge` satisfies `Judge` |
| `choose` | Label returned is one of the options |
| `truth` direction | A true claim scores higher than a false one (catches inverted `"no"` mapping) |
| `rate` | Returned label is in the rubric |
| `batch` | All question keys answered in one pass |
| `@decisive` | Literal, bool, and rated returns work end-to-end |

### E2E tests (live Jev API)

```bash
export TYPESAFE_API_KEY=...
.venv/bin/python -m pytest -m e2e
```

## Confidence note

GLiNER2 confidence is softmax over label logits — **not calibrated**. Thresholds
must be tuned per task. Jev confidence is RLCD-trained. Comparing the two
numbers directly is not meaningful.

## Status

- **Phase 1 (done):** Core protocol, `FakeJudge`, package scaffold.
- **Phase 2 (done):** `Gliner2Judge` against real checkpoints.
- **Phase 3 (done):** `JevJudge` — experimental, behind the `jev` extra.
- **Phase 4 (done):** `@decisive`.
- **Phase 5 (done):** `OllamaJudge` + `Image` annotation for Clef multimodal.

### Validation status

**`Gliner2Judge` validated against real weights** (2026-09-23, H100, `fastino/gliner2.5-small-v1`,
`transformers==4.52.4`):

- Protocol conformance, `choose`, `truth` (direction), `rate`, `batch`,
  `@decisive`: all pass.
- The decision-function example scripts run end-to-end with `--real`.

**Known compatibility issue:** `transformers>=4.53` has a tokenizer bug with
DeBERTa-v2 checkpoints (`extra_special_tokens` is a list but `.keys()` is
called). Pin to `transformers>=4.50,<4.53` and install `protobuf` +
`sentencepiece`. The `scripts/run_integration.sh` runner applies this
automatically.

**`JevJudge` against the live API:** Written against the SDK's response
models and exercised with stubs built from those same types; no request has
been made to the hosted service.
