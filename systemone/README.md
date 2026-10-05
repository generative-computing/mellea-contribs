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

