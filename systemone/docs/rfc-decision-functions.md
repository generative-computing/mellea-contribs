# RFC: Decision Function Decorators for IBM Mellea

**Status:** Proposal  
**Date:** 2026-10-05  
**Package:** `mellea-contribs-systemone`  
**Author:** Shivdeep Singh

---

## Problem

Mellea's `@generative` decorator turns a type-annotated Python stub into an LLM call. This works well for open-ended generation, but many production tasks are closed decisions — routing a ticket, flagging urgency, rating severity — where the answer is one of N known options, not free text.

`@generative` already handles the *shape* of these answers: it builds a pydantic response model from the return annotation and passes it to the backend as `format=`, so a `-> Literal["billing", "bug", "feature"]` stub gets structured output rather than prose. What it does not address:

1. **Price and latency.** Every call is a full LLM generation, even when the answer is one label. A closed decision over a short document does not need a multi-billion-parameter decoder.
2. **No confidence.** `@generative` returns the value only. It does not tell you *how sure* the model is, so you cannot distinguish a clear-cut billing ticket from an ambiguous one without a second LLM call or backend-specific logprob plumbing.

Non-generative decision models (GLiNER2, Jev/TypeSafe) return typed values with per-prediction confidence in a single forward pass. They are a worse fit for generation and a better fit for these closed decisions.

## Proposal

Add a `@decisive` decorator as a sibling of `@generative`, backed by a `Judge` protocol that any decision model with per-prediction confidence can implement.

**Scope.** The decorator ships in `mellea-contribs-systemone`, not in Mellea core. This RFC asks for agreement on the API shape. Moving them into core is a separate decision, to be taken once the open questions below have answers and at least one provider other than GLiNER2 has been validated. Until then the API is experimental and may change between minor versions.

### `@decisive` — closed decisions with confidence

A `@decisive` function's return annotation selects the judge primitive:

```python
from typing import Literal
from mellea_contribs.systemone import decisive, Gliner2Judge

judge = Gliner2Judge()

@decisive
def triage(ticket: str) -> Literal["billing", "bug", "feature"]:
    """Route a support ticket to the owning team."""

# Returns the bare value
team = triage(judge=judge, ticket=text)        # "billing"

# Returns the full Verdict with confidence
v = triage.verdict(judge=judge, ticket=text)   # Verdict(value="billing", confidence=0.93, ...)
```

**Return type mapping:**

| Return annotation | Judge primitive | What the model does |
|---|---|---|
| `Literal["a", "b"]` / `StrEnum` | `Judge.choose` | Pick one option from a closed set |
| `bool` | `Judge.truth` | Score a boolean claim, threshold to True/False |
| `float` | `Judge.truth` | Return raw 0–1 probability |
| `Literal[...]` with `rubric=` | `Judge.rate` | Assign ordinal rating against a rubric |

Anything else raises `UnsupportedReturnType` **at decoration time** — a mis-typed stub fails at import, not in production.

The `Verdict` carries the confidence, so the caller decides what an unsure answer means for their task — accept it, route it for review, or hand it to a `@generative` call.

## The `Judge` protocol

The decorator is provider-agnostic. Any model implementing the protocol can serve them:

```python
class Judge(Protocol):
    name: str
    def choose(self, state, options, instructions, *, images=None) -> Verdict[str]: ...
    def truth(self, state, instructions, *, images=None) -> Verdict[float]: ...
    def rate(self, state, rubric, instructions, *, images=None) -> Verdict[str]: ...
    def batch(self, state, questions, *, images=None) -> dict[str, Verdict]: ...
```

`images` is a list of raw image bytes. Providers that cannot read images ignore it.

`batch` is not sugar over a loop — it is one network call for Jev and one forward pass for GLiNER2. N questions about the same document cost one pass, not N.

**Current providers:**

| Provider | Class | Implements | Notes |
|---|---|---|---|
| GLiNER2 | `Gliner2Judge` | `Judge` | Local, Apache-2.0, 74M–340M params |
| Jev (TypeSafe) | `JevJudge` | `Judge` | Hosted API, RLCD-calibrated confidence |
| Ollama tev1/Clef | `OllamaJudge` | `Judge` (+ images for Clef) | Local via Ollama, dedicated class with image support |
| Fake (tests) | `FakeJudge` | `Judge` | Scripted verdict queues for hermetic tests |

The protocol is the integration point — not the provider. `OllamaJudge` subclasses `JevJudge` (Ollama's `/v1/systemone` endpoint speaks the same API as Jev) and adds native image support for Clef multimodal models. The `Image` annotation on `@decisive` parameters routes image bytes through the Judge protocol as base64-encoded payloads in a single API call.

## Design decisions

### Why a sibling of `@generative`, not a flag on it

`@generative` and `@decisive` have different failure modes. A generation can be wrong in unbounded ways, so it needs `requirements=` and a repair loop. A decision is schema-guaranteed to be one of the declared options — it cannot be malformed, only unconfident. So `@decisive` takes no `requirements=` or `strategy=`; low confidence is surfaced on the `Verdict` for the caller to act on.

### Why `UnsupportedReturnType` at decoration time

A function annotated `-> str` applied to `@decisive` cannot work — the judge returns labels, not prose. Failing at import makes it impossible to ship code that only breaks in production. The error message names the offending type and points at `@generative` for prose-returning functions.

## Alternatives considered

### `@generative` with constrained decoding and logprobs

Keep `@generative -> Literal[...]` and derive confidence from the token probabilities of the chosen label. This needs no new decorator. It was not chosen because:

- Every call still pays for a full LLM, which is the cost this proposal removes.
- Logprob access differs per backend, and label probabilities over multi-token labels need extra handling.
- It remains a useful **baseline**. The calibration plan below compares against it.

### A judge backend for `@generative`

Register GLiNER2/Jev as a Mellea `Backend` so that `@generative` dispatches to them. This was not chosen because a `Backend` is expected to generate text. A judge cannot serve `-> str` stubs, and that mismatch would surface at call time rather than at decoration time.

## Calibration and evaluation plan

Every threshold in this proposal depends on confidence, and GLiNER2 confidence is uncalibrated softmax. Before the defaults are trusted, measure them:

- **Closed decisions:** not yet planned. A labelled routing set should report accuracy against confidence (a reliability curve), and accuracy at several confidence cut-offs, with `@generative -> Literal[...]` as the baseline.

## Errors at call time

| Situation | Behaviour |
|---|---|
| Provider SDK not installed | `JudgeUnavailable` with an install hint |
| Jev / Ollama rejects credentials | `JudgeUnavailable` |

Not yet specified: network timeouts and connection errors (currently the SDK's own exceptions propagate), and a provider response that omits the answer for one question in a batch. Both should get a defined behaviour before a move to core.

## Validation status

**`Gliner2Judge`** — validated on H100 with real weights (`fastino/gliner2.5-small-v1`, `transformers==4.52.4`):

- All `Judge` methods, `@decisive`, and the example scripts verified end-to-end

No latency numbers have been measured yet. Any performance claim should cite hardware, checkpoint, and input size once it is.

**`JevJudge`** — implemented against the SDK's response models and exercised with stubs. No request has been made to the live API.

**`OllamaJudge`** — unit-tested with an injected client only. No integration or e2e test runs against a live Ollama server.

## Open questions

1. **Threshold defaults.** `@decisive` defaults the `bool` threshold to 0.5. GLiNER2 confidence is uncalibrated softmax, so these are starting points. Should the framework mandate calibration, or leave it to the user?
2. **Batch composition across stubs.** Currently each `@decisive` call is independent. Could multiple stubs over the same document be composed into a single `Judge.batch` call automatically?
3. **Jev and Ollama validation.** Neither `JevJudge` nor `OllamaJudge` has run against a live service. Should validation against at least one of them be a prerequisite for merging?
