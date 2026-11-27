"""Ollama-backed :class:`Judge` with optional image support.

Ollama's ``/v1/systemone`` endpoint speaks the same API as TypeSafe's Jev.
For text-only models (tev1, nimble), :class:`OllamaJudge` is functionally
identical to ``JevJudge`` pointed at Ollama.  For multimodal Clef models,
it additionally passes base64-encoded images via the ``images`` request field.

Capability boundary
-------------------
Like Jev, Ollama System One models produce **no string output**;
:class:`OllamaJudge` implements :class:`Judge`.

Image support
-------------
Images are passed as ``list[bytes]`` (raw PNG/JPEG/WebP).  OllamaJudge
handles base64 encoding internally.  Images are shared across all questions
in a batch — Ollama's API does not support per-question images.

Only Clef and Clef Flash models process images.  Sending images to a text-only
model (tev1, nimble) is harmless — Ollama ignores the ``images`` field.

Models
------
=============  ==========  =========  ===========================
Model          Parameters  Modality   Notes
=============  ==========  =========  ===========================
tev1           4B          text       Default
tev1:0.8b      0.8B        text       Smaller, faster
nimble         9B          text       Higher quality text
clef           27B         multimodal Accepts images
clef-flash     —           multimodal Smaller multimodal
=============  ==========  =========  ===========================

Installation
------------
::

    pip install "mellea-contribs-systemone[jev]"   # typesafe-sdk
    ollama pull tev1                               # or clef for images
"""

from __future__ import annotations

import base64
import os
from typing import Any

from mellea_contribs.systemone.backends.jev import JevJudge, _sdk

__all__ = ["DEFAULT_HOST", "OLLAMA_HOST_ENV", "OllamaJudge"]

DEFAULT_HOST = "http://localhost:11434"

OLLAMA_HOST_ENV = "OLLAMA_HOST"


class OllamaJudge(JevJudge):
    """A :class:`Judge` backed by a local or remote Ollama instance.

    Subclasses :class:`JevJudge` — the TypeSafe SDK pointed at Ollama's
    ``/v1/systemone`` endpoint.  Adds native image support for Clef models.

    Args:
        host: Ollama server URL.  Defaults to ``http://localhost:11434``
            or the ``OLLAMA_HOST`` environment variable.
        model: Model name.  Defaults to ``"tev1"``.
        client: A pre-built ``TypeSafeClient``.  When given, ``host`` is
            ignored — this is the injection point for unit tests.

    Example::

        judge = OllamaJudge()                       # default: localhost, tev1
        judge = OllamaJudge(model="clef")           # multimodal

        # Text-only — same as JevJudge
        v = judge.truth(state="The sky is blue.", instructions="Is the sky blue?")

        # With images — only Clef models process them
        v = judge.truth(
            state="Describe the scene.",
            instructions="Is there a cat in the photo?",
            images=[open("cat.png", "rb").read()],
        )
    """

    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self.model = model
            self.client = client
            return

        resolved_host = host or os.environ.get(OLLAMA_HOST_ENV) or DEFAULT_HOST
        resolved_model = model or "tev1"

        sdk = _sdk()
        built_client = sdk.TypeSafeClient(
            api_key="ollama",
            base_url=resolved_host,
            model=resolved_model,
        )
        super().__init__(client=built_client, model=resolved_model)

    def _call_system_one(
        self, sdk: Any, state: Any, payload: dict[str, Any], *, images: list[bytes] | None = None
    ) -> Any:
        """Execute the SDK call, injecting base64-encoded images when present."""
        if images:
            encoded = [base64.b64encode(img).decode("ascii") for img in images]
            return self.client.system_one(state, payload, extra_body={"images": encoded})
        return self.client.system_one(state, payload)
