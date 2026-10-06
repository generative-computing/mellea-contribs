"""Image decisions with a multimodal Clef model.

Run it::

    ollama pull clef-flash && ollama serve
    python examples/image_checks.py path/to/receipt.jpg
    python examples/image_checks.py photo1.png photo2.png --model clef

What this shows
---------------
- Annotate a parameter as ``Annotated[bytes, Image()]`` and ``@decisive`` sends
  it as an image, not as text.  Text parameters still go in as context.
- The example is an expense-claim intake check: what kind of document was
  uploaded, whether it can be read, and whether it agrees with what the
  employee typed in.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Annotated, Literal

from mellea_contribs.systemone import Image, OllamaJudge, decisive


@decisive
def document_type(
    upload: Annotated[bytes, Image()],
) -> Literal["receipt", "invoice", "boarding_pass", "screenshot", "photo", "other"]:
    """What kind of document this uploaded image shows."""


@decisive
def is_legible(upload: Annotated[bytes, Image()]) -> bool:
    """Whether the printed text and amounts in the image can be read clearly."""


@decisive
def matches_claim(claim: str, upload: Annotated[bytes, Image()]) -> bool:
    """Whether the merchant and total shown in the image agree with the expense claim."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "images", nargs="+", type=pathlib.Path, help="Image files (PNG/JPEG/WebP)"
    )
    parser.add_argument("--claim", default="Lunch with client, Blue Door Cafe, $48.20")
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama URL (default: $OLLAMA_HOST or localhost:11434)",
    )
    parser.add_argument(
        "--model",
        default="clef-flash",
        help="clef or clef-flash (text-only models ignore images)",
    )
    args = parser.parse_args()

    judge = OllamaJudge(host=args.host, model=args.model)
    print(f"claim: {args.claim!r}\n")

    for path in args.images:
        upload = path.read_bytes()
        kind = document_type.verdict(judge, upload=upload)
        legible = is_legible.verdict(judge, upload=upload)
        matches = matches_claim.verdict(judge, claim=args.claim, upload=upload)

        print(path.name)
        print(f"  document_type = {kind.value}  (confidence {kind.confidence})")
        print(f"  is_legible    = {legible.value}  (score {legible.confidence})")
        print(f"  matches_claim = {matches.value}  (score {matches.confidence})")

        ok = kind.value in ("receipt", "invoice") and legible.value and matches.value
        print(f"  -> {'accept' if ok else 'ask employee to re-upload'}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
