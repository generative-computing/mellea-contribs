"""Generic dataset loader for KG update documents.

Reads a JSONL (or ``.jsonl.bz2``) file and yields normalised document items
suitable for ``orchestrate_kg_update``.

Each yielded item contains:

* ``id``   — document identifier (``interaction_id``, ``id``, or
  ``"doc_{n}"``).
* ``text`` — document body (``text``, ``query``, or ``context`` field).
* ``_raw`` — the original dict from the file.
"""

from typing import Any, Dict, Generator

from mellea_contribs.kg.utils.data_utils import BaseDatasetLoader, load_jsonl
from mellea_contribs.kg.utils.progress import log_progress


class UpdateDatasetLoader(BaseDatasetLoader):
    """Dataset loader for KG update documents.

    Reads a JSONL (or ``.jsonl.bz2``) file and yields normalised document
    items suitable for ``orchestrate_kg_update``.
    """

    def iter_items(self) -> Generator[Dict[str, Any], None, None]:
        """Yield normalised document items from the dataset file.

        Yields:
            Dict with keys ``id``, ``text``, and ``_raw``.
        """
        for doc_num, raw in enumerate(load_jsonl(self.dataset_path), 1):
            doc_id = (
                raw.get("interaction_id")
                or raw.get("id")
                or f"doc_{doc_num}"
            )
            text = (
                raw.get("text")
                or raw.get("query")
                or raw.get("context")
                or ""
            )
            if not text:
                log_progress(f"[{doc_num}] WARNING: empty text for {doc_id}")
                continue
            yield {"id": str(doc_id), "text": text, "_raw": raw}
