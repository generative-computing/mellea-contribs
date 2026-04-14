"""Movie dataset loader for CRAG-format QA data.

Reads the CRAG movie benchmark JSONL (plain or bz2-compressed) and yields
normalised QA items suitable for ``orchestrate_qa_retrieval``.

Expected input format per line::

    {
        "query": "Who directed Inception?",
        "query_time": "2024-03-05 00:00:00",
        "answer": "Christopher Nolan",
        "answer_aliases": ["Christopher Nolan", "Nolan"],
        "search_results": [...]   # ignored
    }

Each yielded item contains:

* ``id``            — ``"{prefix}{index}"`` string used for resumption.
* ``query``         — question text.
* ``query_time``    — original timestamp string.
* ``answer``        — canonical gold answer string.
* ``answer_aliases``— list of acceptable answer strings.
* ``_raw``          — the original dict from the file.
"""

from typing import Any, Dict, Generator

from mellea_contribs.kg.utils.data_utils import BaseDatasetLoader, load_jsonl


class MovieDatasetLoader(BaseDatasetLoader):
    """Dataset loader for the CRAG movie QA benchmark.

    Iterates over a JSONL (or ``.jsonl.bz2``) file and emits normalised QA
    items.  Supports slicing with ``prefix`` / ``postfix`` to process a
    sub-range of the dataset, which is useful for parallel batch jobs.

    Args:
        dataset_path: Path to the JSONL or ``.jsonl.bz2`` dataset file.
        num_workers: Number of parallel async workers (default: ``1``).
        queue_size: Internal asyncio queue capacity (default: ``100``).
        id_prefix: String prepended to the index to form the item ``id``
            (default: ``"q_"``).
        prefix: 0-based index of the first item to include (default: ``0``).
        postfix: Exclusive upper bound; ``None`` means read to end of file
            (default: ``None``).
    """

    def __init__(
        self,
        dataset_path: str,
        num_workers: int = 1,
        queue_size: int = 100,
        id_prefix: str = "q_",
        prefix: int = 0,
        postfix: int | None = None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            num_workers=num_workers,
            queue_size=queue_size,
        )
        self._id_prefix = id_prefix
        self._prefix = prefix
        self._postfix = postfix

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    def iter_items(self) -> Generator[Dict[str, Any], None, None]:
        """Yield normalised QA items from the dataset file.

        Items outside the [``prefix``, ``postfix``) index range are skipped
        silently.

        Yields:
            Dict with keys ``id``, ``query``, ``query_time``, ``answer``,
            ``answer_aliases``, and ``_raw``.
        """
        for global_idx, raw in enumerate(load_jsonl(self.dataset_path)):
            if global_idx < self._prefix:
                continue
            if self._postfix is not None and global_idx >= self._postfix:
                break

            item = self._normalise(raw, global_idx)
            if item is not None:
                yield item

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise(
        self, raw: Dict[str, Any], index: int
    ) -> Dict[str, Any] | None:
        """Convert a raw CRAG record into a normalised QA item.

        Args:
            raw: Raw dict from the JSONL file.
            index: Global 0-based position in the file.

        Returns:
            Normalised dict, or *None* when the record has no ``query``.
        """
        query = raw.get("query") or raw.get("question") or ""
        if not query:
            return None

        query_time = raw.get("query_time") or raw.get("query_date") or ""

        # Canonical answer
        answer = raw.get("answer") or ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        # Acceptable answer aliases
        aliases = raw.get("answer_aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        # Always include the canonical answer
        if answer and answer not in aliases:
            aliases = [answer] + list(aliases)

        item_id = f"{self._id_prefix}{index}"

        return {
            "id": item_id,
            "query": str(query),
            "query_time": str(query_time),
            "answer": str(answer),
            "answer_aliases": [str(a) for a in aliases],
            "_raw": raw,
        }
