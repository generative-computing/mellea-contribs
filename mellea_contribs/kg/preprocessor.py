"""KG Preprocessor: Layer 2 library for preprocessing raw data into KG entities/relations.

This module provides generic preprocessing infrastructure for converting raw documents
into Knowledge Graph entities and relations using LLM-based extraction.

Example::

    import asyncio
    from mellea import start_session
    from mellea_contribs.kg import MockGraphBackend
    from mellea_contribs.kg.preprocessor import KGPreprocessor

    async def main():
        session = start_session(backend_name="litellm", model_id="gpt-4o-mini")
        backend = MockGraphBackend()
        processor = KGPreprocessor(backend=backend, session=session)

        # Process a document
        doc = {"text": "Avatar was directed by James Cameron in 2009."}
        result = await processor.process_document(
            doc_text=doc["text"],
            domain="movies",
            doc_id="doc_1"
        )
        print(f"Extracted {len(result.entities)} entities and {len(result.relations)} relations")
        await backend.close()

    asyncio.run(main())
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

try:
    from mellea import MelleaSession
except ImportError:
    MelleaSession = None

try:
    from mellea_contribs.kg.components import (
        extract_entities_and_relations,
    )
    from mellea_contribs.kg.components.persistence import (
        persist_entities,
        persist_relations,
    )
except ImportError:
    extract_entities_and_relations = None
    persist_entities = None
    persist_relations = None

try:
    from mellea_contribs.kg.graph_dbs.base import GraphBackend
except ImportError:
    GraphBackend = None  # type: ignore[assignment,misc]

from mellea_contribs.kg.models import ExtractionResult

logger = logging.getLogger(__name__)


class KGPreprocessor(ABC):
    """Generic base class for preprocessing raw data into KG entities and relations.

    Orchestrates entity/relation extraction from documents and handles storage.
    Subclasses should override get_hints() and optionally post_process_extraction().

    Extraction is performed via LLM, persistence via the provided GraphBackend.
    """

    def __init__(
        self,
        backend: GraphBackend,
        session: MelleaSession,
        domain: str = "generic",
        batch_size: int = 10,
    ):
        """Initialize the preprocessor.

        Args:
            backend: GraphBackend instance (Layer 4) for storing entities/relations
            session: MelleaSession for LLM operations
            domain: Domain name (used in extraction hints)
            batch_size: Number of documents to process in parallel
        """
        self.backend = backend
        self.session = session
        self.domain = domain
        self.batch_size = batch_size

    @abstractmethod
    def get_hints(self) -> str:
        """Get domain-specific hints for the LLM extraction.

        Should be overridden by subclasses to provide domain-specific guidance.

        Returns:
            String with domain hints for LLM extraction
        """
        pass

    async def process_document(
        self,
        doc_text: str,
        doc_id: Optional[str] = None,
        entity_types: str = "",
        relation_types: str = "",
    ) -> ExtractionResult:
        """Process a single document to extract entities and relations.

        Uses Layer 3 extract_entities_and_relations function to call the LLM.

        Args:
            doc_text: The document text to process
            doc_id: Optional document ID for tracking
            entity_types: Optional comma-separated list of entity types to extract
            relation_types: Optional comma-separated list of relation types to extract

        Returns:
            ExtractionResult with extracted entities and relations
        """
        logger.info(f"Processing document {doc_id} with {len(doc_text)} chars")

        # Layer 3: Extract entities and relations using LLM
        result = await extract_entities_and_relations(
            self.session,
            doc_context=doc_text,
            domain=self.domain,
            hints=self.get_hints(),
            reference=doc_id or "unknown",
            entity_types=entity_types,
            relation_types=relation_types,
        )

        # Post-process if needed (can be overridden by subclasses)
        result = await self.post_process_extraction(result, doc_text)

        logger.info(
            f"Extracted {len(result.entities)} entities and {len(result.relations)} relations"
        )
        return result

    async def post_process_extraction(
        self, result: ExtractionResult, doc_text: str
    ) -> ExtractionResult:
        """Post-process extracted entities and relations.

        Can be overridden by subclasses for domain-specific processing.

        Args:
            result: The extraction result from LLM
            doc_text: The original document text

        Returns:
            Modified extraction result
        """
        # Default: no post-processing
        return result

    async def persist_extraction(
        self,
        result: ExtractionResult,
        doc_id: str,
        merge_strategy: str = "merge_if_similar",
    ) -> dict[str, Any]:
        """Persist extracted entities and relations to the KG.

        Delegates to Layer 3 ``persist_entities`` and ``persist_relations``
        executor functions, which handle all backend interactions.

        Args:
            result: ExtractionResult to persist
            doc_id: Document ID for tracking provenance
            merge_strategy: Strategy for handling existing entities (unused; upsert handles merging)

        Returns:
            Dictionary with persisted entity/edge IDs and statistics
        """
        # Step 1: upsert every entity via Layer 3 executor
        name_to_id = await persist_entities(self.backend, result.entities, doc_id)

        # Step 2: upsert each relation via Layer 3 executor
        edge_ids, skipped_relations = await persist_relations(
            self.backend, result.relations, name_to_id, doc_id
        )

        return {
            "entity_ids": name_to_id,
            "edge_ids": edge_ids,
            "skipped_relations": skipped_relations,
            "stats": {
                "entities_persisted": len(name_to_id),
                "relations_persisted": len(edge_ids),
                "relations_skipped": len(skipped_relations),
            },
        }

    async def close(self):
        """Close connections and cleanup resources."""
        await self.backend.close()
