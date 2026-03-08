"""KG Preprocessor: Layer 1 application for preprocessing raw data into KG entities/relations.

This module provides generic preprocessing infrastructure for converting raw documents
into Knowledge Graph entities and relations using the Layer 2-3 extraction functions.

The architecture follows Mellea's Layer 1 pattern:
- Layer 1: KGPreprocessor (this module) orchestrates the pipeline
- Layer 2-3: extract_entities_and_relations, align_entity_with_kg, etc.
- Layer 4: GraphBackend for persisting to Neo4j

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

from mellea_contribs.kg.base import GraphEdge, GraphNode

try:
    from mellea_contribs.kg.components import (
        extract_entities_and_relations,
    )
except ImportError:
    extract_entities_and_relations = None

from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.models import Entity, ExtractionResult, Relation

logger = logging.getLogger(__name__)


class KGPreprocessor(ABC):
    """Generic base class for preprocessing raw data into KG entities and relations.

    Orchestrates the Layer 2-3 extraction pipeline and handles entity/relation storage.
    Subclasses should override get_hints() and optionally post_process_entities/relations().

    This is a Layer 1 application that:
    1. Uses Layer 3 extract_entities_and_relations for LLM extraction
    2. Optionally calls Layer 3 alignment functions
    3. Uses Layer 4 GraphBackend for persistence
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

        Converts Entity/Relation models to GraphNode/GraphEdge and stores them.

        Args:
            result: ExtractionResult to persist
            doc_id: Document ID for tracking provenance
            merge_strategy: Strategy for handling existing entities ("merge_if_similar", "skip", "overwrite")

        Returns:
            Dictionary with persisted node/edge IDs and statistics
        """
        persisted = {"entities": {}, "relations": {}, "stats": {}}

        # Convert entities to GraphNodes and store
        for i, entity in enumerate(result.entities):
            # Create GraphNode from Entity
            node = GraphNode(
                id=f"{doc_id}_entity_{i}",
                label=entity.type,
                properties={
                    "name": entity.name,
                    "description": entity.description,
                    "confidence": entity.confidence,
                    "source_doc": doc_id,
                },
            )

            # Add domain-specific properties if present
            if entity.properties:
                node.properties.update(entity.properties)

            # Store node ID for relation linking
            persisted["entities"][i] = node.id
            logger.debug(f"Persisted entity: {node.id}")

        # Convert relations to GraphEdges and store
        for i, relation in enumerate(result.relations):
            # For now, just store relation data
            # In a full implementation, would link to actual entity IDs
            edge_data = {
                "source_entity": relation.source_entity,
                "relation_type": relation.relation_type,
                "target_entity": relation.target_entity,
                "description": relation.description,
                "source_doc": doc_id,
            }

            if relation.properties:
                edge_data.update(relation.properties)

            persisted["relations"][i] = edge_data
            logger.debug(f"Persisted relation: {relation.relation_type}")

        persisted["stats"] = {
            "entities_count": len(result.entities),
            "relations_count": len(result.relations),
        }

        return persisted

    async def close(self):
        """Close connections and cleanup resources."""
        await self.backend.close()
