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

        Converts Entity/Relation models to GraphNode/GraphEdge and writes them to
        the backend using upsert semantics. Relations are linked by name lookup.

        Args:
            result: ExtractionResult to persist
            doc_id: Document ID for tracking provenance
            merge_strategy: Strategy for handling existing entities (unused; upsert handles merging)

        Returns:
            Dictionary with persisted entity/edge IDs and statistics
        """
        persisted: dict[str, Any] = {
            "entity_ids": {},   # entity name → backend-assigned ID
            "edge_ids": [],     # list of persisted edge IDs
            "skipped_relations": [],
            "stats": {},
        }

        # Step 1: upsert every entity and collect name → ID map
        name_to_id: dict[str, str] = {}
        for i, entity in enumerate(result.entities):
            node = GraphNode(
                id=f"{doc_id}_entity_{i}",
                label=entity.type,
                properties={
                    "name": entity.name,
                    "description": entity.description,
                    "confidence": entity.confidence,
                    "source_doc": doc_id,
                    **entity.properties,
                },
            )
            assigned_id = await self.backend.upsert_entity(node)
            name_to_id[entity.name] = assigned_id
            persisted["entity_ids"][entity.name] = assigned_id
            logger.debug(f"Upserted entity '{entity.name}' → {assigned_id}")

        # Step 2: upsert each relation, resolving source/target by name
        for i, relation in enumerate(result.relations):
            src_id = name_to_id.get(relation.source_entity)
            tgt_id = name_to_id.get(relation.target_entity)

            # Fall back to a name search for entities not extracted in this doc
            if src_id is None:
                candidates = await self.backend.search_entities_by_name(
                    relation.source_entity, k=1
                )
                if candidates:
                    src_id = candidates[0].id
                    name_to_id[relation.source_entity] = src_id

            if tgt_id is None:
                candidates = await self.backend.search_entities_by_name(
                    relation.target_entity, k=1
                )
                if candidates:
                    tgt_id = candidates[0].id
                    name_to_id[relation.target_entity] = tgt_id

            if src_id is None or tgt_id is None:
                logger.debug(
                    f"Skipping relation '{relation.relation_type}': "
                    f"could not resolve "
                    f"{'source' if src_id is None else 'target'} entity"
                )
                persisted["skipped_relations"].append(
                    {
                        "relation_type": relation.relation_type,
                        "source_entity": relation.source_entity,
                        "target_entity": relation.target_entity,
                    }
                )
                continue

            src_node = GraphNode(id=src_id, label="Entity", properties={})
            tgt_node = GraphNode(id=tgt_id, label="Entity", properties={})
            edge = GraphEdge(
                id=f"{doc_id}_rel_{i}",
                label=relation.relation_type,
                source=src_node,
                target=tgt_node,
                properties={
                    "description": relation.description,
                    "source_doc": doc_id,
                    **relation.properties,
                },
            )
            assigned_edge_id = await self.backend.upsert_relation(edge)
            persisted["edge_ids"].append(assigned_edge_id)
            logger.debug(
                f"Upserted relation '{relation.relation_type}' → {assigned_edge_id}"
            )

        persisted["stats"] = {
            "entities_persisted": len(persisted["entity_ids"]),
            "relations_persisted": len(persisted["edge_ids"]),
            "relations_skipped": len(persisted["skipped_relations"]),
        }
        return persisted

    async def close(self):
        """Close connections and cleanup resources."""
        await self.backend.close()
