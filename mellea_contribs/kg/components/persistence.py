"""Layer 3 persistence executor functions for KG updates.

These executor functions wrap backend write operations for entity and relation upserts.

Two patterns are covered:

* **Simple upsert** (``persist_entities``, ``persist_relations``): used by
  ``KGPreprocessor`` when raw entities/relations should be bulk-inserted
  without LLM alignment.

* **Align + upsert** (``align_and_upsert_entity``,
  ``align_and_upsert_relation``): used by ``orchestrate_kg_update`` when
  extracted entities/relations must first be compared against the existing
  KG via LLM calls before deciding to merge or insert.
"""

import logging
from typing import Any

from mellea_contribs.kg.base import GraphEdge, GraphNode
from mellea_contribs.kg.components.generative import (
    align_entity_with_kg,
    align_relation_with_kg,
    decide_entity_merge,
    decide_relation_merge,
)
from mellea_contribs.kg.models import Entity, Relation

try:
    from mellea_contribs.kg.graph_dbs.base import GraphBackend
except ImportError:
    GraphBackend = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


async def persist_entities(
    backend: "GraphBackend",
    entities: list[Entity],
    doc_id: str,
) -> dict[str, str]:
    """Upsert entities into the KG and return a name-to-ID map.

    Wraps ``backend.upsert_entity()`` (Layer 4) for each entity, building
    a ``{entity_name: assigned_id}`` mapping used to resolve relation
    endpoints in a subsequent ``persist_relations`` call.

    Args:
        backend: Graph database backend.
        entities: Extracted entities to persist.
        doc_id: Document ID for provenance tracking.

    Returns:
        Dictionary mapping entity name → backend-assigned node ID.
    """
    name_to_id: dict[str, str] = {}
    for i, entity in enumerate(entities):
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
        assigned_id = await backend.upsert_entity(node)
        name_to_id[entity.name] = assigned_id
        logger.debug(f"Upserted entity '{entity.name}' → {assigned_id}")
    return name_to_id


async def persist_relations(
    backend: "GraphBackend",
    relations: list[Relation],
    name_to_id: dict[str, str],
    doc_id: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Upsert relations into the KG, resolving entity IDs by name lookup.

    For entity endpoints not already in ``name_to_id``, falls back to
    ``backend.search_entities_by_name()`` (Layer 4). Relations whose
    endpoints cannot be resolved are skipped.

    Wraps:
    - ``backend.search_entities_by_name()`` (Layer 4 fallback lookup)
    - ``backend.upsert_relation()`` (Layer 4 write)

    Args:
        backend: Graph database backend.
        relations: Extracted relations to persist.
        name_to_id: Entity name → ID map from a prior ``persist_entities``
            call.  Updated in-place as additional entities are resolved.
        doc_id: Document ID for provenance tracking.

    Returns:
        Tuple of ``(edge_ids, skipped_relations)`` where ``edge_ids`` is the
        list of persisted edge IDs and ``skipped_relations`` is a list of
        dicts describing relations that could not be resolved.
    """
    edge_ids: list[str] = []
    skipped: list[dict[str, Any]] = []

    for i, relation in enumerate(relations):
        src_id = name_to_id.get(relation.source_entity)
        tgt_id = name_to_id.get(relation.target_entity)

        if src_id is None:
            candidates = await backend.search_entities_by_name(
                relation.source_entity, k=1
            )
            if candidates:
                src_id = candidates[0].id
                name_to_id[relation.source_entity] = src_id

        if tgt_id is None:
            candidates = await backend.search_entities_by_name(
                relation.target_entity, k=1
            )
            if candidates:
                tgt_id = candidates[0].id
                name_to_id[relation.target_entity] = tgt_id

        if src_id is None or tgt_id is None:
            missing = "source" if src_id is None else "target"
            logger.debug(
                f"Skipping relation '{relation.relation_type}': "
                f"could not resolve {missing} entity"
            )
            skipped.append(
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
        assigned_edge_id = await backend.upsert_relation(edge)
        edge_ids.append(assigned_edge_id)
        logger.debug(
            f"Upserted relation '{relation.relation_type}' → {assigned_edge_id}"
        )

    return edge_ids, skipped


# ---------------------------------------------------------------------------
# Align + upsert (Layer 3) — used by orchestrate_kg_update
# ---------------------------------------------------------------------------


async def align_and_upsert_entity(
    backend: "GraphBackend",
    session: Any,
    entity: Entity,
    doc_text: str,
    domain: str,
    align_topk: int = 10,
    confidence_threshold: float = 0.6,
) -> tuple[str, bool]:
    """Search KG for matching entity, then LLM-decide merge vs. insert.

    Combines:
    - ``backend.search_entities_by_name()`` (Layer 4)
    - ``align_entity_with_kg()`` (Layer 3 @generative)
    - ``decide_entity_merge()`` (Layer 3 @generative)
    - ``backend.upsert_entity()`` (Layer 4)

    Args:
        backend: Graph database backend.
        session: Mellea session for LLM calls.
        entity: Extracted entity to align or insert.
        doc_text: Source document excerpt for alignment context.
        domain: Knowledge domain hint.
        align_topk: Number of KG candidates to retrieve for alignment.
        confidence_threshold: Minimum LLM confidence to attempt a merge.

    Returns:
        Tuple of ``(entity_id, was_merged)`` where ``entity_id`` is the
        backend-assigned ID of the node and ``was_merged`` is ``True`` when
        the entity was merged with an existing KG node.
    """
    from mellea_contribs.kg.components.retrieval import node_to_text

    candidates = await backend.search_entities_by_name(entity.name, k=align_topk)
    aligned_id: str | None = None

    if candidates:
        candidates_str = "\n".join(
            f"id={n.id} | {node_to_text(n)}" for n in candidates
        )
        try:
            alignment = await align_entity_with_kg(
                session,
                extracted_entity_name=entity.name,
                extracted_entity_type=entity.type,
                extracted_entity_desc=entity.description,
                candidate_entities=candidates_str,
                domain=domain,
                doc_text=doc_text[:500],
            )
            if (
                alignment.aligned_entity_id
                and alignment.confidence >= confidence_threshold
            ):
                candidate_node = next(
                    (n for n in candidates if n.id == alignment.aligned_entity_id),
                    None,
                )
                if candidate_node is not None:
                    entity_pair = (
                        f"Extracted: {entity.type} '{entity.name}' — {entity.description}\n"
                        f"KG node:   {node_to_text(candidate_node)}"
                    )
                    merge_dec = await decide_entity_merge(
                        session,
                        entity_pair=entity_pair,
                        doc_text=doc_text[:500],
                        domain=domain,
                    )
                    if merge_dec.should_merge:
                        aligned_id = alignment.aligned_entity_id
                        merged_props = dict(candidate_node.properties)
                        merged_props.update(merge_dec.merged_properties)
                        merged_node = GraphNode(
                            id=aligned_id,
                            label=candidate_node.label,
                            properties=merged_props,
                        )
                        await backend.upsert_entity(merged_node)
                        return aligned_id, True
        except Exception:
            pass  # alignment failure — fall through to insert

    # No match — insert as new entity
    new_node = GraphNode(
        id=entity.id or f"entity_{entity.name.lower().replace(' ', '_')}",
        label=entity.type,
        properties={
            "name": entity.name,
            "description": entity.description,
            **entity.properties,
        },
    )
    try:
        new_id = await backend.upsert_entity(new_node)
        return new_id, False
    except Exception:
        return new_node.id, False


async def align_and_upsert_relation(
    backend: "GraphBackend",
    session: Any,
    relation: Relation,
    src_id: str,
    tgt_id: str,
    doc_text: str,
    domain: str,
    align_topk: int = 10,
    confidence_threshold: float = 0.6,
) -> tuple[str | None, bool]:
    """Search KG for matching relation, then LLM-decide merge vs. insert.

    Combines:
    - ``backend.get_triplets()`` (Layer 4 — retrieves existing relations)
    - ``align_relation_with_kg()`` (Layer 3 @generative)
    - ``decide_relation_merge()`` (Layer 3 @generative)
    - ``backend.upsert_relation()`` (Layer 4)

    Args:
        backend: Graph database backend.
        session: Mellea session for LLM calls.
        relation: Extracted relation to align or insert.
        src_id: Backend-assigned ID for the source entity.
        tgt_id: Backend-assigned ID for the target entity.
        doc_text: Source document excerpt for alignment context.
        domain: Knowledge domain hint.
        align_topk: Number of existing relations to retrieve for alignment.
        confidence_threshold: Minimum LLM confidence to attempt a merge.

    Returns:
        Tuple of ``(relation_id | None, was_merged)`` where ``relation_id``
        is the backend-assigned ID of the edge (or ``None`` on failure) and
        ``was_merged`` is ``True`` when the relation was merged.
    """
    from mellea_contribs.kg.components.retrieval import edge_to_triplet_text

    safe_rel = "".join(
        c for c in relation.relation_type if c.isalnum() or c == "_"
    )
    existing = await backend.get_triplets(src_id, safe_rel, k=align_topk)
    src_node = GraphNode(
        id=src_id, label="Entity", properties={"name": relation.source_entity}
    )
    tgt_node = GraphNode(
        id=tgt_id, label="Entity", properties={"name": relation.target_entity}
    )

    aligned_rel_id: str | None = None

    if existing:
        existing_str = "\n".join(
            f"id={e.id} | {edge_to_triplet_text(e)}" for e in existing
        )
        extracted_rel_str = (
            f"({relation.source_entity})-[{relation.relation_type}]"
            f"->({relation.target_entity}) | {relation.description}"
        )
        try:
            rel_alignment = await align_relation_with_kg(
                session,
                extracted_relation=extracted_rel_str,
                candidate_relations=existing_str,
                synonym_relations="",
                domain=domain,
                doc_text=doc_text[:500],
            )
            if (
                rel_alignment.aligned_entity_id
                and rel_alignment.confidence >= confidence_threshold
            ):
                candidate_edge = next(
                    (e for e in existing if e.id == rel_alignment.aligned_entity_id),
                    None,
                )
                if candidate_edge is not None:
                    rel_pair = (
                        f"Extracted: {extracted_rel_str}\n"
                        f"KG edge:   {edge_to_triplet_text(candidate_edge)}"
                    )
                    merge_dec = await decide_relation_merge(
                        session,
                        relation_pair=rel_pair,
                        doc_text=doc_text[:500],
                        domain=domain,
                    )
                    if merge_dec.should_merge:
                        aligned_rel_id = rel_alignment.aligned_entity_id
                        merged_props = dict(candidate_edge.properties)
                        merged_props.update(merge_dec.merged_properties)
                        merged_edge = GraphEdge(
                            id=aligned_rel_id,
                            source=candidate_edge.source,
                            label=candidate_edge.label,
                            target=candidate_edge.target,
                            properties=merged_props,
                        )
                        await backend.upsert_relation(merged_edge)
                        return aligned_rel_id, True
        except Exception:
            pass

    # No match — insert new relation
    new_edge = GraphEdge(
        id=f"{src_id}_{safe_rel}_{tgt_id}",
        source=src_node,
        label=safe_rel,
        target=tgt_node,
        properties={
            "description": relation.description,
            **relation.properties,
        },
    )
    try:
        new_id = await backend.upsert_relation(new_edge)
        return new_id, False
    except Exception:
        return None, False
