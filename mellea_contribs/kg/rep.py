"""Entity and relation representation utilities for KG operations.

Provides utility functions for formatting entities and relations into
human-readable text for LLM prompts and display.
"""

import re
from typing import Optional

from mellea_contribs.kg.models import Entity, Relation


def normalize_entity_name(name: str) -> str:
    """Normalize entity name to a canonical form.

    Handles:
    - Converting to title case
    - Removing extra whitespace
    - Standardizing quotes and punctuation

    Args:
        name: Raw entity name

    Returns:
        Normalized entity name
    """
    # Remove extra whitespace
    normalized = " ".join(name.split())

    # Convert to title case
    normalized = normalized.title()

    # Standardize quotes
    normalized = normalized.replace("'", "'").replace(""", '"').replace(""", '"')

    return normalized


def entity_to_text(entity: Entity, include_confidence: bool = False) -> str:
    """Format entity into human-readable text for LLM prompts.

    Args:
        entity: Entity to format
        include_confidence: Whether to include confidence score

    Returns:
        Formatted text representation of entity
    """
    parts = [f"**{entity.name}** ({entity.type})"]

    if entity.description:
        parts.append(f"Description: {entity.description}")

    if entity.properties:
        prop_str = ", ".join(
            f"{k}: {v}" for k, v in entity.properties.items() if v is not None
        )
        if prop_str:
            parts.append(f"Properties: {prop_str}")

    if include_confidence:
        parts.append(f"Confidence: {entity.confidence:.2%}")

    return "\n".join(parts)


def relation_to_text(relation: Relation, include_confidence: bool = False) -> str:
    """Format relation into human-readable text for LLM prompts.

    Args:
        relation: Relation to format
        include_confidence: Whether to include confidence score

    Returns:
        Formatted text representation of relation
    """
    parts = [
        f"**{relation.source_entity}** --[{relation.relation_type}]--> **{relation.target_entity}**"
    ]

    if relation.description:
        parts.append(f"Description: {relation.description}")

    if relation.properties:
        prop_str = ", ".join(
            f"{k}: {v}" for k, v in relation.properties.items() if v is not None
        )
        if prop_str:
            parts.append(f"Properties: {prop_str}")

    if include_confidence and hasattr(relation, "confidence"):
        parts.append(f"Confidence: {relation.confidence:.2%}")

    return "\n".join(parts)


def format_entity_list(
    entities: list[Entity], include_confidence: bool = False, max_items: Optional[int] = None
) -> str:
    """Format list of entities into readable text.

    Args:
        entities: List of entities to format
        include_confidence: Whether to include confidence scores
        max_items: Maximum number of entities to display (None for all)

    Returns:
        Formatted text with all entities
    """
    display_entities = entities[:max_items] if max_items else entities

    formatted = []
    for i, entity in enumerate(display_entities, 1):
        formatted.append(f"{i}. {entity_to_text(entity, include_confidence)}")

    if max_items and len(entities) > max_items:
        formatted.append(f"\n... and {len(entities) - max_items} more entities")

    return "\n\n".join(formatted)


def format_relation_list(
    relations: list[Relation],
    include_confidence: bool = False,
    max_items: Optional[int] = None,
) -> str:
    """Format list of relations into readable text.

    Args:
        relations: List of relations to format
        include_confidence: Whether to include confidence scores
        max_items: Maximum number of relations to display (None for all)

    Returns:
        Formatted text with all relations
    """
    display_relations = relations[:max_items] if max_items else relations

    formatted = []
    for i, relation in enumerate(display_relations, 1):
        formatted.append(f"{i}. {relation_to_text(relation, include_confidence)}")

    if max_items and len(relations) > max_items:
        formatted.append(f"\n... and {len(relations) - max_items} more relations")

    return "\n\n".join(formatted)


def format_kg_context(
    entities: list[Entity],
    relations: list[Relation],
    include_confidence: bool = False,
    max_entities: Optional[int] = None,
    max_relations: Optional[int] = None,
) -> str:
    """Format knowledge graph context for LLM prompts.

    Combines entities and relations into a structured text representation.

    Args:
        entities: List of entities from KG
        relations: List of relations from KG
        include_confidence: Whether to include confidence scores
        max_entities: Maximum entities to display
        max_relations: Maximum relations to display

    Returns:
        Formatted KG context text
    """
    sections = []

    if entities:
        sections.append("## Entities\n")
        sections.append(format_entity_list(entities, include_confidence, max_entities))

    if relations:
        sections.append("\n## Relations\n")
        sections.append(format_relation_list(relations, include_confidence, max_relations))

    return "\n".join(sections) if sections else "(Empty knowledge graph)"


def camelcase_to_snake_case(name: str) -> str:
    """Convert camelCase to snake_case.

    Args:
        name: Name in camelCase

    Returns:
        Name converted to snake_case
    """
    # Insert underscore before uppercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_case_to_camelcase(name: str, upper_first: bool = False) -> str:
    """Convert snake_case to camelCase.

    Args:
        name: Name in snake_case
        upper_first: Whether to capitalize first letter (PascalCase)

    Returns:
        Name converted to camelCase or PascalCase
    """
    components = name.split("_")
    if upper_first:
        return "".join(x.title() for x in components)
    else:
        return components[0].lower() + "".join(x.title() for x in components[1:])


__all__ = [
    "normalize_entity_name",
    "entity_to_text",
    "relation_to_text",
    "format_entity_list",
    "format_relation_list",
    "format_kg_context",
    "camelcase_to_snake_case",
    "snake_case_to_camelcase",
]
