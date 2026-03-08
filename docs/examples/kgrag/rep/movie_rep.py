"""Movie domain-specific entity and relation representation utilities.

This module demonstrates how to extend the generic representation utilities
for a specific domain (movie) with domain-specific formatting and validation.
"""

from typing import Optional

from docs.examples.kgrag.models import MovieEntity, PersonEntity, AwardEntity
from mellea_contribs.kg.models import Entity, Relation
from mellea_contribs.kg.rep import entity_to_text as base_entity_to_text
from mellea_contribs.kg.rep import format_kg_context as base_format_kg_context
from mellea_contribs.kg.rep import relation_to_text as base_relation_to_text


def movie_entity_to_text(entity: Entity, include_confidence: bool = False) -> str:
    """Format movie entity with domain-specific details.

    Extends the generic formatting with movie-specific fields like release year,
    director, box office, etc.

    Args:
        entity: Entity to format (should be MovieEntity, PersonEntity, or AwardEntity)
        include_confidence: Whether to include confidence score

    Returns:
        Formatted text representation optimized for movie domain
    """
    # Use base formatting
    text = base_entity_to_text(entity, include_confidence)

    # Add domain-specific details if available
    if isinstance(entity, MovieEntity):
        details = []
        if entity.release_year:
            details.append(f"Released: {entity.release_year}")
        if entity.director:
            details.append(f"Director: {entity.director}")
        if entity.box_office:
            details.append(f"Box Office: ${entity.box_office}M")
        if entity.language:
            details.append(f"Language: {entity.language}")
        if entity.rating:
            details.append(f"Rating: {entity.rating}/10")

        if details:
            text += "\n" + "\n".join(details)

    elif isinstance(entity, PersonEntity):
        details = []
        if entity.birth_year:
            details.append(f"Born: {entity.birth_year}")
        if entity.nationality:
            details.append(f"Nationality: {entity.nationality}")
        if entity.profession:
            details.append(f"Profession: {entity.profession}")

        if details:
            text += "\n" + "\n".join(details)

    elif isinstance(entity, AwardEntity):
        details = []
        if entity.ceremony_number:
            details.append(f"Ceremony: #{entity.ceremony_number}")
        if entity.award_type:
            details.append(f"Award: {entity.award_type}")
        if entity.award_year:
            details.append(f"Year: {entity.award_year}")

        if details:
            text += "\n" + "\n".join(details)

    return text


def movie_relation_to_text(
    relation: Relation, include_confidence: bool = False
) -> str:
    """Format movie relation with domain-specific context.

    Extends generic formatting with movie-specific relation handling.

    Args:
        relation: Relation to format
        include_confidence: Whether to include confidence score

    Returns:
        Formatted text representation optimized for movie domain
    """
    # Use base formatting
    return base_relation_to_text(relation, include_confidence)


def format_movie_context(
    entities: list[Entity],
    relations: list[Relation],
    include_confidence: bool = False,
    max_entities: Optional[int] = None,
    max_relations: Optional[int] = None,
) -> str:
    """Format movie KG context with domain-specific formatting.

    Uses movie-specific entity formatting for better readability.

    Args:
        entities: List of entities from movie KG
        relations: List of relations from movie KG
        include_confidence: Whether to include confidence scores
        max_entities: Maximum entities to display
        max_relations: Maximum relations to display

    Returns:
        Formatted movie KG context text
    """
    sections = []

    if entities:
        sections.append("## Entities\n")
        display_entities = entities[:max_entities] if max_entities else entities

        formatted = []
        for i, entity in enumerate(display_entities, 1):
            formatted.append(f"{i}. {movie_entity_to_text(entity, include_confidence)}")

        if max_entities and len(entities) > max_entities:
            formatted.append(f"\n... and {len(entities) - max_entities} more entities")

        sections.append("\n\n".join(formatted))

    if relations:
        sections.append("\n## Relations\n")
        display_relations = (
            relations[:max_relations] if max_relations else relations
        )

        formatted = []
        for i, relation in enumerate(display_relations, 1):
            formatted.append(
                f"{i}. {movie_relation_to_text(relation, include_confidence)}"
            )

        if max_relations and len(relations) > max_relations:
            formatted.append(
                f"\n... and {len(relations) - max_relations} more relations"
            )

        sections.append("\n\n".join(formatted))

    return "\n".join(sections) if sections else "(Empty movie knowledge graph)"


__all__ = [
    "movie_entity_to_text",
    "movie_relation_to_text",
    "format_movie_context",
]
