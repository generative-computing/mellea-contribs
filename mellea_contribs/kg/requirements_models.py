"""Entity and relation validation requirements for KG operations.

Provides Requirement factories for validating entities and relations
against type constraints, schema rules, and data quality standards.
"""

from typing import Optional

from mellea.stdlib.context import Context
from mellea.stdlib.requirements import Requirement, ValidationResult

from mellea_contribs.kg.models import Entity, Relation


def entity_type_valid(allowed_types: list[str]) -> Requirement:
    """Require that an entity type is in the allowed types list.

    Args:
        allowed_types: List of valid entity type names (e.g., ['Movie', 'Person'])

    Returns:
        A Requirement that passes when the entity's type is allowed.
    """

    async def validate(ctx: Context) -> ValidationResult:
        try:
            entity = ctx.last_output()
            if isinstance(entity, Entity):
                is_valid = entity.type in allowed_types
                return ValidationResult(
                    is_valid,
                    reason=(
                        f"Entity type '{entity.type}' is valid"
                        if is_valid
                        else f"Entity type '{entity.type}' not in allowed types: {allowed_types}"
                    ),
                )
            return ValidationResult(False, reason="Output is not an Entity")
        except Exception as e:
            return ValidationResult(False, reason=f"Validation error: {str(e)}")

    return Requirement(
        description=f"Entity type must be one of: {allowed_types}",
        validation_fn=validate,
    )


def entity_has_name(ctx: Context) -> ValidationResult:
    """Require that an entity has a non-empty name.

    Returns a ValidationResult indicating if entity name is valid.
    """
    try:
        entity = ctx.last_output()
        if isinstance(entity, Entity):
            is_valid = bool(entity.name and entity.name.strip())
            return ValidationResult(
                is_valid,
                reason="Entity has valid name" if is_valid else "Entity name is empty",
            )
        return ValidationResult(False, reason="Output is not an Entity")
    except Exception as e:
        return ValidationResult(False, reason=f"Validation error: {str(e)}")


def entity_has_description(ctx: Context) -> ValidationResult:
    """Require that an entity has a non-empty description.

    Returns a ValidationResult indicating if entity description is valid.
    """
    try:
        entity = ctx.last_output()
        if isinstance(entity, Entity):
            is_valid = bool(entity.description and entity.description.strip())
            return ValidationResult(
                is_valid,
                reason=(
                    "Entity has valid description"
                    if is_valid
                    else "Entity description is empty"
                ),
            )
        return ValidationResult(False, reason="Output is not an Entity")
    except Exception as e:
        return ValidationResult(False, reason=f"Validation error: {str(e)}")


def relation_type_valid(allowed_types: list[str]) -> Requirement:
    """Require that a relation type is in the allowed types list.

    Args:
        allowed_types: List of valid relation type names (e.g., ['directed_by', 'acted_in'])

    Returns:
        A Requirement that passes when the relation's type is allowed.
    """

    async def validate(ctx: Context) -> ValidationResult:
        try:
            relation = ctx.last_output()
            if isinstance(relation, Relation):
                is_valid = relation.relation_type in allowed_types
                return ValidationResult(
                    is_valid,
                    reason=(
                        f"Relation type '{relation.relation_type}' is valid"
                        if is_valid
                        else f"Relation type '{relation.relation_type}' not in allowed types: {allowed_types}"
                    ),
                )
            return ValidationResult(False, reason="Output is not a Relation")
        except Exception as e:
            return ValidationResult(False, reason=f"Validation error: {str(e)}")

    return Requirement(
        description=f"Relation type must be one of: {allowed_types}",
        validation_fn=validate,
    )


def relation_entities_exist(entities: list[str]) -> Requirement:
    """Require that relation source and target entities exist in the provided list.

    Args:
        entities: List of valid entity names that can be relation endpoints

    Returns:
        A Requirement that passes when both entities are in the allowed list.
    """

    async def validate(ctx: Context) -> ValidationResult:
        try:
            relation = ctx.last_output()
            if isinstance(relation, Relation):
                source_valid = relation.source_entity in entities
                target_valid = relation.target_entity in entities
                is_valid = source_valid and target_valid

                reasons = []
                if not source_valid:
                    reasons.append(
                        f"Source entity '{relation.source_entity}' not found"
                    )
                if not target_valid:
                    reasons.append(
                        f"Target entity '{relation.target_entity}' not found"
                    )

                return ValidationResult(
                    is_valid,
                    reason=(
                        "Both relation entities are valid"
                        if is_valid
                        else "; ".join(reasons)
                    ),
                )
            return ValidationResult(False, reason="Output is not a Relation")
        except Exception as e:
            return ValidationResult(False, reason=f"Validation error: {str(e)}")

    return Requirement(
        description=f"Relation entities must be in: {entities}",
        validation_fn=validate,
    )


def entity_confidence_threshold(
    min_confidence: float = 0.5,
) -> Requirement:
    """Require that an entity meets a minimum confidence threshold.

    Args:
        min_confidence: Minimum confidence score (0-1)

    Returns:
        A Requirement that passes when entity confidence meets threshold.
    """

    async def validate(ctx: Context) -> ValidationResult:
        try:
            entity = ctx.last_output()
            if isinstance(entity, Entity):
                is_valid = entity.confidence >= min_confidence
                return ValidationResult(
                    is_valid,
                    reason=(
                        f"Entity confidence {entity.confidence:.2f} meets threshold {min_confidence}"
                        if is_valid
                        else f"Entity confidence {entity.confidence:.2f} below threshold {min_confidence}"
                    ),
                )
            return ValidationResult(False, reason="Output is not an Entity")
        except Exception as e:
            return ValidationResult(False, reason=f"Validation error: {str(e)}")

    return Requirement(
        description=f"Entity confidence must be >= {min_confidence}",
        validation_fn=validate,
    )


__all__ = [
    "entity_type_valid",
    "entity_has_name",
    "entity_has_description",
    "relation_type_valid",
    "relation_entities_exist",
    "entity_confidence_threshold",
]
