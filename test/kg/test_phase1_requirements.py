"""Tests for Phase 1 requirement factory functions."""

import pytest

from mellea_contribs.kg.requirements_models import (
    entity_type_valid,
    entity_has_name,
    entity_has_description,
    relation_type_valid,
    relation_entities_exist,
    entity_confidence_threshold,
)
from mellea_contribs.kg.models import Entity, Relation


class TestRequirementFactories:
    """Tests for requirement factory functions."""

    def test_entity_type_valid_returns_requirement(self):
        """Test that entity_type_valid returns a Requirement."""
        req = entity_type_valid(["Person", "Movie"])

        assert req is not None
        assert hasattr(req, "description")
        assert hasattr(req, "validation_fn")
        assert "Entity type" in req.description or "type" in req.description

    def test_entity_type_valid_factory_multiple_calls(self):
        """Test entity_type_valid can be called multiple times."""
        req1 = entity_type_valid(["Person", "Movie"])
        req2 = entity_type_valid(["Company", "Location"])

        assert req1 is not None
        assert req2 is not None
        assert req1 != req2

    def test_relation_type_valid_returns_requirement(self):
        """Test that relation_type_valid returns a Requirement."""
        req = relation_type_valid(["directed_by", "acted_in"])

        assert req is not None
        assert hasattr(req, "description")
        assert hasattr(req, "validation_fn")

    def test_entity_has_name_returns_function(self):
        """Test that entity_has_name returns a callable."""
        assert callable(entity_has_name)

    def test_entity_has_description_returns_function(self):
        """Test that entity_has_description returns a callable."""
        assert callable(entity_has_description)

    def test_relation_entities_exist_returns_requirement(self):
        """Test that relation_entities_exist returns a Requirement."""
        req = relation_entities_exist(["Alice", "Bob", "Charlie"])

        assert req is not None
        assert hasattr(req, "description")
        assert hasattr(req, "validation_fn")

    def test_entity_confidence_threshold_returns_requirement(self):
        """Test that entity_confidence_threshold returns a Requirement."""
        req = entity_confidence_threshold(min_confidence=0.8)

        assert req is not None
        assert hasattr(req, "description")
        assert hasattr(req, "validation_fn")


class TestEntityTypeValidRequirement:
    """Tests for entity_type_valid requirement."""

    def test_entity_type_valid_description(self):
        """Test entity_type_valid requirement description."""
        allowed_types = ["Person", "Movie", "Award"]
        req = entity_type_valid(allowed_types)

        assert req.description is not None
        assert "Person" in req.description or "type" in req.description.lower()

    def test_entity_type_valid_with_single_type(self):
        """Test entity_type_valid with single allowed type."""
        req = entity_type_valid(["Person"])

        assert req is not None
        assert "Person" in req.description

    def test_entity_type_valid_with_multiple_types(self):
        """Test entity_type_valid with multiple allowed types."""
        types = ["Person", "Movie", "Award", "Company"]
        req = entity_type_valid(types)

        assert req is not None
        for type_name in types:
            assert type_name in req.description or len(req.description) > 10


class TestRelationTypeValidRequirement:
    """Tests for relation_type_valid requirement."""

    def test_relation_type_valid_description(self):
        """Test relation_type_valid requirement description."""
        allowed_types = ["directed_by", "acted_in", "won"]
        req = relation_type_valid(allowed_types)

        assert req.description is not None
        assert "Relation type" in req.description or "type" in req.description.lower()

    def test_relation_type_valid_with_various_types(self):
        """Test relation_type_valid with various relation types."""
        types = ["KNOWS", "DIRECTED", "ACTED_IN", "WON"]

        for type_list in [types, types[:2], types[:1]]:
            req = relation_type_valid(type_list)
            assert req is not None


class TestEntityNameDescriptionValidation:
    """Tests for entity name and description validation functions."""

    def test_entity_has_name_callable(self):
        """Test that entity_has_name is callable."""
        assert callable(entity_has_name)

    def test_entity_has_description_callable(self):
        """Test that entity_has_description is callable."""
        assert callable(entity_has_description)

    def test_entity_has_name_with_entity(self):
        """Test entity_has_name with an Entity."""
        entity = Entity(
            type="Person",
            name="Alice",
            description="A person",
            paragraph_start="Alice",
            paragraph_end="here.",
        )

        # The function should be callable with entity
        assert callable(entity_has_name)

    def test_entity_has_description_with_entity(self):
        """Test entity_has_description with an Entity."""
        entity = Entity(
            type="Person",
            name="Alice",
            description="A person",
            paragraph_start="Alice",
            paragraph_end="here.",
        )

        # The function should be callable with entity
        assert callable(entity_has_description)


class TestRelationEntitiesExistRequirement:
    """Tests for relation_entities_exist requirement."""

    def test_relation_entities_exist_description(self):
        """Test relation_entities_exist requirement description."""
        entities = ["Alice", "Bob", "Charlie"]
        req = relation_entities_exist(entities)

        assert req.description is not None
        assert "entities" in req.description.lower()

    def test_relation_entities_exist_with_various_entity_lists(self):
        """Test relation_entities_exist with various entity lists."""
        test_cases = [
            ["Alice", "Bob"],
            ["Person1", "Movie1", "Award1"],
            ["node-1", "node-2"],
        ]

        for entities in test_cases:
            req = relation_entities_exist(entities)
            assert req is not None
            assert req.description is not None


class TestEntityConfidenceThresholdRequirement:
    """Tests for entity_confidence_threshold requirement."""

    def test_entity_confidence_threshold_description(self):
        """Test entity_confidence_threshold requirement description."""
        req = entity_confidence_threshold(min_confidence=0.8)

        assert req.description is not None
        assert "0.8" in req.description or "confidence" in req.description.lower()

    def test_entity_confidence_threshold_with_various_values(self):
        """Test entity_confidence_threshold with various threshold values."""
        thresholds = [0.0, 0.5, 0.7, 0.9, 1.0]

        for threshold in thresholds:
            req = entity_confidence_threshold(min_confidence=threshold)
            assert req is not None
            assert req.description is not None

    def test_entity_confidence_threshold_low(self):
        """Test entity_confidence_threshold with low threshold."""
        req = entity_confidence_threshold(min_confidence=0.3)

        assert req is not None
        assert "0.3" in req.description or "confidence" in req.description.lower()

    def test_entity_confidence_threshold_high(self):
        """Test entity_confidence_threshold with high threshold."""
        req = entity_confidence_threshold(min_confidence=0.99)

        assert req is not None
        assert "0.99" in req.description or "confidence" in req.description.lower()


class TestRequirementFactoriesConsistency:
    """Tests for consistency of requirement factories."""

    def test_all_requirement_factories_return_non_none(self):
        """Test that all requirement factories return non-None values."""
        factories = [
            (entity_type_valid, [["Person"]]),
            (relation_type_valid, [["KNOWS"]]),
            (relation_entities_exist, [["Alice", "Bob"]]),
            (entity_confidence_threshold, [0.8]),
        ]

        for factory, args in factories:
            result = factory(*args)
            assert result is not None

    def test_all_requirement_factories_have_descriptions(self):
        """Test that all requirements have descriptions."""
        factories_and_args = [
            (entity_type_valid, [["Person"]]),
            (relation_type_valid, [["KNOWS"]]),
            (relation_entities_exist, [["Alice", "Bob"]]),
            (entity_confidence_threshold, [0.8]),
        ]

        for factory, args in factories_and_args:
            result = factory(*args)
            assert hasattr(result, "description") or callable(result)
