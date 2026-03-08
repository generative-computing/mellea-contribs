"""Tests for Phase 1 Entity and Relation models."""

import pytest

from mellea_contribs.kg.models import Entity, Relation, DirectAnswer


class TestEntity:
    """Tests for Entity model."""

    def test_create_entity_minimal(self):
        """Test creating Entity with minimal fields."""
        entity = Entity(
            type="Person",
            name="Alice",
            description="A person",
            paragraph_start="Alice is",
            paragraph_end="a character.",
        )

        assert entity.type == "Person"
        assert entity.name == "Alice"
        assert entity.description == "A person"
        assert entity.paragraph_start == "Alice is"
        assert entity.paragraph_end == "a character."
        # Check optional storage fields default
        assert entity.id is None
        assert entity.confidence == 1.0
        assert entity.embedding is None

    def test_create_entity_with_storage_fields(self):
        """Test creating Entity with storage fields."""
        entity = Entity(
            type="Movie",
            name="Oppenheimer",
            description="2023 film",
            paragraph_start="Oppenheimer is",
            paragraph_end="by Nolan.",
            id="neo4j-123",
            confidence=0.95,
            embedding=[0.1, 0.2, 0.3],
        )

        assert entity.id == "neo4j-123"
        assert entity.confidence == 0.95
        assert entity.embedding == [0.1, 0.2, 0.3]

    def test_entity_with_properties(self):
        """Test Entity with properties dict."""
        props = {"director": "Christopher Nolan", "year": 2023}
        entity = Entity(
            type="Movie",
            name="Oppenheimer",
            description="2023 film",
            paragraph_start="Oppenheimer is",
            paragraph_end="by Nolan.",
            properties=props,
        )

        assert entity.properties == props
        assert entity.properties["director"] == "Christopher Nolan"

    def test_entity_confidence_range(self):
        """Test Entity with valid confidence values."""
        entity1 = Entity(
            type="Person",
            name="Bob",
            description="Test",
            paragraph_start="Bob",
            paragraph_end="test.",
            confidence=0.0,
        )
        entity2 = Entity(
            type="Person",
            name="Bob",
            description="Test",
            paragraph_start="Bob",
            paragraph_end="test.",
            confidence=1.0,
        )
        entity3 = Entity(
            type="Person",
            name="Bob",
            description="Test",
            paragraph_start="Bob",
            paragraph_end="test.",
            confidence=0.5,
        )

        assert entity1.confidence == 0.0
        assert entity2.confidence == 1.0
        assert entity3.confidence == 0.5


class TestRelation:
    """Tests for Relation model."""

    def test_create_relation_minimal(self):
        """Test creating Relation with minimal fields."""
        relation = Relation(
            source_entity="Alice",
            relation_type="KNOWS",
            target_entity="Bob",
            description="Alice knows Bob",
            paragraph_start="Alice knows Bob",
            paragraph_end="very well.",
        )

        assert relation.source_entity == "Alice"
        assert relation.relation_type == "KNOWS"
        assert relation.target_entity == "Bob"
        assert relation.description == "Alice knows Bob"
        # Check optional storage fields default
        assert relation.id is None
        assert relation.source_entity_id is None
        assert relation.target_entity_id is None
        assert relation.valid_from is None
        assert relation.valid_until is None

    def test_create_relation_with_storage_fields(self):
        """Test creating Relation with storage fields."""
        relation = Relation(
            source_entity="Alice",
            relation_type="DIRECTED",
            target_entity="Oppenheimer",
            description="Alice directed Oppenheimer",
            paragraph_start="Alice directed",
            paragraph_end="Oppenheimer.",
            id="rel-123",
            source_entity_id="node-alice",
            target_entity_id="node-oppenheimer",
            valid_from="2020-01-01",
            valid_until="2023-12-31",
        )

        assert relation.id == "rel-123"
        assert relation.source_entity_id == "node-alice"
        assert relation.target_entity_id == "node-oppenheimer"
        assert relation.valid_from == "2020-01-01"
        assert relation.valid_until == "2023-12-31"

    def test_relation_with_properties(self):
        """Test Relation with properties dict."""
        props = {"role": "Director", "years": 3}
        relation = Relation(
            source_entity="Christopher",
            relation_type="DIRECTED",
            target_entity="Oppenheimer",
            description="Christopher directed Oppenheimer",
            paragraph_start="Christopher directed",
            paragraph_end="Oppenheimer.",
            properties=props,
        )

        assert relation.properties == props
        assert relation.properties["role"] == "Director"

    def test_relation_temporal_fields(self):
        """Test Relation temporal validity fields."""
        relation = Relation(
            source_entity="CEO",
            relation_type="LEADS",
            target_entity="Company",
            description="CEO leads company",
            paragraph_start="CEO leads",
            paragraph_end="company.",
            valid_from="2020-01-01",
            valid_until="2025-12-31",
        )

        assert relation.valid_from == "2020-01-01"
        assert relation.valid_until == "2025-12-31"


class TestDirectAnswer:
    """Tests for DirectAnswer model."""

    def test_create_direct_answer(self):
        """Test creating a DirectAnswer."""
        answer = DirectAnswer(
            sufficient="Yes",
            reason="LLM has knowledge about this",
            answer="The answer is 42",
        )

        assert answer.sufficient == "Yes"
        assert answer.reason == "LLM has knowledge about this"
        assert answer.answer == "The answer is 42"

    def test_direct_answer_negative_case(self):
        """Test DirectAnswer with negative sufficient case."""
        answer = DirectAnswer(
            sufficient="No",
            reason="Not enough information",
            answer="I don't know",
        )

        assert answer.sufficient == "No"
        assert answer.reason == "Not enough information"
        assert answer.answer == "I don't know"


class TestEntityRelationIntegration:
    """Tests for Entity and Relation integration."""

    def test_create_entity_and_relate(self):
        """Test creating entities and relating them."""
        alice = Entity(
            type="Person",
            name="Alice",
            description="A person",
            paragraph_start="Alice is",
            paragraph_end="a person.",
        )

        oppenheimer = Entity(
            type="Movie",
            name="Oppenheimer",
            description="A film",
            paragraph_start="Oppenheimer is",
            paragraph_end="a film.",
        )

        relation = Relation(
            source_entity=alice.name,
            relation_type="DIRECTED",
            target_entity=oppenheimer.name,
            description="Alice directed Oppenheimer",
            paragraph_start="Alice directed",
            paragraph_end="Oppenheimer.",
        )

        assert alice.type == "Person"
        assert oppenheimer.type == "Movie"
        assert relation.relation_type == "DIRECTED"
        assert relation.source_entity == "Alice"
        assert relation.target_entity == "Oppenheimer"

    def test_entity_extraction_to_storage_progression(self):
        """Test Entity progression from extracted to stored."""
        # Start: extracted entity (no storage fields set)
        extracted = Entity(
            type="Person",
            name="Bob",
            description="Test person",
            paragraph_start="Bob is",
            paragraph_end="a test.",
        )

        assert extracted.id is None
        assert extracted.confidence == 1.0

        # Later: stored entity (storage fields set)
        stored = Entity(
            type="Person",
            name="Bob",
            description="Test person",
            paragraph_start="Bob is",
            paragraph_end="a test.",
            id="neo4j-bob",
            confidence=0.98,
            embedding=[0.1] * 384,
        )

        assert stored.id == "neo4j-bob"
        assert stored.confidence == 0.98
        assert len(stored.embedding) == 384

    def test_empty_properties_dict(self):
        """Test Entity and Relation with empty properties."""
        entity = Entity(
            type="Thing",
            name="Thing1",
            description="A thing",
            paragraph_start="Thing",
            paragraph_end="here.",
            properties={},
        )

        relation = Relation(
            source_entity="A",
            relation_type="RELATES",
            target_entity="B",
            description="A relates to B",
            paragraph_start="A relates",
            paragraph_end="to B.",
            properties={},
        )

        assert entity.properties == {}
        assert relation.properties == {}
