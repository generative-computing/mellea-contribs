"""Tests for representation utilities (rep.py)."""

import pytest

from mellea_contribs.kg.models import Entity, Relation
from mellea_contribs.kg.rep import (
    normalize_entity_name,
    entity_to_text,
    relation_to_text,
    format_entity_list,
    format_relation_list,
    format_kg_context,
    camelcase_to_snake_case,
    snake_case_to_camelcase,
)


class TestNormalizeEntityName:
    """Tests for normalize_entity_name function."""

    def test_normalize_basic(self):
        """Test normalizing basic names."""
        assert normalize_entity_name("alice") == "Alice"
        assert normalize_entity_name("ALICE") == "Alice"

    def test_normalize_with_spaces(self):
        """Test normalizing names with extra spaces."""
        assert normalize_entity_name("alice  bob") == "Alice Bob"
        assert normalize_entity_name("  alice  ") == "Alice"

    def test_normalize_with_quotes(self):
        """Test normalizing names with quotes."""
        result = normalize_entity_name("alice 'bob' charlie")
        assert "alice" in result.lower()
        assert "bob" in result.lower()

    def test_normalize_title_case(self):
        """Test normalization to title case."""
        assert normalize_entity_name("oppenheimer") == "Oppenheimer"
        assert normalize_entity_name("christopher nolan") == "Christopher Nolan"


class TestEntityToText:
    """Tests for entity_to_text function."""

    def test_entity_to_text_basic(self):
        """Test basic entity formatting."""
        entity = Entity(
            type="Person",
            name="Alice",
            description="A person",
            paragraph_start="Alice is",
            paragraph_end="here.",
        )

        text = entity_to_text(entity)
        assert "Alice" in text
        assert "Person" in text
        assert "A person" in text

    def test_entity_to_text_with_properties(self):
        """Test entity formatting with properties."""
        entity = Entity(
            type="Movie",
            name="Oppenheimer",
            description="A film",
            paragraph_start="Oppenheimer is",
            paragraph_end="great.",
            properties={"year": 2023, "director": "Nolan"},
        )

        text = entity_to_text(entity)
        assert "Oppenheimer" in text
        assert "Movie" in text

    def test_entity_to_text_with_confidence(self):
        """Test entity formatting with confidence score."""
        entity = Entity(
            type="Person",
            name="Bob",
            description="Test",
            paragraph_start="Bob",
            paragraph_end="here.",
            confidence=0.95,
        )

        text = entity_to_text(entity, include_confidence=True)
        assert "Confidence" in text or "0.95" in text

    def test_entity_to_text_no_properties(self):
        """Test entity formatting without properties."""
        entity = Entity(
            type="Thing",
            name="Thing1",
            description="A thing",
            paragraph_start="Thing",
            paragraph_end="here.",
            properties={},
        )

        text = entity_to_text(entity)
        assert "Thing1" in text
        assert "Thing" in text


class TestRelationToText:
    """Tests for relation_to_text function."""

    def test_relation_to_text_basic(self):
        """Test basic relation formatting."""
        relation = Relation(
            source_entity="Alice",
            relation_type="KNOWS",
            target_entity="Bob",
            description="Alice knows Bob",
            paragraph_start="Alice knows",
            paragraph_end="Bob.",
        )

        text = relation_to_text(relation)
        assert "Alice" in text
        assert "KNOWS" in text
        assert "Bob" in text
        assert "Alice knows Bob" in text

    def test_relation_to_text_with_properties(self):
        """Test relation formatting with properties."""
        relation = Relation(
            source_entity="Alice",
            relation_type="DIRECTED",
            target_entity="Oppenheimer",
            description="Alice directed Oppenheimer",
            paragraph_start="Alice directed",
            paragraph_end="Oppenheimer.",
            properties={"year": 2023, "budget": "100M"},
        )

        text = relation_to_text(relation)
        assert "Alice" in text
        assert "DIRECTED" in text
        assert "Oppenheimer" in text

    def test_relation_to_text_with_confidence(self):
        """Test relation formatting with confidence."""
        relation = Relation(
            source_entity="X",
            relation_type="RELATED",
            target_entity="Y",
            description="X related to Y",
            paragraph_start="X related",
            paragraph_end="Y.",
        )

        text = relation_to_text(relation, include_confidence=False)
        assert "X" in text
        assert "RELATED" in text


class TestFormatEntityList:
    """Tests for format_entity_list function."""

    def test_format_single_entity(self):
        """Test formatting single entity list."""
        entity = Entity(
            type="Person",
            name="Alice",
            description="A person",
            paragraph_start="Alice",
            paragraph_end="here.",
        )

        text = format_entity_list([entity])
        assert "1." in text
        assert "Alice" in text

    def test_format_multiple_entities(self):
        """Test formatting multiple entities."""
        entities = [
            Entity(
                type="Person",
                name=f"Person{i}",
                description=f"Person {i}",
                paragraph_start=f"Person{i}",
                paragraph_end="here.",
            )
            for i in range(3)
        ]

        text = format_entity_list(entities)
        assert "1." in text
        assert "2." in text
        assert "3." in text

    def test_format_entity_list_with_max_items(self):
        """Test formatting entity list with max limit."""
        entities = [
            Entity(
                type="Person",
                name=f"Person{i}",
                description=f"Person {i}",
                paragraph_start=f"Person{i}",
                paragraph_end="here.",
            )
            for i in range(5)
        ]

        text = format_entity_list(entities, max_items=3)
        assert "1." in text
        assert "2." in text
        assert "3." in text
        assert "more entities" in text or "..." in text

    def test_format_empty_entity_list(self):
        """Test formatting empty entity list."""
        text = format_entity_list([])
        assert text == ""


class TestFormatRelationList:
    """Tests for format_relation_list function."""

    def test_format_single_relation(self):
        """Test formatting single relation list."""
        relation = Relation(
            source_entity="Alice",
            relation_type="KNOWS",
            target_entity="Bob",
            description="Alice knows Bob",
            paragraph_start="Alice knows",
            paragraph_end="Bob.",
        )

        text = format_relation_list([relation])
        assert "1." in text
        assert "Alice" in text

    def test_format_multiple_relations(self):
        """Test formatting multiple relations."""
        relations = [
            Relation(
                source_entity=f"Entity{i}",
                relation_type="RELATES",
                target_entity=f"Entity{i+1}",
                description=f"Entity {i} relates to Entity {i+1}",
                paragraph_start=f"Entity{i}",
                paragraph_end=f"Entity{i+1}.",
            )
            for i in range(3)
        ]

        text = format_relation_list(relations)
        assert "1." in text
        assert "2." in text
        assert "3." in text

    def test_format_relation_list_with_max_items(self):
        """Test formatting relation list with max limit."""
        relations = [
            Relation(
                source_entity="A",
                relation_type="RELATES",
                target_entity=f"Entity{i}",
                description=f"Relation {i}",
                paragraph_start="A",
                paragraph_end=f"Entity{i}.",
            )
            for i in range(5)
        ]

        text = format_relation_list(relations, max_items=2)
        assert "1." in text
        assert "2." in text
        assert "more relations" in text or "..." in text


class TestFormatKgContext:
    """Tests for format_kg_context function."""

    def test_format_kg_context_entities_only(self):
        """Test formatting KG context with only entities."""
        entities = [
            Entity(
                type="Person",
                name="Alice",
                description="A person",
                paragraph_start="Alice",
                paragraph_end="here.",
            ),
            Entity(
                type="Movie",
                name="Oppenheimer",
                description="A film",
                paragraph_start="Oppenheimer",
                paragraph_end="great.",
            ),
        ]

        text = format_kg_context(entities, [])
        assert "Entities" in text or "entities" in text
        assert "Alice" in text
        assert "Oppenheimer" in text

    def test_format_kg_context_relations_only(self):
        """Test formatting KG context with only relations."""
        relations = [
            Relation(
                source_entity="Alice",
                relation_type="DIRECTED",
                target_entity="Oppenheimer",
                description="Alice directed Oppenheimer",
                paragraph_start="Alice directed",
                paragraph_end="Oppenheimer.",
            ),
        ]

        text = format_kg_context([], relations)
        assert "Relations" in text or "relations" in text
        assert "Alice" in text

    def test_format_kg_context_both(self):
        """Test formatting KG context with entities and relations."""
        entities = [
            Entity(
                type="Person",
                name="Alice",
                description="A person",
                paragraph_start="Alice",
                paragraph_end="here.",
            ),
        ]
        relations = [
            Relation(
                source_entity="Alice",
                relation_type="KNOWS",
                target_entity="Bob",
                description="Alice knows Bob",
                paragraph_start="Alice knows",
                paragraph_end="Bob.",
            ),
        ]

        text = format_kg_context(entities, relations)
        assert "Alice" in text
        assert "Bob" in text
        assert "KNOWS" in text

    def test_format_kg_context_empty(self):
        """Test formatting empty KG context."""
        text = format_kg_context([], [])
        assert "Empty" in text or "empty" in text


class TestCamelcaseToSnakecase:
    """Tests for camelcase_to_snake_case function."""

    def test_simple_camelcase(self):
        """Test converting simple camelCase."""
        assert camelcase_to_snake_case("camelCase") == "camel_case"
        assert camelcase_to_snake_case("myVariableName") == "my_variable_name"

    def test_pascalcase(self):
        """Test converting PascalCase."""
        assert camelcase_to_snake_case("MyClass") == "my_class"

    def test_already_snake_case(self):
        """Test with already snake_case input."""
        assert camelcase_to_snake_case("snake_case") == "snake_case"
        assert camelcase_to_snake_case("my_var") == "my_var"

    def test_single_word(self):
        """Test with single word."""
        assert camelcase_to_snake_case("word") == "word"
        assert camelcase_to_snake_case("Word") == "word"


class TestSnakecaseToCamelcase:
    """Tests for snake_case_to_camelcase function."""

    def test_simple_snake_case(self):
        """Test converting simple snake_case."""
        assert snake_case_to_camelcase("snake_case") == "snakeCase"
        assert snake_case_to_camelcase("my_variable") == "myVariable"

    def test_snake_case_to_pascalcase(self):
        """Test converting snake_case to PascalCase."""
        assert snake_case_to_camelcase("my_class", upper_first=True) == "MyClass"
        assert snake_case_to_camelcase("my_variable_name", upper_first=True) == "MyVariableName"

    def test_single_word(self):
        """Test with single word."""
        assert snake_case_to_camelcase("word") == "word"
        assert snake_case_to_camelcase("word", upper_first=True) == "Word"
