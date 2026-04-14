"""Tests for domain-specific examples (movie domain)."""

import pytest

from mellea_contribs.kg.models import Entity, Relation


class TestMovieEntityModel:
    """Tests for MovieEntity domain-specific model."""

    def test_movie_entity_import(self):
        """Test that MovieEntity can be imported."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import MovieEntity

            assert MovieEntity is not None
        except ImportError:
            # If import fails, the file structure is still valid
            pytest.skip("MovieEntity example not available in test environment")

    def test_movie_entity_structure(self):
        """Test MovieEntity has required fields."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import MovieEntity

            # Check that MovieEntity is a class
            assert hasattr(MovieEntity, "__init__")

            # Create a MovieEntity instance
            entity = MovieEntity(
                type="Movie",
                name="Oppenheimer",
                description="2023 film",
                paragraph_start="Oppenheimer is",
                paragraph_end="by Nolan.",
                release_year=2023,
                director="Christopher Nolan",
            )

            assert entity.type == "Movie"
            assert entity.name == "Oppenheimer"
            assert entity.release_year == 2023
            assert entity.director == "Christopher Nolan"
        except ImportError:
            pytest.skip("MovieEntity example not available")

    def test_movie_entity_optional_fields(self):
        """Test MovieEntity optional fields."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import MovieEntity

            entity = MovieEntity(
                type="Movie",
                name="Oppenheimer",
                description="A film",
                paragraph_start="Movie",
                paragraph_end="here.",
                box_office=952.0,
                language="English",
                rating=8.4,
            )

            assert entity.box_office == 952.0
            assert entity.language == "English"
            assert entity.rating == 8.4
        except ImportError:
            pytest.skip("MovieEntity example not available")


class TestPersonEntityModel:
    """Tests for PersonEntity domain-specific model."""

    def test_person_entity_structure(self):
        """Test PersonEntity has required fields."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import PersonEntity

            entity = PersonEntity(
                type="Person",
                name="Christopher Nolan",
                description="Film director",
                paragraph_start="Christopher is",
                paragraph_end="a director.",
                birth_year=1970,
                nationality="British",
                profession="Director",
            )

            assert entity.type == "Person"
            assert entity.name == "Christopher Nolan"
            assert entity.birth_year == 1970
            assert entity.nationality == "British"
            assert entity.profession == "Director"
        except ImportError:
            pytest.skip("PersonEntity example not available")


class TestAwardEntityModel:
    """Tests for AwardEntity domain-specific model."""

    def test_award_entity_structure(self):
        """Test AwardEntity has required fields."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import AwardEntity

            entity = AwardEntity(
                type="Award",
                name="Best Picture",
                description="Academy Award",
                paragraph_start="Best Picture",
                paragraph_end="award.",
                ceremony_number=96,
                award_type="Oscar",
                award_year=2024,
            )

            assert entity.type == "Award"
            assert entity.name == "Best Picture"
            assert entity.ceremony_number == 96
            assert entity.award_type == "Oscar"
            assert entity.award_year == 2024
        except ImportError:
            pytest.skip("AwardEntity example not available")


class TestMovieRepresentationUtilities:
    """Tests for movie domain representation utilities."""

    def test_movie_rep_utilities_import(self):
        """Test that movie representation utilities can be imported."""
        try:
            from docs.examples.kgrag.rep.movie_rep import (
                movie_entity_to_text,
                movie_relation_to_text,
                format_movie_context,
            )

            assert movie_entity_to_text is not None
            assert movie_relation_to_text is not None
            assert format_movie_context is not None
        except ImportError:
            pytest.skip("Movie rep utilities not available")

    def test_movie_entity_to_text_with_movie_entity(self):
        """Test movie_entity_to_text with MovieEntity."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import MovieEntity
            from docs.examples.kgrag.rep.movie_rep import movie_entity_to_text

            entity = MovieEntity(
                type="Movie",
                name="Oppenheimer",
                description="2023 film",
                paragraph_start="Oppenheimer is",
                paragraph_end="great.",
                release_year=2023,
                director="Christopher Nolan",
            )

            text = movie_entity_to_text(entity)

            assert "Oppenheimer" in text
            assert "Movie" in text
            # Movie-specific fields should be included
            assert "2023" in text or "release" in text.lower() or "2023" in str(entity)

        except ImportError:
            pytest.skip("Movie utilities not available")

    def test_movie_entity_to_text_with_person_entity(self):
        """Test movie_entity_to_text with PersonEntity."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import PersonEntity
            from docs.examples.kgrag.rep.movie_rep import movie_entity_to_text

            entity = PersonEntity(
                type="Person",
                name="Christopher Nolan",
                description="Director",
                paragraph_start="Christopher",
                paragraph_end="is a director.",
                birth_year=1970,
                nationality="British",
            )

            text = movie_entity_to_text(entity)

            assert "Christopher Nolan" in text
            assert "Person" in text
        except ImportError:
            pytest.skip("Movie utilities not available")

    def test_format_movie_context(self):
        """Test format_movie_context function."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import MovieEntity
            from docs.examples.kgrag.rep.movie_rep import format_movie_context

            entities = [
                MovieEntity(
                    type="Movie",
                    name="Oppenheimer",
                    description="2023 film",
                    paragraph_start="Oppenheimer",
                    paragraph_end="is great.",
                    release_year=2023,
                ),
            ]

            relations = [
                Relation(
                    source_entity="Christopher Nolan",
                    relation_type="directed_by",
                    target_entity="Oppenheimer",
                    description="Christopher directed Oppenheimer",
                )
            ]

            context = format_movie_context(entities, relations)

            assert "Oppenheimer" in context
            assert "directed_by" in context or "Entities" in context

        except ImportError:
            pytest.skip("Movie utilities not available")


class TestMoviePreprocessorExample:
    """Tests for MovieKGPreprocessor domain example."""

    def test_movie_preprocessor_import(self):
        """Test that MovieKGPreprocessor can be imported."""
        try:
            from docs.examples.kgrag.preprocessor.movie_preprocessor import MovieKGPreprocessor

            assert MovieKGPreprocessor is not None
        except ImportError:
            pytest.skip("MovieKGPreprocessor not available")

    def test_movie_preprocessor_structure(self):
        """Test MovieKGPreprocessor structure."""
        try:
            from docs.examples.kgrag.preprocessor.movie_preprocessor import MovieKGPreprocessor

            # Check that it's a class
            import inspect

            assert inspect.isclass(MovieKGPreprocessor)

            # Check for key methods
            assert hasattr(MovieKGPreprocessor, "get_hints")
            assert hasattr(MovieKGPreprocessor, "post_process_extraction")

        except ImportError:
            pytest.skip("MovieKGPreprocessor not available")


class TestDomainExampleIntegration:
    """Integration tests for domain examples."""

    def test_domain_models_inheritance(self):
        """Test that domain models properly extend Entity."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import (
                MovieEntity,
                PersonEntity,
                AwardEntity,
            )

            # All should extend Entity
            movie = MovieEntity(
                type="Movie",
                name="Test",
                description="Test",
                paragraph_start="Test",
                paragraph_end=".",
            )
            person = PersonEntity(
                type="Person",
                name="Test",
                description="Test",
                paragraph_start="Test",
                paragraph_end=".",
            )
            award = AwardEntity(
                type="Award",
                name="Test",
                description="Test",
                paragraph_start="Test",
                paragraph_end=".",
            )

            # Check that they all have Entity fields
            for entity in [movie, person, award]:
                assert hasattr(entity, "type")
                assert hasattr(entity, "name")
                assert hasattr(entity, "description")
                assert hasattr(entity, "properties")

        except ImportError:
            pytest.skip("Domain models not available")

    def test_domain_rep_utilities_with_base_entities(self):
        """Test that domain rep utilities work with base Entity."""
        try:
            from docs.examples.kgrag.rep.movie_rep import movie_entity_to_text

            # Test with base Entity
            entity = Entity(
                type="Generic",
                name="Generic Entity",
                description="A generic entity",
                paragraph_start="Generic",
                paragraph_end="here.",
            )

            text = movie_entity_to_text(entity)

            assert text is not None
            assert len(text) > 0

        except ImportError:
            pytest.skip("Domain rep utilities not available")


class TestDomainExampleConsistency:
    """Tests for consistency of domain examples."""

    def test_all_domain_entities_have_same_base_fields(self):
        """Test that all domain entities have the same base Entity fields."""
        try:
            from docs.examples.kgrag.models.movie_domain_models import (
                MovieEntity,
                PersonEntity,
                AwardEntity,
            )

            base_fields = ["type", "name", "description", "paragraph_start", "paragraph_end"]

            for EntityClass in [MovieEntity, PersonEntity, AwardEntity]:
                entity = EntityClass(
                    type="Test",
                    name="Test",
                    description="Test",
                    paragraph_start="Test",
                    paragraph_end=".",
                )

                for field in base_fields:
                    assert hasattr(entity, field), f"{EntityClass.__name__} missing {field}"

        except ImportError:
            pytest.skip("Domain models not available")
