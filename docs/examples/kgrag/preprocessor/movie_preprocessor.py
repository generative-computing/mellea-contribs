"""Domain-specific KG Preprocessor for the movie domain.

This module demonstrates how to extend the generic KGPreprocessor for a specific domain
by providing domain-specific hints and post-processing logic.

Example::

    import asyncio
    from mellea import start_session
    from mellea_contribs.kg import MockGraphBackend
    from movie_preprocessor import MovieKGPreprocessor

    async def main():
        session = start_session(backend_name="litellm", model_id="gpt-4o-mini")
        backend = MockGraphBackend()
        processor = MovieKGPreprocessor(backend=backend, session=session)

        # Process a movie document
        doc_text = \"\"\"Avatar is a 2009 science fiction film directed by James Cameron.
        It stars Sam Worthington, Zoe Saldana, and Sigourney Weaver.
        The film was nominated for multiple Academy Awards.
        \"\"\"

        result = await processor.process_document(
            doc_text=doc_text,
            doc_id="avatar_wiki"
        )
        print(f"Extracted {len(result.entities)} entities and {len(result.relations)} relations")
        await backend.close()

    asyncio.run(main())
"""

from typing import Optional

from mellea_contribs.kg.models import Entity, ExtractionResult
from mellea_contribs.kg.preprocessor import KGPreprocessor


class MovieKGPreprocessor(KGPreprocessor):
    """Domain-specific KG preprocessor for the movie domain.

    Extends the generic KGPreprocessor with movie-specific extraction hints and
    post-processing logic. Demonstrates how to customize preprocessing for a domain.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the movie preprocessor."""
        # Set domain to "movies" if not specified
        if "domain" not in kwargs:
            kwargs["domain"] = "movies"
        super().__init__(*args, **kwargs)

    def get_hints(self) -> str:
        """Get movie-domain-specific hints for LLM extraction.

        Provides guidance on what entity and relation types to look for in movie texts.

        Returns:
            String with movie domain hints
        """
        return """
Movie Domain Extraction Guide:

ENTITY TYPES to extract:
- Movie: Film titles, release years, budgets, box office
- Person: Actors, directors, producers, writers, cinematographers
- Award: Academy Awards, Golden Globes, BAFTA, Cannes Film Festival awards
- Studio: Production studios, distributors
- Genre: Film genres (Action, Drama, Comedy, etc.)
- Character: Character names and roles

RELATION TYPES to extract:
- directed_by: Movie → Director
- acted_in: Actor → Movie
- produced_by: Movie → Producer
- written_by: Movie → Writer
- distributed_by: Movie → Studio
- nominated_for: Movie → Award
- won_award: Movie → Award
- starred_as: Actor → Character (in specific movie)
- belongs_to_genre: Movie → Genre
- prequel_of: Movie → Movie
- sequel_of: Movie → Movie

EXTRACTION PRIORITIES:
1. Movie title and release year (most important)
2. Director and main cast
3. Awards and nominations
4. Production company
5. Box office and budget (if mentioned)
6. Plot and characters (optional)

FORMATTING NOTES:
- Use standard English names for entities
- Include full movie titles (e.g., "Avatar: The Way of Water")
- For actors, use their professional names
- Include award year and category if available
"""

    async def post_process_extraction(
        self, result: ExtractionResult, doc_text: str
    ) -> ExtractionResult:
        """Post-process extraction results for the movie domain.

        Applies movie-specific cleaning and enrichment to extracted entities and relations.

        Args:
            result: The raw extraction result from LLM
            doc_text: The original document text

        Returns:
            Enriched extraction result with movie-specific post-processing
        """
        # Clean up entity names and types
        for entity in result.entities:
            # Standardize entity types
            entity.type = self._standardize_entity_type(entity.type)

            # Clean up names (trim whitespace, fix common issues)
            entity.name = entity.name.strip()

            # Add movie-specific properties if possible
            if entity.type == "Movie":
                entity = self._enrich_movie_entity(entity, doc_text)
            elif entity.type == "Person":
                entity = self._enrich_person_entity(entity, doc_text)

        # Clean up relation types
        for relation in result.relations:
            relation.relation_type = self._standardize_relation_type(relation.relation_type)

        return result

    def _standardize_entity_type(self, entity_type: str) -> str:
        """Standardize entity type names to movie domain vocabulary.

        Args:
            entity_type: Raw entity type from LLM

        Returns:
            Standardized entity type
        """
        type_map = {
            "film": "Movie",
            "movie": "Movie",
            "cinema": "Movie",
            "actor": "Person",
            "actress": "Person",
            "director": "Person",
            "producer": "Person",
            "writer": "Person",
            "cinematographer": "Person",
            "composer": "Person",
            "performer": "Person",
            "studio": "Studio",
            "production_studio": "Studio",
            "distributor": "Studio",
            "award": "Award",
            "oscar": "Award",
            "golden_globe": "Award",
            "award_nomination": "Award",
            "genre": "Genre",
            "character": "Character",
            "role": "Character",
        }

        # Case-insensitive lookup
        normalized = entity_type.lower().replace(" ", "_")
        return type_map.get(normalized, entity_type)

    def _standardize_relation_type(self, relation_type: str) -> str:
        """Standardize relation type names to movie domain vocabulary.

        Args:
            relation_type: Raw relation type from LLM

        Returns:
            Standardized relation type
        """
        type_map = {
            "directed": "directed_by",
            "direct": "directed_by",
            "acted": "acted_in",
            "acted_in": "acted_in",
            "starred_in": "acted_in",
            "starring": "acted_in",
            "produced": "produced_by",
            "written": "written_by",
            "distributed": "distributed_by",
            "nominated_for": "nominated_for",
            "nominated": "nominated_for",
            "won": "won_award",
            "won_award": "won_award",
            "prequel": "prequel_of",
            "sequel": "sequel_of",
            "spinoff": "spinoff_of",
            "based_on": "based_on",
            "remake_of": "remake_of",
        }

        # Case-insensitive lookup
        normalized = relation_type.lower().replace(" ", "_")
        return type_map.get(normalized, relation_type)

    def _enrich_movie_entity(self, entity: Entity, doc_text: str) -> Entity:
        """Enrich a movie entity with additional extracted information.

        Args:
            entity: The movie entity to enrich
            doc_text: The source document text

        Returns:
            Enriched entity with additional properties
        """
        # This would be implemented with more sophisticated extraction logic
        # For now, just return as-is
        return entity

    def _enrich_person_entity(self, entity: Entity, doc_text: str) -> Entity:
        """Enrich a person entity with additional extracted information.

        Args:
            entity: The person entity to enrich
            doc_text: The source document text

        Returns:
            Enriched entity with additional properties
        """
        # This would be implemented with more sophisticated extraction logic
        # For now, just return as-is
        return entity


__all__ = ["MovieKGPreprocessor"]
