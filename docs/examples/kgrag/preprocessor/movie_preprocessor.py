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
        doc_text = '''Avatar is a 2009 science fiction film directed by James Cameron.
        It stars Sam Worthington, Zoe Saldana, and Sigourney Weaver.
        The film was nominated for multiple Academy Awards.
        '''

        result = await processor.process_document(
            doc_text=doc_text,
            doc_id="avatar_wiki"
        )
        print(f"Extracted {len(result.entities)} entities and {len(result.relations)} relations")
        await backend.close()

    asyncio.run(main())
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mellea_contribs.kg.components.query import CypherQuery
from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.models import Entity, ExtractionResult
from mellea_contribs.kg.preprocessor import KGPreprocessor
from mellea_contribs.kg.utils import log_progress


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


@dataclass
class PreprocessingStats:
    """Statistics for a predefined-data preprocessing run."""

    domain: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    entities_loaded: int
    entities_inserted: int
    relations_loaded: int
    relations_inserted: int
    success: bool
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "domain": self.domain,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "entities_loaded": self.entities_loaded,
            "entities_inserted": self.entities_inserted,
            "relations_loaded": self.relations_loaded,
            "relations_inserted": self.relations_inserted,
            "success": self.success,
            "error_message": self.error_message,
        }

    def __str__(self) -> str:
        """Format statistics for display."""
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        lines = [
            f"Domain: {self.domain}",
            f"Status: {status}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Entities loaded: {self.entities_loaded}",
            f"Entities inserted: {self.entities_inserted}",
            f"Relations loaded: {self.relations_loaded}",
            f"Relations inserted: {self.relations_inserted}",
        ]
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        return "\n".join(lines)


class PredefinedDataPreprocessor:
    """Loads predefined movie/person JSON databases into the knowledge graph.

    Reads ``movie_db.json`` and ``person_db.json`` from *data_dir* and inserts
    entities and relations via the graph backend's Cypher execution API.

    Args:
        backend: Graph database backend.
        data_dir: Directory containing ``movie_db.json`` and ``person_db.json``.
        batch_size: Number of records per Cypher batch (default: 50).
    """

    def __init__(
        self,
        backend: GraphBackend,
        data_dir: Path,
        batch_size: int = 50,
    ):
        """Initialize preprocessor."""
        self.backend = backend
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.movie_db: Dict[str, Dict] = {}
        self.person_db: Dict[str, Dict] = {}

    async def preprocess(self) -> PreprocessingStats:
        """Run the full preprocessing pipeline.

        Returns:
            PreprocessingStats with counts and timing.
        """
        start_time = datetime.now()
        try:
            log_progress("Loading movie database...")
            self.movie_db = self._load_json_file("movie_db.json")
            log_progress(f"✓ Loaded {len(self.movie_db)} movies")

            log_progress("Loading person database...")
            self.person_db = self._load_json_file("person_db.json")
            log_progress(f"✓ Loaded {len(self.person_db)} persons")

            log_progress("\nInserting movie entities...")
            movies_inserted = await self._insert_movies()
            log_progress("Inserting person entities...")
            persons_inserted = await self._insert_persons()
            log_progress("\nInserting movie-person relations...")
            relations_inserted = await self._insert_movie_relations()

            end_time = datetime.now()
            return PreprocessingStats(
                domain="movie",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                entities_loaded=len(self.movie_db) + len(self.person_db),
                entities_inserted=movies_inserted + persons_inserted,
                relations_loaded=0,
                relations_inserted=relations_inserted,
                success=True,
            )
        except Exception as e:
            end_time = datetime.now()
            log_progress(f"✗ Preprocessing failed: {e}")
            return PreprocessingStats(
                domain="movie",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                entities_loaded=0,
                entities_inserted=0,
                relations_loaded=0,
                relations_inserted=0,
                success=False,
                error_message=str(e),
            )
        finally:
            await self.backend.close()

    def _load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load a JSON file from the data directory."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    async def _execute_cypher_batch(self, cypher_query: str, batch: List[Dict]) -> None:
        """Execute a parameterised Cypher query with a batch of items."""
        if not batch:
            return
        try:
            query = CypherQuery(cypher_query, parameters={"batch": batch})
            await self.backend.execute_query(query)
        except Exception as e:
            log_progress(f"  Warning: Batch insert failed: {e}")

    async def _insert_movies(self) -> int:
        """Insert movie entities. Returns count inserted."""
        count = 0
        batch: List[Dict] = []
        for movie_id, movie_data in self.movie_db.items():
            batch.append({
                "name": movie_data.get("title", f"Movie_{movie_id}").upper(),
                "release_date": movie_data.get("release_date"),
                "original_language": movie_data.get("original_language"),
                "budget": str(movie_data.get("budget")) if movie_data.get("budget") else None,
                "revenue": str(movie_data.get("revenue")) if movie_data.get("revenue") else None,
                "rating": str(movie_data.get("rating")) if movie_data.get("rating") else None,
            })
            count += 1
            if len(batch) >= self.batch_size:
                await self._execute_cypher_batch(
                    """
                    UNWIND $batch AS movie
                    MERGE (m:Movie {name: movie.name})
                    SET m.release_date = movie.release_date,
                        m.original_language = movie.original_language,
                        m.budget = movie.budget,
                        m.revenue = movie.revenue,
                        m.rating = movie.rating
                    """,
                    batch,
                )
                log_progress(f"  Inserted {count} movies...")
                batch = []
        if batch:
            await self._execute_cypher_batch(
                """
                UNWIND $batch AS movie
                MERGE (m:Movie {name: movie.name})
                SET m.release_date = movie.release_date,
                    m.original_language = movie.original_language,
                    m.budget = movie.budget,
                    m.revenue = movie.revenue,
                    m.rating = movie.rating
                """,
                batch,
            )
        log_progress(f"✓ Inserted {count} movie entities")
        return count

    async def _insert_persons(self) -> int:
        """Insert person entities. Returns count inserted."""
        count = 0
        batch: List[Dict] = []
        for person_id, person_data in self.person_db.items():
            batch.append({
                "name": person_data.get("name", f"Person_{person_id}").upper(),
                "birthday": person_data.get("birthday"),
            })
            count += 1
            if len(batch) >= self.batch_size:
                await self._execute_cypher_batch(
                    """
                    UNWIND $batch AS person
                    MERGE (p:Person {name: person.name})
                    SET p.birthday = person.birthday
                    """,
                    batch,
                )
                log_progress(f"  Inserted {count} persons...")
                batch = []
        if batch:
            await self._execute_cypher_batch(
                """
                UNWIND $batch AS person
                MERGE (p:Person {name: person.name})
                SET p.birthday = person.birthday
                """,
                batch,
            )
        log_progress(f"✓ Inserted {count} person entities")
        return count

    async def _insert_movie_relations(self) -> int:
        """Insert relations between movies and persons. Returns count inserted."""
        count = 0
        cast_batch: List[Dict] = []
        director_batch: List[Dict] = []
        genre_batch: List[Dict] = []

        for movie_id, movie_data in self.movie_db.items():
            movie_name = movie_data.get("title", f"Movie_{movie_id}").upper()

            for cast_member in movie_data.get("cast") or []:
                if not isinstance(cast_member, dict):
                    continue
                person_name = cast_member.get("name", "").upper()
                if not person_name:
                    continue
                cast_batch.append({
                    "person_name": person_name,
                    "movie_name": movie_name,
                    "character": cast_member.get("character", ""),
                    "order": cast_member.get("order", 0),
                })
                count += 1
                if len(cast_batch) >= self.batch_size:
                    await self._execute_cypher_batch(
                        """
                        UNWIND $batch AS item
                        MATCH (m:Movie {name: item.movie_name})
                        MATCH (p:Person {name: item.person_name})
                        MERGE (p)-[:ACTED_IN {character: item.character, order: item.order}]->(m)
                        """,
                        cast_batch,
                    )
                    cast_batch = []

            for crew_member in movie_data.get("crew") or []:
                if not isinstance(crew_member, dict):
                    continue
                person_name = crew_member.get("name", "").upper()
                job = crew_member.get("job", "").lower()
                if not person_name or not job:
                    continue
                if "director" in job:
                    director_batch.append({"person_name": person_name, "movie_name": movie_name})
                    if len(director_batch) >= self.batch_size:
                        await self._execute_cypher_batch(
                            """
                            UNWIND $batch AS item
                            MATCH (m:Movie {name: item.movie_name})
                            MATCH (p:Person {name: item.person_name})
                            MERGE (p)-[:DIRECTED]->(m)
                            """,
                            director_batch,
                        )
                        director_batch = []
                count += 1

            for genre in movie_data.get("genres") or []:
                genre_name = (genre.get("name", "") if isinstance(genre, dict) else str(genre)).upper()
                if not genre_name:
                    continue
                genre_batch.append({"movie_name": movie_name, "genre_name": genre_name})
                count += 1
                if len(genre_batch) >= self.batch_size:
                    await self._execute_cypher_batch(
                        """
                        UNWIND $batch AS item
                        MATCH (m:Movie {name: item.movie_name})
                        MERGE (g:Genre {name: item.genre_name})
                        MERGE (m)-[:BELONGS_TO_GENRE]->(g)
                        """,
                        genre_batch,
                    )
                    genre_batch = []

        # Flush remaining batches
        if cast_batch:
            await self._execute_cypher_batch(
                """
                UNWIND $batch AS item
                MATCH (m:Movie {name: item.movie_name})
                MATCH (p:Person {name: item.person_name})
                MERGE (p)-[:ACTED_IN {character: item.character, order: item.order}]->(m)
                """,
                cast_batch,
            )
        if director_batch:
            await self._execute_cypher_batch(
                """
                UNWIND $batch AS item
                MATCH (m:Movie {name: item.movie_name})
                MATCH (p:Person {name: item.person_name})
                MERGE (p)-[:DIRECTED]->(m)
                """,
                director_batch,
            )
        if genre_batch:
            await self._execute_cypher_batch(
                """
                UNWIND $batch AS item
                MATCH (m:Movie {name: item.movie_name})
                MERGE (g:Genre {name: item.genre_name})
                MERGE (m)-[:BELONGS_TO_GENRE]->(g)
                """,
                genre_batch,
            )

        log_progress(f"✓ Inserted {count} relations")
        return count


__all__ = ["MovieKGPreprocessor", "PredefinedDataPreprocessor", "PreprocessingStats"]
