#!/usr/bin/env python3
"""Knowledge Graph Preprocessing from Predefined Data.

Loads movie and person databases and inserts them into Neo4j using mellea-contribs
library components. This follows the mellea project pattern but uses mellea-contribs'
Entity/Relation models and GraphBackend for persistence.

Usage:
    python run_kg_preprocess.py --data-dir ./dataset/movie --neo4j-uri bolt://localhost:7687
    python run_kg_preprocess.py --data-dir ./dataset/movie --mock
    python run_kg_preprocess.py --data-dir ./dataset/movie --verbose
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mellea_contribs.kg.models import Entity, Relation
from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.utils import (
    create_session,
    create_backend,
    log_progress,
    output_json,
    print_stats,
)


@dataclass
class PreprocessingStats:
    """Statistics for preprocessing operations."""

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
    """Preprocessor for inserting predefined movie/person data into KG.

    Loads movie and person databases from JSON files and inserts them into Neo4j
    using mellea-contribs Entity/Relation models and GraphBackend.
    """

    def __init__(
        self,
        backend: GraphBackend,
        session: Any,
        data_dir: Path,
        batch_size: int = 50,
    ):
        """Initialize preprocessor.

        Args:
            backend: Graph database backend
            session: Mellea session for LLM operations
            data_dir: Directory containing movie_db.json and person_db.json
            batch_size: Batch size for inserting entities
        """
        self.backend = backend
        self.session = session
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

        self.movie_db: Dict[str, Dict] = {}
        self.person_db: Dict[str, Dict] = {}

    async def preprocess(self) -> PreprocessingStats:
        """Run the full preprocessing pipeline."""
        start_time = datetime.now()

        try:
            # Load data
            log_progress("Loading movie database...")
            self.movie_db = self._load_json_file("movie_db.json")
            log_progress(f"✓ Loaded {len(self.movie_db)} movies")

            log_progress("Loading person database...")
            self.person_db = self._load_json_file("person_db.json")
            log_progress(f"✓ Loaded {len(self.person_db)} persons")

            # Insert entities
            log_progress("\nInserting movie entities...")
            movies_inserted = await self._insert_movies()

            log_progress("Inserting person entities...")
            persons_inserted = await self._insert_persons()

            # Insert relations
            log_progress("\nInserting movie-person relations...")
            relations_inserted = await self._insert_movie_relations()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return PreprocessingStats(
                domain="movie",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                entities_loaded=len(self.movie_db) + len(self.person_db),
                entities_inserted=movies_inserted + persons_inserted,
                relations_loaded=0,  # Count calculated during insertion
                relations_inserted=relations_inserted,
                success=True,
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            log_progress(f"✗ Preprocessing failed: {e}")

            return PreprocessingStats(
                domain="movie",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
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
        """Load JSON file from data directory.

        Args:
            filename: Name of JSON file to load

        Returns:
            Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, "r") as f:
            return json.load(f)

    async def _insert_movies(self) -> int:
        """Insert movie entities into the graph using Cypher queries.

        Returns:
            Number of movies inserted
        """
        count = 0
        batch = []

        for movie_id, movie_data in self.movie_db.items():
            try:
                # Prepare movie data (using Entity model for structure)
                movie_dict = {
                    "name": movie_data.get("title", f"Movie_{movie_id}").upper(),
                    "release_date": movie_data.get("release_date"),
                    "original_language": movie_data.get("original_language"),
                    "budget": str(movie_data.get("budget")) if movie_data.get("budget") else None,
                    "revenue": str(movie_data.get("revenue")) if movie_data.get("revenue") else None,
                    "rating": str(movie_data.get("rating")) if movie_data.get("rating") else None,
                }

                batch.append(movie_dict)
                count += 1

                # Execute batch when size reached or at end
                if len(batch) >= self.batch_size:
                    await self._execute_batch_insert_movies(batch)
                    log_progress(f"  Inserted {count} movies...")
                    batch = []

            except Exception as e:
                log_progress(f"  Warning: Failed to prepare movie {movie_id}: {e}")
                continue

        # Insert remaining batch
        if batch:
            await self._execute_batch_insert_movies(batch)

        log_progress(f"✓ Inserted {count} movie entities")
        return count

    async def _execute_cypher_batch(self, cypher_query: str, batch: List[Dict]) -> None:
        """Execute a parameterised Cypher query with a batch of items.

        Args:
            cypher_query: Cypher query with ``$batch`` parameter.
            batch: List of parameter dicts to pass as ``batch``.
        """
        if not batch:
            return
        try:
            if self.backend.backend_id == "neo4j" and hasattr(self.backend, '_async_driver'):
                async with self.backend._async_driver.session() as session:
                    await session.run(cypher_query, batch=batch)
            # Mock backend: no-op (data lives in memory)
        except Exception as e:
            log_progress(f"  Warning: Batch insert failed: {e}")

    async def _execute_batch_insert_movies(self, batch: List[Dict]) -> None:
        """Execute Cypher query to insert batch of movies."""
        await self._execute_cypher_batch(
            """
            UNWIND $batch AS movie
            MERGE (m:Movie {name: movie.name})
            SET m.release_date = movie.release_date,
                m.original_language = movie.original_language,
                m.budget = movie.budget,
                m.revenue = movie.revenue,
                m.rating = movie.rating
            RETURN count(m) as inserted
            """,
            batch,
        )

    async def _insert_persons(self) -> int:
        """Insert person entities into the graph using Cypher queries.

        Returns:
            Number of persons inserted
        """
        count = 0
        batch = []

        for person_id, person_data in self.person_db.items():
            try:
                # Prepare person data (using Entity model for structure)
                person_dict = {
                    "name": person_data.get("name", f"Person_{person_id}").upper(),
                    "birthday": person_data.get("birthday"),
                }

                batch.append(person_dict)
                count += 1

                # Execute batch when size reached or at end
                if len(batch) >= self.batch_size:
                    await self._execute_batch_insert_persons(batch)
                    log_progress(f"  Inserted {count} persons...")
                    batch = []

            except Exception as e:
                log_progress(f"  Warning: Failed to prepare person {person_id}: {e}")
                continue

        # Insert remaining batch
        if batch:
            await self._execute_batch_insert_persons(batch)

        log_progress(f"✓ Inserted {count} person entities")
        return count

    async def _execute_batch_insert_persons(self, batch: List[Dict]) -> None:
        """Execute Cypher query to insert batch of persons."""
        await self._execute_cypher_batch(
            """
            UNWIND $batch AS person
            MERGE (p:Person {name: person.name})
            SET p.birthday = person.birthday
            RETURN count(p) as inserted
            """,
            batch,
        )

    async def _insert_movie_relations(self) -> int:
        """Insert relations between movies and persons using Cypher queries.

        Handles cast (ACTED_IN), crew/directors (DIRECTED), and genres (BELONGS_TO_GENRE).

        Returns:
            Number of relations inserted
        """
        count = 0
        cast_batch = []
        director_batch = []
        genre_batch = []

        for movie_id, movie_data in self.movie_db.items():
            movie_name = movie_data.get("title", f"Movie_{movie_id}").upper()

            # Process cast (actors) - cast is array of {name, character, order, ...}
            if movie_data.get("cast") and isinstance(movie_data["cast"], list):
                for cast_member in movie_data["cast"]:
                    try:
                        if not isinstance(cast_member, dict):
                            continue

                        person_name = cast_member.get("name", "").upper()
                        if not person_name:
                            continue

                        # Prepare relation data using Relation model for structure
                        cast_batch.append({
                            "person_name": person_name,
                            "movie_name": movie_name,
                            "character": cast_member.get("character", ""),
                            "order": cast_member.get("order", 0),
                        })
                        count += 1

                        # Execute batch when full
                        if len(cast_batch) >= self.batch_size:
                            await self._execute_batch_insert_cast(cast_batch)
                            cast_batch = []

                    except Exception as e:
                        log_progress(f"  Warning: Failed to prepare cast for {movie_name}: {e}")
                        continue

            # Process crew (directors, writers, etc.) - crew is array of {name, job, id, ...}
            if movie_data.get("crew") and isinstance(movie_data["crew"], list):
                for crew_member in movie_data["crew"]:
                    try:
                        if not isinstance(crew_member, dict):
                            continue

                        person_name = crew_member.get("name", "").upper()
                        job = crew_member.get("job", "").lower()

                        if not person_name or not job:
                            continue

                        # Extract directors from crew
                        if "director" in job or job == "director":
                            director_batch.append({
                                "person_name": person_name,
                                "movie_name": movie_name,
                            })

                            # Execute batch when full
                            if len(director_batch) >= self.batch_size:
                                await self._execute_batch_insert_directors(director_batch)
                                director_batch = []

                        count += 1

                    except Exception as e:
                        log_progress(f"  Warning: Failed to prepare crew for {movie_name}: {e}")
                        continue

            # Process genres - genres is array of {id, name, ...}
            if movie_data.get("genres") and isinstance(movie_data["genres"], list):
                for genre in movie_data["genres"]:
                    try:
                        if isinstance(genre, dict):
                            genre_name = genre.get("name", "").upper()
                        else:
                            genre_name = str(genre).upper()

                        if not genre_name:
                            continue

                        genre_batch.append({
                            "movie_name": movie_name,
                            "genre_name": genre_name,
                        })

                        # Execute batch when full
                        if len(genre_batch) >= self.batch_size:
                            await self._execute_batch_insert_genres(genre_batch)
                            genre_batch = []

                        count += 1

                    except Exception as e:
                        log_progress(f"  Warning: Failed to prepare genre for {movie_name}: {e}")
                        continue

        # Execute remaining batches
        if cast_batch:
            await self._execute_batch_insert_cast(cast_batch)
        if director_batch:
            await self._execute_batch_insert_directors(director_batch)
        if genre_batch:
            await self._execute_batch_insert_genres(genre_batch)

        log_progress(f"✓ Inserted {count} relations")
        return count

    async def _execute_batch_insert_cast(self, batch: List[Dict]) -> None:
        """Execute Cypher query to insert batch of cast relations."""
        await self._execute_cypher_batch(
            """
            UNWIND $batch AS item
            MATCH (m:Movie {name: item.movie_name})
            MATCH (p:Person {name: item.person_name})
            MERGE (p)-[:ACTED_IN {character: item.character, order: item.order}]->(m)
            """,
            batch,
        )

    async def _execute_batch_insert_directors(self, batch: List[Dict]) -> None:
        """Execute Cypher query to insert batch of director relations."""
        await self._execute_cypher_batch(
            """
            UNWIND $batch AS item
            MATCH (m:Movie {name: item.movie_name})
            MATCH (p:Person {name: item.person_name})
            MERGE (p)-[:DIRECTED]->(m)
            """,
            batch,
        )

    async def _execute_batch_insert_genres(self, batch: List[Dict]) -> None:
        """Execute Cypher query to insert batch of genre relations."""
        await self._execute_cypher_batch(
            """
            UNWIND $batch AS item
            MATCH (m:Movie {name: item.movie_name})
            MERGE (g:Genre {name: item.genre_name})
            MERGE (m)-[:BELONGS_TO_GENRE]->(g)
            """,
            batch,
        )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess and load predefined movie data into KG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data-dir ./data/movie                                    # Load from data directory
  %(prog)s --data-dir ./data/movie --mock                             # Use mock backend
  %(prog)s --data-dir ./data/movie --neo4j-uri bolt://localhost:7687  # Custom Neo4j URI
  %(prog)s --data-dir ./data/movie --verbose                          # Verbose logging
        """,
    )

    # Data configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing movie_db.json and person_db.json",
    )

    # Backend configuration
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)",
    )

    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)",
    )

    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password (default: password)",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockGraphBackend instead of Neo4j (no database needed)",
    )

    # Other options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for inserting entities (default: 50)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log_progress(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    required_files = ["movie_db.json", "person_db.json"]
    for filename in required_files:
        if not (data_dir / filename).exists():
            log_progress(f"ERROR: Required file not found: {data_dir / filename}")
            sys.exit(1)

    try:
        # Initialize backend and session
        backend = create_backend(
            backend_type="neo4j" if not args.mock else "mock",
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        )
        session = create_session(model_id=args.model)

        # Create and run preprocessor
        log_progress("=" * 60)
        log_progress("KG Preprocessing from Predefined Data")
        log_progress("=" * 60)
        log_progress(f"Data directory: {data_dir}")
        log_progress(f"Backend: {'Neo4j' if not args.mock else 'Mock'}")
        log_progress("")

        preprocessor = PredefinedDataPreprocessor(
            backend=backend,
            session=session,
            data_dir=data_dir,
            batch_size=args.batch_size,
        )

        # Run preprocessing
        stats = await preprocessor.preprocess()

        # Print statistics
        log_progress("")
        log_progress("=" * 60)
        log_progress("PREPROCESSING SUMMARY")
        log_progress("=" * 60)
        log_progress(str(stats))
        log_progress("=" * 60)
        log_progress("")

        # Output JSON
        print(json.dumps(stats.to_dict()))

        # Return appropriate exit code
        sys.exit(0 if stats.success else 1)

    except KeyboardInterrupt:
        log_progress("\n⚠️  Preprocessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_progress(f"❌ Preprocessing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
