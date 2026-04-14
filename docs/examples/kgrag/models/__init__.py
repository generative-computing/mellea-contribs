"""Domain-specific entity models for KG-RAG examples.

This package contains example implementations of domain-specific entity models.
Each module demonstrates how to extend the base Entity class for different domains.

Example:
    To use the movie domain models::

        from docs.examples.kgrag.models import MovieEntity, PersonEntity, AwardEntity

        movie = MovieEntity(
            type="Movie",
            name="Oppenheimer",
            description="2023 biographical film",
            paragraph_start="Oppenheimer is",
            paragraph_end="by Nolan.",
            release_year=2023,
            director="Christopher Nolan"
        )
"""

from docs.examples.kgrag.models.movie_domain_models import (
    AwardEntity,
    MovieEntity,
    PersonEntity,
)

__all__ = [
    "MovieEntity",
    "PersonEntity",
    "AwardEntity",
]
