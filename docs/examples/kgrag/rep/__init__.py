"""Domain-specific KG representation examples.

This package contains example implementations of domain-specific entity
and relation representation utilities for different domains.

Example Usage::

    from movie_rep import movie_entity_to_text, format_movie_context
    from docs.examples.kgrag.models import MovieEntity

    movie = MovieEntity(
        type="Movie",
        name="Oppenheimer",
        description="2023 film",
        paragraph_start="Oppenheimer is",
        paragraph_end="by Nolan.",
        release_year=2023,
        director="Christopher Nolan"
    )

    # Format for LLM prompts
    text = movie_entity_to_text(movie, include_confidence=True)
    print(text)
"""

from movie_rep import (
    format_movie_context,
    movie_entity_to_text,
    movie_relation_to_text,
)

__all__ = [
    "movie_entity_to_text",
    "movie_relation_to_text",
    "format_movie_context",
]
