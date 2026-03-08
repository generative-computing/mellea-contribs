"""Domain-specific KG preprocessor examples.

This package contains example implementations of domain-specific KG preprocessors.
Each module demonstrates how to extend the generic KGPreprocessor for different domains.

Available Examples:
    MovieKGPreprocessor: Example preprocessor for the movie domain

Example Usage::

    from movie_preprocessor import MovieKGPreprocessor
    from mellea import start_session
    from mellea_contribs.kg import MockGraphBackend

    async def process_movies():
        session = start_session(backend_name="litellm", model_id="gpt-4o-mini")
        backend = MockGraphBackend()
        processor = MovieKGPreprocessor(backend=backend, session=session)

        result = await processor.process_document(
            doc_text="Avatar directed by James Cameron was released in 2009.",
            doc_id="avatar_1"
        )
        print(f"Extracted {len(result.entities)} entities and {len(result.relations)} relations")
"""

from .movie_preprocessor import MovieKGPreprocessor

__all__ = [
    "MovieKGPreprocessor",
]
