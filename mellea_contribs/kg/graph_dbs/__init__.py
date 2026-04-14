"""Graph database backend implementations.

Provides backend abstraction for different graph database systems.
"""

from typing import TYPE_CHECKING

from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend
except ImportError:
    if not TYPE_CHECKING:
        Neo4jBackend = None  # type: ignore[assignment]

__all__ = ["GraphBackend", "MockGraphBackend", "Neo4jBackend"]
