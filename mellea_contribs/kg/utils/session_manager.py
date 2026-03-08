"""Session and backend management utilities.

Provides factory functions for creating Mellea sessions and graph backends.
"""

import sys
from typing import Optional, Tuple

try:
    from mellea import start_session, MelleaSession
except ImportError:
    MelleaSession = None  # type: ignore

from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend
except ImportError:
    Neo4jBackend = None


def create_session(
    backend_name: str = "litellm",
    model_id: str = "gpt-4o-mini",
    temperature: float = 0.7,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> "MelleaSession":
    """Create a Mellea session.

    Args:
        backend_name: Backend name (default: "litellm").
        model_id: Model ID to use (default: "gpt-4o-mini").
        temperature: Temperature for generation (default: 0.7).
        api_base: Optional API base URL.
        api_key: Optional API key.

    Returns:
        MelleaSession object.

    Raises:
        ImportError: If mellea is not installed.
    """
    if MelleaSession is None:
        print("Error: mellea not installed. Run: pip install mellea[litellm]")
        sys.exit(1)

    return start_session(backend_name=backend_name, model_id=model_id)


def create_backend(
    backend_type: str = "mock",
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
) -> GraphBackend:
    """Create a graph backend.

    Args:
        backend_type: Type of backend ("mock" or "neo4j", default: "mock").
        neo4j_uri: Neo4j connection URI (default: "bolt://localhost:7687").
        neo4j_user: Neo4j username (default: "neo4j").
        neo4j_password: Neo4j password (default: "password").

    Returns:
        GraphBackend instance.

    Raises:
        SystemExit: If Neo4j backend requested but not available.
    """
    if backend_type == "mock":
        return MockGraphBackend()

    if backend_type == "neo4j":
        if Neo4jBackend is None:
            print(
                "Error: Neo4j backend not available. "
                "Install: pip install mellea-contribs[kg]"
            )
            sys.exit(1)

        neo4j_uri = neo4j_uri or "bolt://localhost:7687"
        neo4j_user = neo4j_user or "neo4j"
        neo4j_password = neo4j_password or "password"

        return Neo4jBackend(
            connection_uri=neo4j_uri,
            auth=(neo4j_user, neo4j_password),
        )

    raise ValueError(f"Unknown backend type: {backend_type}")


class MelleaResourceManager:
    """Async context manager for managing Mellea session and backend resources.

    Usage:
        async with MelleaResourceManager(backend_type="mock") as manager:
            session = manager.session
            backend = manager.backend
            # Use session and backend
    """

    def __init__(
        self,
        backend_type: str = "mock",
        model_id: str = "gpt-4o-mini",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """Initialize resource manager.

        Args:
            backend_type: Type of backend ("mock" or "neo4j", default: "mock").
            model_id: Model ID for session (default: "gpt-4o-mini").
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
        """
        self.backend_type = backend_type
        self.model_id = model_id
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.session: Optional[MelleaSession] = None
        self.backend: Optional[GraphBackend] = None

    async def __aenter__(self) -> "MelleaResourceManager":
        """Enter async context and create resources."""
        self.session = create_session(model_id=self.model_id)
        self.backend = create_backend(
            backend_type=self.backend_type,
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup resources."""
        if self.backend:
            await self.backend.close()
