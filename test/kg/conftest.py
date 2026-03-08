"""Pytest configuration for KG tests.

Reads Neo4j credentials from environment variables for testing.
"""

import os

import pytest

try:
    from mellea_contribs.kg.components.query import GraphQuery
except ImportError:
    # Allow tests to run even if mellea is not fully installed
    GraphQuery = None

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


@pytest.fixture
def neo4j_credentials():
    """Get Neo4j credentials from environment variables.

    Set these environment variables before running tests:
        NEO4J_URI: Connection URI (default: bolt://localhost:7687)
        NEO4J_USER: Username (default: neo4j)
        NEO4J_PASSWORD: Password (default: testpassword)

    Example:
        export NEO4J_PASSWORD="your_password"
        uv run pytest test/contribs/kg/test_neo4j_backend.py -v
    """
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "testpassword"),
    }


@pytest.fixture
async def neo4j_backend(neo4j_credentials):
    """Create a Neo4j backend for testing.

    Uses credentials from environment variables via neo4j_credentials fixture.
    Tests will be skipped if Neo4j is not available or connection fails.
    """
    if not NEO4J_AVAILABLE:
        pytest.skip("Neo4j driver not installed")

    backend = Neo4jBackend(
        connection_uri=neo4j_credentials["uri"],
        auth=(neo4j_credentials["user"], neo4j_credentials["password"]),
    )

    # Test connection
    try:
        await backend.get_schema()
        yield backend
    except Exception as e:
        pytest.skip(f"Could not connect to Neo4j: {e}")
    finally:
        await backend.close()


@pytest.fixture
async def populated_neo4j_backend(neo4j_backend):
    """Create a Neo4j backend with test data.

    Clears existing data, populates with test data, and cleans up after tests.
    """
    if GraphQuery is None:
        pytest.skip("GraphQuery not available (mellea not fully installed)")

    # Clear any existing data
    clear_query = GraphQuery(query_string="MATCH (n) DETACH DELETE n")
    await neo4j_backend.execute_query(clear_query)

    # Create test data
    create_query = GraphQuery(
        query_string="""
        CREATE (alice:Person {name: 'Alice', age: 30})
        CREATE (bob:Person {name: 'Bob', age: 35})
        CREATE (matrix:Movie {title: 'The Matrix', year: 1999})
        CREATE (alice)-[:ACTED_IN {role: 'Trinity'}]->(matrix)
        CREATE (bob)-[:ACTED_IN {role: 'Morpheus'}]->(matrix)
        RETURN alice, bob, matrix
        """
    )
    await neo4j_backend.execute_query(create_query)

    yield neo4j_backend

    # Cleanup
    await neo4j_backend.execute_query(clear_query)
