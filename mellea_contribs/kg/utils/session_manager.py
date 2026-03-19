"""Session and backend management utilities.

Provides factory functions for creating Mellea sessions and graph backends.
"""

import sys
from typing import Optional

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
) -> "MelleaSession":
    """Create a Mellea session.

    For OpenAI-compatible endpoints with custom base URL or API key, use
    :func:`create_session_from_env` or :func:`create_openai_session` instead.

    Args:
        backend_name: Backend name (default: "litellm").
        model_id: Model ID to use (default: "gpt-4o-mini").
        temperature: Temperature for generation (default: 0.7).

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


def create_openai_session(
    model_id: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 1800,
    extra_headers: Optional[dict] = None,
    force_openai_schema: bool = True,
) -> "MelleaSession":
    """Create a Mellea session backed by an OpenAI-compatible endpoint.

    Unlike :func:`create_session` (which uses the generic LiteLLM backend),
    this function wires up ``OpenAIBackend`` directly, allowing fine-grained
    control over base URL, API key, timeout, and custom headers.  Suitable
    for Azure OpenAI, vLLM, and IBM RITS endpoints.

    Args:
        model_id: Model identifier recognised by the endpoint
            (e.g. ``"gpt-4o-mini"``).
        api_base: Base URL for the OpenAI-compatible API.  Falls back to the
            ``OPENAI_API_BASE`` environment variable when *None*.
        api_key: API key.  Falls back to the ``OPENAI_API_KEY`` environment
            variable when *None*.
        timeout: Request timeout in seconds (default: ``1800``).
        extra_headers: Additional HTTP headers forwarded with every request.
        force_openai_schema: When *True* (default), override Mellea's
            server-type detection so it always uses the strict OpenAI JSON
            schema format (``additionalProperties=False``, ``strict=True``).
            Mellea's auto-detection classifies non-``api.openai.com`` URLs as
            ``UNKNOWN`` and sends schemas without ``additionalProperties``,
            which many OpenAI-compatible endpoints (including IBM RITS) reject.

    Returns:
        A configured :class:`MelleaSession`.

    Raises:
        SystemExit: If ``mellea`` or ``mellea.backends.openai`` is not
            installed.
    """
    import os

    if MelleaSession is None:
        print("Error: mellea not installed. Run: pip install mellea[litellm]")
        sys.exit(1)

    try:
        from mellea import MelleaSession as _MS
        from mellea.backends.openai import OpenAIBackend, TemplateFormatter
    except ImportError:
        print("Error: mellea.backends.openai not available.")
        sys.exit(1)

    resolved_base = api_base or os.environ.get("OPENAI_API_BASE")
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")

    backend = OpenAIBackend(
        model_id=model_id,
        formatter=TemplateFormatter(model_id=model_id),
        base_url=resolved_base,
        api_key=resolved_key,
        timeout=timeout,
        default_headers=extra_headers or {},
    )

    if force_openai_schema and resolved_base is not None:
        try:
            from mellea.helpers.server_type import _ServerType
            backend._server_type = _ServerType.OPENAI
        except ImportError:
            pass

    if resolved_base is not None:
        _patch_openai_backend_error_logging()

    return _MS(backend=backend)


def _patch_openai_backend_error_logging() -> None:
    """Patch ``send_to_queue`` in the OpenAI backend module so API exceptions
    are logged before being silently swallowed by Mellea's async queue.

    Mellea catches all exceptions from the LLM call inside ``send_to_queue``
    and puts them on a queue.  The backend's ``processing()`` function ignores
    non-response objects, so the original error is permanently lost and
    ``post_processing()`` raises a confusing ``KeyError: 'oai_chat_response'``
    instead.  This patch logs the real exception at ERROR level first.
    """
    import logging
    import traceback as _tb
    from collections.abc import AsyncIterator, Coroutine

    try:
        import mellea.backends.openai as _mo
        import mellea.helpers.async_helpers as _ah
    except ImportError:
        return

    _logger = logging.getLogger("mellea_contribs.kg")

    async def _logged_send_to_queue(co, aqueue) -> None:  # type: ignore[type-arg]
        try:
            aresponse = await co if isinstance(co, Coroutine) else co
            if isinstance(aresponse, AsyncIterator):
                async for item in aresponse:
                    await aqueue.put(item)
            else:
                await aqueue.put(aresponse)
            await aqueue.put(None)
        except Exception as exc:
            _logger.error(
                f"[API] Call failed: {type(exc).__name__}: {exc}"
            )
            _logger.debug(_tb.format_exc())
            await aqueue.put(exc)

    _mo.send_to_queue = _logged_send_to_queue
    _ah.send_to_queue = _logged_send_to_queue


def create_session_from_env(
    default_model: str = "gpt-4o-mini",
    timeout: int = 1800,
    env_prefix: str = "",
) -> tuple:
    """Create a Mellea session from standard environment variables.

    Reads ``{prefix}API_BASE``, ``{prefix}API_KEY``, ``{prefix}MODEL_NAME``,
    and ``{prefix}RITS_API_KEY`` from the environment and delegates to
    :func:`create_openai_session`.  Suitable for any OpenAI-compatible
    endpoint including IBM RITS.

    Args:
        default_model: Model to use when ``MODEL_NAME`` is not set.
        timeout: Request timeout in seconds (default: ``1800``).
        env_prefix: Optional prefix for all env var names (e.g. ``"EVAL_"``
            reads ``EVAL_API_BASE``, ``EVAL_API_KEY``, etc.).

    Returns:
        ``(session, model_id)`` tuple — the configured
        :class:`MelleaSession` and the resolved model name string.
    """
    import os

    import logging
    _log = logging.getLogger("mellea_contribs.kg")

    p = env_prefix
    api_base = os.getenv(f"{p}API_BASE")
    api_key = os.getenv(f"{p}API_KEY", "dummy")
    model_id = os.getenv(f"{p}MODEL_NAME", default_model)
    # Fall back to the unprefixed RITS_API_KEY when the prefixed one isn't set,
    # since the same RITS credentials are typically shared across all sessions.
    rits_api_key = os.getenv(f"{p}RITS_API_KEY") or (
        os.getenv("RITS_API_KEY") if p else None
    )

    _log.info(
        f"create_session_from_env(prefix={repr(env_prefix)}): "
        f"api_base={'set' if api_base else 'MISSING'}, "
        f"api_key={'set' if api_key != 'dummy' else 'dummy/unset'}, "
        f"rits_api_key={'set' if rits_api_key else 'MISSING'}"
    )

    extra_headers: dict = {}
    if rits_api_key:
        extra_headers["RITS_API_KEY"] = rits_api_key

    session = create_openai_session(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
        extra_headers=extra_headers or None,
    )
    return session, model_id


def create_embedding_client(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "text-embedding-3-small",
    timeout: int = 1800,
):
    """Create an async OpenAI-compatible embedding client.

    The returned client exposes ``client.embeddings.create(input, model)``
    just like the official ``openai.AsyncOpenAI`` client, making it usable
    with any OpenAI-compatible embedding endpoint (Azure, vLLM, IBM RITS,
    etc.).

    Args:
        api_base: Base URL of the embedding endpoint.  Falls back to
            ``OPENAI_API_BASE`` when *None*.
        api_key: API key.  Falls back to ``OPENAI_API_KEY`` when *None*.
        model_name: Default model name attached to the client as
            ``client._model_name`` for convenience.
        timeout: HTTP timeout in seconds (default: ``1800``).

    Returns:
        An ``openai.AsyncOpenAI`` instance (or ``None`` when *openai* is not
        installed).
    """
    import os

    resolved_base = api_base or os.environ.get("OPENAI_API_BASE")
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=resolved_base,
            api_key=resolved_key,
            timeout=timeout,
        )
        # Attach model name so callers can read it without a separate arg
        client._model_name = model_name  # type: ignore[attr-defined]
        return client
    except ImportError:
        print("Warning: openai package not installed; embedding client unavailable.")
        return None


async def generate_embeddings(
    client,
    texts: list,
    model_name: Optional[str] = None,
) -> list:
    """Generate embeddings for a list of texts using an async OpenAI client.

    Args:
        client: Async OpenAI-compatible client (from
            :func:`create_embedding_client`).
        texts: List of strings to embed.
        model_name: Override the model name.  Uses ``client._model_name``
            when *None*, falling back to ``"text-embedding-3-small"``.

    Returns:
        List of embedding vectors (one per input text), or a list of
        ``None`` values when the client is unavailable or the call fails.
    """
    if client is None or not texts:
        return [None] * len(texts)

    resolved_model = (
        model_name
        or getattr(client, "_model_name", None)
        or "text-embedding-3-small"
    )
    try:
        response = await client.embeddings.create(input=texts, model=resolved_model)
        return [item.embedding for item in response.data]
    except Exception as exc:
        print(f"Warning: embedding call failed — {exc}", file=sys.stderr)
        return [None] * len(texts)
