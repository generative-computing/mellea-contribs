"""ACPBackend — Mellea backend that drives Bob via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
import datetime
import os
from contextlib import AsyncExitStack
from typing import Any, Sequence

from acp import Client, NewSessionResponse
from acp.interfaces import Agent
from acp.schema import AgentMessageChunk, TextContentBlock
from acp.stdio import spawn_agent_process

from mellea.core import C, BaseModelSubclass, CBlock, Component, Context, ModelOutputThunk
from mellea.core.backend import Backend
from mellea.core.base import GenerateLog
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components import Message


class _BufferingClient:
    """Minimal ACP Client that accumulates AgentMessageChunk text."""

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def clear(self) -> None:
        self._chunks.clear()

    def collected_text(self) -> str:
        return "".join(self._chunks)

    # --- ACP Client protocol ---

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(update, AgentMessageChunk):
            content = update.content
            if isinstance(content, TextContentBlock):
                self._chunks.append(content.text)

    async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ACPBackend does not handle permission requests")


class ACPBackend(Backend):
    """Mellea Backend that drives any ACP-compatible stdio agent subprocess.

    By default spawns Bob (``bob acp``). Pass a different ``command`` to use
    any other ACP agent — e.g. ``command=["vibe-acp"]`` for Mistral Vibe.

    Usage::

        from mellea.stdlib.session import MelleaSession
        from mellea_contribs.acp import ACPBackend

        # Bob (default)
        m = MelleaSession(ACPBackend())

        # Mistral Vibe
        m = MelleaSession(ACPBackend(command=["vibe-acp"]))

        result = m.instruct("Write a haiku about Python.")
    """

    _provider: str = "acp"

    def __init__(self, command: list[str] | None = None) -> None:
        """Create an ACPBackend.

        Args:
            command: The command used to launch the ACP agent subprocess.
                     Defaults to ``["bob", "acp"]``.
                     Example: ``["vibe-acp"]``, ``["bob", "acp", "--trust"]``.
        """
        self._command: list[str] = command if command is not None else ["bob", "acp"]
        self._model_id: str = self._command[0]
        self._formatter = TemplateFormatter(model_id=self._model_id)
        self._client = _BufferingClient()
        self._conn: Any = None  # ClientSideConnection, set after first use
        self._session_id: str | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()  # ACP allows only one prompt per session at a time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        """Spawn the ACP agent (once) and open a session (once)."""
        if self._session_id is not None:
            return

        self._exit_stack = AsyncExitStack()
        conn, _proc = await self._exit_stack.enter_async_context(
            spawn_agent_process(
                self._client,
                self._command[0],
                *self._command[1:],
                cwd=os.getcwd(),
            )
        )
        self._conn = conn

        resp: NewSessionResponse = await conn.new_session(cwd=os.getcwd())
        self._session_id = resp.session_id

    def _render_prompt(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
    ) -> str:
        """Render context + action into a plain-text prompt string.

        Uses to_chat_messages() so that role information (user/assistant/system)
        from ChatContext history is preserved as role-prefixed lines.
        For SimpleContext, view_for_generation() returns [] so only the action
        is rendered.
        """
        prior = ctx.view_for_generation() or []
        all_spans = [*prior, action]
        messages: list[Message] = self._formatter.to_chat_messages(all_spans)

        parts: list[str] = []
        for msg in messages:
            role = msg.role
            content = self._formatter.print(msg)
            # Only prefix with role when there's actual history (multi-turn)
            if len(prior) > 0:
                parts.append(f"{role}: {content}")
            else:
                parts.append(content)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Backend protocol
    # ------------------------------------------------------------------

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        await self._ensure_session()

        prompt_text = self._render_prompt(action, ctx)

        async with self._lock:
            self._client.clear()
            date = datetime.datetime.now()
            await self._conn.prompt(
                session_id=self._session_id,
                prompt=[TextContentBlock(type="text", text=prompt_text)],
            )
            response_text = self._client.collected_text()

        thunk: ModelOutputThunk[C] = ModelOutputThunk(value=response_text)
        thunk.parsed_repr = Message(role="assistant", content=response_text)
        log = GenerateLog()
        log.date = date
        log.prompt = prompt_text
        log.backend = f"acp::{self._model_id}"
        log.model_output = response_text
        log.action = action
        thunk._generate_log = log
        return thunk, ctx

    async def _generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        """Batch generation — delegates each action to _generate_from_context."""
        results = []
        for action in actions:
            thunk, _ = await self._generate_from_context(
                action, ctx, format=format, model_options=model_options, tool_calls=tool_calls
            )
            results.append(thunk)
        return results, None

    async def aclose(self) -> None:
        """Tear down the ACP agent subprocess and session."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._conn = None
            self._session_id = None

    def __del__(self) -> None:
        """Best-effort synchronous cleanup when the object is garbage-collected."""
        if self._exit_stack is not None:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.aclose())
                else:
                    loop.run_until_complete(self.aclose())
            except Exception:
                pass  # best-effort: never raise from __del__
