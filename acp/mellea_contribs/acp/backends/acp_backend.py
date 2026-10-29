"""ACPBackend — Mellea backend that drives an ACP stdio agent (Bob, Mistral Vibe, ...)."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Any, Self

from mellea.core import (
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    ModelOutputThunk,
)
from mellea.core.backend import Backend
from mellea.core.base import GenerateLog, GenerateType
from mellea.formatters import TemplateFormatter
from mellea.helpers.async_helpers import send_to_queue
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext

from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
    TextContentBlock,
)
from acp.stdio import spawn_agent_process

_logger = logging.getLogger(__name__)


class _BufferingClient:
    """Minimal ACP Client that accumulates AgentMessageChunk text per session."""

    def __init__(self) -> None:
        self._chunks: dict[str, list[str]] = {}

    def pop_text(self, session_id: str) -> str:
        """Return and discard the text collected so far for ``session_id``."""
        return "".join(self._chunks.pop(session_id, []))

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
                self._chunks.setdefault(session_id, []).append(content.text)

    async def request_permission(
        self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any
    ) -> RequestPermissionResponse:
        """Deny tool-permission requests; this backend runs headless.

        Selects a ``reject_once`` option if offered (agent continues and can adapt),
        otherwise cancels. ``*_always`` options are never selected.
        """
        for option in options:
            if option.kind == "reject_once":
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(
                        outcome="selected", option_id=option.option_id
                    )
                )
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


def _extract_json(text: str) -> str:
    """Return the first JSON object or array embedded in ``text``.

    ACP agents often wrap replies in prose, code fences or persona prefixes.
    If no valid JSON value is found, ``text`` is returned unchanged.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                _, end = decoder.raw_decode(text, i)
            except ValueError:
                continue
            return text[i:end]
    return text


class ACPBackend(Backend):
    """Mellea Backend that drives any ACP-compatible stdio agent subprocess.

    By default spawns Bob (``bob acp``). Pass a different ``command`` to use
    any other ACP agent — e.g. ``command=["vibe-acp"]`` for Mistral Vibe.

    One agent subprocess is spawned lazily and shared. Every generation runs in
    its own fresh ACP session, so the agent never carries hidden history between
    calls: Mellea's context is the single source of truth, and rejection-sampling
    retries do not pollute each other.

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

    def __init__(
        self,
        command: list[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> None:
        """Create an ACPBackend.

        Args:
            command: The command used to launch the ACP agent subprocess.
                     Defaults to ``["bob", "acp"]``.
                     Example: ``["vibe-acp"]``, ``["bob", "acp", "--trust"]``.
            cwd: Working directory for the agent and its sessions. This is the
                 directory the agent can see. Defaults to ``os.getcwd()`` at the
                 time the agent is first used.
        """
        self._command: list[str] = command if command is not None else ["bob", "acp"]
        self._cwd: str | None = os.fspath(cwd) if cwd is not None else None
        self._model_id: str = self._command[0]
        self._formatter = TemplateFormatter(model_id=self._model_id)
        self._client = _BufferingClient()
        self._conn: Any = None  # ClientSideConnection, set after first use
        self._exit_stack: AsyncExitStack | None = None
        self._spawn_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        """Enter async context; the agent is spawned lazily on first use."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Tear down the agent subprocess."""
        await self.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_agent(self) -> Any:
        """Spawn the shared ACP agent subprocess once and return its connection."""
        async with self._spawn_lock:
            if self._conn is None:
                cwd = self._cwd if self._cwd is not None else os.getcwd()
                stack = AsyncExitStack()
                try:
                    conn, _proc = await stack.enter_async_context(
                        spawn_agent_process(
                            self._client,
                            self._command[0],
                            *self._command[1:],
                            cwd=cwd,
                        )
                    )
                except BaseException:
                    await stack.aclose()
                    raise
                self._exit_stack = stack
                self._conn = conn
            return self._conn

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

    def _build_prompt(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        format: type[BaseModelSubclass] | None,
    ) -> str:
        prompt_text = self._render_prompt(action, ctx)
        if format is not None:
            # Best-effort: ACP has no constrained decoding, so ask in the prompt.
            prompt_text += (
                "\n\nRespond with ONLY a JSON object (no prose, no code fences) "
                f"matching this JSON schema:\n{json.dumps(format.model_json_schema())}"
            )
        return prompt_text

    async def _run_prompt(
        self, prompt_text: str, format: type[BaseModelSubclass] | None
    ) -> str:
        """Run one prompt in a fresh ACP session and return the reply text."""
        conn = await self._ensure_agent()
        cwd = self._cwd if self._cwd is not None else os.getcwd()
        session = await conn.new_session(cwd=cwd)
        session_id: str = session.session_id
        try:
            resp = await conn.prompt(
                session_id=session_id,
                prompt=[TextContentBlock(type="text", text=prompt_text)],
            )
            text = self._client.pop_text(session_id)
        finally:
            self._client.pop_text(session_id)
            try:
                await conn.close_session(session_id=session_id)
            except Exception:  # noqa: BLE001
                _logger.debug("ACP agent did not close session %s", session_id)

        if resp.stop_reason in ("cancelled", "refusal"):
            raise RuntimeError(
                f"ACP agent {self._model_id!r} stopped with "
                f"stop_reason={resp.stop_reason!r}"
            )
        if not text.strip():
            raise RuntimeError(
                f"ACP agent {self._model_id!r} returned an empty reply "
                f"(stop_reason={resp.stop_reason!r})"
            )
        return _extract_json(text) if format is not None else text

    @staticmethod
    def _warn_ignored(model_options: dict | None, tool_calls: bool) -> None:
        if model_options or tool_calls:
            _logger.warning(
                "ACPBackend ignores model_options and tool_calls; the ACP agent "
                "controls its own model and tools."
            )

    def _make_generate_log(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        prompt_text: str,
        date: datetime.datetime,
        response_text: str,
    ) -> GenerateLog:
        log = GenerateLog()
        log.date = date
        log.prompt = prompt_text
        log.backend = f"acp::{self._model_id}"
        log.model_output = response_text
        log.action = action
        return log

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
        self._warn_ignored(model_options, tool_calls)
        prompt_text = self._build_prompt(action, ctx, format)

        output: ModelOutputThunk[C] = ModelOutputThunk(None)
        output._gen.start = datetime.datetime.now()
        output._call.action = action
        output._call.model_options = model_options or {}
        output.generation.model = self._model_id
        output.generation.provider = self._provider
        output.raw.provider = self._provider

        async def process(thunk: ModelOutputThunk, chunk: str) -> None:
            thunk._underlying_value = (thunk._underlying_value or "") + chunk

        async def post_process(thunk: ModelOutputThunk) -> None:
            thunk._generate_log = self._make_generate_log(
                action,
                prompt_text,
                thunk._gen.start or datetime.datetime.now(),
                thunk.value or "",
            )
            thunk._generate_log.result = thunk
            thunk._gen.generate = None

        output._gen.process = process
        output._gen.post_process = post_process
        output._gen.generate = asyncio.create_task(
            send_to_queue(self._run_prompt(prompt_text, format), output)
        )
        output._gen.generate_type = GenerateType.ASYNC

        return output, ctx.add(action).add(output)

    async def _generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        """Batch generation: one fresh session per action, run concurrently.

        ``ctx`` is ignored: each action is rendered on its own.
        """
        self._warn_ignored(model_options, tool_calls)
        empty = SimpleContext()
        prompts = [self._build_prompt(a, empty, format) for a in actions]
        dates = [datetime.datetime.now() for _ in actions]
        texts = await asyncio.gather(*(self._run_prompt(p, format) for p in prompts))

        results: list[ModelOutputThunk] = []
        for action, prompt_text, date, text in zip(actions, prompts, dates, texts):
            thunk: ModelOutputThunk = ModelOutputThunk(value=text)
            thunk.parsed_repr = Message(role="assistant", content=text)
            thunk.generation.model = self._model_id
            thunk.generation.provider = self._provider
            thunk._generate_log = self._make_generate_log(
                action, prompt_text, date, text
            )
            results.append(thunk)
        return results, None

    async def aclose(self) -> None:
        """Tear down the ACP agent subprocess."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._conn = None
