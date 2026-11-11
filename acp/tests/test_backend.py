"""Behavioural tests for ACPBackend using a fake ACP connection (no subprocess)."""

import asyncio
import os
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from acp.schema import (
    AgentMessageChunk,
    PermissionOption,
    PromptResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallUpdate,
)
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.components.genstub import generative
from mellea.stdlib.context import ChatContext
from mellea.stdlib.session import MelleaSession
from pydantic import BaseModel

from mellea_contribs.acp import ACPBackend
from mellea_contribs.acp.backends import acp_backend as module


class FakeConn:
    """Stands in for ClientSideConnection; streams a canned or echoed reply."""

    def __init__(self, client: Any, state: dict[str, Any]) -> None:
        self.client = client
        self.state = state

    async def initialize(self, protocol_version: int = 1, **kwargs: Any) -> Any:
        if self.state.get("init_hang"):
            await asyncio.sleep(3600)
        self.state["initialized"] = True
        self.state["client_info"] = kwargs.get("client_info")
        return SimpleNamespace(
            protocol_version=self.state.get("protocol_version", protocol_version)
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> Any:
        assert self.state.get("initialized"), "new_session before initialize"
        await asyncio.sleep(0)  # yield, so concurrent callers interleave
        self.state["session_cwds"].append(cwd)
        return SimpleNamespace(session_id=f"s{len(self.state['session_cwds'])}")

    async def close_session(self, session_id: str, **kwargs: Any) -> None:
        if not self.state["close_supported"]:
            raise RuntimeError("method not found")
        self.state["closed"].append(session_id)

    async def prompt(self, prompt: list[Any], session_id: str, **kwargs: Any) -> Any:
        text = prompt[0].text
        st = self.state
        st["prompts"].append((session_id, text))
        if st.get("hang"):
            await asyncio.sleep(3600)
        if st.get("die"):
            st["die"] = False
            raise ConnectionError("Connection closed")
        st["inflight"] += 1
        st["max_inflight"] = max(st["max_inflight"], st["inflight"])
        await asyncio.sleep(0.01)
        reply = st["reply"] if st["reply"] is not None else f"echo:{text}"
        for extra in st.get("extra_updates", []):
            await self.client.session_update(session_id, extra)
        await self.client.session_update(
            session_id,
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text=reply),
            ),
        )
        st["inflight"] -= 1
        return PromptResponse(stop_reason=st["stop"])


@pytest.fixture
def spawn(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "spawns": 0,
        "spawn_cwds": [],
        "spawn_envs": [],
        "session_cwds": [],
        "prompts": [],
        "inflight": 0,
        "max_inflight": 0,
        "reply": "hello",  # None => echo the prompt
        "stop": "end_turn",
        "closed": [],
        "hang": False,
        "die": False,
        "close_supported": True,
        "initialized": False,
    }

    @asynccontextmanager
    async def fake_spawn(client: Any, *args: Any, **kwargs: Any):
        state["spawns"] += 1
        state["spawn_cwds"].append(kwargs.get("cwd"))
        state["spawn_envs"].append(kwargs.get("env"))
        await asyncio.sleep(0)
        yield FakeConn(client, state), None

    monkeypatch.setattr(module, "spawn_agent_process", fake_spawn)
    return state


# --- context / thunk -------------------------------------------------------


async def test_context_is_extended(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    action = Instruction("say hi")
    thunk, new_ctx = await backend._generate_from_context(action, ChatContext())
    items = new_ctx.as_list()
    assert action in items
    assert thunk in items


async def test_thunk_is_lazy_then_resolves_via_astream(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(
        Instruction("say hi"), ChatContext()
    )
    assert not thunk.is_computed()
    assert await thunk.avalue() == "hello"
    assert thunk.is_computed()
    assert thunk.parsed_repr is not None
    assert thunk.generate_log is not None
    assert thunk.generate_log.backend == "acp::bob"
    assert thunk.generate_log.model_output == "hello"


async def test_lazy_grounding_context_is_resolved_before_rendering(
    spawn: dict[str, Any],
) -> None:
    backend = ACPBackend()
    lazy, _ = await backend._generate_from_context(Instruction("inner"), ChatContext())
    assert not lazy.is_computed()
    action = Instruction("Summarise", grounding_context={"doc": lazy})
    thunk, _ = await backend._generate_from_context(action, ChatContext())
    await thunk.avalue()
    assert "hello" in spawn["prompts"][-1][1]


async def test_lazy_grounding_context_is_resolved_before_rendering_batch(
    spawn: dict[str, Any],
) -> None:
    backend = ACPBackend()
    lazy, _ = await backend._generate_from_context(Instruction("inner"), ChatContext())
    assert not lazy.is_computed()
    action = Instruction("Summarise", grounding_context={"doc": lazy})
    results, _ = await backend._generate_from_raw([action], ChatContext())
    assert len(results) == 1
    assert "hello" in spawn["prompts"][-1][1]


# --- sessions / concurrency ------------------------------------------------


async def test_concurrent_calls_spawn_once(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    ctx = ChatContext()
    thunks = await asyncio.gather(
        backend._generate_from_context(Instruction("a"), ctx),
        backend._generate_from_context(Instruction("b"), ctx),
    )
    await asyncio.gather(*(t.avalue() for t, _ in thunks))
    assert spawn["spawns"] == 1


async def test_each_prompt_gets_fresh_session(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    for text in ("a", "b"):
        thunk, _ = await backend._generate_from_context(
            Instruction(text), ChatContext()
        )
        await thunk.avalue()
    assert spawn["spawns"] == 1
    session_ids = [sid for sid, _ in spawn["prompts"]]
    assert len(set(session_ids)) == 2


async def test_parent_env_is_forwarded_to_agent(
    spawn: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOB_API_KEY", "secret")
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()
    assert spawn["spawn_envs"][0]["BOB_API_KEY"] == "secret"


async def test_explicit_env_replaces_parent_env(
    spawn: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOB_API_KEY", "secret")
    backend = ACPBackend(env={"FOO": "bar"})
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()
    assert spawn["spawn_envs"][0] == {"FOO": "bar"}


def test_close_is_sync_idempotent_and_blocks_reuse(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(backend._ensure_agent())


def test_close_tears_down_running_agent(spawn: dict[str, Any]) -> None:
    from mellea.helpers.event_loop_helper import _run_async_in_thread

    backend = ACPBackend()

    async def use() -> None:
        thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
        await thunk.avalue()

    _run_async_in_thread(use())
    assert backend._conn is not None
    backend.close()
    assert backend._conn is None
    assert backend._exit_stack is None


async def test_prompts_run_concurrently_without_mixing_replies(
    spawn: dict[str, Any],
) -> None:
    spawn["reply"] = None  # echo
    backend = ACPBackend()
    ctx = ChatContext()
    (t1, _), (t2, _) = await asyncio.gather(
        backend._generate_from_context(Instruction("first"), ctx),
        backend._generate_from_context(Instruction("second"), ctx),
    )
    v1, v2 = await asyncio.gather(t1.avalue(), t2.avalue())
    assert "first" in v1 and "second" not in v1
    assert "second" in v2 and "first" not in v2
    assert spawn["max_inflight"] == 2


# --- cwd -------------------------------------------------------------------


async def test_cwd_defaults_to_scratch_dir(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()
    (scratch,) = spawn["spawn_cwds"]
    assert scratch != os.getcwd()
    assert os.path.isdir(scratch)
    assert os.listdir(scratch) == []
    assert spawn["session_cwds"] == [scratch]
    await backend.aclose()
    assert not os.path.exists(scratch)


async def test_cwd_override(spawn: dict[str, Any]) -> None:
    backend = ACPBackend(cwd="/tmp/acp-sandbox")
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()
    assert spawn["spawn_cwds"] == ["/tmp/acp-sandbox"]
    assert spawn["session_cwds"] == ["/tmp/acp-sandbox"]


# --- permissions -----------------------------------------------------------


async def test_permission_denied_with_reject_once() -> None:
    client = module._BufferingClient()
    options = [
        PermissionOption(option_id="a", name="Allow", kind="allow_once"),
        PermissionOption(option_id="r", name="Reject", kind="reject_once"),
        PermissionOption(option_id="ra", name="Reject always", kind="reject_always"),
    ]
    resp = await client.request_permission(
        session_id="s1", tool_call=ToolCallUpdate(tool_call_id="t"), options=options
    )
    assert resp.outcome.outcome == "selected"
    assert resp.outcome.option_id == "r"


async def test_permission_cancelled_without_reject_option() -> None:
    client = module._BufferingClient()
    options = [PermissionOption(option_id="a", name="Allow", kind="allow_once")]
    resp = await client.request_permission(
        session_id="s1", tool_call=ToolCallUpdate(tool_call_id="t"), options=options
    )
    assert resp.outcome.outcome == "cancelled"


# --- error handling --------------------------------------------------------


@pytest.mark.parametrize("stop", ["cancelled", "refusal"])
async def test_bad_stop_reason_raises(spawn: dict[str, Any], stop: str) -> None:
    spawn["stop"] = stop
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    with pytest.raises(RuntimeError, match=stop):
        await thunk.avalue()


@pytest.mark.parametrize("reply", ["", "   \n"])
async def test_empty_reply_raises(spawn: dict[str, Any], reply: str) -> None:
    spawn["reply"] = reply
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    with pytest.raises(RuntimeError, match="empty"):
        await thunk.avalue()


async def test_timeout_raises_and_respawns(spawn: dict[str, Any]) -> None:
    spawn["hang"] = True
    backend = ACPBackend(timeout=0.05)
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    with pytest.raises(TimeoutError, match="did not reply"):
        await thunk.avalue()
    assert spawn["spawns"] == 1
    spawn["hang"] = False
    thunk, _ = await backend._generate_from_context(Instruction("y"), ChatContext())
    assert await thunk.avalue() == "hello"
    assert spawn["spawns"] == 2


async def test_dead_agent_fails_call_then_respawns(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    spawn["die"] = True
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    with pytest.raises(ConnectionError):
        await thunk.avalue()
    assert spawn["spawns"] == 1
    thunk, _ = await backend._generate_from_context(Instruction("y"), ChatContext())
    assert await thunk.avalue() == "hello"
    assert spawn["spawns"] == 2


async def test_agent_exited_while_idle_respawns(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()
    backend._proc = SimpleNamespace(returncode=1)
    thunk, _ = await backend._generate_from_context(Instruction("y"), ChatContext())
    assert await thunk.avalue() == "hello"
    assert spawn["spawns"] == 2


async def test_timeout_none_passes_none_to_asyncio(
    spawn: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verify that timeout=None threads through to asyncio.timeout(None) rather
    # than being coerced to 0 or some other value.
    import asyncio as _asyncio

    import mellea_contribs.acp.backends.acp_backend as _mod

    delays: list[Any] = []
    real_timeout = _asyncio.timeout

    def capturing_timeout(delay: Any) -> Any:
        delays.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(_mod.asyncio, "timeout", capturing_timeout)
    backend = ACPBackend(timeout=None)
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()
    assert delays[0] is None  # generation timeout — first asyncio.timeout() call


# --- format ----------------------------------------------------------------


class Answer(BaseModel):
    city: str


async def test_format_schema_added_to_prompt(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(
        Instruction("capital of France?"), ChatContext(), format=Answer
    )
    await thunk.avalue()
    sent = spawn["prompts"][0][1]
    assert "city" in sent
    assert "JSON" in sent


@pytest.mark.parametrize(
    "reply",
    [
        '🧞‍♀️ {"city": "Paris"}',
        'Sure!\n```json\n{"city": "Paris"}\n```\nDone.',
        '{"city": "Paris"}',
    ],
)
async def test_format_extracts_json(spawn: dict[str, Any], reply: str) -> None:
    spawn["reply"] = reply
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(
        Instruction("capital?"), ChatContext(), format=Answer
    )
    assert await thunk.avalue() == '{"city": "Paris"}'


async def test_format_leaves_non_json_untouched(spawn: dict[str, Any]) -> None:
    spawn["reply"] = "no json here"
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(
        Instruction("capital?"), ChatContext(), format=Answer
    )
    assert await thunk.avalue() == "no json here"


# --- prompt rendering ------------------------------------------------------


def test_render_prompt_single_turn_has_no_role_prefix() -> None:
    backend = ACPBackend()
    prompt = backend._render_prompt(Instruction("say hi"), ChatContext())
    assert "say hi" in prompt
    assert "user:" not in prompt


def test_render_prompt_multi_turn_has_role_prefixes() -> None:
    backend = ACPBackend()
    ctx = (
        ChatContext()
        .add(Message("user", "My name is Alice"))
        .add(Message("assistant", "Hi Alice"))
    )
    prompt = backend._render_prompt(Message("user", "What is my name?"), ctx)
    assert prompt.splitlines() == [
        "user: My name is Alice",
        "assistant: Hi Alice",
        "user: What is my name?",
    ]


# --- raw generation --------------------------------------------------------


async def test_generate_from_raw_is_concurrent_and_ignores_ctx(
    spawn: dict[str, Any],
) -> None:
    spawn["reply"] = None  # echo
    backend = ACPBackend()
    ctx = ChatContext().add(Message("user", "SECRET HISTORY"))
    results, usage = await backend._generate_from_raw(
        [Instruction("one"), Instruction("two")], ctx
    )
    assert usage is None
    assert [r.is_computed() for r in results] == [True, True]
    assert "one" in str(results[0].value) and "two" in str(results[1].value)
    assert all("SECRET HISTORY" not in text for _, text in spawn["prompts"])
    assert spawn["max_inflight"] == 2


async def test_sessions_are_closed_after_use(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()
    assert spawn["closed"] == [sid for sid, _ in spawn["prompts"]]


async def test_agent_without_close_session_still_works(spawn: dict[str, Any]) -> None:
    spawn["close_supported"] = False
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    assert await thunk.avalue() == "hello"


async def test_pop_text_called_exactly_once_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pop_text must be called exactly once per session — not twice.

    Regression test for the double-pop bug: the old code called pop_text()
    inside the try block and again in finally. The second call always returned
    an empty string (the entry was already removed), but it masked any chunks
    that could accumulate between the two calls. The fix moves pop_text()
    exclusively into finally so it is guaranteed to run exactly once.
    """
    pop_counts: dict[str, int] = {}
    original_pop = module._BufferingClient.pop_text

    def counting_pop(self: Any, session_id: str) -> str:
        pop_counts[session_id] = pop_counts.get(session_id, 0) + 1
        return original_pop(self, session_id)

    monkeypatch.setattr(module._BufferingClient, "pop_text", counting_pop)

    state: dict[str, Any] = {
        "spawns": 0,
        "spawn_cwds": [],
        "spawn_envs": [],
        "session_cwds": [],
        "prompts": [],
        "inflight": 0,
        "max_inflight": 0,
        "reply": "hello",
        "stop": "end_turn",
        "closed": [],
        "close_supported": True,
        "initialized": False,
    }

    @asynccontextmanager
    async def fake_spawn(client: Any, *args: Any, **kwargs: Any):
        state["spawns"] += 1
        state["spawn_cwds"].append(kwargs.get("cwd"))
        state["spawn_envs"].append(kwargs.get("env"))
        await asyncio.sleep(0)
        yield FakeConn(client, state), None

    monkeypatch.setattr(module, "spawn_agent_process", fake_spawn)

    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()

    assert len(pop_counts) == 1
    session_id = next(iter(pop_counts))
    assert pop_counts[session_id] == 1, (
        f"pop_text called {pop_counts[session_id]} times for session {session_id!r}; expected 1"
    )


def test_client_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
    assert module._CLIENT_VERSION == pyproject["project"]["version"]


async def test_initialize_passes_client_info(spawn: dict[str, Any]) -> None:
    backend = ACPBackend()
    thunk, _ = await backend._generate_from_context(Instruction("hi"), ChatContext())
    await thunk.avalue()
    assert spawn["initialized"] is True
    client_info = spawn.get("client_info")
    assert client_info is not None
    assert client_info.name == module._CLIENT_NAME
    assert client_info.version == module._CLIENT_VERSION


async def test_initialize_unsupported_protocol_version_raises(
    spawn: dict[str, Any],
) -> None:
    spawn["protocol_version"] = 99
    backend = ACPBackend()
    with pytest.raises(RuntimeError, match="unsupported protocol version 99"):
        thunk, _ = await backend._generate_from_context(
            Instruction("hi"), ChatContext()
        )
        await thunk.avalue()


async def test_initialize_timeout_raises_and_respawns(
    spawn: dict[str, Any],
) -> None:
    spawn["init_hang"] = True
    backend = ACPBackend(timeout=0.05)
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    with pytest.raises(TimeoutError, match="failed to initialize within"):
        await thunk.avalue()
    assert spawn["spawns"] == 1
    spawn["init_hang"] = False
    thunk, _ = await backend._generate_from_context(Instruction("y"), ChatContext())
    assert await thunk.avalue() == "hello"
    assert spawn["spawns"] == 2


# --- @generative -----------------------------------------------------------


@generative
async def is_positive(text: str) -> bool:
    """Return True if the sentiment of the text is positive."""


@pytest.mark.parametrize(
    "reply",
    [
        '{"result": true}',
        'Sure!\n```json\n{"result": true}\n```',
        '🧞‍♀️ {"result": true}',
    ],
)
async def test_generative_stub_parses_result(spawn: dict[str, Any], reply: str) -> None:
    spawn["reply"] = reply
    m = MelleaSession(ACPBackend())
    assert await is_positive(m, text="I love this!") is True
    sent = spawn["prompts"][0][1]
    assert "I love this!" in sent
    assert "result" in sent  # response schema included in the prompt


# --- on_update -------------------------------------------------------------


def _tool_update(call_id: str) -> ToolCallProgress:
    return ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id=call_id,
        status="completed",
        raw_output={"result": "42"},
    )


async def test_on_update_sees_every_update_with_session_id(
    spawn: dict[str, Any],
) -> None:
    spawn["extra_updates"] = [_tool_update("t1")]
    seen: list[tuple[str, Any]] = []
    backend = ACPBackend(on_update=lambda sid, upd: seen.append((sid, upd)))
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    assert await thunk.avalue() == "hello"
    assert [type(u) for _, u in seen] == [ToolCallProgress, AgentMessageChunk]
    assert {sid for sid, _ in seen} == {spawn["prompts"][0][0]}
    assert seen[0][1].raw_output == {"result": "42"}


async def test_on_update_may_be_async(spawn: dict[str, Any]) -> None:
    spawn["extra_updates"] = [_tool_update("t1")]
    seen: list[str] = []

    async def observer(session_id: str, update: Any) -> None:
        await asyncio.sleep(0)
        seen.append(update.session_update)

    backend = ACPBackend(on_update=observer)
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    await thunk.avalue()
    assert seen == ["tool_call_update", "agent_message_chunk"]


async def test_on_update_concurrent_calls_are_told_apart_by_session_id(
    spawn: dict[str, Any],
) -> None:
    seen: list[str] = []
    backend = ACPBackend(on_update=lambda sid, upd: seen.append(sid))
    results = await backend.generate_from_raw(
        [Instruction("a"), Instruction("b")], ChatContext()
    )
    assert len(results) == 2
    assert sorted(seen) == sorted(sid for sid, _ in spawn["prompts"])
    assert len(set(seen)) == 2


async def test_on_update_exception_does_not_break_generation(
    spawn: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    def boom(session_id: str, update: Any) -> None:
        raise ValueError("observer bug")

    backend = ACPBackend(on_update=boom)
    thunk, _ = await backend._generate_from_context(Instruction("x"), ChatContext())
    assert await thunk.avalue() == "hello"
    assert "on_update callback raised" in caplog.text
