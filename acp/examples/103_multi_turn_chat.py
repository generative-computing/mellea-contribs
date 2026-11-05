"""Multi-turn conversation with ACPBackend using m.chat().

Demonstrates that ACPBackend supports ChatContext and m.chat(). Mellea
accumulates the turn history in the context and ACPBackend renders it into
each prompt; the ACP session on the agent side also retains prior turns.

Usage:
    uv run python examples/103_multi_turn_chat.py
"""

from mellea.stdlib.context import ChatContext
from mellea.stdlib.session import MelleaSession
from mellea_contribs.acp import ACPBackend

m = MelleaSession(ACPBackend(), ctx=ChatContext())

m.chat("My name is Alice and I work in the AI team.")
reply = m.chat("What is my name and what team do I work in?")

print(
    f"***** reply (tests memory) *****\n{reply.content}\n********************************"
)
