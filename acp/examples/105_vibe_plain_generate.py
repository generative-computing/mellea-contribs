"""Plain generation using Mistral Vibe (vibe-acp) via ACPBackend.

Mirrors 102_plain_generate.py but targets Mistral's vibe-acp agent
instead of Bob. Demonstrates that ACPBackend works with any ACP-compatible
stdio agent — just pass a different command.

Prerequisites:
    pip install vibe-acp
    vibe-acp --setup   # configure your Mistral API key

Usage:
    uv run python examples/105_vibe_plain_generate.py
"""

from mellea.stdlib.session import MelleaSession
from mellea_contribs.acp import ACPBackend

m = MelleaSession(ACPBackend(command=["vibe-acp"]))

haiku = m.instruct("Write a haiku about Python programming.")

print(f"***** haiku *****\n{haiku}\n*****************")
