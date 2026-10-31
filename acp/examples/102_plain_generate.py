"""Plain generation with ACPBackend — no requirements, no strategy.

The simplest possible use of ACPBackend: a single m.instruct() call
with no validation or sampling.

Usage:
    uv run python examples/102_plain_generate.py
"""

from mellea.stdlib.session import MelleaSession
from mellea_contribs.acp import ACPBackend

m = MelleaSession(ACPBackend())

haiku = m.instruct("Write a haiku about Python programming.")

print(f"***** haiku *****\n{haiku}\n*****************")
