"""Email generation constrained by natural language requirements — using ACPBackend.

Mirrors mellea/docs/examples/instruct_validate_repair/101_email_with_requirements.py
but drives generation through Bob running as an ACP stdio subprocess instead of
a local model backend.

Usage:
    uv run python examples/101_email_with_requirements.py

Prerequisites:
    - ``bob`` must be installed and on PATH
    - ``bob acp`` must be available (Bob >= version that ships ACP support)
"""

from mellea.stdlib.session import MelleaSession

from mellea_contribs.acp import ACPBackend

m = MelleaSession(ACPBackend())

email_v1 = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["be formal", "Use 'Dear Interns' as greeting."],
)

print(f"***** email ****\n{email_v1}\n*******")
