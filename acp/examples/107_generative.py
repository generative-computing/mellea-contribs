"""@generative functions with ACPBackend.

``@generative`` turns a typed, documented function stub into an LLM call. It
passes a Pydantic response model as ``format=``, which ACPBackend appends to the
prompt as a JSON schema (best effort; see example 106). Add a sampling strategy
so malformed replies are retried.

Usage:
    uv run python examples/107_generative.py
"""

from mellea.stdlib.components.genstub import generative
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import MelleaSession
from mellea_contribs.acp import ACPBackend


@generative
def is_positive(text: str) -> bool:
    """Return True if the sentiment of the text is positive."""


@generative
def extract_cities(text: str) -> list[str]:
    """Return the names of all cities mentioned in the text."""


m = MelleaSession(ACPBackend())

print(f"positive? {is_positive(m, text='I love this!')}")

cities = extract_cities(
    m,
    text="I flew from Paris to Tokyo, then took a train to Osaka.",
    requirements=[req("Only include city names, no countries.")],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)
print(f"cities: {cities}")
