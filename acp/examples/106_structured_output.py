"""Structured output with ACPBackend using format=.

ACP has no constrained decoding, so ACPBackend appends the JSON schema to the
prompt (best effort). Pair it with a validating requirement and a sampling
strategy so malformed replies are retried.

Usage:
    uv run python examples/106_structured_output.py
"""

from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import MelleaSession
from mellea_contribs.acp import ACPBackend
from pydantic import BaseModel, ValidationError


class Capital(BaseModel):
    """A country and its capital city."""

    country: str
    city: str


def is_valid_capital(text: str) -> bool:
    """Return True if ``text`` parses as a ``Capital`` JSON object."""
    try:
        Capital.model_validate_json(text)
    except ValidationError:
        return False
    return True


m = MelleaSession(ACPBackend())

result = m.instruct(
    "What is the capital of France?",
    requirements=[
        req(
            "The reply must be a JSON object matching the schema",
            validation_fn=simple_validate(is_valid_capital),
        )
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    format=Capital,
    return_sampling_results=True,
)

if result.success:
    capital = Capital.model_validate_json(str(result.result))
    print(f"***** parsed *****\n{capital!r}\n******************")
else:
    print(
        f"***** no valid JSON after retries *****\n{result.sample_generations[0].value}\n***************************************"
    )
