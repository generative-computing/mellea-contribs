"""Custom validation with ACPBackend.

Demonstrates using req() with a simple_validate() lambda to enforce
a programmatic constraint alongside natural-language requirements.

Here we require the output to be all lower-case, verified both by an
LLM-as-judge (natural language) and a deterministic Python check.

Usage:
    uv run python examples/104_custom_validation.py
"""

from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import MelleaSession
from mellea_contribs.acp import ACPBackend

m = MelleaSession(ACPBackend())

requirements = [
    req("The email should have a salutation"),
    req(
        "Use only lower-case letters",
        validation_fn=simple_validate(lambda x: x.lower() == x),
    ),
]

result = m.instruct(
    "Write a short email to Olivia thanking her for organising the intern events.",
    requirements=requirements,
    strategy=RejectionSamplingStrategy(loop_budget=5),
    return_sampling_results=True,
)

if result.success:
    print(
        f"***** email (all requirements met) *****\n{result.result}\n****************************************"
    )
else:
    print(
        f"***** email (best effort) *****\n{result.sample_generations[0].value}\n*******************************"
    )
