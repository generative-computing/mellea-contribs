import logging

from mellea import start_session
from mellea.backends import ModelOption

from mellea_contribs.agent_utilities.core.double_round_robin import double_round_robin

logging.basicConfig(level=logging.WARNING)

ITEMS = [
    {
        "name": "flagd-config",
        "latency": "500",
        "severity": "3",
        "logs": ["error connecting to database", "timeout on API call"],
        "severe_signal": "True",
    },
    {
        "name": "adService",
        "latency": "50",
        "severity": "1",
        "logs": ["all systems normal"],
        "severe_signal": "False",
    },
    {
        "name": "payment-svc",
        "latency": "900",
        "severity": "4",
        "logs": ["authorization timeout", "spike in errors"],
        "severe_signal": "True",
    },
]


def test_generic_double_round_robin():
    # Thinking off: each pairwise comparison only needs a single "A"/"B" token,
    # and a double round robin over N items issues N*(N-1) of them. Granite 4.2
    # (mellea's default since v0.8.0) reasons unless told not to, which costs
    # ~190x the output tokens per comparison and made these calls exceed the
    # backend's 300s timeout on CPU-only CI runners (httpx.ReadTimeout).
    m = start_session(model_options={ModelOption.THINKING: False})

    comparison_prompt = """
        Select which option is more likely to be the primary root-cause
        based on severity and the signals in each option's grounding context.
    """

    results = double_round_robin(items=ITEMS, comparison_prompt=comparison_prompt, m=m)

    print("\nDRR Results:")
    for item, score in results:
        print(f"{item['name']}: {score}")

    assert len(results) == 3
    assert all(isinstance(score, int) for _, score in results)


if __name__ == "__main__":
    test_generic_double_round_robin()
