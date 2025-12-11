import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mellea_contribs.tools.double_round_robin import double_round_robin
from mellea import start_session
from mellea.helpers.fancy_logger import FancyLogger

FancyLogger.get_logger().setLevel("WARNING")

ITEMS = [
    {
        "name": "flagd-config",
        "latency": "500",
        "severity": "3",
        "logs": ["error connecting to database", "timeout on API call"],
        "severe_signal": "True"
    },
    {
        "name": "adService",
        "latency": "50",
        "severity": "1",
        "logs": ["all systems normal"],
        "severe_signal": "False"
    },
    {
        "name": "payment-svc",
        "latency": "900",
        "severity": "4",
        "logs": ["authorization timeout", "spike in errors"],
        "severe_signal": "True"
    }
]


def test_generic_double_round_robin():
    m = start_session()

    comparison_prompt = """
        Select which option is more likely to be the primary root-cause
        based on severity and the signals in each option's grounding context.
    """

    results = double_round_robin(
        items=ITEMS,
        comparison_prompt=comparison_prompt,
        m=m
    )

    print("\nDRR Results:")
    for item, score in results:
        print(f"{item['name']}: {score}")

    assert len(results) == 3
    assert all(isinstance(score, int) for _, score in results)

if __name__ == "__main__":
    test_generic_double_round_robin()

