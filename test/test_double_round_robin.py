import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mellea_contribs.reqlib.double_round_robin import double_round_robin
from mellea import start_session
from mellea.helpers.fancy_logger import FancyLogger

FancyLogger.get_logger().setLevel("WARNING")


ITEM_CONTEXTS = {
    "flagd-config": {
        "entity_name": "flagd-config",
        "logs": ["error connecting to database", "timeout on API call"],
        "metrics": {"latency": "500", "error_rate": "0.3"},
        "severe_signal": "True"
    },
    "adService": {
        "entity_name": "adService",
        "logs": ["all systems normal"],
        "metrics": {"latency": "50", "error_rate": "0.01"},
        "severe_signal": "False"
    },
    "payment-svc": {
        "entity_name": "payment-svc",
        "logs": ["authorization timeout", "spike in errors"],
        "metrics": {"latency": "900", "error_rate": "0.47"},
        "severe_signal": "True"
    }
}

ITEMS = [
    {"name": "flagd-config", "latency": 500, "severity": 3},
    {"name": "adService",    "latency":  50, "severity": 1},
    {"name": "payment-svc",  "latency": 900, "severity": 4},
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
        m=m,
        context=ITEM_CONTEXTS  # <-- now a mapping, not a single dict
    )

    print("\nDRR Results:")
    for item, score in results:
        print(f"{item['name']}: {score}")

    assert len(results) == 3
    assert all(isinstance(score, int) for _, score in results)

if __name__ == "__main__":
    test_generic_double_round_robin()

