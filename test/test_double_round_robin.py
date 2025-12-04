import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the double round robin library
from mellea_contribs.reqlib.double_round_robin import run_double_round_robin, LABELS
from mellea import start_session

from mellea.helpers.fancy_logger import FancyLogger

FancyLogger.get_logger().setLevel("WARNING")


# Dummy observability context for testing
TEST_CONTEXTS = [
    {
        "entity_name": "flagd-config",
        "logs": ["error connecting to database", "timeout on API call"],
        "metrics": {"latency": 500, "error_rate": 0.3},
        "severe_signal": True
    },
    {
        "entity_name": "adService",
        "logs": ["all systems normal"],
        "metrics": {"latency": 50, "error_rate": 0.01},
        "severe_signal": False
    },
]

def test_double_round_robin():
    m = start_session()

    for ctx in TEST_CONTEXTS:
        entity_id = ctx["entity_name"]
        result = run_double_round_robin(entity_id, ctx, m)
        print(f"Entity: {entity_id}, Double Round Robin Result: {result}")
        assert result in LABELS, f"Invalid label returned: {result}"

if __name__ == "__main__":
    test_double_round_robin()
