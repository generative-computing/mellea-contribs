import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mellea import start_session
from mellea_contribs.reqlib.top_k import top_k

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



def test_top_k_selection():
    m = start_session()

    comparison_prompt = """
    Select which items are the most severe issues.
    Higher severity and latency indicate a worse issue.
    """

    results = top_k(items=ITEMS, comparison_prompt=comparison_prompt, m=m, k=2)

    print("\nTop-K Results:")
    for item, score in results:
        print(f"{item['name']}: {score}")

    assert len(results) == 3
    assert all(isinstance(score, int) for _, score in results)
    assert sum(score for _, score in results) == 2  # top-2 selected


if __name__ == "__main__":
    test_top_k_selection()
