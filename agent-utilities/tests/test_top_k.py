from mellea import start_session
from mellea.backends import ModelOption

from mellea_contribs.agent_utilities.core.top_k import top_k

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


def test_top_k_selection():
    # Thinking off: top_k's contract is a bare JSON array, so a reasoning trace
    # is pure overhead. Granite 4.2 (mellea's default since v0.8.0) reasons
    # unless told not to, which costs ~36x the output tokens here and pushes
    # each generation past the backend's 300s timeout on CPU-only CI runners.
    m = start_session(model_options={ModelOption.THINKING: False})

    comparison_prompt = """
    Select which items are the most severe issues.
    Higher severity and latency indicate a worse issue.
    """

    results = top_k(items=ITEMS, comparison_prompt=comparison_prompt, m=m, k=2)

    print("\nTop-K Results:")
    for item in results:
        print(f"{item['name']}")

    assert all(item in ITEMS for item in results)
    assert len(results) == 2


if __name__ == "__main__":
    test_top_k_selection()
