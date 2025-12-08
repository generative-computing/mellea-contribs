from mellea_contribs.reqlib.dynamic_context_builder import DynamicContextBuilder
from mellea.stdlib.base import TemplateRepresentation


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_skips_empty_fields():
    print_header("TEST: Skips empty fields")

    comp = DynamicContextBuilder(
        logs=[],
        events=None,
        meta={"id": 5},
        causal=""
    )

    result = comp.format_for_llm()
    print("OUTPUT:\n", result)

    expected = "### meta\n{\n  \"id\": 5\n}"
    print("EXPECTED:\n", expected)

    print("PASS?", result.strip() == expected)


def test_serializes_multiple_fields():
    print_header("TEST: Serializes multiple fields")

    comp = DynamicContextBuilder(
        logs=["fail", "retry"],
        metrics={"latency": 300, "errors": 0.12}
    )

    output = comp.format_for_llm()
    print("OUTPUT:\n", output)

    pass_condition = (
        ("### logs" in output) and
        ("fail" in output) and
        ("### metrics" in output) and
        ("\"latency\"" in output)
    )
    print("PASS?", pass_condition)


def test_returns_template_representation():
    print_header("TEST: Returns TemplateRepresentation")

    comp = DynamicContextBuilder(
        summary="ok!",
        return_template=True
    )

    result = comp.format_for_llm()

    print("OUTPUT TYPE:", type(result))

    # Adapted for current Mellea TemplateRepresentation
    content = ""
    if isinstance(result, TemplateRepresentation):
        # Template string is usually stored inside args under some key, often 'text' or similar
        # Check the keys in args
        if hasattr(result, "args") and isinstance(result.args, dict):
            if len(result.args) > 0:
                # take the first value in args as the text
                content = list(result.args.values())[0]
    print("TEMPLATE:\n", content)

    pass_condition = isinstance(result, TemplateRepresentation) and "ok!" in content
    print("PASS?", pass_condition)


def test_long_user_prompt_and_context():
    print_header("TEST: Long user_prompt and grounding_context")

    user_prompt = """Analyze the entity and update the investigation summary.
    ================================================================
    ## ENTITY FOR ANALYSIS
    Entity: flagd-5c56f7c9db-wwc9k
    Type: Pod
    Depth in exploration: 2
    TRAVERSAL PATH (Primary Failure → Current): [flagd-config]
    Alert Context for this entity: [service is down]
    Gathered Data for this entity: {"__documentation__": {
    "description": "Comprehensive observability report for a Kubernetes entity with contextual explosion",
    "sections": {
      "focal_entity": "The primary entity being analyzed",
      "stack_report": {
        "description": "Hierarchical relationships and service interactions",
        "physical": "Physical infrastructure stack (Cluster -> Node -> Pod -> Container)",
        "controller": {
          "description": "Kubernetes controller hierarchy",
          "self": "The focal entity itself - this is the entity you're analyzing",
          "controllers": "Other controller entities in the hierarchy (Deployment/StatefulSet/DaemonSet)"}  
    """

    grounding_context = {
        "observability_raw": {
            "focal_entity": {
                "id": "2bbba94bf8d4ab09",
                "name": "flagd-5c56f7c9db-wwc9k",
                "type": "Pod",
                "namespace": "otel-demo"
            },
            "stack_report": {
                "physical": {"cluster": {"id": "3ec5b5c761c1afad", "name": "k8s-cluster-root"}},
                "controller": {"self": {"id": "2bbba94bf8d4ab09", "name": "flagd-5c56f7c9db-wwc9k"}},
                "service_interactions": {"incoming": [], "outgoing": []}
            }
        }
    }

    comp = DynamicContextBuilder(
        user_prompt=user_prompt,
        grounding_context=grounding_context,
        return_template=True  # optional if you want TemplateRepresentation
    )

    result = comp.format_for_llm()
    print("OUTPUT TYPE:", type(result))

    content = ""
    if hasattr(result, "args") and isinstance(result.args, dict) and len(result.args) > 0:
        content = list(result.args.values())[0]

    print("TEMPLATE / CONTENT:\n", content[:500], "...\n")  # Print first 500 chars for brevity

    pass_condition = (
            "flagd-5c56f7c9db-wwc9k" in content and
            "observability_raw" in content
    )
    print("PASS?", pass_condition)


def test_empty_context_renders_empty():
    print_header("TEST: Empty context renders empty string")

    comp = DynamicContextBuilder(
        logs=[],
        events=None,
        extra={}
    )

    result = comp.format_for_llm()
    print("OUTPUT: ", repr(result))

    print("PASS?", result == "")


if __name__ == "__main__":
    print("\nRunning DynamicContextBuilder tests...\n")

    test_skips_empty_fields()
    test_serializes_multiple_fields()
    test_returns_template_representation()
    test_empty_context_renders_empty()

    test_long_user_prompt_and_context()

    print("\nAll tests completed.\n")
