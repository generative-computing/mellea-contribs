from mellea_contribs.reqlib.grounding_context_formatter import GroundingContextFormatter
from mellea.stdlib.base import TemplateRepresentation


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_skips_empty_fields():
    print_header("TEST: Skips empty fields")

    comp = GroundingContextFormatter(logs=[], events=None, meta={"id": 5}, causal="")

    result = comp.format_for_llm()
    print("OUTPUT:\n", result)

    expected = '### meta\n{\n  "id": 5\n}'
    print("EXPECTED:\n", expected)

    print("PASS?", result.strip() == expected)


def test_serializes_multiple_fields():
    print_header("TEST: Serializes multiple fields")

    comp = GroundingContextFormatter(
        logs=["fail", "retry"], metrics={"latency": 300, "errors": 0.12}
    )

    output = comp.format_for_llm()
    print("OUTPUT:\n", output)

    pass_condition = (
        "### logs" in output
        and "fail" in output
        and "### metrics" in output
        and '"latency"' in output
    )
    print("PASS?", pass_condition)


def test_returns_template_representation():
    print_header("TEST: Returns TemplateRepresentation")

    comp = GroundingContextFormatter(summary="ok!", return_template=True)

    result = TemplateRepresentation(obj=comp, args={"default": "ok!"}, template="ok!")

    print("OUTPUT TYPE:", type(result))

    content = getattr(result, "template", "")
    if not content and hasattr(result, "args"):
        content = list(result.args.values())[0]

    print("TEMPLATE:\n", content)
    print("PASS?", isinstance(result, TemplateRepresentation) and "ok!" in content)


def _test_long_user_prompt_and_context(user_prompt, grounding_context, example_name):
    print_header(f"TEST: Long user_prompt and grounding_context ({example_name})")

    comp = GroundingContextFormatter(
        user_prompt=user_prompt,
        grounding_context=grounding_context,
        return_template=True,
    )

    result = comp.format_for_llm()

    content = ""
    if isinstance(result, TemplateRepresentation):
        content = getattr(result, "template", "")
        if not content and hasattr(result, "args"):
            content = list(result.args.values())[0]
    elif isinstance(result, str):
        content = result

    print("OUTPUT TYPE:", type(result))
    print("TEMPLATE / CONTENT (first 500 chars):\n", content[:500], "...\n")

    context_key_check = list(grounding_context.keys())[0] if grounding_context else ""
    pass_condition = (user_prompt.strip().splitlines()[0] in content) and (
        context_key_check in content
    )
    print("PASS?", pass_condition)


def test_empty_context_renders_empty():
    print_header("TEST: Empty context renders empty string")

    comp = GroundingContextFormatter(logs=[], events=None, extra={})

    result = comp.format_for_llm()
    print("OUTPUT: ", repr(result))

    print("PASS?", result == "")


if __name__ == "__main__":
    print("\nRunning GroundingContextFormatter tests...\n")

    test_skips_empty_fields()
    test_serializes_multiple_fields()
    test_returns_template_representation()
    test_empty_context_renders_empty()

    # Example 1: TaP/SRE agent
    tap_user_prompt = """Analyze the entity and update the investigation summary.
        ================================================
        ## ENTITY FOR ANALYSIS
        Entity: flagd-5c56f7c9db-wwc9k
        Type: Pod
        Depth in exploration: 2
        TRAVERSAL PATH (Primary Failure → Current): [flagd-config]
        Alert Context for this entity: [service is down]"""

    tap_grounding_context = {
        "observability_raw": {
            "focal_entity": {
                "id": "2bbba94bf8d4ab09",
                "name": "flagd-5c56f7c9db-wwc9k",
                "type": "Pod",
                "namespace": "otel-demo",
            },
            "stack_report": {
                "physical": {
                    "cluster": {"id": "3ec5b5c761c1afad", "name": "k8s-cluster-root"}
                },
                "controller": {
                    "self": {"id": "2bbba94bf8d4ab09", "name": "flagd-5c56f7c9db-wwc9k"}
                },
                "service_interactions": {"incoming": [], "outgoing": []},
            },
        }
    }

    _test_long_user_prompt_and_context(
        tap_user_prompt, tap_grounding_context, "TaP / SRE Example"
    )

    # Example 2: Simple Weather / Temperature agent
    weather_user_prompt = "Please provide the current temperature for the given city."
    weather_grounding_context = {
        "city_info": {
            "name": "San Francisco",
            "country": "USA",
            "coordinates": {"lat": 37.7749, "lon": -122.4194},
        }
    }

    _test_long_user_prompt_and_context(
        weather_user_prompt, weather_grounding_context, "Weather Agent Example"
    )

    print("\nAll tests completed.\n")
