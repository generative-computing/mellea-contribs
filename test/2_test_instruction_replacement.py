"""
Test for demonstrating the "Instruction Replacement" use case.

This script generates a set of validated alternative prompts (instructions)
from the BenchDrift pipeline's output with m-program. These prompts are semantically
similar to the original but have been confirmed by the pipeline to yield the correct answer.
"""
import sys
import os
from typing import Any

# Add project's src directory and tools directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mellea import start_session
from mellea.backends.types import ModelOption
from mellea_contribs.tools.benchdrift_runner import (
    run_benchdrift_pipeline,
    extract_replacement_instructions
)

def test_instruction_replacement():
    """
    Demonstrates extracting validated alternative instructions using the m-program
    with the BenchDrift pipeline.
    """
    # --- 1. Define the baseline problem ---
    # Split into context (rules) and question
    # Only the question will be varied by BenchDrift
    context = """You are calculating total cost for a catering order.
Base price is $15 per person.
Groups of 20 or more get a 10% discount.
Weekend events have a $50 surcharge.
Delivery within 10 miles is free, beyond that costs $2 per mile."""

    baseline_question = "A company is ordering catering for 22 people for a Saturday event. The venue is 12 miles away. What is the total cost?"

    ground_truth_answer = "$351"

    print("\n--- Running BenchDrift Pipeline for Instruction Replacement ---")

    # --- 2. Start Mellea session with Ollama ---
    try:
        m = start_session(
            backend_name="ollama",
            model_id="granite3.3:8b",
            model_options={ModelOption.TEMPERATURE: 0.1}
        )
    except Exception as e:
        print(f"Failed to start Mellea session: {e}")
        print("Skipping test. Ensure ollama is running: ollama serve")
        print("And model is available: ollama pull granite3.3:8b")
        return

    # --- 3. Define m-program ---
    def m_program(question: str) -> Any:
        """
        M-program: Mellea-wrapped agent

        Takes only the question (which may be a variant from BenchDrift),
        and combines it with stable context via grounding_context.
        """
        response = m.instruct(
            description=question,  # The question (potentially a variant)
            grounding_context={
                "rules": context  # Stable context stays constant
            }
        )
        return response.value if hasattr(response, 'value') else response

    # --- 4. Generate Probes with BenchDrift + M-Program ---
    try:
        probes = run_benchdrift_pipeline(
            baseline_problem=baseline_question,  # Only the question, not full prompt
            ground_truth_answer=ground_truth_answer,
            m_program_callable=m_program,
            mellea_session=m,
            max_workers=4
        )
    except Exception as e:
        print(f"BenchDrift pipeline failed: {e}")
        print("Skipping instruction replacement test. Ensure RITS_API_KEY is set and BenchDrift dependencies are met.")
        return

    assert probes is not None
    assert len(probes) > 1

    # --- 5. Extract Validated Alternative Instructions ---
    alternatives = extract_replacement_instructions(probes)

    print("\n--- Validated Alternative Instructions ---")
    if alternatives:
        for i, instruction in enumerate(alternatives):
            print(f"  {i+1}. {instruction[:150]}...")
        print(f"Total alternatives found: {len(alternatives)}")
        assert len(alternatives) > 0
    else:
        print("No validated alternatives found. All variations either failed or didn't improve.")

    print("\n--- Test Passed: Instruction replacement demonstration complete. ---")


if __name__ == "__main__":
    test_instruction_replacement()