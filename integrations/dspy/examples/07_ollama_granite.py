"""Example: Using Mellea DSPy with Ollama backend and granite4:micro-h model.

This example demonstrates how to configure a Mellea session with the Ollama
backend using a specific model (granite4:micro-h), and use it with DSPy.

Note: The 'model' parameter in MelleaLM constructor is only used for metadata
in response objects. The actual model used for generation is determined by
the mellea_session configuration.
"""

import dspy
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea_dspy import MelleaLM


def main():
    """Run example with Ollama backend and granite4:micro-h model."""
    # Create Mellea session with Ollama backend using granite4:micro-h model
    # The model specified here is what will actually be used for generation
    backend = OllamaModelBackend(model_id="granite4:micro-h")
    m = MelleaSession(backend=backend)

    # Create DSPy LM
    # The model parameter here is only for metadata in responses
    lm = MelleaLM(mellea_session=m)

    # Configure DSPy to use Mellea
    dspy.configure(lm=lm)

    # Use with DSPy - will use granite4:micro-h model from session
    print("=== Basic Question Answering ===")
    qa = dspy.Predict("question -> answer")
    result = qa(question="What is the capital of France?")
    print("Question: What is the capital of France?")
    print(f"Answer: {result.answer}")
    print()

    # Another example
    print("=== Code Generation ===")
    code_gen = dspy.Predict("task -> code")
    result = code_gen(task="Write a Python function to calculate factorial")
    print("Task: Write a Python function to calculate factorial")
    print(f"Code:\n{result.code}")


if __name__ == "__main__":
    main()
