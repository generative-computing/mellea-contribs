"""Synchronous and asynchronous usage examples for Mellea Chat Model."""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from mellea import start_session

from mellea_langchain import MelleaChatModel


async def async_invoke_example():
    """Demonstrate async invoke."""
    print("\n" + "=" * 60)
    print("Async Invoke Example")
    print("=" * 60)

    # Create Mellea session and chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Async invoke
    print("\nAsync response to: 'Write a short poem about AI'")
    print("-" * 60)

    messages = [HumanMessage(content="Write a short poem about AI")]
    response = await chat_model.ainvoke(messages)

    print(response.content)
    print("-" * 60)


async def async_batch_example():
    """Demonstrate async batch processing."""
    print("\n" + "=" * 60)
    print("Async Batch Example")
    print("=" * 60)

    # Create Mellea session and chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Process multiple requests concurrently
    print("\nProcessing multiple requests concurrently...")
    print("-" * 60)

    message_batches = [
        [HumanMessage(content="What is 2+2?")],
        [HumanMessage(content="What is the capital of France?")],
        [HumanMessage(content="Name a programming language")],
    ]

    responses = await chat_model.abatch(message_batches)

    for i, response in enumerate(responses, 1):
        print(f"\nResponse {i}: {response.content}")

    print("-" * 60)


def sync_invoke_example():
    """Demonstrate synchronous invoke."""
    print("\n" + "=" * 60)
    print("Synchronous Invoke Example")
    print("=" * 60)

    # Create Mellea session and chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Sync invoke
    print("\nSynchronous response to: 'Count from 1 to 5'")
    print("-" * 60)

    messages = [HumanMessage(content="Count from 1 to 5")]
    response = chat_model.invoke(messages)

    print(response.content)
    print("-" * 60)


def sync_with_system_message_example():
    """Demonstrate sync invoke with system message."""
    print("\n" + "=" * 60)
    print("Sync with System Message Example")
    print("=" * 60)

    # Create Mellea session and chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Invoke with system message
    print("\nResponse with system message context...")
    print("-" * 60)

    messages = [
        SystemMessage(content="You are a helpful assistant that responds concisely."),
        HumanMessage(content="What is Python?"),
    ]
    response = chat_model.invoke(messages)

    print(response.content)
    print("-" * 60)


def main():
    """Run sync and async examples."""
    print("=" * 60)
    print("Mellea Chat Model - Sync & Async Examples")
    print("=" * 60)

    # Run sync examples
    sync_invoke_example()
    sync_with_system_message_example()

    # Run async examples
    asyncio.run(async_invoke_example())
    asyncio.run(async_batch_example())

    print("\n" + "=" * 60)
    print("Sync and async examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
