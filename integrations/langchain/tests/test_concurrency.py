"""Tests for concurrent request handling in MelleaChatModel.

Tests thread safety and async concurrency to ensure the chat model
can handle multiple simultaneous requests without issues.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import HumanMessage

from mellea_langchain import MelleaChatModel


class ThreadSafeMelleaSession:
    """Thread-safe mock Mellea session for concurrency testing."""

    def __init__(self, response_content="Response", delay=0.1):
        self.response_content = response_content
        self.delay = delay
        self.call_count = 0
        self.concurrent_calls = 0
        self.max_concurrent_calls = 0
        self.lock = threading.Lock()
        self.call_history = []

    def chat(self, message, model_options=None, tool_calls=False):
        """Mock sync chat with tracking."""
        with self.lock:
            self.call_count += 1
            self.concurrent_calls += 1
            self.max_concurrent_calls = max(self.max_concurrent_calls, self.concurrent_calls)
            call_id = self.call_count
            self.call_history.append({"id": call_id, "message": message, "thread": threading.current_thread().name})

        # Simulate processing time
        time.sleep(self.delay)

        with self.lock:
            self.concurrent_calls -= 1

        class MockResponse:
            def __init__(self, content):
                self.content = content
                self._tool_calls = None

        return MockResponse(f"{self.response_content} #{call_id}")

    async def achat(self, message, model_options=None, tool_calls=False):
        """Mock async chat with tracking."""
        with self.lock:
            self.call_count += 1
            self.concurrent_calls += 1
            self.max_concurrent_calls = max(self.max_concurrent_calls, self.concurrent_calls)
            call_id = self.call_count
            self.call_history.append({"id": call_id, "message": message, "async": True})

        # Simulate async processing time
        await asyncio.sleep(self.delay)

        with self.lock:
            self.concurrent_calls -= 1

        class MockResponse:
            def __init__(self, content):
                self.content = content
                self._tool_calls = None

        return MockResponse(f"{self.response_content} #{call_id}")


class TestConcurrentSyncRequests:
    """Test concurrent synchronous requests."""

    def test_multiple_threads_basic(self):
        """Test that multiple threads can make requests simultaneously."""
        session = ThreadSafeMelleaSession(delay=0.05)
        chat_model = MelleaChatModel(mellea_session=session)

        def make_request(thread_id):
            messages = [HumanMessage(content=f"Request from thread {thread_id}")]
            result = chat_model.invoke(messages)
            return result.content

        # Run 5 concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(5)]
            results = [f.result() for f in futures]

        # All requests should complete
        assert len(results) == 5
        assert session.call_count == 5
        # Should have had concurrent calls
        assert session.max_concurrent_calls > 1

    def test_thread_safety_no_data_corruption(self):
        """Test that concurrent requests don't corrupt shared state."""
        session = ThreadSafeMelleaSession(delay=0.02)
        chat_model = MelleaChatModel(mellea_session=session)

        results = []
        errors = []

        def make_request(thread_id):
            try:
                messages = [HumanMessage(content=f"Message {thread_id}")]
                result = chat_model.invoke(messages)
                results.append(result.content)
            except Exception as e:
                errors.append(e)

        # Run 10 concurrent requests
        threads = [threading.Thread(target=make_request, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # All requests should complete
        assert len(results) == 10
        assert session.call_count == 10

    def test_high_concurrency_stress(self):
        """Stress test with many concurrent requests."""
        session = ThreadSafeMelleaSession(delay=0.01)
        chat_model = MelleaChatModel(mellea_session=session)

        num_requests = 50

        def make_request(req_id):
            messages = [HumanMessage(content=f"Request {req_id}")]
            return chat_model.invoke(messages).content

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [f.result() for f in futures]

        # All requests should complete successfully
        assert len(results) == num_requests
        assert session.call_count == num_requests
        # Verify high concurrency was achieved
        assert session.max_concurrent_calls >= 5

    def test_concurrent_with_different_messages(self):
        """Test that different messages are handled correctly concurrently."""
        session = ThreadSafeMelleaSession(delay=0.03)
        chat_model = MelleaChatModel(mellea_session=session)

        messages_to_send = [
            "What is Python?",
            "Explain async programming",
            "How does threading work?",
            "What is concurrency?",
            "Describe parallelism",
        ]

        def make_request(message_text):
            messages = [HumanMessage(content=message_text)]
            result = chat_model.invoke(messages)
            return (message_text, result.content)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, msg) for msg in messages_to_send]
            results = [f.result() for f in futures]

        # Verify all messages were sent
        sent_messages = [r[0] for r in results]
        assert set(sent_messages) == set(messages_to_send)

        # Verify all got responses
        assert all(r[1] for r in results)


class TestConcurrentAsyncRequests:
    """Test concurrent asynchronous requests."""

    @pytest.mark.asyncio
    async def test_multiple_async_requests(self):
        """Test that multiple async requests can run concurrently."""
        session = ThreadSafeMelleaSession(delay=0.05)
        chat_model = MelleaChatModel(mellea_session=session)

        async def make_request(req_id):
            messages = [HumanMessage(content=f"Async request {req_id}")]
            result = await chat_model.ainvoke(messages)
            return result.content

        # Run 5 concurrent async requests
        results = await asyncio.gather(*[make_request(i) for i in range(5)])

        # All requests should complete
        assert len(results) == 5
        assert session.call_count == 5
        # Should have had concurrent calls
        assert session.max_concurrent_calls > 1

    @pytest.mark.asyncio
    async def test_async_high_concurrency(self):
        """Stress test with many concurrent async requests."""
        session = ThreadSafeMelleaSession(delay=0.01)
        chat_model = MelleaChatModel(mellea_session=session)

        num_requests = 100

        async def make_request(req_id):
            messages = [HumanMessage(content=f"Request {req_id}")]
            result = await chat_model.ainvoke(messages)
            return result.content

        # Run many concurrent requests
        results = await asyncio.gather(*[make_request(i) for i in range(num_requests)])

        # All requests should complete
        assert len(results) == num_requests
        assert session.call_count == num_requests
        # Verify high concurrency was achieved
        assert session.max_concurrent_calls >= 10

    @pytest.mark.asyncio
    async def test_async_with_errors(self):
        """Test that errors in one async request don't affect others."""

        class ErrorSession(ThreadSafeMelleaSession):
            async def achat(self, message, model_options=None, tool_calls=False):
                # Fail on specific messages
                if "error" in message.lower():
                    raise RuntimeError("Simulated error")
                return await super().achat(message, model_options, tool_calls)

        session = ErrorSession(delay=0.02)
        chat_model = MelleaChatModel(mellea_session=session)

        async def make_request(req_id):
            message_text = f"error request {req_id}" if req_id % 3 == 0 else f"good request {req_id}"
            messages = [HumanMessage(content=message_text)]
            try:
                result = await chat_model.ainvoke(messages)
                return ("success", result.content)
            except RuntimeError:
                return ("error", None)

        # Run requests, some will fail
        results = await asyncio.gather(*[make_request(i) for i in range(9)])

        # Count successes and errors
        successes = [r for r in results if r[0] == "success"]
        errors = [r for r in results if r[0] == "error"]

        # Should have both successes and errors
        assert len(successes) == 6  # 0, 3, 6 fail; others succeed
        assert len(errors) == 3

    @pytest.mark.asyncio
    async def test_async_streaming_concurrent(self):
        """Test concurrent async streaming requests."""
        session = ThreadSafeMelleaSession(delay=0.02)
        chat_model = MelleaChatModel(mellea_session=session)

        async def stream_request(req_id):
            messages = [HumanMessage(content=f"Stream {req_id}")]
            chunks = []
            async for chunk in chat_model.astream(messages):
                chunks.append(chunk.content)
            return "".join(chunks)

        # Run 5 concurrent streaming requests
        results = await asyncio.gather(*[stream_request(i) for i in range(5)])

        # All streams should complete
        assert len(results) == 5
        assert all(r for r in results)


class TestMixedConcurrency:
    """Test mixed sync and async concurrent requests."""

    @pytest.mark.asyncio
    async def test_sync_and_async_together(self):
        """Test that sync and async requests can coexist."""
        session = ThreadSafeMelleaSession(delay=0.03)
        chat_model = MelleaChatModel(mellea_session=session)

        sync_results = []

        def sync_request(req_id):
            messages = [HumanMessage(content=f"Sync {req_id}")]
            result = chat_model.invoke(messages)
            sync_results.append(result.content)

        async def async_request(req_id):
            messages = [HumanMessage(content=f"Async {req_id}")]
            result = await chat_model.ainvoke(messages)
            return result.content

        # Start sync requests in threads
        threads = [threading.Thread(target=sync_request, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()

        # Run async requests concurrently
        async_results = await asyncio.gather(*[async_request(i) for i in range(3)])

        # Wait for sync threads to complete
        for t in threads:
            t.join()

        # All requests should complete
        assert len(sync_results) == 3
        assert len(async_results) == 3
        assert session.call_count == 6


class TestConcurrencyWithToolBinding:
    """Test concurrency with tool-bound models."""

    def test_concurrent_tool_bound_requests(self):
        """Test concurrent requests with tool-bound models."""
        session = ThreadSafeMelleaSession(delay=0.02)
        chat_model = MelleaChatModel(mellea_session=session)

        class MockTool:
            name = "test_tool"
            description = "A test tool"

        bound_model = chat_model.bind_tools([MockTool()])

        def make_request(req_id):
            messages = [HumanMessage(content=f"Use tool {req_id}")]
            result = bound_model.invoke(messages)
            return result.content

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All requests should complete
        assert len(results) == 10
        assert session.call_count == 10

    @pytest.mark.asyncio
    async def test_async_concurrent_tool_bound_requests(self):
        """Test async concurrent requests with tool-bound models."""
        session = ThreadSafeMelleaSession(delay=0.02)
        chat_model = MelleaChatModel(mellea_session=session)

        class MockTool:
            name = "async_tool"
            description = "An async test tool"

        bound_model = chat_model.bind_tools([MockTool()])

        async def make_request(req_id):
            messages = [HumanMessage(content=f"Use async tool {req_id}")]
            result = await bound_model.ainvoke(messages)
            return result.content

        results = await asyncio.gather(*[make_request(i) for i in range(10)])

        # All requests should complete
        assert len(results) == 10
        assert session.call_count == 10


class TestConcurrencyPerformance:
    """Test performance characteristics under concurrent load."""

    def test_concurrent_faster_than_sequential(self):
        """Verify concurrent requests are faster than sequential."""
        session = ThreadSafeMelleaSession(delay=0.1)
        chat_model = MelleaChatModel(mellea_session=session)

        num_requests = 5

        # Sequential execution
        start_time = time.time()
        for i in range(num_requests):
            messages = [HumanMessage(content=f"Sequential {i}")]
            chat_model.invoke(messages)
        sequential_time = time.time() - start_time

        # Reset session
        session.call_count = 0

        # Concurrent execution
        def make_request(req_id):
            messages = [HumanMessage(content=f"Concurrent {req_id}")]
            return chat_model.invoke(messages)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            [f.result() for f in futures]
        concurrent_time = time.time() - start_time

        # Concurrent should be significantly faster
        # With 0.1s delay and 5 requests: sequential ~0.5s, concurrent ~0.1s
        assert concurrent_time < sequential_time * 0.5

    @pytest.mark.asyncio
    async def test_async_concurrent_faster_than_sequential(self):
        """Verify async concurrent requests are faster than sequential."""
        session = ThreadSafeMelleaSession(delay=0.1)
        chat_model = MelleaChatModel(mellea_session=session)

        num_requests = 5

        # Sequential execution
        start_time = time.time()
        for i in range(num_requests):
            messages = [HumanMessage(content=f"Sequential {i}")]
            await chat_model.ainvoke(messages)
        sequential_time = time.time() - start_time

        # Reset session
        session.call_count = 0

        # Concurrent execution
        async def make_request(req_id):
            messages = [HumanMessage(content=f"Concurrent {req_id}")]
            return await chat_model.ainvoke(messages)

        start_time = time.time()
        await asyncio.gather(*[make_request(i) for i in range(num_requests)])
        concurrent_time = time.time() - start_time

        # Concurrent should be significantly faster
        assert concurrent_time < sequential_time * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
