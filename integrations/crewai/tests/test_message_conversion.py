"""Tests for message conversion functionality."""


class TestCrewAIToMelleaConversion:
    """Tests for converting CrewAI messages to Mellea format."""

    def test_import_message_conversion(self):
        """Test that message conversion module can be imported."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        assert CrewAIMessageConverter is not None

    def test_convert_string_message(self):
        """Test converting a simple string message."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        result = converter.to_mellea("Hello, world!")

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "Hello, world!"

    def test_convert_single_user_message(self):
        """Test converting a single user message dict."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        messages = [{"role": "user", "content": "Test message"}]
        result = converter.to_mellea(messages)

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "Test message"

    def test_convert_system_message(self):
        """Test converting a system message."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        messages = [{"role": "system", "content": "You are a helpful assistant"}]
        result = converter.to_mellea(messages)

        assert len(result) == 1
        assert result[0].role == "system"
        assert result[0].content == "You are a helpful assistant"

    def test_convert_assistant_message(self):
        """Test converting an assistant message."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        messages = [{"role": "assistant", "content": "I can help you with that"}]
        result = converter.to_mellea(messages)

        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "I can help you with that"

    def test_convert_multiple_messages(self):
        """Test converting multiple messages."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = converter.to_mellea(messages)

        assert len(result) == 4
        assert result[0].role == "system"
        assert result[1].role == "user"
        assert result[2].role == "assistant"
        assert result[3].role == "user"

    def test_convert_empty_list(self):
        """Test converting empty message list."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        result = converter.to_mellea([])

        assert len(result) == 0

    def test_message_content_preserved(self):
        """Test that message content is preserved exactly."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        content = "This is a test message with special chars: !@#$%^&*()"
        messages = [{"role": "user", "content": content}]
        result = converter.to_mellea(messages)

        assert result[0].content == content

    def test_multiline_message_content(self):
        """Test converting messages with multiline content."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        content = """This is a multiline message.
It has multiple lines.
And should be preserved."""
        messages = [{"role": "user", "content": content}]
        result = converter.to_mellea(messages)

        assert result[0].content == content


class TestMelleaToCrewAIConversion:
    """Tests for converting Mellea responses to CrewAI format."""

    def test_import_response_conversion(self):
        """Test that response conversion can be imported."""
        from mellea_crewai.message_conversion import CrewAIMessageConverter

        assert CrewAIMessageConverter is not None

    def test_convert_simple_response(self):
        """Test converting a simple Mellea response."""
        from unittest.mock import Mock

        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        mock_response = Mock()
        mock_response.content = "This is a response"

        result = converter.from_mellea(mock_response)

        assert result == "This is a response"

    def test_convert_response_with_special_chars(self):
        """Test converting response with special characters."""
        from unittest.mock import Mock

        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        mock_response = Mock()
        mock_response.content = "Response with special chars: !@#$%^&*()"

        result = converter.from_mellea(mock_response)

        assert result == "Response with special chars: !@#$%^&*()"

    def test_convert_multiline_response(self):
        """Test converting multiline response."""
        from unittest.mock import Mock

        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        content = """This is a multiline response.
It has multiple lines.
All should be preserved."""
        mock_response = Mock()
        mock_response.content = content

        result = converter.from_mellea(mock_response)

        assert result == content

    def test_convert_empty_response(self):
        """Test converting empty response."""
        from unittest.mock import Mock

        from mellea_crewai.message_conversion import CrewAIMessageConverter

        converter = CrewAIMessageConverter()
        mock_response = Mock()
        mock_response.content = ""

        result = converter.from_mellea(mock_response)

        assert result == ""


class TestMessageConversionIntegration:
    """Integration tests for message conversion in LLM calls."""

    def test_string_message_in_llm_call(self):
        """Test that string messages are properly converted in LLM calls."""
        from unittest.mock import Mock

        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "Response"
        mock_response._tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_session)
        result = llm.call("Test message")

        # Verify chat was called
        mock_session.chat.assert_called_once()
        assert result == "Response"

    def test_list_messages_in_llm_call(self):
        """Test that list messages are properly converted in LLM calls."""
        from unittest.mock import Mock

        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "Response"
        mock_response._tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_session)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = llm.call(messages)

        # Verify chat was called
        mock_session.chat.assert_called_once()
        assert result == "Response"

    def test_conversation_flow(self):
        """Test a full conversation flow with message conversion."""
        from unittest.mock import Mock

        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "I'm doing well, thank you!"
        mock_response._tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage
        mock_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_session)

        # Simulate a conversation
        messages = [
            {"role": "system", "content": "You are a friendly assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = llm.call(messages)

        assert result == "I'm doing well, thank you!"
        mock_session.chat.assert_called_once()
