import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain.messages import HumanMessage

from atlasai.service.graph_service import GraphService


class TestGraphService(unittest.TestCase):
    def test_agent_llm_node_includes_thread_summary_in_system_prompt(self):
        vector_service = SimpleNamespace(asimilarity_search=AsyncMock(return_value=[]))
        tool_retriever = SimpleNamespace(as_retriever=lambda **_: SimpleNamespace())
        llm_response = SimpleNamespace(
            tool_calls=[],
            usage_metadata=None,
            response_metadata={},
        )
        captured_messages: list[object] = []

        async def ainvoke(messages, config=None):
            del config
            captured_messages.extend(messages)
            return llm_response

        llm = SimpleNamespace(
            bind_tools=lambda _tools: SimpleNamespace(
                ainvoke=ainvoke,
            )
        )
        config = {
            "model": {
                "provider": "openai",
                "model": "gpt-5",
                "base_url": "https://example.com",
                "api_key": "test-key",
            },
            "soul": "Atlas soul",
            "user_instruction_prompt": "Atlas instructions",
            "cg_api_key": "demo",
            "pg_store": "postgresql://demo",
            "pg_vector_store": "postgresql://demo",
        }
        runtime = SimpleNamespace(
            context={
                "soul": "Atlas soul",
                "user_instruction_prompt": "Atlas instructions",
            },
            stream_writer=lambda _event: None,
        )

        with (
            patch(
                "atlasai.service.graph_service.tool_store",
                return_value=tool_retriever,
            ),
            patch("atlasai.service.graph_service.system_model", return_value=llm),
            patch(
                "atlasai.service.graph_service.build_document_context_prompt",
                return_value="",
            ),
            patch(
                "atlasai.service.graph_service.build_retrieved_image_assets",
                return_value=[],
            ),
        ):
            service = GraphService(config=config, vector_service=vector_service)
            asyncio.run(
                service.agent_llm_node(
                    {
                        "user_input": "Hello",
                        "user_id": "user-42",
                        "summary": "User is building Atlas Chat.",
                        "messages": [HumanMessage(content="Hello")],
                        "resolved_tools": [],
                    },
                    runtime,
                )
            )

        self.assertTrue(captured_messages)
        system_message = captured_messages[0]
        self.assertIn("CONVERSATION SUMMARY", system_message.content)
        self.assertIn("User is building Atlas Chat.", system_message.content)

    def test_agent_llm_node_filters_rag_search_by_user_id(self):
        vector_service = SimpleNamespace(asimilarity_search=AsyncMock(return_value=[]))
        tool_retriever = SimpleNamespace(as_retriever=lambda **_: SimpleNamespace())
        llm_response = SimpleNamespace(
            tool_calls=[],
            usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
            response_metadata={"model_name": "gpt-5.4-mini-2026-03-17"},
        )
        llm = SimpleNamespace(
            bind_tools=lambda _tools: SimpleNamespace(
                ainvoke=AsyncMock(return_value=llm_response)
            )
        )
        config = {
            "model": {
                "provider": "openai",
                "model": "gpt-5",
                "base_url": "https://example.com",
                "api_key": "test-key",
            },
            "soul": "Atlas soul",
            "user_instruction_prompt": "Atlas instructions",
            "cg_api_key": "demo",
            "pg_store": "postgresql://demo",
            "pg_vector_store": "postgresql://demo",
        }
        runtime = SimpleNamespace(
            context={
                "soul": "Atlas soul",
                "user_instruction_prompt": "Atlas instructions",
            },
            stream_writer=lambda _event: None,
        )

        with (
            patch(
                "atlasai.service.graph_service.tool_store",
                return_value=tool_retriever,
            ),
            patch("atlasai.service.graph_service.system_model", return_value=llm),
            patch(
                "atlasai.service.graph_service.build_document_context_prompt",
                return_value="",
            ),
            patch(
                "atlasai.service.graph_service.build_retrieved_image_assets",
                return_value=[],
            ),
        ):
            service = GraphService(config=config, vector_service=vector_service)
            result = asyncio.run(
                service.agent_llm_node(
                    {
                        "user_input": "Hello",
                        "user_id": "user-42",
                        "messages": [HumanMessage(content="Hello")],
                        "resolved_tools": [],
                    },
                    runtime,
                )
            )

        vector_service.asimilarity_search.assert_awaited_once_with(
            table_name="raggidy_docs",
            query="Hello",
            filter={"user_id": "user-42"},
        )
        self.assertEqual(result["messages"], [llm_response])
        self.assertEqual(
            result["usage_payload"],
            {
                "model_name": "gpt-5.4-mini-2026-03-17",
                "usage_metadata": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                },
            },
        )

    def test_agent_llm_node_filters_rag_search_by_user_even_with_document_id(self):
        vector_service = SimpleNamespace(asimilarity_search=AsyncMock(return_value=[]))
        tool_retriever = SimpleNamespace(as_retriever=lambda **_: SimpleNamespace())
        llm_response = SimpleNamespace(
            tool_calls=[],
            usage_metadata=None,
            response_metadata={},
        )
        llm = SimpleNamespace(
            bind_tools=lambda _tools: SimpleNamespace(
                ainvoke=AsyncMock(return_value=llm_response)
            )
        )
        config = {
            "model": {
                "provider": "openai",
                "model": "gpt-5",
                "base_url": "https://example.com",
                "api_key": "test-key",
            },
            "soul": "Atlas soul",
            "user_instruction_prompt": "Atlas instructions",
            "cg_api_key": "demo",
            "pg_store": "postgresql://demo",
            "pg_vector_store": "postgresql://demo",
        }
        runtime = SimpleNamespace(
            context={
                "soul": "Atlas soul",
                "user_instruction_prompt": "Atlas instructions",
            },
            stream_writer=lambda _event: None,
        )

        with (
            patch(
                "atlasai.service.graph_service.tool_store",
                return_value=tool_retriever,
            ),
            patch("atlasai.service.graph_service.system_model", return_value=llm),
            patch(
                "atlasai.service.graph_service.build_document_context_prompt",
                return_value="",
            ),
            patch(
                "atlasai.service.graph_service.build_retrieved_image_assets",
                return_value=[],
            ),
        ):
            service = GraphService(config=config, vector_service=vector_service)
            asyncio.run(
                service.agent_llm_node(
                    {
                        "user_input": "Hello",
                        "user_id": "user-42",
                        "document_id": "document-7",
                        "messages": [HumanMessage(content="Hello")],
                        "resolved_tools": [],
                    },
                    runtime,
                )
            )

        vector_service.asimilarity_search.assert_awaited_once_with(
            table_name="raggidy_docs",
            query="Hello",
            filter={"user_id": "user-42"},
        )
