import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain.messages import HumanMessage

from atlasai.service.graph_service import GraphService


class TestGraphService(unittest.TestCase):
    def test_build_graph_config_includes_langgraph_user_id(self):
        service = object.__new__(GraphService)

        config = service.build_graph_config(
            "thread-1",
            "atlas-main",
            user_id="user-42",
        )

        self.assertEqual(
            config["configurable"],
            {
                "thread_id": "thread-1",
                "langgraph_user_id": "user-42",
            },
        )
        self.assertEqual(config["run_name"], "atlas-main")

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
        streamed_events: list[dict] = []
        image_assets = [{"asset_id": "asset-1", "mime_type": "image/png"}]
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
            stream_writer=streamed_events.append,
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
                return_value=image_assets,
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
        self.assertNotIn("selected_image_assets", result)
        self.assertIn(
            {"type": "sources", "data": image_assets},
            streamed_events,
        )
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

    def test_stream_keeps_image_assets_local_to_each_run(self):
        first_asset = {"asset_id": "asset-1", "mime_type": "image/png"}
        final_asset = {"asset_id": "asset-2", "mime_type": "image/jpeg"}

        class FakeGraph:
            def __init__(self):
                self.run_count = 0

            async def astream(self, *_args, **_kwargs):
                self.run_count += 1
                if self.run_count == 1:
                    yield {
                        "type": "custom",
                        "data": {"type": "sources", "data": [first_asset]},
                    }
                    yield {
                        "type": "custom",
                        "data": {"type": "sources", "data": [final_asset]},
                    }
                else:
                    yield {
                        "type": "custom",
                        "data": {"type": "sources", "data": []},
                    }

                yield {
                    "type": "values",
                    "data": {
                        "messages": [],
                        "selected_image_assets": [first_asset],
                    },
                }

        async def run_streams():
            service = object.__new__(GraphService)
            service.graph = FakeGraph()
            service.run_summarizer = AsyncMock()
            payload = {
                "user_input": "Question",
                "thread_id": "thread-1",
                "user_id": "user-1",
            }

            first_run = [chunk async for chunk in service.stream(payload)]
            second_run = [chunk async for chunk in service.stream(payload)]
            await asyncio.sleep(0)
            return first_run, second_run

        first_run, second_run = asyncio.run(run_streams())

        self.assertEqual(
            [chunk for chunk in first_run if chunk.get("type") == "sources"],
            [{"type": "sources", "data": [final_asset]}],
        )
        self.assertFalse(
            any(chunk.get("type") == "sources" for chunk in second_run)
        )
