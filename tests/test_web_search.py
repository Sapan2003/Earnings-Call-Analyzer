"""Exercise provider selection and the real MCP transport without paid services."""
import asyncio
import importlib
import json
import sys
import types

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pipeline import web_search


SOURCE_URL = "https://example.com/apple-earnings"
EXCERPT = "Apple reported quarterly revenue of $100 billion."


@pytest.fixture
def mcp_http(monkeypatch):
    """Use the real MCP SDK with an HTTP-level JSON-RPC server fixture."""
    requests = []
    state = {"error": False, "slow": False, "status": 200}
    original_client = httpx.AsyncClient

    async def respond(request):
        requests.append(request)
        assert str(request.url) == web_search.PARALLEL_URL
        assert request.headers["User-Agent"] == web_search.USER_AGENT
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {
                "type": "object", "properties": {}, "additionalProperties": True}}]}
        else:
            assert method == "tools/call"
            assert payload["params"]["name"] == "web_search"
            args = payload["params"]["arguments"]
            assert args["objective"] == args["search_queries"][0]
            assert len(args["session_id"]) == 36
            if state["slow"]:
                await asyncio.sleep(10)
            if state["status"] != 200:
                return httpx.Response(state["status"], text="Service unavailable")
            text = "Quota exceeded" if state["error"] else json.dumps({
                "results": [{"title": "Apple earnings", "url": SOURCE_URL,
                             "excerpts": [EXCERPT]}]})
            result = {"content": [{"type": "text", "text": text}], "isError": state["error"]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is True
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(web_search.httpx, "AsyncClient", client)
    # Ambient credentials must not change the anonymous MCP request.
    monkeypatch.setenv("PARALLEL_API_KEY", "unused-ambient-key")
    return requests, state


class ScriptedSearchModel(BaseChatModel):
    """A scripted tool call, then an answer consuming the real ToolMessage."""
    bound_names: list[str] = []

    @property
    def _llm_type(self):
        return "scripted-search-test"

    def bind_tools(self, tools, **kwargs):
        self.bound_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            answer = AIMessage(content="Search evidence: " + messages[-1].content)
        elif "search_financial_news" in self.bound_names:
            answer = AIMessage(content="", tool_calls=[{
                "name": "search_financial_news", "args": {"query": "Apple recent earnings news"},
                "id": "news-1", "type": "tool_call"}])
        else:
            answer = AIMessage(content="Web search is disabled; use filings and market data.")
        return ChatResult(generations=[ChatGeneration(message=answer)])


@pytest.fixture
def agent_module(monkeypatch):
    # Isolate unrelated ChromaDB/model downloads and market-data services.
    # LangGraph, LangChain tools and the MCP client remain real.
    retriever = types.ModuleType("pipeline.retriever")
    retriever.retrieve_chunks = lambda *args, **kwargs: []
    retriever.format_context = lambda chunks: ""
    financial = types.ModuleType("pipeline.financial_data")
    financial.get_financial_metrics = lambda ticker: {}
    financial.format_metrics_for_llm = lambda metrics: ""
    monkeypatch.setitem(sys.modules, "pipeline.retriever", retriever)
    monkeypatch.setitem(sys.modules, "pipeline.financial_data", financial)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    sys.modules.pop("pipeline.agent", None)
    module = importlib.import_module("pipeline.agent")
    model = ScriptedSearchModel()
    monkeypatch.setattr(module, "get_llm", lambda question: model)
    yield module, model
    sys.modules.pop("pipeline.agent", None)


def test_agent_consumes_parallel_search(agent_module, mcp_http, monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "parallel")
    agent, model = agent_module
    result = agent.run_agent("What is the latest Apple earnings news?")
    assert SOURCE_URL in result["answer"]
    assert EXCERPT in result["answer"]
    assert "search_financial_news" in model.bound_names
    methods = [json.loads(r.content)["method"] for r in mcp_http[0]]
    assert "initialize" in methods and "tools/call" in methods


@pytest.mark.parametrize("provider", [None, "duckduckgo", " DuckDuckGo "])
def test_default_and_explicit_duckduckgo(provider, monkeypatch, mcp_http):
    if provider is None:
        monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("SEARCH_PROVIDER", provider)
    from langchain_community.tools import DuckDuckGoSearchRun
    monkeypatch.setattr(DuckDuckGoSearchRun, "run", lambda self, query: "DuckDuckGo: " + query)
    assert web_search.search_news("Apple news") == "DuckDuckGo: Apple news"
    assert not mcp_http[0]


def test_disabled_removes_tool(agent_module, monkeypatch, mcp_http):
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    agent, model = agent_module
    assert "disabled" in agent.run_agent("Apple news")["answer"]
    assert "search_financial_news" not in model.bound_names
    assert web_search.search_news("Apple news") == "Web search is disabled."
    assert not mcp_http[0]


def test_invalid_provider_never_falls_back(agent_module, monkeypatch, mcp_http):
    monkeypatch.setenv("SEARCH_PROVIDER", "unknown")
    assert "SEARCH_PROVIDER must be" in agent_module[0].run_agent("Apple news")["answer"]
    assert not mcp_http[0]


@pytest.mark.parametrize("failure", ["error", "status", "slow"])
def test_agent_reports_failures_without_fallback(agent_module, monkeypatch, mcp_http, failure):
    monkeypatch.setenv("SEARCH_PROVIDER", "parallel")
    requests, state = mcp_http
    state[failure] = 503 if failure == "status" else True
    if failure == "slow":
        monkeypatch.setattr(web_search, "SEARCH_TIMEOUT", 0.05)
    from langchain_community.tools import DuckDuckGoSearchRun

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit Parallel failure must not invoke DuckDuckGo")

    monkeypatch.setattr(DuckDuckGoSearchRun, "run", forbidden)
    answer = agent_module[0].run_agent("Apple news")["answer"]
    assert "Web search failed:" in answer
    if failure == "slow":
        assert "timed out" in answer
    elif failure == "error":
        assert "Quota exceeded" in answer
    else:
        assert "unavailable" in answer
    assert requests


def test_session_reuse_and_reset(monkeypatch, mcp_http):
    monkeypatch.setenv("SEARCH_PROVIDER", "parallel")
    with web_search.search_session():
        web_search.search_news("Apple earnings news")
        web_search.search_news("Apple analyst reaction")
    calls = [json.loads(r.content) for r in mcp_http[0] if json.loads(r.content)["method"] == "tools/call"]
    assert calls[0]["params"]["arguments"]["session_id"] == calls[1]["params"]["arguments"]["session_id"]
    assert web_search._session_id.get() is None


def test_parallel_cancellation_closes_transport(mcp_http):
    mcp_http[1]["slow"] = True

    async def cancel():
        task = asyncio.create_task(web_search.parallel_search("Apple earnings news"))
        for _ in range(100):
            if any(json.loads(r.content)["method"] == "tools/call" for r in mcp_http[0]):
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("Search did not start")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

    asyncio.run(cancel())


def test_agent_does_not_swallow_cancellation(agent_module, monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "parallel")

    async def cancelled(query):
        raise asyncio.CancelledError()

    monkeypatch.setattr(web_search, "parallel_search", cancelled)
    with pytest.raises(asyncio.CancelledError):
        agent_module[0].run_agent("Apple news")
