"""Web search selection for the financial agent."""

import asyncio
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


PARALLEL_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "Earnings-Call-Analyzer/1.0 (+https://github.com/Sapan2003/Earnings-Call-Analyzer)"
SEARCH_TIMEOUT = 30
_session_id = ContextVar("web_search_session_id", default=None)


def get_search_provider() -> str:
    """Keep DuckDuckGo as the default; require an explicit Parallel opt-in."""
    provider = os.getenv("SEARCH_PROVIDER", "duckduckgo").strip().lower()
    if provider not in {"duckduckgo", "parallel", "none"}:
        raise ValueError("SEARCH_PROVIDER must be duckduckgo, parallel, or none")
    return provider


@contextmanager
def search_session():
    """Reuse one anonymous session identifier across an agent's tool calls."""
    token = _session_id.set(str(uuid4()))
    try:
        yield
    finally:
        _session_id.reset(token)


async def parallel_search(query: str) -> str:
    """Call the anonymous Parallel Search MCP over Streamable HTTP."""
    arguments = {
        "objective": query,
        "search_queries": [query],
        "session_id": _session_id.get() or str(uuid4()),
    }
    try:
        # Bound the whole exchange, including discovery and transport cleanup.
        with anyio.fail_after(SEARCH_TIMEOUT):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=SEARCH_TIMEOUT,
                follow_redirects=False,
                trust_env=True,
            ) as client:
                async with streamable_http_client(
                    PARALLEL_URL, http_client=client
                ) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool("web_search", arguments)
    except TimeoutError as error:
        raise RuntimeError("Parallel Search MCP timed out. Try again later.") from error
    except Exception as error:
        raise RuntimeError("Parallel Search MCP is unavailable. Check the connection and try again.") from error

    text = "\n\n".join(
        block.text for block in result.content if block.type == "text"
    )
    if result.isError:
        raise RuntimeError(f"Parallel Search MCP error: {text[:1000]}")
    if not text and result.structuredContent:
        text = json.dumps(result.structuredContent, ensure_ascii=False)
    return text or "No relevant web results found."


def search_news(query: str) -> str:
    """Dispatch without falling back after an explicitly selected provider fails."""
    provider = get_search_provider()
    if provider == "none":
        return "Web search is disabled."
    if provider == "parallel":
        return asyncio.run(parallel_search(query))

    from langchain_community.tools import DuckDuckGoSearchRun

    return DuckDuckGoSearchRun().run(query)
