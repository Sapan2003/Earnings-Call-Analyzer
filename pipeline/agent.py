import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from pipeline.retriever import retrieve_chunks, format_context
from pipeline.financial_data import (get_financial_metrics,
                                     format_metrics_for_llm)
from utils.logger import get_logger

load_dotenv()
logger = get_logger("agent", "pipeline.log")

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()


def get_api_key() -> str:
    """Get API key from environment or Streamlit secrets."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    return key


API_KEY = get_api_key()

# Fast model for simple factual questions
llm_fast = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    anthropic_api_key=API_KEY
)

# Powerful model for complex analysis
llm_powerful = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=1500,
    anthropic_api_key=API_KEY
)


def get_llm(question: str):
    """
    Routes to appropriate model based on question complexity.
    Simple factual questions use Haiku (fast, cheap).
    Complex analysis questions use Sonnet (powerful).
    """
    simple_keywords = [
        "price", "pe ratio", "market cap",
        "dividend", "recommendation", "eps",
        "52 week", "current", "ratio", "yield"
    ]
    question_lower = question.lower()
    if any(kw in question_lower for kw in simple_keywords):
        logger.info("Routing to Haiku (simple question)")
        return llm_fast
    logger.info("Routing to Sonnet (complex question)")
    return llm_powerful

# ── TOOL 1: SEC Filings RAG ──────────────────────────────────────


@tool
def search_sec_filings(query: str) -> str:
    """
    Search through SEC filings and earnings transcripts
    for a company. Use this for questions about:
    - Revenue, profit, earnings trends
    - Management commentary and strategy
    - Risk factors and challenges
    - Quarterly performance comparisons
    - Business segment breakdowns
    Input should be a search query with the company ticker.
    Example: 'AAPL revenue growth Q1 2026'
    """
    logger.info(f"Tool: search_sec_filings | query='{query}'")

    words = query.upper().split()
    ticker = None
    common_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN",
                      "TSLA", "META", "NVDA"]
    for word in words:
        if word in common_tickers:
            ticker = word
            break

    chunks = retrieve_chunks(query, ticker=ticker, k=4)

    if not chunks:
        return "No relevant SEC filing information found."

    context = format_context(chunks)
    logger.info(f"Tool: search_sec_filings returned "
                f"{len(chunks)} chunks")
    return context


# ── TOOL 2: Live Financial Data ──────────────────────────────────


@tool
def get_live_financial_data(ticker: str) -> str:
    """
    Fetch live financial metrics for a publicly traded company.
    Use this for questions about:
    - Current stock price
    - P/E ratio and valuation
    - Market capitalization
    - Analyst recommendations and price targets
    - Profit margins and revenue (trailing twelve months)
    - 52 week high/low
    Input should be just the ticker symbol.
    Example: 'AAPL' or 'MSFT'
    """
    logger.info(f"Tool: get_live_financial_data | ticker='{ticker}'")
    ticker = ticker.strip().upper()
    metrics = get_financial_metrics(ticker)

    if not metrics:
        return f"Could not fetch financial data for {ticker}"

    formatted = format_metrics_for_llm(metrics)
    logger.info(f"Tool: get_live_financial_data returned "
                f"metrics for {ticker}")
    return formatted


# ── TOOL 3: Web Search ───────────────────────────────────────────
if SEARCH_PROVIDER == "tavily":
    from tavily import TavilyClient
    _tavily_client = TavilyClient()
    logger.info("Using Tavily as search provider")
else:
    search = DuckDuckGoSearchRun()
    logger.info("Using DuckDuckGo as search provider")


@tool
def search_financial_news(query: str) -> str:
    """
    Search the web for latest financial news and analysis.
    Use this for questions about:
    - Recent news affecting a company
    - Latest analyst reports
    - Industry trends and competitors
    - Recent events not in SEC filings
    - Current market sentiment
    Input should be a specific search query.
    Example: 'Apple earnings Q1 2026 results analyst reaction'
    """
    logger.info(f"Tool: search_financial_news | query='{query}'")
    try:
        if SEARCH_PROVIDER == "tavily":
            response = _tavily_client.search(
                query=query,
                max_results=5,
                search_depth="basic",
                topic="news",
            )
            results = []
            for r in response["results"]:
                results.append(f"{r['title']}\n{r['content']}")
            result = "\n\n".join(results) if results else "No results found."
        else:
            result = search.run(query)
        logger.info("Tool: search_financial_news returned results")
        return result
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search failed: {str(e)}"


# ── AGENT SETUP ──────────────────────────────────────────────────
tools = [search_sec_filings,
         get_live_financial_data,
         search_financial_news]


AGENT_PROMPT = """You are an expert financial analyst with access
to three powerful tools:

1. search_sec_filings: Search through official SEC filings and
   earnings transcripts for detailed financial analysis
2. get_live_financial_data: Get current stock price, valuation
   metrics, and analyst recommendations
3. search_financial_news: Search the web for latest news
   and market developments

Always cite which source each piece of information came from.
Be precise with numbers and include units."""


def run_agent(question: str) -> dict:
    """
    Main function - runs the LangChain agent.
    Dynamically selects model based on question complexity.
    """
    logger.info(f"Agent query: '{question}'")

    try:
        # Pick model based on question complexity
        llm = get_llm(question)

        # Create agent with selected model
        agent = create_react_agent(llm, tools, prompt=AGENT_PROMPT)

        result = agent.invoke(
            {"messages": [{"role": "user",
                          "content": question}]},
            config={"recursion_limit": 10}
        )

        answer = result["messages"][-1].content
        logger.info("Agent completed successfully")

        return {
            "answer": answer,
            "question": question
        }

    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return {
            "answer": f"Agent error: {str(e)}",
            "question": question
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.agent <QUESTION>")
        print('Example: python -m pipeline.agent '
              '"Is Apple a good investment right now?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nQuestion: {question}")
    print("\nAgent thinking...\n")

    result = run_agent(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["answer"])
