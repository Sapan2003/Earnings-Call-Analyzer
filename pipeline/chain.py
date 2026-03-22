import os
from anthropic import Anthropic
from dotenv import load_dotenv
from pipeline.retriever import retrieve_chunks, format_context
from utils.logger import get_logger

load_dotenv()
logger = get_logger("chain", "pipeline.log")

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert financial analyst specializing in
analyzing SEC filings and earnings call data for publicly traded companies.

Your job is to answer questions about company financial performance
based ONLY on the provided SEC filing excerpts.

Rules you must follow:
1. Only use information from the provided context
2. Always cite your sources (company, filing date, section)
3. If the context doesn't contain enough information, say so clearly
4. When discussing numbers, be precise and include units
5. For trend questions, compare across the provided quarters
6. Never make up or assume financial data not in the context

Response format:
- Give a clear direct answer first
- Support with specific numbers from the filings
- End with source citations
"""


def ask(question: str, ticker: str = None, k: int = 4) -> dict:
    """
    Main function - takes a question and returns
    a grounded answer from Claude based on SEC filings.

    Args:
        question: User's natural language question
        ticker: Optional company ticker to filter by
        k: Number of chunks to retrieve

    Returns:
        Dictionary with answer, sources, and metadata
    """
    logger.info(f"Question received | ticker={ticker} | "
                f"question='{question}'")

    # Step 1 - Retrieve relevant chunks from ChromaDB
    chunks = retrieve_chunks(question, ticker=ticker, k=k)

    if not chunks:
        logger.warning("No chunks retrieved - cannot answer question")
        return {
            "answer": "I could not find relevant information in the "
                     "SEC filings to answer this question. Please make "
                     "sure the company data has been ingested first.",
            "sources": [],
            "chunks_used": 0
        }

    # Step 2 - Format chunks into context string
    context = format_context(chunks)

    # Step 3 - Build user message with context + question
    user_message = f"""Based on the following SEC filing excerpts,
please answer this question:

QUESTION: {question}

SEC FILING CONTEXT:
{context}

Please provide a detailed answer with specific numbers and cite
which filing each piece of information comes from."""

    logger.info(f"Sending request to Claude | "
                f"chunks={len(chunks)} | "
                f"context_length={len(context)} chars")

    # Step 4 - Call Claude API
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        answer = response.content[0].text

        # Extract source information from chunks
        sources = []
        for chunk in chunks:
            source = (f"{chunk['metadata']['ticker']} | "
                     f"{chunk['metadata']['filed_date']} | "
                     f"{chunk['metadata']['form_type']} | "
                     f"Section: {chunk['metadata']['section']} | "
                     f"Relevance: {chunk['relevance_score']}")
            if source not in sources:
                sources.append(source)

        logger.info(f"Answer generated successfully | "
                   f"tokens_used={response.usage.input_tokens + response.usage.output_tokens}")

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "tokens_used": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens
            }
        }

    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return {
            "answer": f"Error calling Claude API: {str(e)}",
            "sources": [],
            "chunks_used": 0
        }


def ask_with_history(messages: list, ticker: str = None) -> dict:
    """
    Handles multi-turn conversations - remembers previous
    questions and answers in the same session.

    Args:
        messages: List of previous messages in conversation
        ticker: Optional company ticker to filter by

    Returns:
        Dictionary with answer and updated messages
    """
    logger.info(f"Multi-turn query | ticker={ticker} | "
                f"history_length={len(messages)}")

    # Get latest question from messages
    latest_question = messages[-1]["content"]

    # Retrieve chunks for latest question
    chunks = retrieve_chunks(latest_question, ticker=ticker, k=4)
    context = format_context(chunks)

    # Add context to the latest message
    messages[-1]["content"] = f"""Based on the following SEC filing excerpts,
please answer this question:

QUESTION: {latest_question}

SEC FILING CONTEXT:
{context}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        answer = response.content[0].text
        logger.info("Multi-turn answer generated successfully")

        return {
            "answer": answer,
            "sources": [c["metadata"] for c in chunks],
            "chunks_used": len(chunks)
        }

    except Exception as e:
        logger.error(f"Multi-turn Claude API call failed: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "chunks_used": 0
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.chain <QUESTION> [TICKER]")
        print('Example: python -m pipeline.chain "How did Apple revenue trend?" AAPL')
        sys.exit(1)

    question = sys.argv[1]
    ticker = sys.argv[2].upper() if len(sys.argv) > 2 else None

    print(f"\nQuestion: {question}")
    print(f"Ticker: {ticker}")
    print("\nThinking...\n")

    result = ask(question, ticker=ticker)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "=" * 60)
    print("SOURCES:")
    print("=" * 60)
    for source in result["sources"]:
        print(f"  - {source}")
    print(f"\nChunks used: {result['chunks_used']}")
    if "tokens_used" in result:
        print(f"Tokens used: {result['tokens_used']['total']}")
        print(f"Estimated cost: ${result['tokens_used']['total'] * 0.000003:.6f}")
        