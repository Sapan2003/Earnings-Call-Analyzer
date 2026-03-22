import chromadb
from chromadb.utils import embedding_functions
from utils.logger import get_logger

logger = get_logger("retriever", "pipeline.log")

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "earnings_filings"


def get_collection():
    """
    Connects to existing ChromaDB collection.
    This is the same collection embedder.py created.
    """
    logger.info("Connecting to ChromaDB collection")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def retrieve_chunks(question: str, ticker: str = None,
                    filed_date: str = None, k: int = 4, collection=None) -> list:
    """
    Searches ChromaDB for chunks most relevant to the question.

    This is the core of RAG — semantic search.
    Instead of keyword matching, it finds chunks that are
    semantically similar to the question.

    Args:
        question: User's natural language question
        ticker: Optional - filter by company ticker
        filed_date: Optional - filter by specific filing date
        k: Number of chunks to retrieve

    Returns:
        List of relevant chunks with text and metadata
    """
    logger.info(f"Retrieving chunks | question='{question}' | "
                f"ticker={ticker} | k={k}")

    if collection is None:
        collection = get_collection()

    # Build metadata filter if ticker provided
    where_filter = None
    if ticker and filed_date:
        where_filter = {
            "$and": [
                {"ticker": {"$eq": ticker.upper()}},
                {"filed_date": {"$eq": filed_date}}
            ]
        }
    elif ticker:
        where_filter = {"ticker": {"$eq": ticker.upper()}}

    try:
        # Query ChromaDB
        results = collection.query(
            query_texts=[question],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results into clean list
        chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append({
                "text": doc,
                "metadata": meta,
                "relevance_score": round(1 - dist, 4)
            })

        logger.info(f"Retrieved {len(chunks)} chunks")

        for chunk in chunks:
            logger.debug(f"Chunk | ticker={chunk['metadata']['ticker']} | "
                        f"date={chunk['metadata']['filed_date']} | "
                        f"section={chunk['metadata']['section']} | "
                        f"score={chunk['relevance_score']}")

        return chunks

    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {e}")
        return []


def format_context(chunks: list) -> str:
    """
    Formats retrieved chunks into a single context string
    that gets passed to Claude.

    Each chunk is labeled with its source so Claude
    can cite where the information came from.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant information found."

    context_parts = []

    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        source_label = (f"[Source {i+1}: {meta['ticker']} | "
                       f"{meta['filed_date']} | "
                       f"{meta['form_type']} | "
                       f"Section: {meta['section']}]")

        context_parts.append(f"{source_label}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    logger.debug(f"Formatted context length: {len(context)} chars")
    return context


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.retriever <QUESTION> [TICKER]")
        print('Example: python -m pipeline.retriever "How did Apple revenue trend?" AAPL')
        sys.exit(1)

    question = sys.argv[1]
    ticker = sys.argv[2].upper() if len(sys.argv) > 2 else None

    chunks = retrieve_chunks(question, ticker=ticker, k=4)

    if chunks:
        print(f"\nFound {len(chunks)} relevant chunks:\n")
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1} ---")
            print(f"Source: {chunk['metadata']['ticker']} | "
                  f"{chunk['metadata']['filed_date']} | "
                  f"{chunk['metadata']['section']}")
            print(f"Relevance: {chunk['relevance_score']}")
            print(f"Text: {chunk['text'][:200]}...")
            print()
    else:
        print("No chunks found")
        