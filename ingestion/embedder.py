import os
import chromadb
from chromadb.utils import embedding_functions
from utils.logger import get_logger

logger = get_logger("embedder", "ingestion.log")

# Path where ChromaDB saves data locally
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "earnings_filings"


def get_chroma_client():
    """
    Creates and returns a persistent ChromaDB client.
    Persistent means data is saved to disk - 
    so you don't re-embed every time you restart.
    """
    logger.info(f"Connecting to ChromaDB at {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client


def get_collection(client):
    """
    Gets or creates the ChromaDB collection.
    Think of a collection like a table in a database
    - it holds all our embedded chunks.
    """
    # Using HuggingFace embedding model - completely free
    # Runs locally on your machine, no API calls needed
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info(f"Connected to collection: {COLLECTION_NAME}")
    return collection


def embed_chunks(chunks: list, collection) -> int:
    """
    Takes parsed chunks and stores them in ChromaDB.
    ChromaDB automatically converts text to vectors.

    Args:
        chunks: List of chunk dicts with text + metadata
        collection: ChromaDB collection to store in

    Returns:
        Number of chunks successfully embedded
    """
    if not chunks:
        logger.warning("No chunks to embed")
        return 0

    logger.info(f"Embedding {len(chunks)} chunks into ChromaDB")

    # Prepare data for ChromaDB
    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        ticker = chunk["metadata"]["ticker"]
        filed_date = chunk["metadata"]["filed_date"]
        section = chunk["metadata"]["section"]
        chunk_index = chunk["metadata"]["chunk_index"]

        # Create unique ID for each chunk
        chunk_id = f"{ticker}_{filed_date}_{section}_{chunk_index}"

        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])
        ids.append(chunk_id)

    try:
        # Add to ChromaDB in batches of 100
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            collection.upsert(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )

            logger.info(f"Embedded batch {i // batch_size + 1} "
                       f"({len(batch_docs)} chunks)")

        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to embed chunks: {e}")
        return 0


def embed_company(ticker: str, quarters: int = 8) -> int:
    """
    Main function - fetches, parses and embeds all filings
    for a company in one go.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        quarters: Number of quarters to process

    Returns:
        Total number of chunks embedded
    """
    from ingestion.sec_fetcher import fetch_company_filings
    from ingestion.transcript_parser import parse_filing

    logger.info(f"=== Starting embedding pipeline for {ticker} ===")

    # Step 1 - Fetch filings
    filings = fetch_company_filings(ticker, quarters=quarters)
    if not filings:
        logger.error(f"No filings found for {ticker}")
        return 0

    # Step 2 - Connect to ChromaDB
    client = get_chroma_client()
    collection = get_collection(client)

    # Step 3 - Parse and embed each filing
    total_chunks = 0

    for filing in filings:
        chunks = parse_filing(filing)
        embedded = embed_chunks(chunks, collection)
        total_chunks += embedded
        logger.info(f"Processed {filing['filed_date']}: "
                   f"{embedded} chunks embedded")

    logger.info(f"=== Completed: {total_chunks} total chunks "
               f"embedded for {ticker} ===")
    return total_chunks


def get_collection_stats():
    """
    Shows how many chunks are stored in ChromaDB.
    Useful for debugging and monitoring.
    """
    client = get_chroma_client()
    collection = get_collection(client)
    count = collection.count()
    logger.info(f"ChromaDB collection has {count} chunks")
    print(f"\n📊 ChromaDB Stats:")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Total chunks stored: {count}")
    return count


if __name__ == "__main__":
    # Test - embed Apple filings
    print("🚀 Starting embedding pipeline for AAPL...")
    total = embed_company("AAPL", quarters=2)
    print(f"\n✅ Embedding complete!")
    get_collection_stats()