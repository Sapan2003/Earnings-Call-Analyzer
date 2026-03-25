import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from pipeline.agent import run_agent  # noqa: E402
from pipeline.financial_data import (get_financial_metrics,  # noqa: E402
                                     get_quarterly_financials)
from ingestion.embedder import (get_chroma_client,  # noqa: E402
                                get_collection,
                                embed_company)
from utils.logger import get_logger  # noqa: E402

logger = get_logger("streamlit_app", "pipeline.log")


@st.cache_resource
def load_embedding_model():
    """
    Loads embedding model once at startup.
    Persists across all reruns and user sessions.
    Never reloads after first load — saves 6-8 seconds per query.
    """
    from chromadb.utils import embedding_functions
    logger.info("Loading embedding model (one time only)...")
    embedding_fn = (
        embedding_functions
        .SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    logger.info("Embedding model loaded and cached")
    return embedding_fn


@st.cache_resource
def load_chroma_collection():
    """
    Opens ChromaDB connection once at startup.
    Reused across all queries — saves 2-3 seconds per query.
    """
    import chromadb
    logger.info("Connecting to ChromaDB (one time only)...")
    client = chromadb.PersistentClient(path="data/chroma_db")
    embedding_fn = load_embedding_model()
    collection = client.get_or_create_collection(
        name="earnings_filings",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info("ChromaDB collection loaded and cached")
    return collection


# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Call Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background: #f0f7ff;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        color: #555;
    }
    .stChatMessage {
    font-size: 1.2rem !important;
    }

    .stMarkdown p {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    .stChatInput input {
        font-size: 1rem !important;
    }

    /* Sidebar text */
    .sidebar .stMarkdown {
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "quarterly_data" not in st.session_state:
    st.session_state.quarterly_data = None


# ── HELPER FUNCTIONS ─────────────────────────────────────────────


def check_data_exists(ticker: str) -> bool:
    """Check if ticker data is already in ChromaDB"""
    try:
        client = get_chroma_client()
        collection = get_collection(client)
        existing = collection.get(
            where={"ticker": ticker.upper()}
        )
        return len(existing["ids"]) > 0
    except Exception:
        return False


def ingest_ticker(ticker: str, quarters: int = 8):
    """Ingest company data with progress indicator"""
    with st.spinner(f"Fetching {ticker} SEC filings "
                   f"({quarters} quarters + annual reports)..."):
        try:
            embed_company(ticker, quarters=quarters)
            return True
        except Exception as e:
            st.error(f"Failed to ingest data: {e}")
            return False


@st.cache_data(ttl=3600)
def load_financial_data(ticker: str):
    """Load live financial metrics and quarterly data.
       Cached for 1 hour to avoid rate limiting."""
    metrics = get_financial_metrics(ticker)
    quarterly = get_quarterly_financials(ticker)
    return metrics, quarterly


def create_revenue_chart(quarterly_data: dict) -> go.Figure:
    """Creates quarterly revenue trend chart"""
    if not quarterly_data or not quarterly_data.get("quarters"):
        return None

    quarters = quarterly_data["quarters"]
    revenue = quarterly_data["revenue"]

    # Filter out None values
    valid_data = [(q, r) for q, r in zip(quarters, revenue)
                 if r is not None]

    if not valid_data:
        return None

    quarters_clean, revenue_clean = zip(*valid_data)
    revenue_billions = [r / 1e9 for r in revenue_clean]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(quarters_clean),
        y=revenue_billions,
        marker_color="#1f77b4",
        name="Revenue",
        hovertemplate="Quarter: %{x}<br>Revenue: $%{y:.2f}B<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=list(quarters_clean),
        y=revenue_billions,
        mode="lines+markers",
        line=dict(color="#ff7f0e", width=2),
        marker=dict(size=8),
        name="Trend",
        hovertemplate="Quarter: %{x}<br>Revenue: $%{y:.2f}B<extra></extra>"
    ))

    fig.update_layout(
        title="Quarterly Revenue Trend",
        xaxis_title="Quarter",
        yaxis_title="Revenue (Billions USD)",
        height=400,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", tickangle=45),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    )

    return fig


def create_margins_chart(metrics: dict) -> go.Figure:
    """Creates profit margins comparison chart"""
    if not metrics:
        return None

    margin_data = {
        "Gross Margin": metrics.get("gross_margin", 0),
        "Profit Margin": metrics.get("profit_margin", 0)
    }

    labels = []
    values = []
    for label, value in margin_data.items():
        if value and value != "N/A":
            try:
                labels.append(label)
                values.append(float(value) * 100)
            except Exception:
                pass

    if not values:
        return None

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=["#2ecc71", "#3498db"],
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title="Profit Margins",
        yaxis_title="Percentage (%)",
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    )

    return fig


# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Earnings Call Analyzer")
    st.markdown("---")

    # Ticker input
    ticker_input = st.text_input(
        "Enter Stock Ticker",
        placeholder="e.g. AAPL, MSFT, TSLA",
        help="Enter any US publicly traded company ticker"
    ).upper().strip()

    quarters = st.slider(
        "Quarters to analyze",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of quarterly filings to ingest"
    )

    analyze_button = st.button(
        "Analyze Company",
        type="primary",
        use_container_width=True
    )

    if analyze_button and ticker_input:
        if ticker_input != st.session_state.current_ticker:
            # Reset chat history for new ticker
            st.session_state.messages = []
            st.session_state.current_ticker = ticker_input
            st.session_state.metrics = None
            st.session_state.quarterly_data = None

        # Check if data exists
        if not check_data_exists(ticker_input):
            ingest_ticker(ticker_input, quarters)

        # Load financial metrics
        with st.spinner(f"Loading live financial data for {ticker_input}..."):
            metrics, quarterly = load_financial_data(ticker_input)

        if metrics:
            st.session_state.metrics = metrics
        else:
            st.warning("Live market data temporarily unavailable. "
                      "Try again in a few minutes.")

        if quarterly:
            st.session_state.quarterly_data = quarterly

    st.markdown("---")

    # Example questions
    st.markdown("### Example Questions")
    example_questions = [
        "What was the most recent quarterly revenue?",
        "Is this a good investment right now?",
        "What are the main risk factors?",
        "How has revenue trended recently?",
        "What is the current P/E ratio?",
        "Which segment is driving growth?"
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True, key=q):
            if st.session_state.current_ticker:
                full_question = (
                    f"{st.session_state.current_ticker}: {q}"
                )
                # Add to messages
                st.session_state.messages.append({
                    "role": "user",
                    "content": q
                })
                # Process through agent immediately
                with st.spinner("Analyzing..."):
                    try:
                        result = run_agent(full_question)
                        answer = result["answer"]
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {e}"
                        })
                st.rerun()
            else:
                st.warning("Please analyze a company first!")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool analyzes SEC filings and live financial
    data to answer questions about public companies.

    **Data Sources:**
    - SEC EDGAR (official filings)
    - Yahoo Finance (live metrics)
    - Web search (latest news)

    **Built with:**
    - Claude API (Anthropic)
    - LangChain agents
    - ChromaDB vector store
    """)


# ── MAIN CONTENT ─────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Earnings Call Analyzer</div>',
           unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered financial analysis '
           'using SEC filings, live data, and web search</div>',
           unsafe_allow_html=True)

# Show metrics dashboard if company is loaded
if st.session_state.current_ticker and (
    st.session_state.metrics
    or st.session_state.quarterly_data
):
    metrics = st.session_state.metrics
    ticker = st.session_state.current_ticker

    st.markdown(f"### {metrics.get('company_name', ticker)} "
               f"({ticker})")

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        price = metrics.get("current_price", "N/A")
        st.metric("Current Price",
                 f"${price}" if price != "N/A" else "N/A")

    with col2:
        mktcap = metrics.get("market_cap", "N/A")
        if mktcap != "N/A":
            mktcap = f"${float(mktcap)/1e12:.2f}T"
        st.metric("Market Cap", mktcap)

    with col3:
        pe = metrics.get("pe_ratio", "N/A")
        st.metric("P/E Ratio",
                 f"{float(pe):.1f}x" if pe != "N/A" else "N/A")

    with col4:
        margin = metrics.get("profit_margin", "N/A")
        st.metric("Profit Margin",
                 f"{float(margin)*100:.1f}%"
                 if margin != "N/A" else "N/A")

    with col5:
        rec = metrics.get("recommendation", "N/A")
        if rec != "N/A":
            # Convert STRONG_BUY → Strong Buy
            rec_formatted = rec.replace("_", " ").title()
        else:
            rec_formatted = "N/A"
        st.metric("Analyst Rec", rec_formatted)

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.quarterly_data:
            fig = create_revenue_chart(
                st.session_state.quarterly_data
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = create_margins_chart(metrics)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

        # Additional metrics
        st.markdown("**Key Metrics**")
        target = metrics.get("analyst_target", "N/A")
        high = metrics.get("52_week_high", "N/A")
        low = metrics.get("52_week_low", "N/A")
        div = metrics.get("dividend_yield", "N/A")

        if target != "N/A":
            st.write(f"Analyst Target: **${float(target):.2f}**")
        if high != "N/A":
            st.write(f"52W High: **${float(high):.2f}**")
        if low != "N/A":
            st.write(f"52W Low: **${float(low):.2f}**")
        if div != "N/A":
            st.write(f"Dividend Yield: **{float(div):.2f}%**")

    st.markdown("---")

# Chat interface
st.markdown("### Ask Anything About This Company")

if not st.session_state.current_ticker:
    st.info("Enter a stock ticker in the sidebar and click "
           "'Analyze Company' to get started.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(
        f"Ask about {st.session_state.current_ticker}..."
    ):
        # Add user message
        full_prompt = (f"{st.session_state.current_ticker}: "
                      f"{prompt}"
                      if st.session_state.current_ticker
                      not in prompt
                      else prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    result = run_agent(full_prompt)
                    answer = result["answer"]

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    logger.info(f"Chat response generated | "
                               f"ticker={st.session_state.current_ticker}")

                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    logger.error(error_msg)

    # Clear chat button
    if st.session_state.messages:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# ── ADMIN: Pre-ingestion (hidden page) ──────────────────────────
if st.query_params.get("admin") == "true":
    st.markdown("---")
    st.markdown("### Admin: Pre-embed Companies")

    if st.button("Pre-embed All Blue Chip Companies"):
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "BAC", "GS",
            "JNJ", "UNH", "WMT", "KO", "PG", "BA", "CAT"
        ]

        progress = st.progress(0)
        status = st.empty()

        for i, ticker in enumerate(tickers):
            status.info(f"Ingesting {ticker}... ({i+1}/{len(tickers)})")
            try:
                if not check_data_exists(ticker):
                    embed_company(ticker, quarters=8)
                    status.success(f" {ticker} ingested")
                else:
                    status.success(f" {ticker} already exists")
            except Exception as e:
                status.error(f" {ticker} failed: {e}")
            progress.progress((i + 1) / len(tickers))

        st.success("Pre-embedding complete!")
