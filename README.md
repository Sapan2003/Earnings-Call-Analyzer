# 📈 Earnings Call Analyzer

> An AI-powered financial research tool that analyzes SEC filings 
> and live market data to answer questions about publicly traded 
> companies — built with RAG, LangChain agents, and Claude API.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-green)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)

---

## What It Does

Ask natural language questions about any publicly traded company 
and get grounded, cited answers backed by real SEC filings and 
live financial data.

**Example queries:**
- *"Has Apple's revenue been growing over the last 6 quarters?"*
- *"What risks did Microsoft's management highlight?"*
- *"Is Tesla a good investment right now?"*
- *"How has Google's cloud segment grown?"*

---

## Architecture
```
User Question
      │
      ▼
┌─────────────────┐     ┌──────────────────┐
│  Streamlit UI   │────▶│  LangChain Agent  │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │ ChromaDB │ │ yFinance │ │  Claude  │
             │  (RAG)   │ │  (Live   │ │   API    │
             │          │ │  Data)   │ │  (LLM)   │
             └──────────┘ └──────────┘ └──────────┘
                    │            │            │
                    └────────────┴────────────┘
                                 │
                    Grounded Answer + Citations
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | Claude Sonnet / Haiku (Anthropic) | Answer generation |
| Orchestration | LangChain + LangGraph | Agent framework |
| Vector Database | ChromaDB | Semantic document storage |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Text vectorization |
| Financial Data | yFinance | Live market metrics |
| SEC Filings | SEC EDGAR API | 10-Q + 10-K documents |
| Backend | FastAPI | REST API endpoints |
| Frontend | Streamlit | Interactive UI |
| Deployment | Streamlit Cloud | Free hosting |
| Logging | Python logging | Full pipeline observability |
| Containerization | Docker | Reproducible environment |

---

## Key Features

**Multi-Source RAG Pipeline**
Combines SEC filing text (10-Q + 10-K), live financial 
metrics from yFinance, and real-time web search into 
one grounded answer.

**LangChain Agent with 3 Tools**
Claude autonomously decides which tools to call based 
on the question — SEC filings for historical analysis, 
yFinance for live metrics, DuckDuckGo for recent news.

**Dynamic Model Routing**
Simple factual questions route to Claude Haiku (fast, cheap). 
Complex analysis routes to Claude Sonnet (powerful). 
Reduces avg response time and cost without sacrificing quality.

**Table-Aware Document Parsing**
Two-pass HTML parser extracts financial tables before 
stripping HTML — preserving revenue figures and segment 
data that naive parsers miss.

**LLM-as-Judge Evaluation**
Uses Claude Haiku as an independent judge to evaluate 
answer quality. More reliable than keyword matching — 
judges meaning not wording.

**Auto-Ingestion**
First query for any company automatically fetches and 
embeds SEC filings. No manual setup required.

**Pre-loaded Blue Chip Companies**
17 major companies pre-embedded at deployment:
AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, 
JPM, BAC, GS, JNJ, UNH, WMT, KO, PG, BA, CAT

---

## Project Structure
```
earnings-call-analyzer/
│
├── ingestion/
│   ├── sec_fetcher.py        # SEC EDGAR API + 10-Q/10-K fetching
│   ├── transcript_parser.py  # Table-aware HTML cleaning + chunking
│   └── embedder.py           # ChromaDB embedding pipeline
│
├── pipeline/
│   ├── retriever.py          # Semantic search over ChromaDB
│   ├── chain.py              # Basic Claude RAG chain
│   ├── agent.py              # LangChain agent with 3 tools
│   └── financial_data.py     # yFinance live data fetcher
│
├── evaluation/
│   └── eval.py               # LLM-as-judge evaluation framework
│
├── app/
│   └── streamlit_app.py      # Streamlit frontend
│
├── api/
│   └── main.py               # FastAPI backend
│
├── scripts/
│   └── preembed.py           # Pre-embed blue chip companies
│
├── utils/
│   └── logger.py             # Centralized logging utility
│
├── logs/                     # Pipeline execution logs
├── data/chroma_db/           # Local vector store
├── .env.example              # Environment variable template
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com))

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Sapan2003/Earnings-analyzer.git
cd Earnings-analyzer
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Add your Anthropic API key to .env
```

**5. Pre-embed blue chip companies**
```bash
python scripts/preembed.py
```

**6. Run the app**
```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Docker
```bash
docker build -t earnings-analyzer .
docker run -p 8501:8501 --env-file .env earnings-analyzer
```

---

## Evaluation Results

Two evaluation approaches were implemented, progressing 
from simple to industry-standard:

### Approach 1 — Keyword Matching (Baseline)
Initial evaluation used keyword presence to check answers.

**Limitation discovered:** Keywords test wording not correctness.
```
Claude: "iPhone revenue was $85,269 million"  → FAIL  
Claude: "iPhone revenue was $85.3 billion"    → PASS
```
Both answers identical — keyword matching failed unfairly.

### Approach 2 — LLM as Judge (Industry Standard)
Replaced keyword matching with Claude Haiku as independent 
judge evaluating meaning, completeness, and accuracy.

**Results:**

| Company | Accuracy | Avg Response | Notes |
|---|---|---|---|
| AAPL | 95% | 33s | Keyword baseline |
| MSFT | 80% | 28s | LLM judge, pre-optimization |
| MSFT | 80% | 12s | LLM judge, post-optimization |
| GOOGL | 55% | 12s | LLM judge |

**Response time improved 61%** after:
- Embedding model caching (`@st.cache_resource`)
- ChromaDB connection pooling
- Dynamic model routing (Haiku vs Sonnet)

**Profitability and valuation questions: 100%** across 
all companies via live yFinance data.

---

## Known Issues & Roadmap

**1. Revenue extraction varies by company (High Priority)**

The pipeline achieves 95% accuracy on Apple and 80% on
Microsoft but drops to 55% on Google due to differences
in SEC filing structure.

Root cause: Each company uses different labels for revenue:
| Company | Revenue Label |
- Apple: "Total net sales"
- Microsoft: "Total revenue"
- Google: "Revenues"

Planned fix: Add company-specific extraction patterns.

**2. Analyst source attribution**

yFinance free tier lacks source attribution for analyst 
consensus data. Production solution: Bloomberg or FactSet API for verified analyst data.

**3. ChromaDB Persistence**

Streamlit Cloud free tier resets storage periodically.
Current workaround: admin page at `/?admin=true` to
re-ingest companies when needed.

Planned fix: Migrate from local ChromaDB to Pinecone
cloud vector database for persistent storage across
deployments.

### Planned Features
- [ ] Company-specific SEC filing parsers for consistent accuracy
- [ ] Migrate to Pinecone for persistent cloud vector storage
- [ ] Earnings call audio transcript integration
- [ ] CEO sentiment scoring over time
- [ ] Competitor cross-analysis (compare 2 tickers)
- [ ] Fine-tune embeddings on financial domain text
- [ ] Bloomberg/FactSet API for verified analyst data
- [ ] Email alerts for significant trend changes

---

## CI/CD Pipeline

GitHub Actions runs automated checks on every push:

| Trigger | Action |
|---|---|
| Every push | Lint check + unit tests |
| Merge to main | Auto-deploy to Streamlit Cloud |

[![CI Pipeline](https://github.com/Sapan2003/Earnings-Call-Analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/Sapan2003/Earnings-Call-Analyzer/actions)

---

## Logging

All pipeline activity logged to `logs/` directory:
```
logs/
├── ingestion.log   # Data fetching + chunking
├── pipeline.log    # Every query and response
└── eval.log        # Evaluation run results
```

Sample output:
```
2026-03-21 10:23:41 | INFO | sec_fetcher | Fetching AAPL filings
2026-03-21 10:23:43 | INFO | sec_fetcher | Fetched 10 filings
2026-03-21 10:24:01 | INFO | agent | Routing to Haiku (simple)
2026-03-21 10:24:04 | INFO | agent | Agent completed | time=3.2s
```

---

## Author

**Sapan Parikh**
- MS Machine Learning Engineering @ Drexel University (GPA: 4.0)
- [LinkedIn](https://www.linkedin.com/in/sapan-parikh13/)
- [GitHub](https://github.com/Sapan2003)
- sapanparikh13@gmail.com

---

## License

MIT License — feel free to use this project as a reference.

---

## Acknowledgements

- [SEC EDGAR](https://www.sec.gov/edgar) — free public financial data
- [Anthropic](https://www.anthropic.com) — Claude API
- [LangChain](https://langchain.com) — agent orchestration
- [ChromaDB](https://www.trychroma.com) — vector database
- [yFinance](https://pypi.org/project/yfinance/) — live market data