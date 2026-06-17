# Antinomai — Institutional Multi-Agent Research Platform

> **Antinomai** is a production-grade, intent-driven investment research platform built on a decoupled Multi-Agent System (MAS) using LangGraph. Instead of a single monolithic LLM call, an Orchestrator instantly triages every user query into one of five optimised execution pipelines — each with its own model configuration, toolset, and safety logic.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finai-pk.streamlit.app)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2%2B-purple)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [The Five Execution Routes](#the-five-execution-routes)
- [Tech Stack](#tech-stack)
- [Key Safety Systems](#key-safety-systems)
- [User Flow](#user-flow)
- [Folder Structure](#folder-structure)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud)
- [Design Decisions](#design-decisions)

---

## Overview

Most LLM-based research tools route every query through the same heavyweight pipeline — wasting latency and compute on simple questions while under-investing in complex ones. Antinomai fixes this with a **router-first architecture**:

```
User Query
    │
    ▼
┌──────────────────┐
│   Orchestrator   │  ← Groq Llama-3.3-70b, temperature=0, structured output
│  (Intent Router) │
└────────┬─────────┘
         │
   ┌─────┴──────┬──────────┬─────────────┬────────────┐
   ▼            ▼          ▼             ▼            ▼
 Debate       Vision     Trivia     Fundamental   Portfolio
 Panel        Node       Node          Node         Node
(Route 1)  (Route 2)  (Route 3)    (Route 4)    (Route 5)
```

Each route has its own agent configuration, tools, model temperature, and output format — so the system is both **fast** (trivia is one LLM call) and **thorough** (fundamental analysis runs multi-stage synthesis).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT FRONTEND                          │
│   Controls Row  │  Chart Uploader Expander  │  Chat Interface       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  HumanMessage (text or multimodal)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH STATE MACHINE                        │
│                                                                     │
│  ApplicationState (TypedDict)                                       │
│  ├── messages:         List[BaseMessage]                            │
│  ├── next_destination: str                                          │
│  ├── user_input_type:  str                                          │
│  └── portfolio_report: str                                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ORCHESTRATOR NODE                         │   │
│  │  Model: Llama-3.3-70b-versatile  │  Temp: 0.0               │   │
│  │  Output: IntentRoute (Pydantic)  │  Retries: 3              │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                           │
│      ┌──────────────────┼──────────────────────┐                   │
│      │          ┌───────┤───────┐               │                   │
│      ▼          ▼       ▼       ▼               ▼                   │
│  ┌────────┐ ┌──────┐ ┌──────┐ ┌──────────┐ ┌─────────┐           │
│  │  Bull  │ │Vision│ │Trivia│ │Fundament.│ │Portfolio│           │
│  │ Agent  │ │ Node │ │ Node │ │   Node   │ │  Node   │           │
│  └───┬────┘ └──────┘ └──────┘ └──────────┘ └─────────┘           │
│      ▼                                                              │
│  ┌────────┐                                                         │
│  │  Bear  │                                                         │
│  │ Agent  │                                                         │
│  └───┬────┘                                                         │
│      ▼                                                              │
│  ┌─────────┐                                                        │
│  │  Judge  │                                                        │
│  │  Agent  │                                                        │
│  └────┬────┘                                                        │
│       └──────────────────────────────────────────► END             │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │   Groq Cloud     │           │   Tavily Search  │
    │ Llama-3.3-70b    │           │  (live web data) │
    │ Llama-4-Scout    │           └──────────────────┘
    │ (vision)         │
    └──────────────────┘
              │
    ┌──────────────────┐
    │  yfinance / NSE  │
    │  (market data)   │
    └──────────────────┘
```

### State Management

- **`@st.cache_resource`** — The LangGraph compiled app and `MemorySaver` are instantiated once per server lifetime, not re-created on every Streamlit rerun. This prevents checkpoint loss between interactions.
- **Per-message `thread_id`** — Each query gets a unique `f"ant_{session_id}_{msg_count}"` thread ID, preventing state bleed across unrelated queries in the same session.
- **Session isolation** — `session_id` is a `hashlib.md5` of the session start timestamp, scoped to the browser tab.

---

## The Five Execution Routes

### Route 1 — Debate Panel
**Trigger:** "Should I buy/sell/hold [STOCK]?"

An adversarial three-agent pipeline designed to eliminate LLM sycophancy:

```
Bull Agent  ──► Bear Agent  ──► Judge Agent
(Temp: 0.4)    (Temp: 0.4)     (Temp: 0.0)
    │               │                │
    └── get_company_metrics (Tavily) ─┘
```

- **Bull Agent** — Fetches live metrics via Tavily, builds a 2-paragraph data-backed bullish thesis. Every claim must cite a specific figure.
- **Bear Agent** — Reads the Bull thesis, fetches countering data, directly refutes each bullish claim with cited figures.
- **Judge Agent** — Issues a structured CIO Verdict with explicit **sycophancy penalty**: any claim without a specific data point (P/E ratio, named analyst, dated figure) is rated "Weak". Outputs `STRONG BUY / BUY / HOLD / SELL / STRONG SELL`.
- **Optional:** Live Sentiment Cross-check toggle adds a Tavily social/analyst sentiment search layer to both Bull and Bear before they finalise.

---

### Route 2 — Vision / Technical Analysis
**Trigger:** User uploads a chart image and asks for analysis.

```
Uploaded Image (base64)  ──►  Llama-4-Scout-17B (multimodal)
                                      │
                         Fallback guard: len(response) < 60 chars
                                      │
                    "Unresolved Technical Trend" message
```

- Uses **Llama-4-Scout-17b-16e-instruct** (Meta's vision model via Groq).
- Image is base64-encoded and sent as `image_url` content block.
- Provides: Overall Trend, Support/Resistance levels, Chart Patterns, Technical Outlook.
- **Hallucination guard:** If response is under 60 characters (model gave up), it returns a standardised fallback rather than a fabricated analysis.

---

### Route 3 — Trivia / Quick Lookups
**Trigger:** Definitions, quick price checks, general finance questions.

```
User Query  ──►  Llama-3.3-70b (tool_use)
                       │
              get_stock_price tool?
                 yes ──┘
                       ▼
               yfinance → Tavily fallback
```

- Bypasses all agent loops — single LLM call with optional tool use.
- Minimises latency and API cost for low-complexity queries.
- `get_stock_price` tool: tries `yfinance` first, falls back to Tavily web search if no data.

---

### Route 4 — Fundamental Analysis
**Trigger:** "Deep dive on [COMPANY]", "Fundamental analysis of [TICKER]"

```
User Query
    │
    ▼
Ticker Extraction (LLM)
    │
    ▼
clean_ticker() Safety Net ──► 3-stage pipeline:
    │                          1. Known NSE list check
    │                          2. Auto .NS suffix
    │                          3. LLM inference fallback
    ▼
yfinance.info pull
    │
    ▼
Anti-Hallucination Short-Circuit
(checks 6 fields: P/E, Revenue, Cash,
 Gross Margins, Volume, Market Cap)
 < 2 valid fields? → ABORT with explanation
    │
    ▼
Tavily News Cross-verification
    │
    ▼
Groq Llama-3.3-70b Synthesis (Temp: 0.2)
    │
    ▼
3-Paragraph Analyst Report
(Valuation | Solvency | Forward Outlook)
```

- **Anti-hallucination short-circuit:** Checks 6 core yfinance fields before allowing any LLM synthesis. Fewer than 2 valid → returns an explicit data failure message instead of fabricated analysis.
- **Ticker safety net:** `clean_ticker()` runs a 3-stage resolution pipeline so user typos and informal company names are handled gracefully.

---

### Route 5 — Portfolio Risk Engine
**Trigger:** User lists 2+ stocks with share counts.

```
User Input (natural language holdings)
    │
    ▼
PortfolioExtraction (Pydantic structured output)
    │
    ▼
Per-Asset Python/Pandas Sandbox
    ├── yfinance price fetch
    ├── Beta vs benchmark (^GSPC for USD, ^NSEI for INR)
    ├── Annualised Volatility (30-day rolling, sqrt(252) annualised)
    ├── Portfolio weight calculation
    └── Live USD/INR FX rate (Tavily → static 84.0 fallback)
    │
    ▼
Immutable Math Payload (string, no LLM involvement)
    │
    ▼
Groq Llama-3.3-70b Reasoning Layer (Temp: 0.1)
    │
    ▼
3-Paragraph Risk Commentary + Holdings Dataframe
```

- **Zero math in LLM prompts** — all Beta, volatility, and weight calculations are done in a pure Python/Pandas/NumPy sandbox. The LLM only provides narrative reasoning over the verified numbers.
- **Auto-benchmarking** — `.NS` suffix → Nifty 50 (`^NSEI`); otherwise → S&P 500 (`^GSPC`).
- **Concentration alerts** — flags any position > 40% of portfolio.
- **FX handling** — live USD/INR via Tavily with static fallback; all values normalised to USD for comparison.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit 1.58+ | Chat interface, file uploader, controls |
| **Agent Orchestration** | LangGraph 1.2+ | Stateful multi-agent graph with checkpointing |
| **LLM Provider** | Groq Cloud | Ultra-low latency inference |
| **Text Model** | Llama-3.3-70b-versatile | Orchestrator, Bull/Bear/Judge, Fundamental, Portfolio |
| **Vision Model** | Llama-4-Scout-17b-16e-instruct | Chart image analysis (Route 2) |
| **LangChain Core** | langchain-core 1.4+ | Message types, tool decorators, structured output |
| **LangChain Groq** | langchain-groq 1.1+ | Groq LLM wrapper with `.bind_tools()` |
| **Web Search** | Tavily Python 0.7+ | Live financial news, sentiment, price cross-checks |
| **Market Data** | yfinance 1.4+ | Stock prices, fundamentals, historical OHLCV |
| **Data Processing** | Pandas 3.0+, NumPy 2.4+ | Beta calculation, volatility, portfolio weights |
| **Schema Validation** | Pydantic 2.13+ | Structured LLM outputs (routing, portfolio extraction) |
| **Memory** | LangGraph MemorySaver | In-process graph state checkpointing |
| **Python** | 3.11+ | Runtime |

---

## Key Safety Systems

### 1. `clean_ticker()` — 3-Stage Safety Net
```python
Stage 1: Known NSE list  (50+ major Indian tickers, instant lookup)
Stage 2: .NS auto-suffix  (if no dot present and looks like Indian stock)
Stage 3: LLM inference    (Groq call as last resort for ambiguous inputs)
```
Prevents the LLM from hallucinating tickers or formatting them incorrectly for yfinance.

### 2. Anti-Hallucination Short-Circuit (Fundamental Node)
```python
critical_fields = [trailingPE, totalRevenue, totalCash,
                   grossMargins, volume, marketCap]
if valid_count < 2:
    → ABORT: return explicit data failure message
    → Never invoke LLM synthesis over empty data
```

### 3. Sycophancy Penalty (Judge Agent)
The Judge Agent system prompt explicitly instructs:
> *"Any claim without a specific data point (exact P/E, named analyst, specific %, dated figure) must be rated 'Weak'. Do not reward vague sentiment-based arguments."*

### 4. Vision Hallucination Guard
If the vision model's response is under 60 characters, the system returns a standardised `"Unresolved Technical Trend"` message rather than a potentially fabricated short analysis.

### 5. UTF-8 Console Fix
```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```
Prevents `charmap codec` crashes on Windows cp1252 consoles when logging contains Unicode characters.

### 6. `@st.cache_resource` Graph Compilation
The LangGraph app and `MemorySaver` are wrapped in `@st.cache_resource` so they survive Streamlit reruns. Without this, the in-memory checkpoint is wiped on every rerun.

---

## User Flow

```
┌─────────────────────────────────────────────────────┐
│                   USER OPENS APP                    │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │   Controls Row visible   │
              │  • Sentiment toggle      │
              │  • Route status          │
              │  • Clear Chat button     │
              └────────────┬─────────────┘
                           │
           ┌───────────────▼───────────────┐
           │  Chart upload expander        │  ← optional
           │  (for Vision route)           │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │        Types a query          │
           └───────────────┬───────────────┘
                           │
    ┌──────────────────────▼────────────────────────┐
    │               Orchestrator                    │
    │   "Should I buy TCS?"  ──► debate             │
    │   "Analyze this chart" ──► vision             │
    │   "What is P/E ratio?" ──► trivia             │
    │   "Deep dive on Wipro" ──► fundamental        │
    │   "I have 100 TCS, 50 INFY, 30 RELIANCE"     │
    │                        ──► portfolio          │
    └──────────────────────┬────────────────────────┘
                           │
          Runs appropriate agent pipeline
                           │
           ┌───────────────▼───────────────┐
           │    Response renders in chat   │
           │    (with portfolio table      │
           │     if route = portfolio)     │
           └───────────────────────────────┘
```

### Example Queries

| Route | Example Input |
|---|---|
| Debate | *"Should I buy Reliance Industries right now?"* |
| Vision | *(upload chart image)* → *"Analyze this chart"* |
| Trivia | *"What is the current price of TCS?"* or *"Explain what EBITDA means"* |
| Fundamental | *"Give me a deep fundamental analysis of Wipro"* |
| Portfolio | *"I own 100 shares of TCS, 50 of Infosys, and 200 of Reliance. How is my portfolio looking?"* |

---

## Folder Structure

```
finRAG/
│
├── app.py                    # Main application — all agents, graph, and UI
│
├── .streamlit/
│   └── secrets.toml          # LOCAL ONLY — API keys (gitignored)
│
├── pyproject.toml            # Project metadata and dependencies (uv)
├── requirements.txt          # Pip-compatible requirements
├── uv.lock                   # Locked dependency graph
│
├── .python-version           # Python version pin (3.11)
├── .gitignore                # Excludes .venv, secrets.toml, __pycache__
├── .devcontainer/            # VS Code Dev Container config
│
├── image.png                 # Sample chart for testing Vision route
├── test.jpg                  # Sample chart for testing
└── test1.png                 # Sample chart for testing
```

### `app.py` Internal Structure

```
app.py
│
├── Imports & UTF-8 fix (lines 1–25)
│
├── Page config & CSS (lines 27–52)
│
├── API key loading with graceful fallback (lines 54–68)
│
├── State & Schemas
│   ├── ApplicationState (TypedDict)
│   ├── IntentRoute (Pydantic — router output schema)
│   ├── Asset (Pydantic — portfolio asset)
│   └── PortfolioExtraction (Pydantic — structured portfolio)
│
├── clean_ticker() safety net
│
├── LangChain Tools
│   ├── get_stock_price
│   ├── get_company_metrics
│   └── get_sentiment_crosscheck
│
├── Agent Node Functions
│   ├── orchestrator_router
│   ├── trivia_node
│   ├── vision_node
│   ├── bull_agent
│   ├── bear_agent
│   ├── judge_agent
│   ├── fundamental_node
│   └── portfolio_node
│       ├── _get_live_fx_rate()
│       ├── _compute_beta()
│       └── _compute_vol()
│
├── @st.cache_resource _build_app()
│   └── LangGraph StateGraph compilation
│
└── Streamlit UI
    ├── Title & caption
    ├── Session state initialisation
    ├── Controls row (columns)
    ├── Chart upload expander
    ├── Chat history render loop
    ├── Portfolio dataframe (conditional)
    └── Chat input handler → _app.invoke()
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier available)
- A [Tavily API key](https://app.tavily.com) (free tier available)

### Install with `uv` (recommended)

```bash
# Clone the repo
git clone https://github.com/pk-rcb/finAI.git
cd finAI

# Install uv if you don't have it
pip install uv

# Create venv and install all dependencies
uv sync

# Activate the venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Install with `pip`

```bash
git clone https://github.com/pk-rcb/finAI.git
cd finAI
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Add API Keys

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY   = "gsk_your_key_here"
TAVILY_API_KEY = "tvly_your_key_here"
```

Or export as environment variables:

```bash
# Windows PowerShell
$env:GROQ_API_KEY   = "gsk_..."
$env:TAVILY_API_KEY = "tvly_..."

# macOS/Linux
export GROQ_API_KEY="gsk_..."
export TAVILY_API_KEY="tvly_..."
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Environment Variables

| Variable | Required | Source | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) | LLM inference (Llama-3.3 + Llama-4-Scout) |
| `TAVILY_API_KEY` | ✅ Yes | [app.tavily.com](https://app.tavily.com) | Live web search for news, sentiment, FX rates |

No other environment variables are required. Both keys have free tiers sufficient for development and personal use.

---

## Deploying to Streamlit Cloud

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `master`, main file `app.py`
4. Under **Advanced settings → Secrets**, paste:

```toml
GROQ_API_KEY   = "gsk_your_key_here"
TAVILY_API_KEY = "tvly_your_key_here"
```

5. Click **Deploy**

> **Note:** Do not add `.streamlit/secrets.toml` to git — it is gitignored by default in this repo to protect your API keys.

---

## Design Decisions

### Why LangGraph over a simple LLM chain?
LangGraph gives us a **stateful, directed acyclic graph** with checkpointing. This means:
- Each agent in the Debate Panel (Bull → Bear → Judge) automatically receives the accumulated message history from all previous agents.
- State is isolated per `thread_id`, so concurrent queries don't interfere.
- The graph can be extended with new nodes (e.g., a "Macro Analysis" node) without rewriting the plumbing.

### Why Groq?
Sub-second LLM inference. The Orchestrator's routing call completes in ~200-400ms, making the intent detection feel near-instant even before the main agent runs.

### Why separate the math from the LLM (Portfolio node)?
LLMs are unreliable at arithmetic. By computing Beta, volatility, weights, and FX conversion in a pure Python/Pandas sandbox and passing the verified numbers to the LLM as an immutable string payload, we get:
- Mathematically correct outputs every time
- LLM provides only narrative reasoning (what it's actually good at)
- Easy to audit — the payload is logged to console for every run

### Why `@st.cache_resource` for the graph?
Streamlit re-executes the entire `app.py` script on every user interaction (rerun). Without `@st.cache_resource`, `MemorySaver()` and `workflow.compile()` would be called thousands of times per session, recreating in-memory state and losing all checkpoints. Caching ensures the compiled graph is a true singleton per server process.

### Why remove the HITL interrupt?
LangGraph's `interrupt_before` works by saving state to the `MemorySaver` checkpoint, then resuming on the next `invoke(None, ...)` call. In Streamlit's rerun model, even a `@st.cache_resource`-wrapped checkpointer can have subtle resumption issues across rerun boundaries. Replacing the interrupt with a pure Streamlit session-state gate (store the initial state, re-invoke the full graph on approval) is simpler, more debuggable, and more reliable in production.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with LangGraph · Groq · Streamlit · yfinance · Tavily*
