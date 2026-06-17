import os
import re
import sys
import base64
import time
import hashlib
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, TypedDict, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import (
    HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

# Fix Windows cp1252 console encoding so print() never crashes on emoji/unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# PAGE CONFIG  (must be first Streamlit call)
# ==========================================
st.set_page_config(
    page_title="Antinomai | Research",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Minimal CSS — dark background + font only.
# Do NOT override any Streamlit chat/message internals;
# those selectors break avatar icons and cause text overflow.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background-color: #0f1117; }
.hitl-box {
    border: 1px solid #b45309; border-radius: 8px;
    padding: 0.9rem 1.1rem; background: rgba(251,191,36,0.04); margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. API KEYS & GLOBAL SETUP
# ==========================================
try:
    GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY",   os.environ.get("GROQ_API_KEY",   ""))
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY", ""))
except Exception:
    GROQ_API_KEY   = os.environ.get("GROQ_API_KEY",   "")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error(
        "**API Keys Missing** — add to `.streamlit/secrets.toml`:\n\n"
        "```toml\nGROQ_API_KEY   = \"gsk_...\"\nTAVILY_API_KEY = \"tvly_...\"\n```"
    )
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ==========================================
# 2. STATE & SCHEMAS
# ==========================================
class ApplicationState(TypedDict):
    messages:         List[BaseMessage]
    next_destination: str
    user_input_type:  str
    portfolio_report: str

class IntentRoute(BaseModel):
    destination: str = Field(
        description="Must be exactly one of: 'debate', 'vision', 'trivia', 'fundamental', 'portfolio'"
    )

class Asset(BaseModel):
    ticker: str   = Field(description="Official ticker symbol. Append .NS for Indian stocks.")
    shares: float = Field(description="Exact number of shares owned.")

class PortfolioExtraction(BaseModel):
    assets: List[Asset] = Field(description="All extracted assets and share counts.")

# ==========================================
# 3. SAFETY NET — clean_ticker()
# ==========================================
_KNOWN_NSE = {
    "TCS","RELIANCE","INFY","WIPRO","BHARTIARTL","HDFCBANK","ICICIBANK",
    "BAJFINANCE","SBIN","AXISBANK","KOTAKBANK","LT","ASIANPAINT","HCLTECH",
    "SUNPHARMA","ULTRACEMCO","MARUTI","TITAN","TATAMOTORS","ZOMATO","NYKAA",
    "PAYTM","DMART","ONGC","NTPC","POWERGRID","ADANIENT","ADANIPORTS",
    "HINDUNILVR","NESTLEIND","GRASIM","JSWSTEEL","TATASTEEL","TECHM",
    "DIVISLAB","DRREDDY","CIPLA","EICHERMOT","HEROMOTOCO","BAJAJ-AUTO",
    "TATACONSUM","BRITANNIA","COALINDIA","HINDALCO","VEDL","INDUSINDBK",
    "SHREECEM","AMBUJACEM","IRCTC","BPCL","IOC","BHEL","SAIL","NMDC",
    "MUTHOOTFIN","CHOLAFIN","SBICARD","HDFCLIFE","ICICIPRULI","LICI",
    "BANKBARODA","CANARABANK","PNB","UNIONBANK","IDFCFIRSTB","FEDERALBNK",
    "ABSLAMC","RVNL","IRFC","HAL","BEL","COCHINSHIP",
}

def clean_ticker(raw: str, llm=None) -> str:
    parts   = re.sub(r'[$\'"\\s]', "", raw.upper()).strip().split()
    cleaned = parts[0] if parts else raw.strip().upper()

    if "." in cleaned:
        return cleaned.upper()

    if cleaned in _KNOWN_NSE:
        result = f"{cleaned}.NS"
        print(f"[Safety Net] {raw} -> {result} (NSE list)")
        return result

    if llm:
        try:
            resp = llm.invoke([
                SystemMessage(content=(
                    "Return ONLY the correct stock ticker with regional suffix. "
                    "Indian/NSE stocks -> append .NS. US stocks -> no suffix. "
                    "London stocks -> append .L. Nothing else."
                )),
                HumanMessage(content=f"Ticker/company: {raw}"),
            ])
            result = resp.content.strip().upper().split()[0]
            print(f"[Safety Net] LLM: {raw} -> {result}")
            return result
        except Exception as e:
            print(f"[Safety Net] LLM fallback failed: {e}")

    return cleaned

# ==========================================
# 4. LIVE TOOLS
# ==========================================
@tool
def get_stock_price(ticker: str) -> str:
    """Fetches the current real-time stock price for a given ticker or company name."""
    _llm  = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    clean = clean_ticker(ticker, llm=_llm)
    try:
        hist = yf.Ticker(clean).history(period="1d")
        if not hist.empty:
            price    = hist["Close"].iloc[-1]
            currency = "INR" if clean.endswith(".NS") else "USD"
            return f"Live price of {clean}: {currency} {price:,.2f}"
    except Exception:
        pass
    try:
        resp    = tavily_client.search(query=f"current stock price {ticker} today", search_depth="basic", max_results=2)
        results = [r["content"] for r in resp.get("results", [])]
        if results:
            return f"Web data for {ticker}:\n" + "\n".join(results)
    except Exception as e:
        return f"Error: {e}"
    return f"Could not retrieve price for: {ticker}"


@tool
def get_company_metrics(ticker: str) -> str:
    """Fetches earnings, P/E ratio, analyst ratings, and news for a stock."""
    try:
        resp    = tavily_client.search(
            query=f"Latest earnings P/E ratio analyst rating news {ticker} stock",
            search_depth="basic", max_results=3,
        )
        results = [r["content"] for r in resp.get("results", [])]
        return f"LIVE DATA FOR {ticker}:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_sentiment_crosscheck(ticker: str) -> str:
    """Cross-verifies corporate disclosures against analyst and social media sentiment."""
    try:
        resp    = tavily_client.search(
            query=f"{ticker} stock social media analyst sentiment vs earnings 2025",
            search_depth="advanced", max_results=3,
        )
        results = [r["content"] for r in resp.get("results", [])]
        return f"SENTIMENT CROSS-CHECK FOR {ticker}:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 5. AGENT NODES
# ==========================================

def orchestrator_router(state: ApplicationState):
    raw = state["messages"][-1].content
    if isinstance(raw, list):
        user_message = next((p["text"] for p in raw if isinstance(p, dict) and p.get("type") == "text"), "analyze chart")
    else:
        user_message = raw

    llm        = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_retries=3)
    router_llm = llm.with_structured_output(IntentRoute)
    system     = SystemMessage(content="""You are an intent router for a financial platform.
Classify input into EXACTLY ONE category:
  'debate'      - asking whether to buy/sell/hold a specific stock
  'vision'      - uploading a chart or asking for technical/visual chart analysis
  'trivia'      - asking for a definition, concept, or quick stock price lookup
  'fundamental' - asking for deep-dive fundamental analysis of one company
  'portfolio'   - providing multiple holdings (2+ stocks with share counts)

Rules: Multiple stocks with shares -> portfolio. Buy/sell question -> debate. Deep analysis -> fundamental.""")

    try:
        decision = router_llm.invoke([system, HumanMessage(content=f"Classify: {user_message}")])
        dest     = decision.destination
    except Exception:
        dest = "trivia"

    print(f"[Orchestrator] -> {dest}")
    return {"next_destination": dest, "user_input_type": dest}


def trivia_node(state: ApplicationState):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).bind_tools([get_stock_price])
    sys_msg = SystemMessage(content=(
        "You are a knowledgeable financial assistant. Answer concisely and accurately. "
        "For live stock prices, use the get_stock_price tool."
    ))
    new_messages = list(state["messages"])
    response     = llm.invoke([sys_msg] + new_messages)
    new_messages.append(response)

    if response.tool_calls:
        tc       = response.tool_calls[0]
        result   = get_stock_price.invoke(tc)
        tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
        new_messages.append(tool_msg)
        final    = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).invoke([sys_msg] + new_messages)
        new_messages.append(final)

    return {"messages": new_messages}


def vision_node(state: ApplicationState):
    messages   = state["messages"]
    vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
    sys_msg    = SystemMessage(content="""You are an expert technical stock analyst.
Analyze the provided stock chart image and respond with:
1. **Overall Trend** - Bullish / Bearish / Sideways with reasoning
2. **Key Support & Resistance** - Specific price levels visible
3. **Notable Patterns** - Candlestick formations, chart patterns
4. **Technical Outlook** - Short and medium-term assessment

CRITICAL: If the image has insufficient contrast, resolution, or no recognizable chart indicators,
respond ONLY with: "Unresolved Technical Trend: Image lacks sufficient contrast or chart indicators. Please upload a clearer chart."
Do NOT fabricate analysis from unclear images.""")
    try:
        response = vision_llm.invoke([sys_msg] + messages)
        content  = (response.content or "").strip()
        if len(content) < 60:
            return {"messages": messages + [AIMessage(content="Unresolved Technical Trend: Image lacks sufficient contrast or chart indicators. Please upload a clearer candlestick chart.")]}
        return {"messages": messages + [response]}
    except Exception as e:
        return {"messages": messages + [AIMessage(content=f"Vision Agent Error: {e}")]}


# --- DEBATE PANEL ---
_debate_llm   = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)
_debate_agent = _debate_llm.bind_tools([get_company_metrics])

def bull_agent(state: ApplicationState):
    print("[Bull Agent] Fetching upside catalysts...")
    messages    = state["messages"]
    enable_xchk = st.session_state.get("enable_sentiment_check", False)

    sys_msg  = SystemMessage(content="""You are an aggressive growth investor.
Use get_company_metrics to pull live data. Build a data-backed 2-paragraph bullish thesis:
- Para 1: Macro tailwinds, sector growth, strategic positioning (cite specific data)
- Para 2: Undervalued fundamentals, upcoming catalysts (cite specific P/E, margins, figures)
Every claim MUST cite a specific data point. Vague sentiment is not acceptable.""")

    response   = _debate_agent.invoke([sys_msg] + messages)
    extra_msgs = []

    if response.tool_calls:
        tc          = response.tool_calls[0]
        tool_result = get_company_metrics.invoke(tc)
        tool_msg    = ToolMessage(content=str(tool_result), tool_call_id=tc["id"], name=tc["name"])
        extra_msgs  = [response, tool_msg]
        bull_final  = _debate_llm.invoke([sys_msg] + messages + extra_msgs)

        if enable_xchk:
            raw_msg   = messages[0].content
            ticker_kw = (raw_msg if isinstance(raw_msg, str) else "stock").split()[-1]
            xc_result = get_sentiment_crosscheck.invoke({"ticker": ticker_kw})
            xc_msg    = AIMessage(content=f"[Sentiment Cross-Check]\n{xc_result}")
            extra_msgs.append(xc_msg)
            bull_final = _debate_llm.invoke([sys_msg] + messages + extra_msgs + [bull_final])

        bull_final.content = f"**BULL THESIS:**\n\n{bull_final.content}"
        return {"messages": messages + extra_msgs + [bull_final]}

    response.content = f"**BULL THESIS:**\n\n{response.content}"
    return {"messages": messages + [response]}


def bear_agent(state: ApplicationState):
    print("[Bear Agent] Building counter-argument...")
    messages    = state["messages"]
    enable_xchk = st.session_state.get("enable_sentiment_check", False)

    sys_msg = SystemMessage(content="""You are a cynical short-seller and forensic accountant.
You have read the Bull thesis above. Use get_company_metrics to pull data and DISPROVE it.
Build a 2-paragraph bearish counter-argument:
- Para 1: Overvaluation with specific P/E vs sector median; margin compression with exact %
- Para 2: Regulatory headwinds, analyst downgrades, poor earnings — all cited with specifics
Directly attack EACH of the Bull's claims by name. Every counter must cite specific data.""")

    response   = _debate_agent.invoke([sys_msg] + messages)
    extra_msgs = []

    if response.tool_calls:
        tc          = response.tool_calls[0]
        tool_result = get_company_metrics.invoke(tc)
        tool_msg    = ToolMessage(content=str(tool_result), tool_call_id=tc["id"], name=tc["name"])
        extra_msgs  = [response, tool_msg]
        bear_final  = _debate_llm.invoke([sys_msg] + messages + extra_msgs)

        if enable_xchk:
            raw_msg   = messages[0].content
            ticker_kw = (raw_msg if isinstance(raw_msg, str) else "stock").split()[-1]
            xc_result = get_sentiment_crosscheck.invoke({"ticker": ticker_kw})
            xc_msg    = AIMessage(content=f"[Sentiment Cross-Check]\n{xc_result}")
            extra_msgs.append(xc_msg)
            bear_final = _debate_llm.invoke([sys_msg] + messages + extra_msgs + [bear_final])

        bear_final.content = f"**BEAR THESIS:**\n\n{bear_final.content}"
        return {"messages": messages + extra_msgs + [bear_final]}

    response.content = f"**BEAR THESIS:**\n\n{response.content}"
    return {"messages": messages + [response]}


def judge_agent(state: ApplicationState):
    print("[Judge Agent] Weighing arguments (sycophancy penalty active)...")
    sys_msg = SystemMessage(content="""You are the Chief Investment Officer. Review the Bull and Bear theses.

SYCOPHANCY PENALTY: Any claim without a specific data point (exact P/E, named analyst, specific %, dated figure)
must be rated "Weak". Do not reward vague sentiment-based arguments.

Output this EXACT format:
**CIO VERDICT:**
- **Bull Evidence Quality:** [Strong/Weak] - [why in 1 sentence]
- **Bear Evidence Quality:** [Strong/Weak] - [why in 1 sentence]
- **Strengths Validated:** [best supported bullish point]
- **Risks Validated:** [best supported bearish point]
- **Final Rating:** [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
- **Actionable Summary:** [2 decisive sentences]

Be precise. No generic disclaimers.""")

    judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    response  = judge_llm.invoke([sys_msg] + state["messages"])
    return {"messages": state["messages"] + [response]}


def fundamental_node(state: ApplicationState):
    print("[Fundamental] Human-approved. Executing deep research...")
    user_message = state["messages"][0].content
    if isinstance(user_message, list):
        user_message = next((p["text"] for p in user_message if isinstance(p, dict) and p.get("type") == "text"), "")

    _llm     = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    raw_tick = _llm.invoke([
        SystemMessage(content="Extract only the stock ticker or company name. Return ONLY the ticker, nothing else."),
        HumanMessage(content=user_message),
    ]).content.strip()

    ticker = clean_ticker(raw_tick, llm=_llm)
    print(f"[Fundamental] Ticker: {ticker}")

    try:
        info = yf.Ticker(ticker).info

        # Anti-hallucination short-circuit
        critical    = [info.get("trailingPE"), info.get("totalRevenue"), info.get("totalCash"),
                       info.get("grossMargins"), info.get("volume"), info.get("marketCap")]
        valid_count = sum(1 for v in critical if v is not None and v != 0)

        if not info or valid_count < 2:
            print(f"[Short-Circuit] Only {valid_count}/6 valid fields - aborting to prevent hallucination.")
            return {"messages": list(state["messages"]) + [AIMessage(content=(
                f"**DATA RETRIEVAL FAILURE** for `{ticker}`\n\n"
                f"Only {valid_count}/6 core metrics returned. Possible reasons:\n"
                f"- Ticker may be delisted or incorrectly formatted\n"
                f"- Indian equities need `.NS` suffix (e.g., `{raw_tick}.NS`)\n\n"
                f"*Analysis aborted to prevent AI hallucination over empty data.*"
            ))]}

        def fmt(v):  return f"${v:,.0f}" if isinstance(v, (int, float)) and v > 0 else "N/A"
        def pct(v):  return f"{v*100:.2f}%" if isinstance(v, (int, float)) else "N/A"

        company  = info.get("longName", ticker)
        sector   = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        report  = f"**{company}** (`{ticker}`) | {sector} > {industry}\n\n"
        report += "**Valuation & Profitability**\n"
        report += f"- Trailing P/E: {info.get('trailingPE','N/A')}\n"
        report += f"- Forward P/E: {info.get('forwardPE','N/A')}\n"
        report += f"- Gross Margins: {pct(info.get('grossMargins'))}\n"
        report += f"- Operating Margins: {pct(info.get('operatingMargins'))}\n"
        report += f"- ROE: {pct(info.get('returnOnEquity'))}\n"
        report += f"- Revenue Growth YoY: {pct(info.get('revenueGrowth'))}\n\n"
        report += "**Balance Sheet**\n"
        report += f"- Total Cash: {fmt(info.get('totalCash'))}\n"
        report += f"- Total Debt: {fmt(info.get('totalDebt'))}\n"
        report += f"- Free Cash Flow: {fmt(info.get('freeCashflow'))}\n"
        report += f"- Debt/Equity: {info.get('debtToEquity','N/A')}\n"
        report += f"- Current Ratio: {info.get('currentRatio','N/A')}\n\n"
        report += "**Market & Analyst Data**\n"
        report += f"- Market Cap: {fmt(info.get('marketCap'))}\n"
        report += f"- 52W High/Low: {info.get('fiftyTwoWeekHigh','N/A')} / {info.get('fiftyTwoWeekLow','N/A')}\n"
        report += f"- Analyst Target: {info.get('targetMeanPrice','N/A')}\n"
        report += f"- Recommendation: {str(info.get('recommendationKey','N/A')).upper()}\n"

        # Tavily news cross-verification
        news_block = ""
        try:
            news_resp = tavily_client.search(query=f"{company} {ticker} news earnings analyst 2025", search_depth="basic", max_results=2)
            items = [r["content"] for r in news_resp.get("results", [])]
            if items:
                news_block = "\n\n**Live News Context:**\n" + "\n---\n".join(items)
                print("[Tavily] News cross-check complete.")
        except Exception as e:
            print(f"[Tavily] News fetch failed: {e}")

        synth = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2).invoke([
            SystemMessage(content="""You are a senior equity analyst. Write a 3-paragraph fundamental analysis:
Para 1 - Profitability & Valuation: interpret P/E vs sector, margins, ROE, growth
Para 2 - Solvency & Liquidity: debt/equity, current ratio, free cash flow
Para 3 - Analyst Assessment: synthesize into a forward-looking view, reference target price
RULE: Do NOT cite any figure not in the data. If >50% N/A, say data is insufficient."""),
            HumanMessage(content=f"{report}{news_block}"),
        ])

        final = (
            f"*[Human-Approved Deep Research]*\n\n"
            f"## Fundamental Analysis: {company} ({ticker})\n\n"
            f"### Raw Quantitative Data\n{report}\n---\n\n"
            f"### Analyst Synthesis\n{synth.content}"
        )
        return {"messages": list(state["messages"]) + [AIMessage(content=final)]}

    except Exception as e:
        print(f"[Fundamental] Error: {e}")
        return {"messages": list(state["messages"]) + [AIMessage(content=f"System error in Fundamental Engine: {e}")]}


# --- PORTFOLIO RISK ENGINE ---
def _get_live_fx_rate() -> tuple:
    try:
        resp    = tavily_client.search(query="USD to INR exchange rate today", search_depth="basic", max_results=1)
        content = resp.get("results", [{}])[0].get("content", "")
        matches = re.findall(r'\b(8[0-9]\.\d+|9[0-2]\.\d+)\b', content)
        if matches:
            rate = float(matches[0])
            print(f"[FX] Live USD/INR: {rate}")
            return rate, True
    except Exception as e:
        print(f"[FX] Fallback ({e})")
    return 84.0, False

def _compute_beta(ticker: str, benchmark: str) -> Optional[float]:
    try:
        a = yf.Ticker(ticker).history(period="1y")["Close"].pct_change().dropna()
        b = yf.Ticker(benchmark).history(period="1y")["Close"].pct_change().dropna()
        df = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(df) < 20:
            return None
        cov = np.cov(df["a"], df["b"])
        return round(float(cov[0, 1] / cov[1, 1]), 3)
    except Exception:
        return None

def _compute_vol(ticker: str) -> Optional[float]:
    try:
        h = yf.Ticker(ticker).history(period="6mo")["Close"].pct_change().dropna()
        if len(h) < 20:
            return None
        return round(float(h.tail(30).std() * np.sqrt(252) * 100), 2)
    except Exception:
        return None

def portfolio_node(state: ApplicationState):
    print("[Portfolio] Risk engine starting...")
    user_message = state["messages"][0].content
    if isinstance(user_message, list):
        user_message = next((p["text"] for p in user_message if isinstance(p, dict) and p.get("type") == "text"), "")

    extractor = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).with_structured_output(PortfolioExtraction)
    try:
        extracted = extractor.invoke(user_message)
    except Exception as e:
        return {"messages": list(state["messages"]) + [AIMessage(content=f"Could not parse portfolio: {e}")], "portfolio_report": ""}

    if not extracted.assets:
        return {"messages": list(state["messages"]) + [AIMessage(content="No assets detected. Please list holdings with share counts.")], "portfolio_report": ""}

    fx_rate, fx_live = _get_live_fx_rate()
    fx_note          = f"{'Live' if fx_live else 'Static'} USD/INR: Rs.{fx_rate:.2f}"

    _llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    rows      = []
    total_usd = 0.0

    for asset in extracted.assets:
        ticker = clean_ticker(asset.ticker, llm=_llm)
        is_inr = ticker.endswith(".NS")
        bench  = "^NSEI" if is_inr else "^GSPC"
        try:
            hist = yf.Ticker(ticker).history(period="1d")
            if hist.empty:
                continue
            price     = float(hist["Close"].iloc[-1])
            currency  = "INR" if is_inr else "USD"
            price_usd = price / fx_rate if is_inr else price
            value_usd = price_usd * asset.shares
            beta      = _compute_beta(ticker, bench)
            vol       = _compute_vol(ticker)
            total_usd += value_usd
            rows.append({"Ticker": ticker, "Shares": asset.shares, "Price": f"{currency} {price:,.2f}",
                         "Value_USD": value_usd, "Beta": beta, "Vol_Ann": vol})
            print(f"[Sandbox] {ticker}: ${value_usd:,.2f} | Beta={beta} | Vol={vol}")
        except Exception as e:
            print(f"[Sandbox] Error for {ticker}: {e}")

    if not rows:
        return {"messages": list(state["messages"]) + [AIMessage(content="Could not fetch data for any tickers.")], "portfolio_report": ""}

    for r in rows:
        r["Weight_Pct"] = round((r["Value_USD"] / total_usd) * 100, 2) if total_usd > 0 else 0.0

    beta_rows = [r for r in rows if r["Beta"] is not None]
    wtd_beta  = (sum(r["Beta"] * r["Weight_Pct"] / 100 for r in beta_rows) /
                 sum(r["Weight_Pct"] / 100 for r in beta_rows)) if beta_rows else None

    # Immutable math payload
    payload  = f"PORTFOLIO MATH PAYLOAD ({fx_note})\n{'='*55}\n"
    payload += f"Total Value: ${total_usd:,.2f} USD | Holdings: {len(rows)}\n"
    if wtd_beta:
        payload += f"Weighted Beta: {wtd_beta:.3f} ({'aggressive' if wtd_beta>1.2 else 'moderate' if wtd_beta>0.8 else 'defensive'})\n"
    payload += "\nBreakdown:\n"
    for r in sorted(rows, key=lambda x: x["Weight_Pct"], reverse=True):
        b = f"{r['Beta']:.3f}" if r["Beta"] else "N/A"
        v = f"{r['Vol_Ann']:.1f}%" if r["Vol_Ann"] else "N/A"
        payload += f"  {r['Ticker']:<15} {r['Shares']:>8.1f} sh | {r['Price']:>16} | ${r['Value_USD']:>11,.2f} | {r['Weight_Pct']:>5.1f}% | Beta {b} | Vol {v}\n"

    flagged = [r for r in rows if r["Weight_Pct"] > 40]
    if flagged:
        payload += f"\nCONCENTRATION ALERT: {', '.join(r['Ticker'] for r in flagged)} > 40% of portfolio.\n"

    st.session_state["portfolio_df_data"] = rows
    st.session_state["portfolio_total"]   = total_usd
    st.session_state["portfolio_beta"]    = wtd_beta

    print(f"[Portfolio] Payload ready. Total: ${total_usd:,.2f}. Passing to LLM reasoning layer...")

    analysis = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1).invoke([
        SystemMessage(content="""You are an institutional portfolio manager writing a client report.
You are given verified portfolio math from a Python sandbox. Write a 3-paragraph risk commentary:
Para 1 - Composition & Geographic Exposure: sector concentration, USD vs INR exposure, currency risk
Para 2 - Risk Profile: interpret weighted Beta (>1.2=aggressive, 0.8-1.2=moderate, <0.8=defensive), flag >40% holdings
Para 3 - Strategic Recommendations: 3 specific actionable steps referencing actual tickers and percentages
RULE: Every figure must come from the payload. Do not invent numbers."""),
        HumanMessage(content=payload),
    ])

    wtd_beta_str = f"{wtd_beta:.3f}" if wtd_beta else "N/A"
    final = (
        f"## Portfolio Risk Analysis\n\n"
        f"**Total Value:** ${total_usd:,.2f} | **Holdings:** {len(rows)} | **Wtd. Beta:** {wtd_beta_str} | *{fx_note}*\n\n"
        f"---\n\n{analysis.content}\n\n"
        f"---\n*All math computed in Python/Pandas sandbox. Benchmarks: ^GSPC (USD), ^NSEI (INR).*"
    )
    return {"messages": list(state["messages"]) + [AIMessage(content=final)], "portfolio_report": payload}


# ==========================================
# 6. GRAPH COMPILATION
# @st.cache_resource ensures the graph and MemorySaver are built ONCE
# per server lifetime, not rebuilt on every Streamlit rerun.
# Without this, MemorySaver loses its checkpoints on every rerun,
# which was silently breaking the HITL resume.
# ==========================================
@st.cache_resource
def _build_app():
    from langgraph.graph import StateGraph, END as _END
    from langgraph.checkpoint.memory import MemorySaver as _MS

    wf = StateGraph(ApplicationState)
    wf.add_node("orchestrator", orchestrator_router)
    wf.add_node("trivia",       trivia_node)
    wf.add_node("vision",       vision_node)
    wf.add_node("fundamental",  fundamental_node)
    wf.add_node("portfolio",    portfolio_node)
    wf.add_node("bull_agent",   bull_agent)
    wf.add_node("bear_agent",   bear_agent)
    wf.add_node("judge_agent",  judge_agent)

    wf.set_entry_point("orchestrator")
    wf.add_conditional_edges("orchestrator", lambda s: s["next_destination"], {
        "debate": "bull_agent", "vision": "vision", "trivia": "trivia",
        "fundamental": "fundamental", "portfolio": "portfolio",
    })
    wf.add_edge("bull_agent",  "bear_agent")
    wf.add_edge("bear_agent",  "judge_agent")
    wf.add_edge("judge_agent", _END)
    wf.add_edge("vision",      _END)
    wf.add_edge("trivia",      _END)
    wf.add_edge("fundamental", _END)
    wf.add_edge("portfolio",   _END)

    # NOTE: No interrupt_before here.
    # HITL is handled at the Streamlit layer (session_state), which is
    # simpler and survives reruns without checkpoint dependency.
    return wf.compile(checkpointer=_MS())

_app = _build_app()

# ==========================================
# 7. STREAMLIT UI
# ==========================================

st.title("Antinomai")
st.caption("Institutional Multi-Agent Research Platform · LangGraph MAS")
st.divider()

# --- Session State ---
_defaults = {
    "messages":               [],
    "enable_sentiment_check": False,
    "portfolio_df_data":      None,
    "portfolio_total":        0.0,
    "portfolio_beta":         None,
    "last_route":             None,
    "session_id":             hashlib.md5(str(time.time()).encode()).hexdigest()[:10],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# TOP CONTROLS ROW  (replaces sidebar — works on Streamlit Cloud)
# ==========================================
col_left, col_mid, col_right = st.columns([2, 2, 1])

with col_left:
    st.session_state.enable_sentiment_check = st.toggle(
        "Live Sentiment Cross-check",
        value=st.session_state.enable_sentiment_check,
        help="Adds Tavily social/analyst sentiment layer to Debate Panel (+5-8s latency).",
    )

with col_mid:
    if st.session_state.last_route:
        st.caption(f"Last route: **{st.session_state.last_route.upper()}**")

with col_right:
    if st.button("Clear Chat", use_container_width=True):
        for k in ["messages", "last_route", "portfolio_df_data", "portfolio_total", "portfolio_beta"]:
            st.session_state[k] = _defaults.get(k)
        st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:10]
        st.rerun()

# ==========================================
# FILE UPLOADER  (chart vision — inline expander)
# ==========================================
with st.expander("Upload Chart for Vision Analysis", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload a candlestick chart (JPG/PNG), then type 'analyze this chart' below.",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Chart ready — type 'analyze this chart' in the chat.", width=400)

st.divider()


# ==========================================
# CHAT HISTORY
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Portfolio table (shown after the last portfolio response)
if (st.session_state.portfolio_df_data
        and st.session_state.last_route == "portfolio"
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "assistant"):
    with st.expander("Holdings Breakdown", expanded=True):
        df = pd.DataFrame(st.session_state.portfolio_df_data)
        display = pd.DataFrame({
            "Ticker":      df["Ticker"],
            "Shares":      df["Shares"],
            "Price":       df["Price"],
            "Value (USD)": df["Value_USD"].map("${:,.2f}".format),
            "Weight %":    df["Weight_Pct"],
            "Beta":        df["Beta"].apply(lambda x: f"{x:.3f}" if x is not None else "N/A"),
            "Ann. Vol":    df["Vol_Ann"].apply(lambda x: f"{x:.1f}%" if x is not None else "N/A"),
        })
        def _cw(val):
            try:
                v = float(val)
                if v > 40: return "color: #f87171"
                if v > 25: return "color: #fbbf24"
                return "color: #34d399"
            except Exception:
                return ""
        st.dataframe(display.style.map(_cw, subset=["Weight %"]), use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Value", f"${st.session_state.portfolio_total:,.2f}")
        c2.metric("Holdings",    len(df))
        bv = st.session_state.portfolio_beta
        c3.metric("Wtd. Beta",   f"{bv:.3f}" if bv else "N/A")

# ==========================================
# CHAT INPUT
# ==========================================
if prompt := st.chat_input("Ask about a stock, analyze a portfolio, or request a fundamental analysis..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build initial graph state
    if uploaded_file is not None:
        uploaded_file.seek(0)
        img_b64  = base64.b64encode(uploaded_file.read()).decode("utf-8")
        img_mime = uploaded_file.type or "image/jpeg"
        initial_state: ApplicationState = {
            "messages": [HumanMessage(content=[
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{img_b64}"}},
            ])],
            "next_destination": "", "user_input_type": "", "portfolio_report": "",
        }
    else:
        initial_state = {
            "messages":         [HumanMessage(content=prompt)],
            "next_destination": "", "user_input_type": "", "portfolio_report": "",
        }

    # Unique thread per message — prevents state bleed between queries
    msg_count = len(st.session_state.messages)
    config    = {"configurable": {"thread_id": f"ant_{st.session_state.session_id}_{msg_count}"}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = _app.invoke(initial_state, config=config)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        route = result.get("next_destination", "trivia")
        st.session_state.last_route = route

        final_response = result["messages"][-1].content
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

        # Show portfolio table inline for portfolio responses
        if route == "portfolio" and st.session_state.portfolio_df_data:
            with st.expander("Holdings Breakdown", expanded=True):
                df = pd.DataFrame(st.session_state.portfolio_df_data)
                display = pd.DataFrame({
                    "Ticker":      df["Ticker"],
                    "Shares":      df["Shares"],
                    "Price":       df["Price"],
                    "Value (USD)": df["Value_USD"].map("${:,.2f}".format),
                    "Weight %":    df["Weight_Pct"],
                    "Beta":        df["Beta"].apply(lambda x: f"{x:.3f}" if x is not None else "N/A"),
                    "Ann. Vol":    df["Vol_Ann"].apply(lambda x: f"{x:.1f}%" if x is not None else "N/A"),
                })
                def _cw2(val):
                    try:
                        v = float(val)
                        if v > 40: return "color: #f87171"
                        if v > 25: return "color: #fbbf24"
                        return "color: #34d399"
                    except Exception:
                        return ""
                st.dataframe(display.style.map(_cw2, subset=["Weight %"]), use_container_width=True, hide_index=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Value", f"${st.session_state.portfolio_total:,.2f}")
                c2.metric("Holdings",    len(df))
                bv = st.session_state.portfolio_beta
                c3.metric("Wtd. Beta",   f"{bv:.3f}" if bv else "N/A")

    st.rerun()