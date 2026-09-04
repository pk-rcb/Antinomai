"""
agents.py — All LangGraph agent nodes (no Streamlit dependencies).
"""
import re
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional

from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
import os

from backend.state import (
    ApplicationState, IntentRoute, Asset, PortfolioExtraction,
    get_primary_model, get_vision_model,
)
from backend.tools import (
    clean_ticker, get_stock_price, get_company_metrics, get_sentiment_crosscheck
)


def _llm(temperature: float = 0.0):
    return ChatGroq(model=get_primary_model(), temperature=temperature)


def _tavily():
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ── Orchestrator ───────────────────────────────────────────────────────────────
def orchestrator_router(state: ApplicationState):
    raw = state["messages"][-1].content
    has_image = False
    
    if isinstance(raw, list):
        user_message = next(
            (p["text"] for p in raw if isinstance(p, dict) and p.get("type") == "text"),
            "analyze chart"
        )
        has_image = any(isinstance(p, dict) and p.get("type") == "image_url" for p in raw)
    else:
        user_message = raw

    if has_image:
        print("[Orchestrator] Image detected -> vision")
        return {"next_destination": "vision", "user_input_type": "vision"}

    router_llm = _llm().with_structured_output(IntentRoute)
    system = SystemMessage(content="""You are an intent router for a financial platform.
Classify input into EXACTLY ONE category:
  'debate'      - asking whether to buy/sell/hold a specific stock
  'vision'      - uploading a chart or asking for technical/visual chart analysis
  'trivia'      - asking for a definition, concept, or quick stock price lookup
  'fundamental' - asking for deep-dive fundamental analysis of one company
  'portfolio'   - providing multiple holdings (2+ stocks with share counts)
  'research'    - asking about uploaded documents, reports, earnings transcripts, filings, or notes in the research vault

Rules: Multiple stocks with shares -> portfolio. Buy/sell question -> debate. Deep analysis -> fundamental. Document/vault/transcript/filing questions -> research.""")

    try:
        decision = router_llm.invoke([system, HumanMessage(content=f"Classify: {user_message}")])
        dest     = decision.destination
    except Exception:
        dest = "trivia"

    print(f"[Orchestrator] -> {dest}")
    return {"next_destination": dest, "user_input_type": dest}


# ── Trivia ─────────────────────────────────────────────────────────────────────
def trivia_node(state: ApplicationState):
    llm = _llm().bind_tools([get_stock_price])
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
        final    = _llm().invoke([sys_msg] + new_messages)
        new_messages.append(final)

    return {"messages": new_messages}


# ── Vision ─────────────────────────────────────────────────────────────────────
def vision_node(state: ApplicationState):
    messages   = state["messages"]
    vision_llm = ChatGroq(
        model=get_vision_model(),
        temperature=0.1,
    )
    sys_msg = SystemMessage(content="""You are an expert technical stock analyst.
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
        content  = re.sub(r"<think>.*?</think>", "", response.content or "", flags=re.DOTALL).strip()
        if len(content) < 60:
            return {"messages": messages + [AIMessage(content="Unresolved Technical Trend: Image lacks sufficient contrast or chart indicators. Please upload a clearer candlestick chart.")]}
        return {"messages": messages + [AIMessage(content=content)]}
    except Exception as e:
        return {"messages": messages + [AIMessage(content=f"Vision Agent Error: {e}")]}


# ── Debate Panel ───────────────────────────────────────────────────────────────
def bull_agent(state: ApplicationState):
    print("[Bull Agent] Fetching upside catalysts...")
    messages    = state["messages"]
    enable_xchk = state.get("enable_sentiment_check", False)

    debate_llm   = _llm(0.4)
    debate_agent = debate_llm.bind_tools([get_company_metrics])

    sys_msg = SystemMessage(content="""You are an aggressive growth investor.
Use get_company_metrics to pull live data. Build a data-backed 2-paragraph bullish thesis:
- Para 1: Macro tailwinds, sector growth, strategic positioning (cite specific data)
- Para 2: Undervalued fundamentals, upcoming catalysts (cite specific P/E, margins, figures)
Every claim MUST cite a specific data point. Vague sentiment is not acceptable.""")

    response   = debate_agent.invoke([sys_msg] + messages)
    extra_msgs = []

    if response.tool_calls:
        tc          = response.tool_calls[0]
        tool_result = get_company_metrics.invoke(tc)
        tool_msg    = ToolMessage(content=str(tool_result), tool_call_id=tc["id"], name=tc["name"])
        extra_msgs  = [response, tool_msg]
        bull_final  = debate_llm.invoke([sys_msg] + messages + extra_msgs)

        if enable_xchk:
            raw_msg   = messages[0].content
            ticker_kw = (raw_msg if isinstance(raw_msg, str) else "stock").split()[-1]
            xc_result = get_sentiment_crosscheck.invoke({"ticker": ticker_kw})
            xc_msg    = AIMessage(content=f"[Sentiment Cross-Check]\n{xc_result}")
            extra_msgs.append(xc_msg)
            bull_final = debate_llm.invoke([sys_msg] + messages + extra_msgs + [bull_final])

        bull_final.content = f"**BULL THESIS:**\n\n{bull_final.content}"
        return {"messages": messages + extra_msgs + [bull_final]}

    response.content = f"**BULL THESIS:**\n\n{response.content}"
    return {"messages": messages + [response]}


def bear_agent(state: ApplicationState):
    print("[Bear Agent] Building counter-argument...")
    messages    = state["messages"]
    enable_xchk = state.get("enable_sentiment_check", False)

    debate_llm   = _llm(0.4)
    debate_agent = debate_llm.bind_tools([get_company_metrics])

    sys_msg = SystemMessage(content="""You are a cynical short-seller and forensic accountant.
You have read the Bull thesis above. Use get_company_metrics to pull data and DISPROVE it.
Build a 2-paragraph bearish counter-argument:
- Para 1: Overvaluation with specific P/E vs sector median; margin compression with exact %
- Para 2: Regulatory headwinds, analyst downgrades, poor earnings — all cited with specifics
Directly attack EACH of the Bull's claims by name. Every counter must cite specific data.""")

    response   = debate_agent.invoke([sys_msg] + messages)
    extra_msgs = []

    if response.tool_calls:
        tc          = response.tool_calls[0]
        tool_result = get_company_metrics.invoke(tc)
        tool_msg    = ToolMessage(content=str(tool_result), tool_call_id=tc["id"], name=tc["name"])
        extra_msgs  = [response, tool_msg]
        bear_final  = debate_llm.invoke([sys_msg] + messages + extra_msgs)

        if enable_xchk:
            raw_msg   = messages[0].content
            ticker_kw = (raw_msg if isinstance(raw_msg, str) else "stock").split()[-1]
            xc_result = get_sentiment_crosscheck.invoke({"ticker": ticker_kw})
            xc_msg    = AIMessage(content=f"[Sentiment Cross-Check]\n{xc_result}")
            extra_msgs.append(xc_msg)
            bear_final = debate_llm.invoke([sys_msg] + messages + extra_msgs + [bear_final])

        bear_final.content = f"**BEAR THESIS:**\n\n{bear_final.content}"
        return {"messages": messages + extra_msgs + [bear_final]}

    response.content = f"**BEAR THESIS:**\n\n{response.content}"
    return {"messages": messages + [response]}


def judge_agent(state: ApplicationState):
    print("[Judge Agent] Weighing arguments...")
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

    response = _llm().invoke([sys_msg] + state["messages"])
    return {"messages": state["messages"] + [response]}


# ── Fundamental ────────────────────────────────────────────────────────────────
def fundamental_node(state: ApplicationState):
    print("[Fundamental] Executing deep research...")
    user_message = state["messages"][0].content
    if isinstance(user_message, list):
        user_message = next(
            (p["text"] for p in user_message if isinstance(p, dict) and p.get("type") == "text"), ""
        )

    lm       = _llm()
    raw_tick = lm.invoke([
        SystemMessage(content="Extract only the stock ticker or company name. Return ONLY the ticker, nothing else."),
        HumanMessage(content=user_message),
    ]).content.strip()

    ticker = clean_ticker(raw_tick, llm=lm)
    print(f"[Fundamental] Ticker: {ticker}")

    try:
        info = yf.Ticker(ticker).info

        critical    = [info.get("trailingPE"), info.get("totalRevenue"), info.get("totalCash"),
                       info.get("grossMargins"), info.get("volume"), info.get("marketCap")]
        valid_count = sum(1 for v in critical if v is not None and v != 0)

        if not info or valid_count < 2:
            return {"messages": list(state["messages"]) + [AIMessage(content=(
                f"**DATA RETRIEVAL FAILURE** for `{ticker}`\n\n"
                f"Only {valid_count}/6 core metrics returned. Possible reasons:\n"
                f"- Ticker may be delisted or incorrectly formatted\n"
                f"- Indian equities need `.NS` suffix (e.g., `{raw_tick}.NS`)\n\n"
                f"*Analysis aborted to prevent AI hallucination over empty data.*"
            ))]}

        def fmt(v): return f"${v:,.0f}" if isinstance(v, (int, float)) and v > 0 else "N/A"
        def pct(v): return f"{v*100:.2f}%" if isinstance(v, (int, float)) else "N/A"

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

        news_block = ""
        try:
            news_resp = _tavily().search(
                query=f"{company} {ticker} news earnings analyst 2025",
                search_depth="basic", max_results=2,
            )
            items = [r["content"] for r in news_resp.get("results", [])]
            if items:
                news_block = "\n\n**Live News Context:**\n" + "\n---\n".join(items)
        except Exception as e:
            print(f"[Tavily] News fetch failed: {e}")

        synth = _llm(0.2).invoke([
            SystemMessage(content="""You are a senior equity analyst. Write a 3-paragraph fundamental analysis:
Para 1 - Profitability & Valuation: interpret P/E vs sector, margins, ROE, growth
Para 2 - Solvency & Liquidity: debt/equity, current ratio, free cash flow
Para 3 - Analyst Assessment: synthesize into a forward-looking view, reference target price
RULE: Do NOT cite any figure not in the data. If >50% N/A, say data is insufficient."""),
            HumanMessage(content=f"{report}{news_block}"),
        ])

        final = (
            f"*[Deep Research]*\n\n"
            f"## Fundamental Analysis: {company} ({ticker})\n\n"
            f"### Raw Quantitative Data\n{report}\n---\n\n"
            f"### Analyst Synthesis\n{synth.content}"
        )
        return {"messages": list(state["messages"]) + [AIMessage(content=final)]}

    except Exception as e:
        print(f"[Fundamental] Error: {e}")
        return {"messages": list(state["messages"]) + [AIMessage(content=f"System error in Fundamental Engine: {e}")]}


# ── Portfolio ──────────────────────────────────────────────────────────────────
def _get_live_fx_rate() -> tuple:
    try:
        resp    = _tavily().search(query="USD to INR exchange rate today", search_depth="basic", max_results=1)
        content = resp.get("results", [{}])[0].get("content", "")
        matches = re.findall(r'\b(8[0-9]\.\d+|9[0-2]\.\d+)\b', content)
        if matches:
            rate = float(matches[0])
            return rate, True
    except Exception as e:
        print(f"[FX] Fallback ({e})")
    return 84.0, False


def _compute_beta(ticker: str, benchmark: str) -> Optional[float]:
    try:
        a  = yf.Ticker(ticker).history(period="1y")["Close"].pct_change().dropna()
        b  = yf.Ticker(benchmark).history(period="1y")["Close"].pct_change().dropna()
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
        user_message = next(
            (p["text"] for p in user_message if isinstance(p, dict) and p.get("type") == "text"), ""
        )

    extractor = _llm().with_structured_output(PortfolioExtraction)
    try:
        extracted = extractor.invoke(user_message)
    except Exception as e:
        return {
            "messages":        list(state["messages"]) + [AIMessage(content=f"Could not parse portfolio: {e}")],
            "portfolio_report": "",
        }

    if not extracted.assets:
        return {
            "messages":        list(state["messages"]) + [AIMessage(content="No assets detected. Please list holdings with share counts.")],
            "portfolio_report": "",
        }

    fx_rate, fx_live = _get_live_fx_rate()
    fx_note          = f"{'Live' if fx_live else 'Static'} USD/INR: Rs.{fx_rate:.2f}"
    lm               = _llm()
    rows             = []
    total_usd        = 0.0

    for asset in extracted.assets:
        ticker = clean_ticker(asset.ticker, llm=lm)
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
            rows.append({
                "Ticker":    ticker,
                "Shares":    asset.shares,
                "Price":     f"{currency} {price:,.2f}",
                "Value_USD": value_usd,
                "Beta":      beta,
                "Vol_Ann":   vol,
            })
        except Exception as e:
            print(f"[Portfolio] Error for {ticker}: {e}")

    if not rows:
        return {
            "messages":        list(state["messages"]) + [AIMessage(content="Could not fetch data for any tickers.")],
            "portfolio_report": "",
        }

    for r in rows:
        r["Weight_Pct"] = round((r["Value_USD"] / total_usd) * 100, 2) if total_usd > 0 else 0.0

    beta_rows = [r for r in rows if r["Beta"] is not None]
    wtd_beta  = (
        sum(r["Beta"] * r["Weight_Pct"] / 100 for r in beta_rows)
        / sum(r["Weight_Pct"] / 100 for r in beta_rows)
    ) if beta_rows else None

    payload  = f"PORTFOLIO MATH PAYLOAD ({fx_note})\n{'='*55}\n"
    payload += f"Total Value: ${total_usd:,.2f} USD | Holdings: {len(rows)}\n"
    if wtd_beta:
        label   = "aggressive" if wtd_beta > 1.2 else "moderate" if wtd_beta > 0.8 else "defensive"
        payload += f"Weighted Beta: {wtd_beta:.3f} ({label})\n"
    payload += "\nBreakdown:\n"
    for r in sorted(rows, key=lambda x: x["Weight_Pct"], reverse=True):
        b = f"{r['Beta']:.3f}" if r["Beta"] else "N/A"
        v = f"{r['Vol_Ann']:.1f}%" if r["Vol_Ann"] else "N/A"
        payload += f"  {r['Ticker']:<15} {r['Shares']:>8.1f} sh | {r['Price']:>16} | ${r['Value_USD']:>11,.2f} | {r['Weight_Pct']:>5.1f}% | Beta {b} | Vol {v}\n"

    flagged = [r for r in rows if r["Weight_Pct"] > 40]
    if flagged:
        payload += f"\nCONCENTRATION ALERT: {', '.join(r['Ticker'] for r in flagged)} > 40% of portfolio.\n"

    analysis = _llm(0.1).invoke([
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
        f"**Total Value:** ${total_usd:,.2f} | **Holdings:** {len(rows)} | "
        f"**Wtd. Beta:** {wtd_beta_str} | *{fx_note}*\n\n"
        f"---\n\n{analysis.content}\n\n"
        f"---\n*All math computed in Python/Pandas sandbox. Benchmarks: ^GSPC (USD), ^NSEI (INR).*"
    )

    # Return portfolio rows in state for the API to send to frontend
    return {
        "messages":        list(state["messages"]) + [AIMessage(content=final)],
        "portfolio_report": payload,
        "portfolio_rows":   rows,          # extra key consumed by main.py
        "portfolio_total":  total_usd,
        "portfolio_beta":   wtd_beta,
    }


# ── Research (RAG) ────────────────────────────────────────────────────────────
def research_node(state: ApplicationState):
    """
    Research Vault route — answers from privately ingested documents.
    Uses ChromaDB (dev) or Qdrant (prod) semantic search.
    Follows the same citation-mandatory philosophy as the Judge Agent.
    """
    print("[Research] Querying vault...")
    from backend.vault import retrieve, vault_doc_count

    # Extract plain-text query
    raw = state["messages"][0].content
    if isinstance(raw, list):
        query = next(
            (p["text"] for p in raw if isinstance(p, dict) and p.get("type") == "text"),
            "summarize"
        )
    else:
        query = raw

    # Check if vault has anything at all
    session_id = state.get("session_id", "default")
    doc_count = vault_doc_count(session_id=session_id)
    if doc_count == 0:
        empty_msg = (
            "**🗂️ Research Vault is empty.**\n\n"
            "No documents have been ingested yet. Upload documents using the "
            "**Research Vault** panel in the sidebar to enable this mode.\n\n"
            "*Supported formats: PDF, TXT, MD — e.g. earnings transcripts, "
            "annual reports, research notes, SEC filings.*"
        )
        return {"messages": list(state["messages"]) + [AIMessage(content=empty_msg)]}

    # Optional: extract ticker hint from query for filtered search
    ticker_hint: str | None = None
    try:
        raw_tick = _llm().invoke([
            SystemMessage(content=(
                "Extract a stock ticker from the user's query if one is mentioned. "
                "Return ONLY the ticker symbol (e.g. TCS.NS, AAPL). "
                "If no ticker is mentioned, return the single word: NONE"
            )),
            HumanMessage(content=query),
        ]).content.strip().upper().split()[0]
        if raw_tick != "NONE" and len(raw_tick) <= 12:
            ticker_hint = raw_tick
    except Exception:
        pass

    # Retrieve relevant chunks (try ticker-filtered first, fall back to global)
    chunks = retrieve(query=query, ticker_filter=ticker_hint, n_results=6, session_id=session_id)
    if not chunks and ticker_hint:
        chunks = retrieve(query=query, ticker_filter=None, n_results=6, session_id=session_id)

    if not chunks:
        no_match_msg = (
            f"**🔍 No relevant documents found** for your query.\n\n"
            f"The vault has **{doc_count} document(s)** but none matched your question. "
            f"Try uploading a more specific document or rephrasing your query."
        )
        return {"messages": list(state["messages"]) + [AIMessage(content=no_match_msg)]}

    # Build grounded context block (doc_type | source | date)
    context_parts = [
        f"[{c.doc_type.upper()} | {c.source} | {c.date_added}]\n{c.content}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_parts)
    sources  = list(dict.fromkeys(c.source for c in chunks))  # unique, order-preserving

    # LLM synthesis — mandatory citation rule (same as Judge Agent philosophy)
    synthesis = _llm(0.1).invoke([
        SystemMessage(content="""You are a financial research analyst with access to a private document vault.
Answer the user's question ONLY using the provided context below.

CITATION RULE (non-negotiable, same as the CIO Judge Agent):
- Every factual claim must be followed by [Source: filename] inline.
- If the vault context does not contain the answer, say exactly: "Not found in vault context."
- Do NOT use your parametric knowledge to fill gaps — only what is in the context.
- Structure your answer clearly. Use bullet points for lists of facts."""),
        HumanMessage(content=f"Vault Context:\n{context}\n\n---\n\nUser Question: {query}"),
    ])

    # Format final response with vault provenance header
    source_list = ", ".join(f"`{s}`" for s in sources)
    final = (
        f"*[Research Vault — {len(chunks)} chunks from {len(sources)} doc(s)]*\n\n"
        f"{synthesis.content}\n\n"
        f"---\n**Sources:** {source_list}"
    )
    return {"messages": list(state["messages"]) + [AIMessage(content=final)]}


# ── Graph compilation (singleton) ──────────────────────────────────────────────
_APP_INSTANCE = None


def get_app():
    global _APP_INSTANCE
    if _APP_INSTANCE is None:
        wf = StateGraph(ApplicationState)
        wf.add_node("orchestrator", orchestrator_router)
        wf.add_node("trivia",       trivia_node)
        wf.add_node("vision",       vision_node)
        wf.add_node("fundamental",  fundamental_node)
        wf.add_node("portfolio",    portfolio_node)
        wf.add_node("bull_agent",   bull_agent)
        wf.add_node("bear_agent",   bear_agent)
        wf.add_node("judge_agent",  judge_agent)
        wf.add_node("research",     research_node)   # RAG vault route

        wf.set_entry_point("orchestrator")
        wf.add_conditional_edges("orchestrator", lambda s: s["next_destination"], {
            "debate":      "bull_agent",
            "vision":      "vision",
            "trivia":      "trivia",
            "fundamental": "fundamental",
            "portfolio":   "portfolio",
            "research":    "research",
        })
        wf.add_edge("bull_agent",  "bear_agent")
        wf.add_edge("bear_agent",  "judge_agent")
        wf.add_edge("judge_agent", END)
        wf.add_edge("vision",      END)
        wf.add_edge("trivia",      END)
        wf.add_edge("fundamental", END)
        wf.add_edge("portfolio",   END)
        wf.add_edge("research",    END)

        _APP_INSTANCE = wf.compile(checkpointer=MemorySaver())
        print("[Graph] LangGraph app compiled successfully.")
    return _APP_INSTANCE

