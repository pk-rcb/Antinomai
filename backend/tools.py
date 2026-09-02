"""
tools.py — Live financial data tools (extracted from app.py, no Streamlit deps).
"""
import re
import os
from tavily import TavilyClient
import yfinance as yf

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# Tavily client — initialised lazily from env so imports never crash.
def _tavily():
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

_KNOWN_NSE = {
    # Nifty 50 / large-cap
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
    # Mid / small-cap NSE additions
    "TBZ","JUBLFOOD","BERGEPAINT","PIDILITIND","HAVELLS","VOLTAS","WHIRLPOOL",
    "BATAINDIA","RELAXO","PAGEIND","MCDOWELL-N","RADICO","UNITDSPR",
    "TATACOMM","MPHASIS","COFORGE","PERSISTENT","LTIM","KPITTECH",
    "POLYCAB","KEI","FINOLEX","CUMMINSIND","THERMAX","GRINDWELL",
    "APLAPOLLO","JINDALSAW","WELSPUNIND","TRIDENT","RAYMOND",
    "KAJARIACER","GREENPLY","CENTURYPLY","ASTRAL","SUPREMEIND",
    "TRENT","VSTIND","MANYAVAR","ABFRL","SHOPERSTOP",
    "CHOLAHLDNG","MANAPPURAM","IIFL","APTUS","HOMEFIRST",
    "FORTIS","MAXHEALTH","APOLLOHOSP","METROPOLIS","THYROCARE",
    "ZYDUSLIFE","ALKEM","AUROPHARMA","IPCALAB","NATCOPHARM",
    "DALBHARAT","JKCEMENT","RAMCOCEM","HEIDELBERG","PRISMJOINTS",
    "MOTHERSON","BOSCHLTD","BALKRISIND","APOLLOTYRE","CEATLTD",
    "TATAELXSI","DIXON","AMBER","PGEL","KAYNES",
    "IREDA","SJVN","NHPC","RECLTD","PFC","CESC","TORNTPOWER",
    "GSPL","IGL","MGL","ATGL","GAIL",
    "CONCOR","BLUEDART","MAHLOG","DELHIVERY",
    "ZEEL","PVRINOX","INOXWIND","NETWRK18","TV18BRDCST",
    "JUSTDIAL","NAUKRI","POLICYBZR","EASEMYTRIP","IXIGO",
}


def clean_ticker(raw: str, llm=None) -> str:
    parts   = re.sub(r'[$\'"\\s]', "", raw.upper()).strip().split()
    cleaned = parts[0] if parts else raw.strip().upper()

    if "." in cleaned:
        return cleaned.upper()

    if cleaned in _KNOWN_NSE:
        return f"{cleaned}.NS"

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
            return resp.content.strip().upper().split()[0]
        except Exception as e:
            print(f"[clean_ticker] LLM fallback failed: {e}")

    return cleaned


@tool
def get_stock_price(ticker: str) -> str:
    """Fetches the current real-time stock price for a given ticker or company name."""
    import datetime
    from langchain_groq import ChatGroq
    from backend.state import get_primary_model
    _llm  = ChatGroq(model=get_primary_model(), temperature=0.0)
    clean = clean_ticker(ticker, llm=_llm)

    today = datetime.date.today().strftime("%Y-%m-%d")

    # ── Attempt 1: fast_info (truly live / last market price, no caching) ──
    try:
        t = yf.Ticker(clean)
        fi = t.fast_info
        # fast_info.last_price is the last traded price (real-time during market hours)
        price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
        if price and price > 0:
            currency = "INR" if clean.endswith(".NS") or clean.endswith(".BO") else "USD"
            return (
                f"Live price of {clean}: {currency} {price:,.2f} "
                f"(as of {today}, via yfinance fast_info)"
            )
    except Exception as e:
        print(f"[get_stock_price] fast_info failed for {clean}: {e}")

    # ── Attempt 2: history with 5d window to get latest available close ──
    try:
        hist = yf.Ticker(clean).history(period="5d", auto_adjust=False)
        if not hist.empty:
            latest_row  = hist.iloc[-1]
            price       = latest_row["Close"]
            date_str    = str(latest_row.name.date())
            currency    = "INR" if clean.endswith(".NS") or clean.endswith(".BO") else "USD"
            stale_warn  = "" if date_str == today else f" ⚠️ last trading close ({date_str}, market may be closed)"
            return (
                f"Price of {clean}: {currency} {price:,.2f}{stale_warn}"
            )
    except Exception as e:
        print(f"[get_stock_price] history failed for {clean}: {e}")

    # ── Attempt 3: Tavily live web search ──
    try:
        resp = _tavily().search(
            query=f"{ticker} stock price today {today} NSE BSE live",
            search_depth="advanced",
            max_results=3,
        )
        results = [r["content"] for r in resp.get("results", [])]
        if results:
            return f"Web search result for {ticker} (today={today}):\n" + "\n".join(results)
    except Exception as e:
        return f"Error fetching price: {e}"

    return f"Could not retrieve price for: {ticker}"


@tool
def get_company_metrics(ticker: str) -> str:
    """Fetches earnings, P/E ratio, analyst ratings, and news for a stock."""
    try:
        resp    = _tavily().search(
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
        resp    = _tavily().search(
            query=f"{ticker} stock social media analyst sentiment vs earnings 2025",
            search_depth="advanced", max_results=3,
        )
        results = [r["content"] for r in resp.get("results", [])]
        return f"SENTIMENT CROSS-CHECK FOR {ticker}:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Error: {e}"

