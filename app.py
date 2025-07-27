import re
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 🎛️  CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="FinScope AI",
    page_icon="📈",
    layout="wide",
)

# -----------------------------------------------------------------------------
# ⚙️  HELPERS & CACHING
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False, ttl=None)
def get_model():
    """Load FinBERT sentiment model (cached across sessions)."""
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model


def _clean_headline(h: str) -> str:
    return re.sub(r"\s+-\s+[A-Za-z &]+$", "", h).strip()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker: str, api_key: str) -> list[str]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 20,
        "apiKey": api_key,
    }
    r = httpx.get(url, params=params, timeout=10)
    r.raise_for_status()
    return [_clean_headline(a.get("title", "")) for a in r.json().get("articles", [])]


@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_list() -> list[str]:
    import requests
    from bs4 import BeautifulSoup

    try:
        html = requests.get("https://www.fool.com/investing/", timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return [a.text for a in soup.find_all("a") if a.text.isupper() and len(a.text) <= 5][:10]
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


@st.cache_data(ttl=300, show_spinner=False)
def load_price_history(symbol: str, period: str = "6mo") -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, auto_adjust=False)


@st.cache_data(ttl=120, show_spinner=False)
def load_intraday(symbol: str) -> pd.DataFrame:
    """1‑minute candles for the current trading day (Yahoo)."""
    df = yf.download(symbol, interval="1m", period="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi‑level columns that Yahoo returns when group_by="ticker"
        df.columns = df.columns.get_level_values(-1)
    df = df.tz_localize(None)
    return df

# -----------------------------------------------------------------------------
# 📈  SENTIMENT ANALYSIS
# -----------------------------------------------------------------------------

def score_sentiment(headlines: list[str]) -> tuple[list[str], float]:
    tokenizer, model = get_model()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    weights = {"positive": 1, "neutral": 0, "negative": -1}

    inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).numpy()

    labels = [id2label[int(idx)] for idx in probs.argmax(axis=1)]
    compound = float(np.mean([weights[l] for l in labels])) if labels else 0.0
    return [l.capitalize() for l in labels], compound


def advice_from_score(score: float) -> str:
    return "BUY" if score >= 0.5 else "SELL" if score <= -0.5 else "HOLD"

# -----------------------------------------------------------------------------
# 🔄  TECHNICAL INDICATORS (INTRADAY)
# -----------------------------------------------------------------------------

def compute_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA‑20, RSI‑14 and Entry columns to an intraday DataFrame.

    Handles erratic column casing from yfinance by searching for any column
    whose lowercase name equals "close".
    """
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Find the CLOSE column case‑insensitively, then standardise its name
    close_col = next((c for c in df.columns if c.lower() == "close"), None)
    if close_col is None:
        # Bail gracefully – caller will detect Entry column absence
        return pd.DataFrame()
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    # Calculate indicators
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(14).mean()
    loss = down.rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["RSI_14"] = 100 - 100 / (1 + rs)

    df["Entry"] = (
        (df["Close"].shift(1) < df["SMA_20"].shift(1)) &
        (df["Close"] > df["SMA_20"]) &
        (df["RSI_14"] < 70)
    )
    return df

# -----------------------------------------------------------------------------
# 🖥️  MAIN UI
# -----------------------------------------------------------------------------

def main() -> None:
    st.title("📈 FinScope AI")
    st.markdown("AI‑powered dashboard: news sentiment, price history & intraday signals.")
    st.markdown("---")

    with st.sidebar:
        st.header("🔍 Stock Selection")
        defaults = get_ticker_list()
        ticker = st.text_input("Ticker (e.g. AAPL)", value=defaults[0] if defaults else "AAPL").strip().upper()
        if not re.fullmatch(r"[A-Z.\-]{1,5}", ticker):
            st.warning("Enter a valid ticker (1‑5 capital letters).")
            st.stop()
        period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "5y"], 2)
        refresh = st.button("🔄 Refresh intraday")

    # Secrets
    if "newsapi_key" not in st.secrets:
        st.error("🔑 Add your NewsAPI key to Streamlit secrets to enable sentiment analysis.")
        st.stop()
    news_key = st.secrets["newsapi_key"]

    # Price history
    hist = load_price_history(ticker, period)
    if hist.empty:
        st.error("No price data returned – verify the ticker.")
        st.stop()

    # Headlines & sentiment
    with st.spinner("Fetching headlines …"):
        headlines = fetch_news(ticker, news_key)[:5]
    labels, compound = score_sentiment(headlines) if headlines else ([], 0.0)
    rec = advice_from_score(compound)

    # KPI
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg sentiment", f"{compound:+.2f}")
    if len(hist) > 1:
        day_change = (hist.Close.iloc[-1] - hist.Close.iloc[-2]) / hist.Close.iloc[-2] * 100
        k2.metric("Price Δ 1‑day", f"{day_change:+.2f}%")
    k3.metric("Advice", rec)

    # Tabs
    tab_news, tab_chart, tab_intraday = st.tabs(["📰 News", "📉 Chart", "⏱️ Intraday"])

    with tab_news:
        st.subheader("Latest headlines")
        if not headlines:
            st.write("No recent news.")
        else:
            for h, lbl in zip(headlines, labels):
                st.markdown(f"- **{h}** — *{lbl}*")

    with tab_chart:
        st.subheader(f"{ticker} price history – {period}")
        fig = go.Figure([
            go.Candlestick(x=hist.index, open=hist.Open, high=hist.High, low=hist.Low, close=hist.Close)
        ])
        fig.update_layout(height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_intraday:
        st.subheader("Intraday 1‑minute candles & entry signal")
        if refresh:
            load_intraday.clear()  # invalidate cache on demand
        intraday_df = load_intraday(ticker)
        if intraday_df.empty:
            st.write("Intraday data not available outside market hours.")
        else:
            indf = compute_intraday_indicators(intraday_df)
            if indf.empty or "Close" not in indf.columns:
                st.warning("Unable to compute intraday indicators for this symbol right now.")
            else:
                last = indf.iloc[-1]
                entry_text = "✅ Entry signal!" if last["Entry"] else "No entry signal currently"
                st.write(
                    f"**Current price:** {last['Close']:.2f} | "
                    f"SMA20: {last['SMA_20']:.2f} | "
                    f"RSI14: {last['RSI_14']:.1f}"
                )
                st.success(entry_text) if last["Entry"] else st.info(entry_text)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=indf.index, y=indf["Close"], mode="lines", name="Close"))
                fig2.add_trace(go.Scatter(x=indf.index, y=indf["SMA_20"], mode="lines", name="SMA 20"))
                entry_df = indf[indf["Entry"]]
                if not entry_df.empty:
                    fig2.add_trace(
                        go.Scatter(
                            x=entry_df.index,
                            y=entry_df["Close"],
                            mode="markers",
                            marker_symbol="triangle-up",
                            marker_color="green",
                            marker_size=10,
                            name="Entry",
                        )
                    )
                fig2.update_layout(height=400, xaxis_title="Time", yaxis_title="Price")
                fig2.update_layout(height=400, xaxis_title="Time", yaxis_title="Price")
                st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
