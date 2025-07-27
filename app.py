import re
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
    """Remove publisher suffixes (e.g. " - Bloomberg") and trim whitespace."""
    return re.sub(r"\s+-\s+[A-Za-z &]+$", "", h).strip()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker: str, api_key: str) -> list[str]:
    """Return up to 20 latest English headlines for a ticker."""
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
    """Return a small list of trending symbols from Motley Fool as defaults."""
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
    """Download OHLC price data from Yahoo Finance."""
    return yf.Ticker(symbol).history(period=period, auto_adjust=False)

# -----------------------------------------------------------------------------
# 📈  SENTIMENT ANALYSIS
# -----------------------------------------------------------------------------

def score_sentiment(headlines: list[str]) -> tuple[list[str], float]:
    """Infer per‑headline sentiment using the model's **own** label map
    to avoid hard‑coding the index order (different FinBERT forks swap
    Positive / Negative).
    Returns the list of labels and a compound (+1,0,‑1) mean score.
    """
    tokenizer, model = get_model()
    id2label = model.config.id2label  # e.g. {0: 'neutral', 1: 'positive', 2: 'negative'}
    weights = {"positive": 1, "neutral": 0, "negative": -1}

    inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).numpy()

    labels = [id2label[int(idx)].lower() for idx in probs.argmax(axis=1)]
    compound = float(np.mean([weights[l] for l in labels]))
    # Capitalise first letter for display
    return [l.capitalize() for l in labels], compound


def advice_from_score(score: float) -> str:
    if score >= 0.5:
        return "BUY"
    if score <= -0.5:
        return "SELL"
    return "HOLD"

# -----------------------------------------------------------------------------
# 🖥️  MAIN UI
# -----------------------------------------------------------------------------

def main() -> None:
    st.title("📈 FinScope AI")
    st.markdown("AI‑powered dashboard combining news sentiment and price history.")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("🔍 Stock Selection")
        defaults = get_ticker_list()
        ticker = st.text_input(
            "Company name or ticker (e.g. AAPL)",
            value=defaults[0] if defaults else "AAPL",
        ).strip().upper()
        if not re.fullmatch(r"[A-Z.\-]{1,5}", ticker):
            st.warning("Enter a valid ticker (1‑5 capital letters).")
            st.stop()
        period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "5y"], index=2)

    # Check NewsAPI key
    if "newsapi_key" not in st.secrets:
        st.error("🔑 Add your NewsAPI key to Streamlit secrets to enable sentiment analysis.")
        st.stop()
    news_key = st.secrets["newsapi_key"]

    # Price data
    hist = load_price_history(ticker, period)
    if hist.empty:
        st.error("No price data returned – please verify the ticker.")
        st.stop()

    # Headlines
    with st.spinner("Fetching headlines …"):
        headlines = fetch_news(ticker, news_key)[:5]

    labels, compound = ([], 0.0) if not headlines else score_sentiment(headlines)
    recommendation = advice_from_score(compound)

    # KPI Row
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg sentiment", f"{compound:+.2f}")
    day_change = (hist.Close.iloc[-1] - hist.Close.iloc[-2]) / hist.Close.iloc[-2] * 100
    k2.metric("Price Δ 1‑day", f"{day_change:+.2f}%")
    k3.metric("Advice", recommendation)

    # Tabs
    tab_news, tab_chart = st.tabs(["📰 News", "📉 Chart"])

    # News Tab
    with tab_news:
        st.subheader("Latest headlines")
        if not headlines:
            st.write("No recent news found.")
        else:
            for h, lbl in zip(headlines, labels):
                st.markdown(f"- **{h}** — *{lbl}*")

    # Chart Tab
    with tab_chart:
        st.subheader(f"{ticker} price history – {period}")
        fig = go.Figure(
            [
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name="Price",
                )
            ]
        )
        fig.update_layout(height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
