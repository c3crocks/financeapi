import asyncio
import re

import httpx
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 🎛️  CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="FinScope AI",
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
    """Remove publisher suffixes e.g. ' - Bloomberg' and redundant whitespace."""
    return re.sub(r"\s+-\s+[A-Za-z &]+$", "", h).strip()


@st.cache_data(ttl=900, show_spinner=False)
async def fetch_news(ticker: str, api_key: str) -> list[str]:
    """Asynchronously pull latest news headlines (<=20) for the symbol."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 20,
        "apiKey": api_key,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return [_clean_headline(a["title"]) for a in r.json().get("articles", [])]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_ticker_list() -> list[str]:
    """Scrape a few trending tickers from Motley Fool (lazy fallback list)."""
    import requests
    from bs4 import BeautifulSoup

    try:
        html = requests.get("https://www.fool.com/investing/", timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return list(
            {
                a.text
                for a in soup.find_all("a")
                if a.text.isupper() and len(a.text) <= 5
            }
        )[:10]
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


@st.cache_data(ttl=5 * 60, show_spinner=False)
def load_price_history(symbol: str, period: str = "6mo") -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, auto_adjust=False)


# -----------------------------------------------------------------------------
# 📈  SENTIMENT ANALYSIS
# -----------------------------------------------------------------------------

def score_sentiment(headlines: list[str]) -> tuple[list[str], float]:
    """Return list of sentiment labels and the average compound score."""
    sentiments_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    tokenizer, model = get_model()

    inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()

    labels = [sentiments_map[idx] for idx in probs.argmax(axis=1)]
    # Weighted (+1, 0, -1) mean
    weights = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1,
    }
    compound = float(np.mean([weights[l] for l in labels]))
    return labels, compound


def advice_from_score(score: float) -> str:
    if score >= 0.5:
        return "BUY"
    if score <= -0.5:
        return "SELL"
    return "HOLD"


# -----------------------------------------------------------------------------
# 🔮  FORECASTING (Prophet)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def prophet_forecast(symbol: str, days: int = 7):
    df = (
        yf.download(symbol, period="1y", progress=False)[["Close"]]
        .dropna()
        .reset_index()
        .rename(columns={"Date": "ds", "Close": "y"})
    )
    df["y"] = np.log(df["y"])
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    fcst = model.predict(future)
    fcst[["yhat", "yhat_lower", "yhat_upper"]] = np.exp(
        fcst[["yhat", "yhat_lower", "yhat_upper"]]
    )
    return model, fcst.tail(days)


# -----------------------------------------------------------------------------
# 🖥️   MAIN UI
# -----------------------------------------------------------------------------

def main() -> None:
    st.title("📈 FinScope AI")
    st.markdown(
        "AI‑powered dashboard combining news sentiment, price history and short‑term forecasts."
    )
    st.markdown("---")

    # Sidebar – stock selection ---------------------------------------------
    with st.sidebar:
        st.header("🔍 Stock Selection")
        default_list = get_ticker_list()
        choice = st.text_input(
            "Company name or ticker (e.g. AAPL)", value=default_list[0] if default_list else "AAPL"
        ).strip().upper()

        # Validate ticker pattern
        if not re.fullmatch(r"[A-Z.\-]{1,5}", choice):
            st.warning("Please enter a valid ticker symbol (1–5 capital letters).")
            st.stop()

        period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "5y"], index=2)

    # Secrets gate -----------------------------------------------------------
    if "newsapi_key" not in st.secrets:
        st.error("🔑  Add your NewsAPI key to Streamlit secrets to enable sentiment analysis.")
        st.stop()

    news_key: str = st.secrets["newsapi_key"]

    # ---------------------------------------------------------------------
    # DATA FETCHING
    # ---------------------------------------------------------------------
    hist = load_price_history(choice, period)
    if hist.empty:
        st.error("No price data returned – please verify the ticker.")
        st.stop()

    # Async news fetch
    with st.spinner("Fetching latest headlines …"):
        headlines = asyncio.run(fetch_news(choice, news_key))[:5]

    # Sentiment scoring -----------------------------------------------------
    if headlines:
        labels, compound = score_sentiment(headlines)
        recommendation = advice_from_score(compound)
    else:
        labels, compound, recommendation = [], 0.0, "HOLD"

    # KPI metrics -----------------------------------------------------------
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Avg sentiment", f"{compound:+.2f}")
    day_change = (hist.Close.iloc[-1] - hist.Close.iloc[-2]) / hist.Close.iloc[-2] * 100
    col_kpi2.metric("Price Δ 1‑day", f"{day_change:+.2f}%")
    col_kpi3.metric("Advice", recommendation)

    # ---------------------------------------------------------------------
    # LAYOUT – Tabs
    # ---------------------------------------------------------------------

    tab_news, tab_chart, tab_forecast = st.tabs([
        "📰 News", "📉 Chart", "🔮 7‑day Forecast",
    ])

    # ---------------- News tab -------------------------------------------
    with tab_news:
        st.subheader("Latest headlines")
        if not headlines:
            st.write("No recent news found.")
        else:
            for h, lbl in zip(headlines, labels):
                st.markdown(f"- **{h}** — *{lbl}*")

    # ---------------- Chart tab ------------------------------------------
    with tab_chart:
        st.subheader(f"{choice} price history – {period}")
        fig = go.Figure(
            data=[
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

    # ---------------- Forecast tab ---------------------------------------
    with tab_forecast:
        st.subheader("Prophet 7‑day forecast (experimental)")
        try:
            model, fcst = prophet_forecast
