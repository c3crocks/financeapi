import re
import httpx
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yfinance as yf
import plotly.graph_objects as go
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------------------------------------------------------
# 🔒 Pop‑up Disclaimer (must accept before app runs)
# -----------------------------------------------------------------------------

# Persistent footer disclaimer text
DISCLAIMER_MD = """**CRITICAL RISK DISCLAIMER**  \
FinScope AI is an *experimental* analytics tool provided **“as‑is”**. Nothing here constitutes financial advice; past performance does not guarantee future results. Trading involves substantial risk of loss. Data, headlines and model outputs may be delayed, inaccurate or unavailable. By using this app you accept full responsibility for your decisions and hold the developers and hosts harmless for any losses."""



# -----------------------------------------------------------------------------
# ⚙️ HELPERS & CACHING
# ----------------------------------------------------------------------------- & CACHING
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False, ttl=None)
def get_model():
    tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tok, mdl


def _clean_headline(txt: str) -> str:
    return re.sub(r"\s+-\s+[A-Za-z &]+$", "", txt).strip()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(tkr: str, api_key: str) -> list[str]:
    r = httpx.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": tkr,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 20,
            "apiKey": api_key,
        },
        timeout=10,
    )
    r.raise_for_status()
    return [_clean_headline(a.get("title", "")) for a in r.json().get("articles", [])]


@st.cache_data(ttl=3600, show_spinner=False)
def default_tickers() -> list[str]:
    import requests
    from bs4 import BeautifulSoup
    try:
        html = requests.get("https://www.fool.com/investing/", timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return [a.text for a in soup.find_all("a") if a.text.isupper() and len(a.text) <= 5][:10]
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


@st.cache_data(ttl=300, show_spinner=False)
def load_history(symbol: str, period: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, auto_adjust=False)


@st.cache_data(ttl=120, show_spinner=False)
def load_intraday(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, interval="1m", period="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df.tz_localize(None)

# -----------------------------------------------------------------------------
# 📈 SENTIMENT FUNCTIONS
# -----------------------------------------------------------------------------

def score_sentiment(headlines: list[str]):
    tok, mdl = get_model()
    id2label = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    weights = {"positive": 1, "neutral": 0, "negative": -1}
    inputs = tok(headlines, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        probs = torch.softmax(mdl(**inputs).logits, dim=-1).numpy()
    labels = [id2label[int(i)] for i in probs.argmax(axis=1)]
    compound = float(np.mean([weights[l] for l in labels])) if labels else 0.0
    return [l.capitalize() for l in labels], compound


def advice_from(compound: float) -> str:
    return "BUY" if compound >= 0.5 else "SELL" if compound <= -0.5 else "HOLD"

# -----------------------------------------------------------------------------
# 🔄 INTRADAY TECHNICALS
# -----------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    close_col = next((c for c in df.columns if c.lower() == "close"), None)
    if close_col is None:
        return pd.DataFrame()
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["RSI_14"] = 100 - 100 / (1 + rs)
    df["Entry"] = (
        (df["Close"].shift(1) < df["SMA_20"].shift(1)) &
        (df["Close"] > df["SMA_20"]) &
        (df["RSI_14"] < 70)
    )
    return df

# -----------------------------------------------------------------------------
# 🖥️ MAIN APP
# -----------------------------------------------------------------------------

def main():
    st.title("📈 FinScope AI")
    st.markdown("AI-powered dashboard: news sentiment, price history & intraday signals.")
    st.markdown("---")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("🔍 Stock Selection")
        ticker_defaults = default_tickers()
        ticker = st.text_input("Ticker (e.g. AAPL)", value=ticker_defaults[0] if ticker_defaults else "AAPL").strip().upper()
        if not re.fullmatch(r"[A-Z.\-]{1,5}", ticker):
            st.warning("Enter a valid ticker (1-5 capital letters).")
            st.stop()
        period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "5y"], index=2)
        refresh_intraday = st.button("🔄 Refresh intraday")

    # ---------- API key ----------
    if "newsapi_key" not in st.secrets:
        st.error("🔑 Add your NewsAPI key to Streamlit secrets to enable sentiment analysis.")
        st.stop()
    news_key = st.secrets["newsapi_key"]

    # ---------- Price History ----------
    hist = load_history(ticker, period)
    if hist.empty:
        st.error("No price data returned – verify the ticker.")
        st.stop()

    # ---------- Sentiment ----------
    headlines = fetch_news(ticker, news_key)[:5]
    labels, compound = score_sentiment(headlines) if headlines else ([], 0.0)
    rec = advice_from(compound)

    # ---------- KPIs ----------
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg sentiment", f"{compound:+.2f}", help="–1 (all negative) … +1 (all positive). Based on last 5 headlines.")
    if len(hist) > 1:
        dchg = (hist.Close.iloc[-1] - hist.Close.iloc[-2]) / hist.Close.iloc[-2] * 100
        k2.metric("Price Δ 1-day", f"{dchg:+.2f}%", help="Close-to-close percent change.")
    else:
        k2.metric("Price Δ 1-day", "–")
    k3.metric("Advice", rec, help="BUY if sentiment ≥ +0.5, SELL if ≤ –0.5, else HOLD.")

    # ---------- Tabs ----------
    tab_news, tab_chart, tab_intraday = st.tabs(["📰 News", "📉 Chart", "⏱️ Intraday"])

    # ----- News tab -----
    with tab_news:
        st.subheader("Latest headlines")
        if not headlines:
            st.write("No recent news.")
        else:
            for hl, lbl in zip(headlines, labels):
                st.markdown(f"- **{hl}** — *{lbl}*")

    # ----- Chart tab -----
    with tab_chart:
        st.subheader(f"{ticker} price history – {period}")
        fig = go.Figure([
            go.Candlestick(
                x=hist.index,
                open=hist.Open,
                high=hist.High,
                low=hist.Low,
                close=hist.Close,
            )
        ])
        fig.update_layout(height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # ----- Intraday tab -----
    with tab_intraday:
        st.subheader("Intraday 1‑minute candles & entry signal")
        if refresh_intraday:
            load_intraday.clear()
        intra_raw = load_intraday(ticker)
        if intra_raw.empty:
            st.info("Intraday data unavailable (market closed or API limit).")
        else:
            indf = compute_indicators(intra_raw)
            if indf.empty or "Close" not in indf.columns:
                st.warning("Indicators could not be computed for this symbol at the moment.")
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
                fig2.add_trace(go.Scatter(x=indf.index, y=indf['Close'], mode='lines', name='Close'))
                fig2.add_trace(go.Scatter(x=indf.index, y=indf['SMA_20'], mode='lines', name='SMA 20'))
                entries = indf[indf['Entry']]
                if not entries.empty:
                    fig2.add_trace(
                        go.Scatter(
                            x=entries.index,
                            y=entries['Close'],
                            mode='markers',
                            marker_symbol='triangle-up',
                            marker_color='green',
                            marker_size=10,
                            name='Entry'
                        )
                    )
                fig2.update_layout(height=400, xaxis_title='Time', yaxis_title='Price')
                st.plotly_chart(fig2, use_container_width=True)

    # ---------- Persistent footer disclaimer ----------
    st.markdown(
        "<hr style='margin-top:2em'>"
        "<small><em>See full risk disclosure above. FinScope AI assumes no liability for trading losses.</em></small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
