import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from bs4 import BeautifulSoup
import urllib.request
import numpy as np

# Page setup
st.set_page_config(page_title="FinScope AI", page_icon="📈", layout="wide")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_model()

def get_news(ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])[:5]
    return [a["title"] for a in articles]

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiments = ["Negative", "Neutral", "Positive"]
    return sentiments[torch.argmax(probs)], probs.detach().numpy().flatten()

def summarize_sentiments(sentiment_scores):
    pos = sum(s == "Positive" for s in sentiment_scores)
    neg = sum(s == "Negative" for s in sentiment_scores)
    if pos >= 3:
        return "BUY"
    elif neg >= 3:
        return "SELL"
    return "HOLD"

def scrape_motley_fool():
    url = "https://www.fool.com/investing/"
    try:
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page, "html.parser")
        tickers = list(set([a.text for a in soup.find_all('a') if a.text.isupper() and len(a.text) <= 5]))
        return tickers[:10]
    except:
        return []

# UI
st.title("📈 FinScope AI")
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<p class='big-font'>AI-Powered Stock Sentiment, Technicals & Forecasting Tool</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔍 Stock Selection")
    default_tickers = scrape_motley_fool()
    company_search = st.text_input("Search company name or ticker", value="AAPL")
    ticker = yf.Ticker(company_search).info.get("symbol", company_search.upper())

# News API key
newsapi_key = st.secrets.get("newsapi_key", "YOUR_NEWS_API_KEY")

if ticker and newsapi_key != "YOUR_NEWS_API_KEY":
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    tab1, tab2 = st.tabs(["📊 Stock Analysis", "📐 Technical Analysis"])

    # TAB 1 — Sentiment & Price Chart
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📰 News Sentiment Analysis")
            headlines = get_news(ticker, newsapi_key)
            sentiments = []
            for h in headlines:
                sent, _ = get_sentiment(h)
                sentiments.append(sent)
                st.markdown(f"- **{h}** — *{sent}*")
            recommendation = summarize_sentiments(sentiments)
            st.success(f"### 📊 Recommendation: **{recommendation}**")

        with col2:
            st.subheader("📉 Price Chart")
            if not hist.empty:
                fig, ax = plt.subplots()
                ax.plot(hist.index, hist["Close"], label="Close Price", color='blue')
                ax.set_title(f"{ticker.upper()} - Last 6 Months")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.warning("No price data found.")

    # TAB 2 — Technicals & Forecast
    with tab2:
        st.subheader("📐 Technical Indicators")
        if not hist.empty:
            df = hist.copy()
            df["MA20"] = df["Close"].rolling(window=20).mean()
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axs[0].plot(df.index, df["Close"], label="Close Price")
            axs[0].plot(df.index, df["MA20"], label="MA20", linestyle="--")
            axs[0].set_title("Price & Moving Average")
            axs[0].legend()

            axs[1].plot(df.index, df["RSI"], color="orange", label="RSI")
            axs[1].axhline(70, color='red', linestyle='--')
            axs[1].axhline(30, color='green', linestyle='--')
            axs[1].legend()
            axs[1].set_title("Relative Strength Index")

            axs[2].plot(df.index, df["MACD"], label="MACD", color="blue")
            axs[2].plot(df.index, df["Signal"], label="Signal", color="magenta")
            axs[2].axhline(0, color='black', linestyle='--')
            axs[2].legend()
            axs[2].set_title("MACD")

            st.pyplot(fig)

        # Forecasting with Prophet
        # Forecasting with Linear Regression
        st.subheader("🔮 7-Day Forecast (Linear Regression)")
        try:
            from sklearn.linear_model import LinearRegression

            df_lr = yf.download(ticker, period="6mo", progress=False)[["Close"]].dropna().reset_index()
            df_lr.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
            df_lr["ds"] = pd.to_datetime(df_lr["ds"])
            df_lr["ds_ordinal"] = df_lr["ds"].map(pd.Timestamp.toordinal)

            # Prepare training data
            X = df_lr["ds_ordinal"].values.reshape(-1, 1)
            y = df_lr["y"].values

            model = LinearRegression()
            model.fit(X, y)

            # Predict next 7 days
            last_date = df_lr["ds"].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
            future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            future_preds = model.predict(future_ordinals)

            forecast_display = pd.DataFrame({
                "ds": [d.date() for d in future_dates],
                "yhat": future_preds
            })

            st.write("Forecasted Prices (Next 7 Days):")
            st.dataframe(forecast_display)

            # Plot
            fig, ax = plt.subplots()
            ax.plot(df_lr["ds"], df_lr["y"], label="Historical Close")
            ax.plot(forecast_display["ds"], forecast_display["yhat"], label="Forecast", color="orange", linestyle="--")
            ax.set_title(f"{ticker.upper()} Forecast (Linear Regression)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Forecast error: {e}")




else:
    st.info("Enter a valid stock ticker and set your NewsAPI key in `.streamlit/secrets.toml`.")
