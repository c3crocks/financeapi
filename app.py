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

# Page configuration
st.set_page_config(page_title="FinScope AI", page_icon="📈", layout="wide")

# Load sentiment model (cached)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_model()

# Functions
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

# UI Layout
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

# API key
newsapi_key = st.secrets.get("newsapi_key", "YOUR_NEWS_API_KEY")

if ticker and newsapi_key != "YOUR_NEWS_API_KEY":
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    tab1, tab2 = st.tabs(["📊 Stock Analysis", "📐 Technical Analysis"])

    # TAB 1 — Sentiment & Price Chart
    with tab1:
        col1, col2 = st.columns(2)

        # News & Sentiment
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

        # Price Chart
        with col2:
            st.subheader("📉 Price Chart")
            if not hist.empty:
                fig, ax = plt.subplots()
                ax.plot(hist.index, hist["Close"], label="Closing Price", color='blue')
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
        # Forecasting with Prophet
        st.subheader("🔮 7-Day Forecast (Prophet)")
        try:
            df_prophet = yf.download(ticker, period="1y", progress=False)[["Close"]].dropna().reset_index()
            df_prophet = df_prophet.rename(columns={"Date": "ds", "Close": "y"})
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

            if df_prophet.empty or len(df_prophet) < 30:
                st.warning("Not enough data for forecast.")
            else:
                # Remove outliers
                q1 = df_prophet["y"].quantile(0.25)
                q3 = df_prophet["y"].quantile(0.75)
                iqr = q3 - q1
                df_prophet = df_prophet[(df_prophet["y"] >= q1 - 1.5 * iqr) & (df_prophet["y"] <= q3 + 1.5 * iqr)]

                # Log transform
                df_prophet["y"] = np.log(df_prophet["y"])
                model = Prophet(daily_seasonality=True)
                model.fit(df_prophet)

                future = model.make_future_dataframe(periods=7)
                forecast = model.predict(future)

                # Convert back to original price scale
                forecast[["yhat", "yhat_lower", "yhat_upper"]] = np.exp(forecast[["yhat", "yhat_lower", "yhat_upper"]])

                # Get last actual log-close value and scale
                last_actual = np.exp(df_prophet["y"].iloc[-1])
                forecast_base = forecast["yhat"].iloc[len(df_prophet) - 1]  # match index
                scale_factor = float(last_actual / forecast_base)  # ensure scalar

                # Apply scale
                forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[["yhat", "yhat_lower", "yhat_upper"]] * scale_factor

                # Display forecast
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
                forecast_display['ds'] = forecast_display['ds'].dt.date
                st.write("Forecasted Prices (Next 7 Days):")
                st.dataframe(forecast_display)

                fig4 = plot_plotly(model, forecast)
                st.plotly_chart(fig4)

        except Exception as e:
            st.error(f"⚠️ Forecast error: {e}")


else:
    st.info("Enter a valid stock ticker and set your NewsAPI key in `.streamlit/secrets.toml`.")
