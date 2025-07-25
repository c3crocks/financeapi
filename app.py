import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import urllib.request
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(
    page_title="FinScope AI",
    page_icon="📈",
    layout="wide"
)

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
    headlines = [a["title"] for a in articles]
    return headlines

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
    else:
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

st.title("📈 FinScope AI")
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; color: #1f77b4; }
    .section-header { font-size: 20px; margin-top: 1rem; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='big-font'>AI-Powered Stock Sentiment, Technicals & Forecasting Tool</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("🔍 Stock Selection")
    default_tickers = scrape_motley_fool()
    company_search = st.text_input("Search company name or ticker", value="Apple")

    if company_search:
        search_results = yf.Ticker(company_search).info.get("symbol", company_search.upper())
        ticker = search_results
    else:
        ticker = "AAPL"

newsapi_key = st.secrets.get("newsapi_key", "YOUR_NEWS_API_KEY")

if ticker and newsapi_key != "YOUR_NEWS_API_KEY":
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    tab1, tab2 = st.tabs(["📊 Stock Analysis", "📐 Technical Analysis"])

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
                ax.plot(hist.index, hist["Close"], label="Closing Price", color='blue')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.set_title(f"{ticker.upper()} - Last 6 Months")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.warning("No price data found for this ticker.")

    with tab2:
        st.subheader("🔮 7-Day Forecast (Prophet)")
        try:
            # Fetch fresh data directly, bypassing Ticker cache
            df_prophet = yf.download(ticker, period="1y", progress=False)[["Close"]].dropna().reset_index()
            df_prophet = df_prophet.rename(columns={"Date": "ds", "Close": "y"})
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

            # Optional: Display last available date to verify freshness
            st.write(f"📅 Last available date in data: {df_prophet['ds'].max().date()}")

            # Outlier filtering using IQR
            q1 = df_prophet["y"].quantile(0.25)
            q3 = df_prophet["y"].quantile(0.75)
            iqr = q3 - q1
            df_prophet = df_prophet[(df_prophet["y"] >= q1 - 1.5 * iqr) & (df_prophet["y"] <= q3 + 1.5 * iqr)]

            # Log transform for better forecasting
            df_prophet["y"] = np.log(df_prophet["y"])

            # Build and fit Prophet model
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)

            # Forecast 7 future days
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)

            # Revert log transform
            forecast[["yhat", "yhat_lower", "yhat_upper"]] = np.exp(forecast[["yhat", "yhat_lower", "yhat_upper"]])

            # Rescale forecast to match last actual close price
            last_close = df_prophet["y"].iloc[-1]
            original_last_close = np.exp(last_close)
            forecast_base = forecast.iloc[-8]["yhat"]  # 8th last = last known value before future
            scale_factor = original_last_close / forecast_base
            forecast[["yhat", "yhat_lower", "yhat_upper"]] *= scale_factor

            # Prepare final forecast table
            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
            forecast_display['ds'] = forecast_display['ds'].dt.date

            st.write("Forecasted Prices (next 7 days):")
            st.dataframe(forecast_display)

            # Interactive plot
            fig4 = plot_plotly(model, forecast)
            st.plotly_chart(fig4)

        except Exception as e:
            st.error(f"⚠️ Failed to generate forecast: {e}")


else:
    st.info("Enter a stock ticker and configure your NewsAPI key in secrets to begin.")
