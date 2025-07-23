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

st.title("📈 FinScope AI")
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; color: #1f77b4; }
    .section-header { font-size: 20px; margin-top: 1rem; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='big-font'>AI-Powered Stock Sentiment, Technicals & Forecasting Tool</p>", unsafe_allow_html=True)
st.markdown("---")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL")
newsapi_key = st.secrets.get("newsapi_key", "YOUR_NEWS_API_KEY")

if ticker and newsapi_key != "YOUR_NEWS_API_KEY":

    with st.expander("📰 News Sentiment Analysis", expanded=True):
        headlines = get_news(ticker, newsapi_key)
        sentiments = []
        for h in headlines:
            sent, _ = get_sentiment(h)
            sentiments.append(sent)
            st.markdown(f"- **{h}** — *{sent}*")

        recommendation = summarize_sentiments(sentiments)
        st.success(f"### 📊 Recommendation: **{recommendation}**")

    with st.expander("📉 Price Chart", expanded=True):
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if not hist.empty:
            fig, ax = plt.subplots()
            ax.plot(hist.index, hist["Close"], label="Closing Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.set_title(f"{ticker.upper()} - Last 6 Months")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No price data found for this ticker.")

    with st.expander("📐 Technical Indicators (MA, RSI, MACD)", expanded=False):
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
            axs[0].set_ylabel("Price")
            axs[0].legend()
            axs[0].set_title(f"{ticker.upper()} Price & MA")

            axs[1].plot(df.index, df["RSI"], color="orange", label="RSI")
            axs[1].axhline(70, color='red', linestyle='--')
            axs[1].axhline(30, color='green', linestyle='--')
            axs[1].set_ylabel("RSI")
            axs[1].legend()
            axs[1].set_title("Relative Strength Index (RSI)")

            axs[2].plot(df.index, df["MACD"], label="MACD", color="blue")
            axs[2].plot(df.index, df["Signal"], label="Signal", color="magenta")
            axs[2].axhline(0, color='black', linestyle='--')
            axs[2].set_ylabel("MACD")
            axs[2].legend()
            axs[2].set_title("MACD")

            st.pyplot(fig)

    with st.expander("🔮 7-Day Forecast", expanded=False):
        try:
            df = stock.history(period="1y")
            df = df[["Close"]].reset_index()
            df = df.rename(columns={"Date": "ds", "Close": "y"})
            df["ds"] = df["ds"].dt.tz_localize(None)

            model = Prophet(daily_seasonality=True)
            model.fit(df)

            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)

            st.write("Forecasted Prices (next 7 days):")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))

            fig2 = plot_plotly(model, forecast)
            st.plotly_chart(fig2)

        except Exception as e:
            st.error(f"Failed to generate forecast: {e}")

    with st.expander("🧾 Option Chain & Sensitivity", expanded=False):
        try:
            expirations = stock.options
            if expirations:
                nearest_expiry = expirations[0]
                opt_chain = stock.option_chain(nearest_expiry)
                calls = opt_chain.calls
                puts = opt_chain.puts

                current_price = hist["Close"][-1]
                calls["diff"] = abs(calls["strike"] - current_price)
                puts["diff"] = abs(puts["strike"] - current_price)
                atm_call = calls.sort_values("diff").iloc[0]
                atm_put = puts.sort_values("diff").iloc[0]

                st.markdown(f"**Stock Price:** ${current_price:.2f}")
                st.markdown(f"**ATM Call Strike:** ${atm_call['strike']} — Bid: ${atm_call['bid']}, Ask: ${atm_call['ask']}")
                st.markdown(f"**ATM Put Strike:** ${atm_put['strike']} — Bid: ${atm_put['bid']}, Ask: ${atm_put['ask']}")

                st.markdown("#### 🔁 Simulate Option Sensitivity")
                sim_change = st.slider("Simulate Stock Price Change (%)", -5.0, 5.0, 1.0, step=0.5)

                option_leverage = 5
                call_pct_change = sim_change * option_leverage
                put_pct_change = -sim_change * option_leverage

                call_price = (atm_call['bid'] + atm_call['ask']) / 2
                put_price = (atm_put['bid'] + atm_put['ask']) / 2

                call_estimated = call_price * (1 + call_pct_change / 100)
                put_estimated = put_price * (1 + put_pct_change / 100)

                st.success(f"💰 Call Estimate: `{call_pct_change:.1f}%` → ${call_estimated:.2f}")
                st.error(f"📉 Put Estimate: `{put_pct_change:.1f}%` → ${put_estimated:.2f}")

            else:
                st.warning("No option chain available for this ticker.")

        except Exception as e:
            st.error(f"Failed to simulate options sensitivity: {e}")

else:
    st.info("Enter a stock ticker and configure your NewsAPI key in secrets to begin.")
