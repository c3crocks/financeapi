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


# Load FinBERT model
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

st.title("📈 FinScope AI: NLP-Powered Stock Sentiment Analysis")
st.write("Enter a stock ticker to get a Buy/Sell/Hold recommendation based on recent news.")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT)")
newsapi_key = st.secrets.get("newsapi_key", "YOUR_NEWS_API_KEY")

if ticker and newsapi_key != "YOUR_NEWS_API_KEY":
    with st.spinner("Fetching news and analyzing sentiment..."):
        headlines = get_news(ticker, newsapi_key)
        sentiments = []
        for h in headlines:
            sent, scores = get_sentiment(h)
            sentiments.append(sent)
            st.markdown(f"**{h}** — *{sent}*")

        recommendation = summarize_sentiments(sentiments)
        st.success(f"### 📊 Recommendation: **{recommendation}**")
else:
    st.info("Enter a ticker and set your NewsAPI key to begin.")


st.subheader(f"📊 Price Chart for {ticker.upper()}")

try:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")  # Last 6 months

    if hist.empty:
        st.warning("No price data found for this ticker.")
    else:
        fig, ax = plt.subplots()
        ax.plot(hist.index, hist["Close"], label="Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.set_title(f"{ticker.upper()} - Last 6 Months")
        ax.legend()
        st.pyplot(fig)
except Exception as e:
    st.error(f"Failed to fetch or display chart: {e}")


st.subheader(f"📊 Technical Chart for {ticker.upper()}")

try:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    if hist.empty:
        st.warning("No price data found for this ticker.")
    else:
        df = hist.copy()

        # --- Moving Average ---
        df["MA20"] = df["Close"].rolling(window=20).mean()

        # --- RSI Calculation ---
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # --- MACD Calculation ---
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # --- Plotting ---
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Price and MA
        axs[0].plot(df.index, df["Close"], label="Close Price")
        axs[0].plot(df.index, df["MA20"], label="MA20", linestyle="--")
        axs[0].set_ylabel("Price")
        axs[0].legend()
        axs[0].set_title(f"{ticker.upper()} Price & MA")

        # RSI
        axs[1].plot(df.index, df["RSI"], color="orange", label="RSI")
        axs[1].axhline(70, color='red', linestyle='--')
        axs[1].axhline(30, color='green', linestyle='--')
        axs[1].set_ylabel("RSI")
        axs[1].legend()
        axs[1].set_title("Relative Strength Index (RSI)")

        # MACD
        axs[2].plot(df.index, df["MACD"], label="MACD", color="blue")
        axs[2].plot(df.index, df["Signal"], label="Signal", color="magenta")
        axs[2].axhline(0, color='black', linestyle='--')
        axs[2].set_ylabel("MACD")
        axs[2].legend()
        axs[2].set_title("MACD")

        st.pyplot(fig)

except Exception as e:
    st.error(f"Failed to fetch or display chart: {e}")
#Forecasting code
st.subheader(f"🔮 7-Day Forecast for {ticker.upper()}")

try:
    df = stock.history(period="1y")
    df = df[["Close"]].reset_index()
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df["ds"] = df["ds"].dt.tz_localize(None)  


    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # Show forecast data table (optional)
    st.write("Forecasted Prices:")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))

    # Plot forecast
    fig2 = plot_plotly(model, forecast)
    st.plotly_chart(fig2)

except Exception as e:
    st.error(f"Failed to generate forecast: {e}")

# Options code
# Get available expiration dates
# Get the most recent closing price
current_price = hist["Close"][-1]

st.subheader("🔁 Simulated Option Price Sensitivity")

sim_change = st.slider("Simulate Stock Price Change (%)", min_value=-5.0, max_value=5.0, value=1.0, step=0.5)

# Simulated stock price
simulated_stock_price = current_price * (1 + sim_change / 100)

# Assume leverage: option moves 5x the stock move
option_leverage = 5
call_pct_change = sim_change * option_leverage
put_pct_change = -sim_change * option_leverage

call_estimated_price = atm_call['lastPrice'] * (1 + call_pct_change / 100)
put_estimated_price = atm_put['lastPrice'] * (1 + put_pct_change / 100)

st.markdown(f"📈 **If stock changes by `{sim_change:.1f}%`, then:**")

st.info(f"💰 **Call ({atm_call['strike']}$):** would change by `{call_pct_change:.1f}%` → Estimated price: `${call_estimated_price:.2f}`")
st.info(f"📉 **Put ({atm_put['strike']}$):** would change by `{put_pct_change:.1f}%` → Estimated price: `${put_estimated_price:.2f}`")
