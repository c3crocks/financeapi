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
    if default_tickers:
        ticker = st.selectbox("Select from top tickers or enter manually", default_tickers + ["Other"])
        if ticker == "Other":
            ticker = st.text_input("Enter stock ticker", value="AAPL")
    else:
        ticker = st.text_input("Enter stock ticker", value="AAPL")

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

        st.subheader("🔮 7-Day Forecast (LSTM)")
        try:
            from sklearn.preprocessing import MinMaxScaler
            from keras.models import Sequential
            from keras.layers import LSTM, Dense
            import numpy as np

            df_lstm = stock.history(period="1y")[["Close"]].dropna()
            df_lstm = df_lstm.rename(columns={"Close": "y"})

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_lstm.values)

            seq_len = 60
            x_train, y_train = [], []
            for i in range(seq_len, len(scaled_data)):
                x_train.append(scaled_data[i - seq_len:i, 0])
                y_train.append(scaled_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            model_lstm = Sequential()
            model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model_lstm.add(LSTM(units=50))
            model_lstm.add(Dense(1))
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            model_lstm.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

            inputs = scaled_data[-seq_len:].reshape(1, seq_len, 1)
            forecast = []
            for _ in range(7):
                pred = model_lstm.predict(inputs)[0][0]
                forecast.append(pred)
                inputs = np.append(inputs[:, 1:, :], [[[pred]]], axis=1)

            forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            forecast_dates = pd.date_range(start=df_lstm.index[-1] + pd.Timedelta(days=1), periods=7)
            forecast_df = pd.DataFrame({"Date": forecast_dates.date, "Predicted Close": forecast_prices.flatten()})

            st.write("Forecasted Prices (next 7 days):")
            st.dataframe(forecast_df)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_lstm.index[-30:], y=df_lstm["y"].values[-30:], mode='lines', name='Past Prices'))
            fig3.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices.flatten(), mode='lines+markers', name='Forecast'))
            fig3.update_layout(title="LSTM Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig3)

        except Exception as e:
            st.error(f"Failed to generate LSTM forecast: {e}")

        except Exception as e:
            st.error(f"Failed to generate forecast: {e}")

else:
    st.info("Enter a stock ticker and configure your NewsAPI key in secrets to begin.")
