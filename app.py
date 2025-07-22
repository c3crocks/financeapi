import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yfinance as yf
import matplotlib.pyplot as plt


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
