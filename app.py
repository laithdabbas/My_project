import streamlit as st
import yfinance as yf
import pandas as pd
import pickle

# Load Logistic Regression model
with open("stock_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📈 Stock Price Trend Prediction")
st.write("Predict whether a stock’s next-day closing price will go Up or Down.")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if ticker:
    try:
        data = yf.download(ticker, period="5d", interval="1d")
        if len(data) > 1:
            X_latest = pd.DataFrame({
                "Close": [data["Close"].iloc[-2]],
                "Volume": [data["Volume"].iloc[-2]]
            })
            prediction = model.predict(X_latest)[0]
            prob = model.predict_proba(X_latest)[0]
            if prediction == 1:
                st.success(f"🔼 Predicted trend: UP (Probability: {prob[1]:.2f})")
            else:
                st.error(f"🔽 Predicted trend: DOWN (Probability: {prob[0]:.2f})")
        else:
            st.warning("Not enough data for this ticker.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
