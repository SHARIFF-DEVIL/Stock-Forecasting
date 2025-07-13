import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 Stock Price Predictor with LSTM")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-06-30"))

if st.button("Predict"):
    # 1. Load data
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[["Close"]].dropna()
    
    if len(df) < 100:
        st.error("Not enough data to train model. Choose a different date range.")
        st.stop()

    st.success("Data Loaded")
    st.line_chart(df)

    # 2. Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # 3. Create sequences
    def create_sequences(data, window=60):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 4. Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 5. Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early = EarlyStopping(patience=10, restore_best_weights=True)

    with st.spinner("Training LSTM model..."):
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=100, batch_size=32, callbacks=[early], verbose=0)
    st.success("Model Trained")

    # 6. Predict
    pred_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 7. Forecast next 7 days
    def forecast_next_n(model, scaler, last_sequence, n_days=7):
        seq = last_sequence.copy()
        preds_scaled = []
        for _ in range(n_days):
            pred = model.predict(seq.reshape(1, -1, 1), verbose=0)[0][0]
            preds_scaled.append(pred)
            seq = np.append(seq[1:], pred)
        return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    last_60 = scaled_data[-60:, 0]
    next_7 = forecast_next_n(model, scaler, last_60)

    # 8. Display plots
    future_dates = pd.bdate_range(df.index[-1], periods=8, closed="right")
    forecast_df = pd.DataFrame({"Forecast": next_7}, index=future_dates)

    st.subheader("📉 Actual vs Predicted Closing Prices")
    chart_data = pd.DataFrame({
        "Actual": actual.flatten(),
        "Predicted": predicted.flatten()
    })
    st.line_chart(chart_data[-200:])

    st.subheader("🔮 Next 7-Day Forecast")
    st.line_chart(forecast_df)

    st.write("Forecasted values:", forecast_df)
