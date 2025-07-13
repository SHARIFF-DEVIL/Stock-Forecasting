# main.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Download stock data
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2025-06-30")
df = df[["Close"]]
df.dropna(inplace=True)

# Step 2: Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Create sequences
def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 5: Build and train LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=100, batch_size=32, callbacks=[early_stop])

# Step 6: Predict and visualize
predicted_scaled = model.predict(X_test)
predicted = scaler.inverse_transform(predicted_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.title(f"{ticker} Close Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# ---------- 7‑Day Forecast ----------
def forecast_next_n(model, scaler, last_sequence, n_days=7):
    """
    Predict the next n_days prices.
    last_sequence must be a 1‑D NumPy array of length = look‑back window (60).
    """
    seq = last_sequence.copy()          # shape: (60,)
    preds_scaled = []

    for _ in range(n_days):
        # model expects shape (1, 60, 1)
        pred = model.predict(seq.reshape(1, -1, 1), verbose=0)[0][0]
        preds_scaled.append(pred)

        # slide the window forward
        seq = np.append(seq[1:], pred)

    # rescale back to original price range
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
    return preds.flatten()

# Grab the last 60 scaled closing prices from the *entire* series
last_60_scaled = scaled_data[-60:, 0]

# Forecast next 7 trading days
next_7_prices = forecast_next_n(model, scaler, last_60_scaled, n_days=7)
print("Next‑7‑day forecast:", next_7_prices)
