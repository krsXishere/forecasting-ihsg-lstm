import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model #type: ignore
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ **Load Model dan Scaler**
best_model = load_model("best_lstm_jkse_model.h5")
scaler = joblib.load("scaler.pkl")

print("✅ Model dan scaler berhasil dimuat!")

# 2️⃣ **Load Data dan Preprocessing**
df = pd.read_csv("dataset/jkse_data.csv", parse_dates=["Date"], index_col="Date")
df = df[['Close']]
df = df.dropna()

df_scaled = scaler.transform(df)

# 3️⃣ **Prediksi untuk Seluruh Data**
time_step = 60

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

X, Y = create_dataset(df_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 🔥 Prediksi menggunakan model yang dimuat
predicted = best_model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(df_scaled[time_step:])

# 4️⃣ **Plot Hasil Prediksi vs Data Aktual**
plt.figure(figsize=(12,6))
plt.plot(df.index[time_step:-1], actual_prices[:-1], label="Actual Price", color="blue")
plt.plot(df.index[time_step:-1], predicted_prices, label="Predicted Price", color="orange")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices with Loaded LSTM Model")
plt.xticks(rotation=45)
plt.show()

# 5️⃣ **Prediksi untuk Tahun 2024**
df.index = pd.to_datetime(df.index)
actual_2024 = df[df.index.year == 2024]

future_predictions = []
future_data = df_scaled[-time_step:]  # Mengambil window terakhir untuk prediksi

for _ in range(len(actual_2024)):
    test_input = future_data.reshape(1, time_step, 1)
    pred = best_model.predict(test_input)
    future_predictions.append(pred[0, 0])
    future_data = np.append(future_data[1:], [[pred[0, 0]]], axis=0)

# **Konversi hasil prediksi ke skala asli**
future_predictions = np.array(future_predictions)
future_predictions_inversed = scaler.inverse_transform(future_predictions.reshape(-1,1))

# 6️⃣ **Plot Prediksi untuk Tahun 2024**
plt.figure(figsize=(12,6))
plt.plot(actual_2024.index, actual_2024["Close"], label="Actual Price 2024", color="blue")
plt.plot(actual_2024.index, future_predictions_inversed, label="Predicted Price 2024", linestyle="dashed", color="orange")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices in 2024 using Loaded LSTM Model")
plt.xticks(rotation=45)
plt.show()
