from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore

df = pd.read_csv("dataset/jkse_data.csv", parse_dates=["Date"], index_col="Date")
df = df[['Close']]
df = df.dropna()
df.head()

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(df_scaled, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

loaded_model = load_model("models/lstm_jkse_model.h5")
print("Model berhasil dimuat!")

predicted = loaded_model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(df_scaled[time_step:])

print(f"Prediksi Harga Saham Berikutnya: {predicted_prices[0,0]}")

df.index = pd.to_datetime(df.index)
actual_2024 = df[df.index.year == 2024]

future_predictions = []
future_data = df_scaled[-time_step:]

for _ in range(len(actual_2024)):
    test_input = future_data.reshape(1, time_step, 1)
    pred = loaded_model.predict(test_input)
    future_predictions.append(pred[0, 0])
    
    future_data = np.append(future_data[1:], [[pred[0, 0]]], axis=0)

future_predictions = np.array(future_predictions)
future_predictions_inversed = scaler.inverse_transform(future_predictions.reshape(-1,1))

actual_2024 = df[df.index.year == 2024]

mask_2024 = df.index[time_step:-1].year == 2024

plt.figure(figsize=(12,6))
plt.plot(df.index[time_step:-1][mask_2024], actual_prices[:-1][mask_2024], label="Actual Price", color="blue")
plt.plot(df.index[time_step:-1][mask_2024], predicted_prices[mask_2024], label="Predicted Price", color="orange")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices in 2024 with LSTM")
plt.xticks(rotation=45)
plt.show()
