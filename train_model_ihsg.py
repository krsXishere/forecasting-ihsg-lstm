import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type: ignore
import joblib  # type: ignore

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

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X, Y, epochs=130, batch_size=16, validation_split=0.1, verbose=1)

predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)

joblib.dump(scaler, "scaler.pkl")
model.save("lstm_jkse_model.h5")
print("Model berhasil disimpan sebagai lstm_jkse_model.h5")

actual_prices = scaler.inverse_transform(df_scaled[time_step:])

plt.figure(figsize=(12,6))
plt.plot(df.index[time_step:-1], actual_prices[:-1], label="Actual Price", color="blue")
plt.plot(df.index[time_step:-1], predicted_prices, label="Predicted Price", color="orange")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices with LSTM")
plt.xticks(rotation=45)
plt.show()

df.index = pd.to_datetime(df.index)

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
