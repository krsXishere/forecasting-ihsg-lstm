import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

# Load and prepare data
df = pd.read_csv("dataset/jkse_data.csv", parse_dates=["Date"], index_col="Date")
df = df[['Close']].dropna()

# Scale data
df_values = df.values
n_values = len(df_values)
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_values)

# Create dataset for time-series

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Hyperparameters
time_step = 60
X, Y = create_dataset(df_scaled, time_step)

# Reshape input for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the LSTM model
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

# Train the model
history = model.fit(X, Y, epochs=60, batch_size=16, validation_split=0.1, verbose=1)

# Make predictions
predicted = model.predict(X)
# Inverse transform to original scale
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(df_scaled[time_step:len(predicted) + time_step])

# Calculate RMSE (raw) and log it
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"Raw RMSE: {rmse:.4f}")

# Calculate average of actual prices and ratio
avg_actual = np.mean(actual_prices)
ratio = avg_actual / rmse if rmse != 0 else float('inf')
print(f"Average Actual Price: {avg_actual:.4f}")
print(f"Average Actual / RMSE: {ratio:.4f}")

# (Optional) percentage error
percent_error = (rmse / avg_actual) * 100 if avg_actual != 0 else float('inf')
print(f"RMSE as Percentage of Average: {percent_error:.2f}%")

# Plot overall results
plt.figure(figsize=(12, 6))
plt.plot(df.index[time_step:time_step + len(predicted_prices)], actual_prices.flatten(), label="Actual Price")
plt.plot(df.index[time_step:time_step + len(predicted_prices)], predicted_prices.flatten(), label="Predicted Price")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices with LSTM")
plt.xticks(rotation=45)
plt.show()

# Save the model
model.save("models/lstm_jkse_model2.h5")
print("Model berhasil disimpan sebagai lstm_jkse_model.h5")

# Plot for 2024 only
mask_2024 = df.index[time_step:time_step + len(predicted_prices)].year == 2024
plt.figure(figsize=(12, 6))
plt.plot(df.index[time_step:time_step + len(predicted_prices)][mask_2024],
         actual_prices.flatten()[mask_2024], label="Actual Price")
plt.plot(df.index[time_step:time_step + len(predicted_prices)][mask_2024],
         predicted_prices.flatten()[mask_2024], label="Predicted Price")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices in 2024 with LSTM")
plt.xticks(rotation=45)
plt.show()
