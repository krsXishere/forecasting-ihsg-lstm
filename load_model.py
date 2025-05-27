from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore

df = pd.read_csv("dataset/jkse_test.csv", parse_dates=["Date"], index_col="Date")
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

loaded_model = load_model("models/lstm_jkse_model2.h5")
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

plt.figure(figsize=(12,6))
# Remove [:-1] from actual_prices
plt.plot(df.index[time_step:-1][mask_2024], actual_prices[mask_2024], label="Actual Price", color="blue")
plt.plot(df.index[time_step:-1][mask_2024], predicted_prices[mask_2024], label="Predicted Price", color="orange")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices in 2024 with LSTM")
plt.xticks(rotation=45)
plt.show()

# from tensorflow.keras.models import load_model  # type: ignore
# import matplotlib.pyplot as plt  # type: ignore
# import pandas as pd  # type: ignore
# import numpy as np  # type: ignore
# from sklearn.preprocessing import MinMaxScaler  # type: ignore
# from sklearn.metrics import mean_squared_error

# # 1. Load dan siapkan data
# df = pd.read_csv("dataset/jkse_test.csv", parse_dates=["Date"], index_col="Date")
# df = df[['Close']].dropna()

# # 2. Scaling
# scaler = MinMaxScaler(feature_range=(0, 1))
# df_scaled = scaler.fit_transform(df)

# # 3. Buat dataset untuk LSTM
# def create_dataset(data, time_step=60):
#     X, Y = [], []
#     for i in range(len(data) - time_step):
#         X.append(data[i:(i + time_step), 0])
#         Y.append(data[i + time_step, 0])
#     return np.array(X), np.array(Y)

# time_step = 60
# X, Y = create_dataset(df_scaled, time_step)
# X = X.reshape(X.shape[0], X.shape[1], 1)

# # 4. Load model
# loaded_model = load_model("models/lstm_jkse_model2.h5")
# print("Model berhasil dimuat!")

# # 5. Predict
# predicted = loaded_model.predict(X)
# predicted_prices = scaler.inverse_transform(predicted)
# actual_prices = scaler.inverse_transform(df_scaled[time_step:])

# print(f"Prediksi Harga Saham Berikutnya: {predicted_prices[0,0]}")

# # 6. Siapkan actual yang sepadan: hanya mulai dari index ke-time_step
# actual_prices = df.values[time_step:]                     # shape (N,1)

# future_predictions = []
# future_data = df_scaled[-time_step:]

# for _ in range(len(actual_2024)):
#     test_input = future_data.reshape(1, time_step, 1)
#     pred = loaded_model.predict(test_input)
#     future_predictions.append(pred[0, 0])
    
#     future_data = np.append(future_data[1:], [[pred[0, 0]]], axis=0)

# # 7. Hitung RMSE
# rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
# print(f"Raw RMSE: {rmse:.4f}")

# # 8. Rasio & persentase
# avg_actual = actual_prices.mean()
# ratio = avg_actual / rmse if rmse != 0 else np.inf
# percent_error = (rmse / avg_actual) * 100 if avg_actual != 0 else np.inf

# print(f"Average Actual Price: {avg_actual:.4f}")
# print(f"Average Actual / RMSE: {ratio:.4f}")
# print(f"RMSE as Percentage of Average: {percent_error:.2f}%")

# # 9. Plot Actual vs Predicted untuk tahun 2024 saja
# df.index = pd.to_datetime(df.index)
# dates = df.index[time_step:]
# mask_2024 = dates.year == 2024

# plt.figure(figsize=(12,6))
# plt.plot(dates[mask_2024], actual_prices[mask_2024], label="Actual Price")
# plt.plot(dates[mask_2024], predicted_prices[mask_2024], label="Predicted Price")
# plt.xlabel("Date")
# plt.ylabel("Stock Price")
# plt.title("Actual vs Predicted JKSE Prices in 2024")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

# # Load data
# df = pd.read_csv("dataset/jkse_test.csv", parse_dates=["Date"], index_col="Date")
# df = df[['Close']].dropna()

# # Normalisasi data
# scaler = MinMaxScaler(feature_range=(0, 1))
# df_scaled = scaler.fit_transform(df)

# # Fungsi pembuatan dataset
# def create_dataset(data, time_step=60):
#     X, Y = [], []
#     for i in range(len(data) - time_step - 1):
#         X.append(data[i:(i + time_step), 0])
#         Y.append(data[i + time_step, 0])
#     return np.array(X), np.array(Y)

# time_step = 60
# X, Y = create_dataset(df_scaled, time_step)
# X = X.reshape(X.shape[0], X.shape[1], 1)

# # Load model
# loaded_model = load_model("models/lstm_jkse_model2.h5")
# print("Model berhasil dimuat!")

# # Prediksi
# predicted = loaded_model.predict(X)
# predicted_prices = scaler.inverse_transform(predicted)
# actual_prices = scaler.inverse_transform(df_scaled[time_step:])

# print(f"Prediksi Harga Saham Berikutnya: {predicted_prices[0,0]}")

# # Persiapan prediksi 2024
# actual_2024 = df[df.index.year == 2024]
# future_data = df_scaled[-time_step:]

# # Generate prediksi future
# future_predictions = []
# for _ in range(len(actual_2024)):
#     test_input = future_data.reshape(1, time_step, 1)
#     pred = loaded_model.predict(test_input, verbose=0)
#     future_predictions.append(pred[0, 0])
#     future_data = np.append(future_data[1:], [[pred[0, 0]]], axis=0)

# future_predictions_inversed = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# # Buat DataFrame untuk prediksi
# future_predictions_df = pd.DataFrame(
#     future_predictions_inversed,
#     index=actual_2024.index,
#     columns=['Predicted']
# )

# # Filter untuk Januari 2024
# jan_mask = (actual_2024.index.month == 2) & (actual_2024.index.year == 2024)
# actual_jan_2024 = actual_2024[jan_mask]
# predicted_jan_2024 = future_predictions_df[jan_mask]

# # Hitung metrik untuk Januari
# jan_rmse = np.sqrt(mean_squared_error(actual_jan_2024, predicted_jan_2024))
# jan_mape = np.mean(np.abs((actual_jan_2024.values - predicted_jan_2024.values) / actual_jan_2024.values)) * 100

# print("\n=== Evaluasi Januari 2024 ===")
# print(f"RMSE: {jan_rmse:.2f}")
# print(f"MAPE: {jan_mape:.2f}%")

# # 8. Rasio & persentase
# avg_actual = actual_prices.mean()
# ratio = avg_actual / jan_rmse if jan_rmse != 0 else np.inf
# percent_error = (jan_rmse / avg_actual) * 100 if avg_actual != 0 else np.inf

# print(f"Average Actual Price: {avg_actual:.4f}")
# print(f"Average Actual / RMSE: {ratio:.4f}")
# print(f"RMSE as Percentage of Average: {percent_error:.2f}%")

# # Plotting
# plt.figure(figsize=(14, 7))
# plt.plot(actual_jan_2024.index, actual_jan_2024['Close'], 
#          label='Harga Aktual', marker='o', linewidth=2, color='#1f77b4')
# plt.plot(predicted_jan_2024.index, predicted_jan_2024['Predicted'], 
#          label='Prediksi LSTM', marker='s', linestyle='--', color='#ff7f0e')

# plt.title('Perbandingan Harga Saham JKSE: Aktual vs Prediksi (Januari 2024)', fontsize=16)
# plt.xlabel('Tanggal', fontsize=12)
# plt.ylabel('Harga Penutupan (IDR)', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Tampilkan tabel
# jan_comparison = pd.DataFrame({
#     'Tanggal': actual_jan_2024.index.strftime('%Y-%m-%d'),
#     'Aktual': actual_jan_2024['Close'].round(2),
#     'Prediksi': predicted_jan_2024['Predicted'].round(2),
#     'Selisih': (actual_jan_2024['Close'] - predicted_jan_2024['Predicted']).round(2)
# })

# print("\nTabel Perbandingan Harian:")
# print(jan_comparison.to_string(index=False))

# plt.show()