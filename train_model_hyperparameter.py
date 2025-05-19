import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type: ignore
import joblib
import keras_tuner as kt
from keras_tuner import RandomSearch

# 1️⃣ Load Dataset
df = pd.read_csv("dataset/jkse_data.csv", parse_dates=["Date"], index_col="Date")
df = df[['Close']]
df = df.dropna()

# 2️⃣ Normalisasi Data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# 3️⃣ Buat Dataset untuk LSTM
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(df_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 4️⃣ Definisikan Fungsi Model untuk Tuning
def build_model(hp):
    model = Sequential()
    
    # Layer pertama LSTM
    model.add(LSTM(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        return_sequences=True, 
        input_shape=(time_step, 1)
    ))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # Layer kedua LSTM (opsional)
    if hp.Boolean("second_lstm_layer"):
        model.add(LSTM(
            units=hp.Int('units_2', min_value=32, max_value=128, step=32),
            return_sequences=False
        ))
        model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    else:
        model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), return_sequences=False))
    
    # Fully connected layer
    model.add(Dense(hp.Int('dense_units', min_value=10, max_value=50, step=10), activation='relu'))
    model.add(Dense(1))

    # Optimizer dengan learning rate yang bisa disesuaikan
    optimizer = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# 5️⃣ Hyperparameter Tuning dengan Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='loss',
    max_trials=2,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='lstm_jkse_tuning'
)

# 🔥 Cari kombinasi hyperparameter terbaik
tuner.search(X, Y, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# 6️⃣ Dapatkan Hyperparameter Terbaik dan Bangun Model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# print(f"""
# Best Hyperparameters:
# - LSTM units 1: {best_hps.get('units_1')}
# - Dropout 1: {best_hps.get('dropout_1')}
# - LSTM units 2: {best_hps.get('units_2')}
# - Dropout 2: {best_hps.get('dropout_2')}
# - Dense units: {best_hps.get('dense_units')}
# - Learning rate: {best_hps.get('learning_rate')}
# - Second LSTM Layer: {best_hps.get('second_lstm_layer')}
# """)

best_model = tuner.hypermodel.build(best_hps)

# 🔥 Latih model terbaik dengan hyperparameter optimal
history = best_model.fit(X, Y, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# 7️⃣ Simpan Model dan Scaler
best_model.save("best_lstm_jkse_model.h5")
joblib.dump(scaler, "scaler.pkl")
print("Model dan scaler berhasil disimpan!")

# 8️⃣ Prediksi dengan Model yang Sudah Dituning
predicted = best_model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(df_scaled[time_step:])

# 9️⃣ Plot Hasil Prediksi vs Aktual
plt.figure(figsize=(12,6))
plt.plot(df.index[time_step:-1], actual_prices[:-1], label="Actual Price", color="blue")
plt.plot(df.index[time_step:-1], predicted_prices, label="Predicted Price", color="orange")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices with Optimized LSTM")
plt.xticks(rotation=45)
plt.show()

# 🔟 Prediksi untuk Tahun 2024
df.index = pd.to_datetime(df.index)
actual_2024 = df[df.index.year == 2024]

future_predictions = []
future_data = df_scaled[-time_step:]

for _ in range(len(actual_2024)):
    test_input = future_data.reshape(1, time_step, 1)
    pred = best_model.predict(test_input)
    future_predictions.append(pred[0, 0])
    future_data = np.append(future_data[1:], [[pred[0, 0]]], axis=0)

future_predictions = np.array(future_predictions)
future_predictions_inversed = scaler.inverse_transform(future_predictions.reshape(-1,1))

# 1️⃣1️⃣ Plot Prediksi Tahun 2024
plt.figure(figsize=(12,6))
plt.plot(actual_2024.index, actual_2024["Close"], label="Actual Price 2024", color="blue")
plt.plot(actual_2024.index, future_predictions_inversed, label="Predicted Price 2024", linestyle="dashed", color="orange")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Comparison: Actual vs Predicted JKSE Prices in 2024 with Optimized LSTM")
plt.xticks(rotation=45)
plt.show()
