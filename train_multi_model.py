import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ========== Load and prepare data ==========
df = pd.read_csv("drive/MyDrive/Forecasting-IHSG/dataset/jkse_data_train.csv", parse_dates=["Date"], index_col="Date")
df = df[['Close']].dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df.values)

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(df_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = Y[:train_size], Y[train_size:]

# ========== Model builder ==========
def build_model(arch, input_shape):
    model = Sequential()
    if arch == 'A':
        model.add(LSTM(50, input_shape=input_shape))
        model.add(Dense(1))
    elif arch == 'B':
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
    elif arch == 'C':
        model.add(LSTM(64, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
    elif arch == 'D':
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ========== Training and evaluation ==========
def evaluate_model(model, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0, callbacks=[early_stop])
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse, predictions

architectures = ['A', 'B', 'C', 'D']
results = {}
models = {}

for arch in architectures:
    print(f"Evaluating Model {arch}...")
    model = build_model(arch, input_shape=(time_step, 1))
    rmse, preds = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[arch] = rmse
    models[arch] = model
    print(f"Model {arch} RMSE: {rmse:.4f}")

# ========== Show result ==========
best_arch = min(results, key=results.get)
print(f"\n🏆 Best Model: {best_arch} with RMSE: {results[best_arch]:.4f}")
models[best_arch].save(f"drive/MyDrive/Forecasting-IHSG/models/lstm_jkse_best_{best_arch}.h5")
print(f"✅ Saved best model to drive/MyDrive/Forecasting-IHSG/models/lstm_jkse_best_{best_arch}.h5")

# ========== Visualize RMSE ==========
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title("Perbandingan RMSE antar Arsitektur LSTM")
plt.xlabel("Model")
plt.ylabel("RMSE (semakin rendah semakin baik)")
plt.grid(True)
plt.tight_layout()
plt.savefig("rmse_comparison.png")
plt.show()
