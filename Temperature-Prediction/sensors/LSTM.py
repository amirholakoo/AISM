from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping

# --- Load Data ---
data = pd.read_csv('DHT22-172.16.3.245.csv')
data['Time steps'] = pd.to_datetime(data['Time steps'])
data = data.sort_values('Time steps')  # Ensure chronological order

# --- Extract Humidity and Scale ---
humidity = data['Humidity'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(humidity)

# Save the scaler for future use
joblib.dump(scaler, 'humidity_scaler.pkl')

# --- Train/Validation/Test Split ---
train_ratio, val_ratio = 0.7, 0.15  # 70% train, 15% validation, 15% test
train_len = int(len(scaled_data) * train_ratio)
val_len = int(len(scaled_data) * val_ratio)

train_data = scaled_data[:train_len]
val_data = scaled_data[train_len:train_len + val_len]
test_data = scaled_data[train_len + val_len - 180:]  # Overlap for sequence creation

# --- Prepare Sequences ---
def create_sequences(data, input_steps, forecast_intervals):
    X, y = [], []
    max_forecast = max(forecast_intervals)
    for i in range(input_steps, len(data) - max_forecast):
        X.append(data[i - input_steps:i, 0])
        y.append([data[i + k - 1, 0] for k in forecast_intervals])  # Adjust index for intervals
    return np.array(X), np.array(y)

input_steps = 180  # Past 180 minutes (3 hours)
forecast_intervals = [60]  # Predict at 10-min intervals
num_outputs = len(forecast_intervals)

# Create sequences for train, validation, and test
x_train, y_train = create_sequences(train_data, input_steps, forecast_intervals)
x_val, y_val = create_sequences(val_data, input_steps, forecast_intervals)
x_test, y_test = create_sequences(test_data, input_steps, forecast_intervals)

# Reshape for LSTM: [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# --- Build Model with Regularization ---
model = keras.models.Sequential()
model.add(keras.layers.LSTM(16, activation='relu', input_shape=(input_steps, 1)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(num_outputs))  # Output for 6 points

# --- Compile Model ---
model.compile(optimizer='adam', loss='mse')

# --- Early Stopping ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- Train Model ---
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# --- Save Model ---
model.save("humidity_forecast_6points.h5")

# --- Plot Loss ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# --- Predict on Test Data ---
y_pred_scaled = model.predict(x_test)

# Inverse transform to real humidity
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_real = scaler.inverse_transform(y_test)

# --- Plot Samples ---
time_ahead = forecast_intervals
for index in range(min(3, len(y_test_real))):  # Avoid index error
    plt.figure(figsize=(12, 4))
    plt.plot(time_ahead, y_test_real[index], label='Actual', marker='o')
    plt.plot(time_ahead, y_pred[index], label='Predicted', marker='x')
    plt.title(f'Sample {index} Forecast vs Actual')
    plt.xlabel('Minutes Ahead')
    plt.ylabel('Humidity (%)')
    plt.legend()
    plt.grid(True)
    plt.show()