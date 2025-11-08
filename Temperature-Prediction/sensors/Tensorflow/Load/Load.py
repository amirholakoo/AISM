import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Define constants from training
second = 1
minute = 60 * second
hour = 60 * minute
day = 24 * hour
month = 30 * day
OUT_STEPS = 24
num_features = 10  # 2 original (Temp, Hum) + 8 cyclical

# Load scaler stats
with open('scaler_stats.pkl', 'rb') as f:
    scaler_stats = pickle.load(f)
mean = scaler_stats['mean']
std = scaler_stats['std']

# Define FeedBack class (same as training)
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

# Instantiate model and load weights
# Instantiate model and load weights
model = FeedBack(units=64, out_steps=OUT_STEPS)
model.build(input_shape=(None, 24, 10))  # ✅ BUILD FIRST
model.load_weights("feedback_lstm.weights.h5")  # ✅ NOW WORKS

# Load Excel data
excel_file = 'output.csv'  # Replace with your Excel file path
df = pd.read_csv(excel_file)

# Assume columns: 'Time steps', 'Temperature', 'Humidity'
# Convert 'Time steps' to datetime for easier searching
df['datetime'] = pd.to_datetime(df['Time'])
#df = df.sort_values('Time steps')  # Ensure sorted by time

# Function to add cyclical features
def add_cyclical_features(timestamps_s):
    cyclical = {}
    cyclical['Month sin'] = np.sin(2 * np.pi * timestamps_s / month)
    cyclical['Month cos'] = np.cos(2 * np.pi * timestamps_s / month)
    cyclical['Day sin'] = np.sin(2 * np.pi * timestamps_s / day)
    cyclical['Day cos'] = np.cos(2 * np.pi * timestamps_s / day)
    cyclical['Hour sin'] = np.sin(2 * np.pi * timestamps_s / hour)
    cyclical['Hour cos'] = np.cos(2 * np.pi * timestamps_s / hour)
    cyclical['Minute sin'] = np.sin(2 * np.pi * timestamps_s / minute)
    cyclical['Minute cos'] = np.cos(2 * np.pi * timestamps_s / minute)
    return pd.DataFrame(cyclical)

# Input: date and time as string, e.g., '2023-10-20 12:00:00'
date_time_str = input("Enter date and time (YYYY-MM-DD HH:MM:SS): ")
target_dt = pd.to_datetime(date_time_str)

# Find closest row
idx = (df['datetime'] - target_dt).abs().argmin()

# ✅ NEW: Show exact match!
time_diff = df['datetime'].iloc[idx] - target_dt
print(f"✅ Target: {target_dt}")
print(f"✅ Found:  {df['datetime'].iloc[idx]} (diff: {time_diff.total_seconds():.1f}s)")
print()

# Check if enough previous data (need 24 prior including current)
if idx < 23:
    raise ValueError("Not enough previous data for 24-hour input.")

# Extract 24 previous rows (idx-23 to idx inclusive)
start_idx = idx - 23
input_slice = slice(start_idx, idx + 1)
features = df[['Temperature', 'Humidity']].iloc[input_slice]  # [24, 2]
timestamps_s = df['Time steps'].iloc[input_slice].values

cyclical_df = add_cyclical_features(timestamps_s)  # [24, 8]

input_df = pd.concat([features.reset_index(drop=True),
                      cyclical_df.reset_index(drop=True)], axis=1)  # [24, 10] PERFECT!
print(input_df ['Temperature'])
print(f"✅ Input shape: {input_df.shape}")  # (24, 10)

# Normalize
input_normalized = (input_df - mean[input_df.columns]) / std[input_df.columns]

# Convert to array and add batch dimension
input_array = np.array(input_normalized)[np.newaxis, :, :]  # [1, 24, 10]

# Predict
full_prediction = model(input_array)  # [1, 24, 10]
temp_pred_norm = full_prediction[0, :, 0]  # [24]

# Denormalize
temp_pred = temp_pred_norm * std['Temperature'] + mean['Temperature']

# Extract actual next 24 temperatures
if idx + OUT_STEPS >= len(df):
    raise ValueError("Not enough future data for comparison.")
actual_slice = slice(idx + 1, idx + 1 + OUT_STEPS)
actual_temps = df['Temperature'].iloc[actual_slice].values

# Compare: Calculate MAE
mae = np.mean(np.abs(temp_pred - actual_temps))
print(f"Mean Absolute Error: {mae:.4f} °C")

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(range(1, OUT_STEPS + 1), actual_temps, label='Actual Temperature', marker='o')
plt.plot(range(1, OUT_STEPS + 1), temp_pred, label='Predicted Temperature', marker='x')
plt.xlabel('Future Hours')
plt.ylabel('Temperature (°C)')
plt.title(f'Temperature Forecast vs Actual starting from {target_dt}')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Print values
for i in range(OUT_STEPS):
    print(f"Hour {i+1}: Predicted {temp_pred[i]:.2f} °C, Actual {actual_temps[i]:.2f} °C")