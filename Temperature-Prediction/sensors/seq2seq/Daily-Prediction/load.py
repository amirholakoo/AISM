import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

data = pd.read_csv('Modified_Timeseries_Temperature_and_Humidity_1_Hour.csv')
data['Time steps'] = pd.to_datetime(data['Time steps'])
model = load_model('temperature_prediction_CNN.h5', compile=False)
scaler = joblib.load('temp_scaler.pkl')
temp_min = scaler['temp_min']
temp_max = scaler['temp_max']

Time = '2025-10-10 10:00'
target_time = pd.to_datetime(Time)
nearest_idx = (data['Time steps'] - target_time).abs().idxmin()
start_idx = max(nearest_idx - 24, 0)
previous_temps = data.loc[start_idx:nearest_idx-1, 'Temperature'].values  # Past temps
previous_times = data.loc[start_idx:nearest_idx-1, 'Time steps']
end_idx = min(nearest_idx + 24, len(data) - 1)
next_temps = data.loc[nearest_idx:end_idx-1, 'Temperature'].values  # Future temps (including current? Adjust if needed)

# Normalize past temps (no sine)
past_temp_norm = (previous_temps - temp_min) / (temp_max - temp_min)
past_temp_norm = past_temp_norm.reshape(-1, 1)

# Compute past time features
past_time_minutes = (previous_times.dt.hour * 60 + previous_times.dt.minute).to_numpy()
past_time_sin = np.sin(2 * np.pi * past_time_minutes / 1440).reshape(-1, 1)
past_time_cos = np.cos(2 * np.pi * past_time_minutes / 1440).reshape(-1, 1)

# Features: past temp_norm + past time sin/cos
features = np.concatenate([past_temp_norm, past_time_sin, past_time_cos], axis=1)
features = np.expand_dims(features, axis=0)  # Batch dim

# Predict normalized, then denormalize
predict_norm = model.predict(features)
predict = predict_norm * (temp_max - temp_min) + temp_min

# Fix Y_time to align with predictions (starting from target_time +1h if next_temps starts after)
Y_time = pd.date_range(start=target_time + pd.Timedelta(hours=1), periods=24, freq='H')  # Adjust based on your slicing

plt.figure(figsize=(10, 5))
plt.plot(Y_time, predict.flatten(), marker='o', label='Prediction')
plt.plot(Y_time, next_temps.flatten(), marker='o', label='True')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature (°C)')
plt.title(f"24-Hour Prediction Starting After {target_time.strftime('%Y-%m-%d %H:%M')}")
plt.xticks(Y_time, [t.strftime('%H:%M') for t in Y_time], rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()