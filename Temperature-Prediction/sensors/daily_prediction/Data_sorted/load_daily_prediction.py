import requests
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
import joblib

URL = "http://192.168.2.20:7500/dashboard/device_info_json/8/"

def extract_data(api_url, date, tolerance_minutes=2):
    response = requests.get(api_url)
    response.raise_for_status()
    sensor_data = response.json()
    proceed_data = []

    # Parse JSON data
    for entry in sensor_data:
        raw_data_str = entry.get('data', '{}')
        data_dict = ast.literal_eval(raw_data_str)
        temperature = data_dict.get('temperature')
        last_update_unix = entry.get("LastUpdate")
        last_update_readable = datetime.fromtimestamp(last_update_unix)
        proceed_data.append([last_update_readable, temperature])
        
    data_frame = pd.DataFrame(proceed_data, columns=["Time", "Temperature"])
    data_frame['Time'] = pd.to_datetime(data_frame['Time'])
    data_frame = data_frame.sort_values('Time').reset_index(drop=True)
    
    query_time = pd.to_datetime(date)
    
    # Find closest timestamp within tolerance
    data_frame['time_diff'] = (data_frame['Time'] - query_time).abs().dt.total_seconds() / 60.0
    candidates = data_frame[data_frame['time_diff'] <= tolerance_minutes]
    if candidates.empty:
        return None, None, None, None

    idx = candidates['time_diff'].idxmin()

    # Ensure enough data exists before and after
    if idx - 1440 < 0:
        return None, None, None, None

    # Past 1440 minutes (inputs)
    inputs = data_frame['Temperature'].iloc[idx-1440:idx].values
    input_times = data_frame['Time'].iloc[idx-1440:idx].reset_index(drop=True)

    # Future 1440 minutes (targets) – sample every 60 minutes
    future_window = data_frame.iloc[idx:idx+1440]
    if future_window.empty:
        target, time_step = np.array([]), pd.Series([], dtype='datetime64[ns]')
    else:
        time_step = future_window['Time'].iloc[::60].reset_index(drop=True)
        target = future_window['Temperature'].iloc[::60].dropna().values

    return np.array(inputs), np.array(target), time_step, input_times


# === Load model and scaler ===
model = load_model('tempreture_pretiction_CNN.h5', compile=False)
scaler = joblib.load('scaler.save')

# === Give the specific start time ===
start_time = datetime.strptime("2025-10-1 12:50", "%Y-%m-%d %H:%M")  # ⬅️ change this

inputs, target, time_step, input_times = extract_data(URL, start_time.strftime("%m/%d/%Y %H:%M:%S"))

if inputs is None or len(inputs) < 1440:
    print("❌ Not enough data around the selected time.")
else:
    # Create time features for model (past 1440 minutes)
    first_date = input_times.dt.date.iloc[0]
    time_min = (input_times - pd.Timestamp(first_date)).dt.total_seconds() / 60
    time_sin = np.sin(2 * np.pi * time_min / 1440)
    time_cos = np.cos(2 * np.pi * time_min / 1440)

    # Scale and stack
    inputs_scaled = scaler.transform(inputs.reshape(-1, 1))
    model_input = np.stack([inputs_scaled.reshape(-1), time_sin.values, time_cos.values], axis=1)
    model_input = np.expand_dims(model_input, axis=0)

    # Predict (model outputs 12 points)
    prediction = model.predict(model_input)
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)

    # Prepare results for plotting
    prediction = pd.Series(prediction, index=time_step[:len(prediction)])
    target = pd.Series(target[:len(prediction)], index=time_step[:len(prediction)]) if len(target) > 0 else None

    # === Plot ===
    plt.figure(figsize=(12, 6))
    plt.plot(prediction.index, prediction.values, label='Predicted Temperature', color='b', marker='o')
    if target is not None and not target.empty:
        plt.plot(target.index, target.values, label='True Temperature', color='r', marker='x')
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Temperature Prediction at {start_time.strftime('%Y-%m-%d %H:%M')}")
    plt.legend()
    plt.grid(True)
    plt.show()
