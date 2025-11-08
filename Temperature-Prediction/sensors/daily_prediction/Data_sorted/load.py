from tensorflow.keras.models import load_model
import numpy as np
import requests
import ast
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import joblib
URL = "http://192.168.2.20:7500/dashboard/device_info_json/8/"

def extract_data(api_url, date):
    response = requests.get(api_url)
    response.raise_for_status()
    sensor_data = response.json()
    proceed_data = []
    for entry in sensor_data:
        raw_data_str = entry.get('data', '{}')
        data_dict = ast.literal_eval(raw_data_str)
        temperature = data_dict.get('temperature')
        last_update_unix = entry.get("LastUpdate")
        last_update_readable = datetime.fromtimestamp(last_update_unix).strftime("%Y-%m-%d %H:%M:%S")
        proceed_data.append([last_update_readable, temperature])
        
    data_frame = pd.DataFrame(proceed_data, columns=["Time", "Temperature"])
    data_frame['Time'] = pd.to_datetime(data_frame['Time'])
    query_time = pd.to_datetime(date)
    result = data_frame.loc[data_frame['Time'] == query_time, 'Temperature']
    idx = result.index[0]
    inputs = data_frame['Temperature'].iloc[idx-120:idx].values
    time_step = pd.to_datetime(data_frame['Time'].iloc[idx-120:idx])
    target = data_frame['Temperature'].iloc[idx:idx+60].values
    return np.array(inputs), np.array(target), time_step

inputs, target , time_step = extract_data(URL, '10/1/2025 9:46:11')

model = load_model('tempreture_pretiction_CNN.h5', compile = False)


time_min = time_step.dt.hour * 60 + time_step.dt.minute
time_sin = np.sin(2 * np.pi * time_min / 1440)
time_cos = np.cos(2 * np.pi * time_min / 1440)

scaler = joblib.load('scaler.save')
inputs = scaler.transform(inputs.reshape(-1, 1))
inputs = np.stack([inputs.reshape(-1), time_sin, time_cos], axis=1)
inputs = np.expand_dims(inputs, axis = 0)
prediction = model.predict(inputs)
def inverse_transform_y(y_scaled, scaler):
    
    y_2d = y_scaled.reshape(-1, 1)               # flatten to 2D
    y_inv = scaler.inverse_transform(y_2d)       # inverse scaling
    return y_inv.reshape(y_scaled.shape)         # reshape back

prediction = (inverse_transform_y(prediction, scaler)).reshape(-1)
plt.figure(figsize=(12,5))
plt.plot(target, label='True Temperature', color='r')
plt.plot(prediction, label='Predicted Temperature', color='b')
plt.xlabel("Future Time Steps")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Prediction - First Test Sample")
plt.legend()
plt.show()