import requests
import numpy as np
import pandas as pd
import pytz
from tensorflow import keras
import joblib
from datetime import datetime

# -----------------------
#  API PART (Old ex_data.py)
# -----------------------
url = "http://192.168.2.20:7500/dashboard/device_info_json/4/?month"

def extract_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        sensor_data = response.json()

        proceed_data = []
        target = []

        for datatype in sensor_data:
            for key, value in datatype.items():
                if key == "data":
                    for i, data in enumerate(value):
                        target.append({
                            f"{datatype['type']}": data,
                            "timestamp": datatype["timestamps"][i]
                        })

        i = 0
        for x in target:
            try:
                proceed_data.append([x["temperature"], x["timestamp"]])
            except:
                proceed_data[i].append(x["humidity"])
                i += 1
        return proceed_data

    except Exception as e:
        print("API Error:", e)
        return []

# -----------------------
# ML PREDICTION PIPELINE
# -----------------------

# Tehran timezone
tehran_tz = pytz.timezone('Asia/Tehran')

data = extract_data(url)

if not data:
    raise Exception("No data received from API!")

# convert to DataFrame (Temperature, Timestamp, Humidity)
df = pd.DataFrame(data, columns=["Temperature", "Time steps", "Humidity"])

# latest 72 rows for prediction window
df = df.iloc[-72:]

# Time constants
second = 1
minute = 60 * second
hour = 60 * minute
day = 24 * hour
month = 30 * day

timestamps_s = df["Time steps"]

# Add cyclical time features
df['Month sin'] = np.sin(2*np.pi * timestamps_s / month)
df['Month cos'] = np.cos(2*np.pi * timestamps_s / month)
df['Day sin'] = np.sin(2*np.pi * timestamps_s / day)
df['Day cos'] = np.cos(2*np.pi * timestamps_s / day)
df['Hour sin'] = np.sin(2*np.pi * timestamps_s / hour)
df['Hour cos'] = np.cos(2*np.pi * timestamps_s / hour)
df['Minute sin'] = np.sin(2*np.pi * timestamps_s / minute)
df['Minute cos'] = np.cos(2*np.pi * timestamps_s / minute)

# Input feature order same as training
features = df[['Month sin','Month cos','Day sin','Day cos','Hour sin','Hour cos',
               'Minute sin','Minute cos','Temperature','Humidity']].values

# Load scaler & model
scaler = joblib.load('scaler.save')
model = keras.models.load_model('Temperature_Prediction.keras')

features = scaler.transform(features)
features = np.expand_dims(features, axis=0)

# Predict
predict = model.predict(features)
predictions = predict.flatten()

# Output results
ts = df['Time steps'].iloc[-1]

print(f"Last timestamp: {ts}")
print("Predicted temperatures:")
for i, value in enumerate(predictions, start=1):
    print(f"{i}. {value:.2f}")

"""
target server = 192.168.2.20:7500
strtucture: {
    "device_id": "pot-24",
    "sensor_type": "dht22",
    "data": {"temperature": [10,12,..]} # len -> 24
}
"""
TargetServer = "http://192.168.2.37:7500"
FinalResponse = {
    "device_id": "pot-24",
    "sensor_type": "dht22",
    "target_device_id" : "B4:3A:45:3F:B8:34	",
    "sensor_target": 4,
    "is_ai":True,
    "data": {"temperature": [float(f"{val:.2f}") for i,val in enumerate(predictions, start=1)]} # len -> 24
}

requests.post(TargetServer,json=FinalResponse)
# with open('Temperature_prediction.txt', 'w', encoding='utf-8') as f:
#     f.write(f'Last timestamps : {ts}\n')
#     for i, value in enumerate(predictions, start=1):
#         f.write(f"{i}. {value:.2f}\n")
