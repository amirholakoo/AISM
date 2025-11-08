from flask import Flask
import json, requests, time, threading
import numpy as np
from FinalLoad import Prediction

app = Flask(__name__)

TargetServer = "http://192.168.2.20:7500"
def scheduler():
    while True:
        CONFIG = False
        with open('config.json', 'r') as file:
            CONFIG = json.load(file)
        print("in while")
        now = time.time()
        if CONFIG:
            for i in CONFIG["sensors"]:
                if (now - i["last_prediction"]) >= (CONFIG["Interval_Hour"] * 3600):
                    now = time.time()
                    with open('config.json', 'w') as file:
                        i["last_prediction"] = now
                        json.dump(CONFIG,file)
                    print("time for prediction")
                    print("sensor data:",i)
                    url = f"http://192.168.2.20:7500/dashboard/device_info_json/{i['sensor_target']}/?month"
                    pred_data = Prediction(url,i["sensor_target"])
                    print("output data:",pred_data)
                    if np.any(pred_data):
                        FinalResponse = i
                        FinalResponse["data"] = {"temperature": [float(f"{val:.2f}") for i,val in enumerate(pred_data, start=1)]}
                        requests.post(TargetServer,json=FinalResponse)
                    else:
                        with open('config.json', 'w') as file:
                            i["last_prediction"] = 1
                            json.dump(CONFIG,file)
        time.sleep(60)

def start_scheduler():
    t = threading.Thread(target=scheduler, daemon=True)
    t.start()

if __name__ == "__main__":
    start_scheduler()
    app.run(host="0.0.0.0",port=6005, debug=False)