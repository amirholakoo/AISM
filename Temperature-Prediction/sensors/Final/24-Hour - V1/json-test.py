import requests
import time
url = 'http://192.168.2.20:7500/dashboard/device_info_json/11/?month'
def extract_data(api_url):
    try:
        x = 0
        Is_ok = False
        while x <10:
            response = requests.get(api_url)
            response.raise_for_status()
            sensor_data = response.json()
            if response.ok:
                Is_ok = True
                break
            x += 1
            time.sleep(0.5)
        if not Is_ok:
            print(f"api {url} faild")
            return []
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
                try:
                    proceed_data[i].append(x["humidity"])
                except:
                    break
                i += 1
        return proceed_data

    except Exception as e:
        print("API Error:", e)
        return []
x = extract_data(url)
print(x)