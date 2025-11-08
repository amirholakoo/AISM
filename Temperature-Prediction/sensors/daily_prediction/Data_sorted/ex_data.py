import requests
import csv
import ast
from datetime import datetime

url = "http://192.168.2.20:7500/dashboard/device_info_json/8/"

csv_filename = "DHT22_data"


def extract_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        sensor_data = response.json()

        proceed_data = []

        for entry in sensor_data:
            raw_data_str = entry.get("data", "{}")
            try:
                data_dict = ast.literal_eval(raw_data_str)
            except Exception as e:
                print(f"Skipping invalid data format: {raw_data_str}")

            temperature = data_dict.get('temperature')
            humidity = data_dict.get('humidity')
            last_update_unix = entry.get('LastUpdate')
            last_update_readable = datetime.fromtimestamp(last_update_unix).strftime('%Y-%m-%d %H:%M:%S')
            proceed_data.append([last_update_readable, temperature, humidity])
        return proceed_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        return[]
    

def save_to_csv(data, filename):
    with open(filename, mode = 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time steps" , "Temperature", "Humidity"])
        writer.writerows(data)

    print(f"Data saved to {filename}")

if __name__ =="__main__":
    data = extract_data(url)
    if data:
        save_to_csv(data, csv_filename)

