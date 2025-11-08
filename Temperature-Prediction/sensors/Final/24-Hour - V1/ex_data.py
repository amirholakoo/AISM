import requests
import csv
import ast
from datetime import datetime

url = "http://192.168.2.20:7500/dashboard/device_info_json/4/?month"

csv_filename = "DHT22_data.csv"


def extract_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        sensor_data = response.json()

        proceed_data = []
        target = []
        for datatype in sensor_data:
            datatype_index = 0
            for  key,value in datatype.items():
                index = 0
                if key == "data":
                    for data in value:
                        target.append({f"{datatype["type"]}":data,"timestamp": datatype["timestamps"][index]})
                        index +=1

        i = 0
        for x in target:
            try:
                proceed_data.append([x["temperature"],x["timestamp"]])
            except:
                proceed_data[i].append(x["humidity"])
                i+=1
        return proceed_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        return[]
    

def save_to_csv(data, filename):
    with open(filename, mode = 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time steps", "Temperature", "Humidity"])
        reordered_data = [(row[1], row[0], row[2]) for row in data]
        writer.writerows(reordered_data)

    print(f"Data saved to {filename}")

if __name__ =="__main__":
    data = extract_data(url)
    if data:
        save_to_csv(data, csv_filename)

