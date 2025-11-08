import pandas as pd
from datetime import timedelta
# Load CSV
df = pd.read_csv('DHT22_data.csv')

# Convert Time steps to datetime
df['Time steps'] = pd.to_datetime(df['Time steps'])

new_rows = []

for i in range(len(df) - 1):
    current_row = df.iloc[i]
    next_row = df.iloc[i+1]

    dif = int((next_row['Time steps'] - current_row['Time steps']).total_seconds() / 60)
    

    if dif > 1:
        for m in range (1, dif):
            new_time = current_row['Time steps'] + pd.Timedelta(minutes=m)
            new_temp = df['Temperature'].mean()
            new_hum = df['Humidity'].mean()
            new_rows.append(
                {
                    'Time steps' : new_time,
                    'Temperature' : new_temp,
                    'Humidity' : new_hum
                }
            )

if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index= True)

df = df.sort_values('Time steps').reset_index(drop = True)

df.to_csv('main.csv', index=False)