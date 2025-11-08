import pandas as pd
import numpy as np
df = pd.read_csv('DHT22_data.csv')
df['Time steps'] = pd.to_datetime(df['Time steps'])
df = df.sort_values('Time steps')
df = df.drop_duplicates(subset=['Time steps'], keep = 'first').reset_index(drop = True)


def insert_missing_rows(df):
    all_rows = []
    for i in range(len(df)):
        current_time = df.iloc[i]['Time steps']
        all_rows.append(df.iloc[i].to_dict())

        if i < len(df) - 1:
            next_time = df.iloc[i + 1]['Time steps']
            time_diff = (next_time - current_time).total_seconds() / 60

            if time_diff > 1:
                missing_times = pd.date_range(start=current_time + pd.Timedelta(minutes=1), end = next_time - pd.Timedelta(minutes=1), freq='1T')

                for miss_time in missing_times:
                    nan_row = {'Time steps' : miss_time, 'Temperature': np.nan, 'Humidiy': np.nan}
                    all_rows.append(nan_row)
    

    df_full = pd.DataFrame(all_rows)
    df_full = df_full.sort_values('Time steps').reset_index(drop=True)
    return df_full


df_full = insert_missing_rows(df)
df_full['Temperature'] = df_full['Temperature'].interpolate(method='linear')
df_full['Humidity'] = df_full['Humidity'].interpolate(method='linear')
df_full.to_csv('Modified_Timeseries_Temperature_and_Humidity.csv', index=False)