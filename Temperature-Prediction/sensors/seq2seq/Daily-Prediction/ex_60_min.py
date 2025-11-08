import pandas as pd

# Load data
df = pd.read_csv('Modified_Timeseries_Temperature_and_Humidity.csv')

# Convert 'Time steps' to datetime
df['Time steps'] = pd.to_datetime(df['Time steps'])

# Set datetime as index
df.set_index('Time steps', inplace=True)

# Resample hourly and take the mean (or sum, or first, depending on your data)
df_hourly = df.resample('H').mean()  # 'H' = 1 hour

# Optional: reset index to have 'Time steps' as a column again
df_hourly = df_hourly.reset_index()

# Save to CSV
df_hourly.to_csv('Modified_Timeseries_Temperature_and_Humidity_1_Hour.csv', index=False)
