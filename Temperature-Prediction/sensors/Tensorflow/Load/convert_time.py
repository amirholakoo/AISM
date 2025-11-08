import pandas as pd
from datetime import datetime

# Load your CSV
df = pd.read_csv("DHT22_data.csv")

# Convert timestamps to datetime, rounding down to the hour
df["Time"] = pd.to_datetime(df["Time steps"], unit="s").dt.floor("H")

# Format as desired (YYYY-MM-DD HH:00:00)
df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:00:00")

# Save back to CSV
df.to_csv("output.csv", index=False)