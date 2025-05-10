import pandas as pd
import hopsworks
from pathlib import Path

# -------------------------------
# Step 1: Load the Cleaned Dataset
# -------------------------------
input_file = Path("data/processed/citibike_top3stations_2years.csv")
# Step 1: Load cleaned CSV and ensure datetime conversion
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Explicit conversion
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')

# Drop rows with invalid/missing dates
df = df.dropna(subset=["started_at"])

# Continue with transformation
df['datetime'] = df['started_at'].dt.floor('H')

# -------------------------------
# Step 2: Transform to Hourly Trip Count
# -------------------------------
df['datetime'] = df['started_at'].dt.floor('H')
df_hourly = df.groupby(['start_station_name', 'datetime']).size().reset_index(name='trip_count')
print(f"Transformed to time series format: {df_hourly.shape}")

# -------------------------------
# Step 3: Connect to Hopsworks
# -------------------------------
project = hopsworks.login(project="meghana_spring25_taxi")  # change if needed
fs = project.get_feature_store()

# -------------------------------
# Step 4: Create Feature Group
# -------------------------------
feature_group = fs.get_or_create_feature_group(
    name="citibike_hourly_trips",
    version=1,
    description="Hourly Citi Bike trip counts for top 3 stations (2024-2025)",
    primary_key=["start_station_name", "datetime"],
    event_time="datetime"
)

# -------------------------------
# Step 5: Insert Data
# -------------------------------
feature_group.insert(df_hourly, write_options={"wait_for_job": True})
print("Feature group created and data uploaded to Hopsworks!")
