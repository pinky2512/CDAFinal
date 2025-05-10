import pandas as pd
import zipfile
import io
import requests
from datetime import datetime

# Define range
years = [2024, 2025]
months = list(range(1, 13))
dataframes = []

# Download 2 years: Mar 2023 to Feb 2025
for year in years:
    for month in months:
        if (year == 2025 and month > 4):
            continue  # Skip out-of-range months

        yyyymm = f"{year}{month:02d}"
        url = f"https://s3.amazonaws.com/tripdata/{yyyymm}-citibike-tripdata.csv.zip"

        try:
            print(f"[{datetime.now()}] Downloading {url}...")
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to download {yyyymm}, skipping...")
                print("Trying without .csv")
                url = f"https://s3.amazonaws.com/tripdata/{yyyymm}-citibike-tripdata.zip"
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Failed to download {yyyymm} without .csv, skipping...")
                    continue
                else:
                    print(f"Downloaded {yyyymm} without .csv")

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f)
                    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

                    # Convert datetime fields
                    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
                    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')

                    # Trip duration in minutes
                    df['trip_duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

                    # Basic cleaning
                    df = df.dropna(subset=['started_at', 'ended_at', 'start_station_name', 'end_station_name', 'trip_duration_min'])
                    df = df[(df['trip_duration_min'] >= 1) & (df['trip_duration_min'] <= 120)]

                    dataframes.append(df)

                    print(f"Loaded {yyyymm}: {len(df)} rows")

        except Exception as e:
            print(f"Error processing {yyyymm}: {e}")
            continue

# Combine all months
print(f"[{datetime.now()}] Combining all months...")
df_all = pd.concat(dataframes, ignore_index=True)
print(f"Combined dataset shape: {df_all.shape}")

# Top 3 most frequent start stations
top_stations = df_all['start_station_name'].value_counts().nlargest(3).index.tolist()
print(f"Top 3 Start Stations: {top_stations}")

# Filter to top stations
df_top3 = df_all[df_all['start_station_name'].isin(top_stations)].copy()

# Save to CSV
output_file = "data/processed/citibike_top3stations_2years.csv"
df_top3.to_csv(output_file, index=False)
print(f"Saved cleaned 2-year data to {output_file} with {len(df_top3)} rows.")
