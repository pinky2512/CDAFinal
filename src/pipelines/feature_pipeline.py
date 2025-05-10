# File: src/feature_pipeline.py

import pandas as pd
import os
from dotenv import load_dotenv
import hopsworks
from src.modeling.utils import create_lag_features

# -------------------------------
# Step 1: Load environment vars
# -------------------------------
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

if not api_key or not project_name:
    raise EnvironmentError("Missing HOPSWORKS_API_KEY or HOPSWORKS_PROJECT in .env")

# -------------------------------
# Step 2: Connect to Hopsworks
# -------------------------------
os.environ["HOPSWORKS_API_KEY"] = api_key
project = hopsworks.login(project=project_name)
fs = project.get_feature_store()

# -------------------------------
# Step 3: Load raw hourly trip data
# -------------------------------
fg_raw = fs.get_feature_group("citibike_hourly_trips", version=1)
df_raw = fg_raw.read()
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

# -------------------------------
# Step 4: Create lag features
# -------------------------------
df_lagged = create_lag_features(df_raw, lags=list(range(1, 29)))
print(f"Created lag features: {df_lagged.shape}")

# -------------------------------
# Step 5: Save to new Feature Group
# -------------------------------
fg_lag = fs.get_or_create_feature_group(
    name="citibike_lag_features",
    version=1,
    primary_key=["start_station_name", "datetime"],
    event_time="datetime",
    description="Lag features (1-28 hours) for trip prediction"
)

fg_lag.insert(df_lagged, write_options={"wait_for_job": True})
print("Lag features saved to Hopsworks: citibike_lag_features_v1")
