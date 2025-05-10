# File: pipelines/backfill_inference.py

import os
import pandas as pd
import joblib
from dotenv import load_dotenv
import hopsworks
from datetime import datetime, timezone

# ---------------------------
# Load environment and connect
# ---------------------------
load_dotenv()
os.environ["HOPSWORKS_API_KEY"] = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(project=os.getenv("HOPSWORKS_PROJECT"))
fs = project.get_feature_store()

# ---------------------------
# Load lag features & model
# ---------------------------
fg_lag = fs.get_feature_group("citibike_lag_features", version=1)
df = fg_lag.read()
df['datetime'] = pd.to_datetime(df['datetime'])

model = joblib.load("models/best_model.pkl")
features = [f"lag_{i}" for i in range(1, 29)]

# ---------------------------
# Prepare prediction feature group
# ---------------------------
fg_preds = fs.get_or_create_feature_group(
    name="citibike_predictions",
    version=1,
    primary_key=["start_station_name", "datetime"],
    event_time="prediction_time",
    description="Predictions for each hour from May 1 to now"
)

# ---------------------------
# Backfill range: May 1 to now (UTC)
# ---------------------------
start_time = pd.Timestamp("2025-05-01 00:00:00", tz="UTC")
end_time = datetime.now(timezone.utc)

prediction_times = pd.date_range(start=start_time, end=end_time, freq="H")
print(f"🔁 Backfilling {len(prediction_times)} hours from {start_time} to {end_time}...")

# ---------------------------
# Run predictions per hour
# ---------------------------
df = df.sort_values("datetime")

for dt in prediction_times:
    hour_df = df[df['datetime'] == dt]

    if hour_df.empty or not all(f in hour_df.columns for f in features):
        continue

    try:
        preds = model.predict(hour_df[features])
        result = hour_df[["start_station_name", "datetime"]].copy()
        result["prediction"] = preds
        result["prediction_time"] = pd.Timestamp(datetime.now(timezone.utc))

        fg_preds.insert(result, write_options={"wait_for_job": False})
        print(f"✅ Stored prediction for {dt}")

    except Exception as e:
        print(f"⚠️ Skipped {dt}: {e}")
