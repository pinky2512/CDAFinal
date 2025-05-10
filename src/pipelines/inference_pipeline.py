import os
import pandas as pd
import joblib
from dotenv import load_dotenv
import hopsworks
from datetime import datetime

# ---------------------------
# Step 1: Load environment
# ---------------------------
load_dotenv()
os.environ["HOPSWORKS_API_KEY"] = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(project=os.getenv("HOPSWORKS_PROJECT"))
fs = project.get_feature_store()

# ---------------------------
# Step 2: Load lag feature data
# ---------------------------
fg_lag = fs.get_feature_group("citibike_lag_features", version=1)
df = fg_lag.read()
df['datetime'] = pd.to_datetime(df['datetime'])

# Use only the latest record for each station
latest_df = df.sort_values("datetime").groupby("start_station_name").tail(1)
features = [f"lag_{i}" for i in range(1, 29)]

# ---------------------------
# Step 3: Load best model
# ---------------------------
model = joblib.load("models/best_model.pkl")

# ---------------------------
# Step 4: Make predictions
# ---------------------------
latest_df["prediction"] = model.predict(latest_df[features])
latest_df["prediction_time"] = datetime.utcnow()

predictions_df = latest_df[["start_station_name", "datetime", "prediction", "prediction_time"]]

print("Predictions:\n", predictions_df)

# ---------------------------
# Step 5: Save predictions to Hopsworks
# ---------------------------
fg_preds = fs.get_or_create_feature_group(
    name="citibike_predictions",
    version=1,
    primary_key=["start_station_name", "datetime"],
    event_time="prediction_time",
    description="Predicted Citi Bike trip count (1-hour ahead) for each station"
)

fg_preds.insert(predictions_df, write_options={"wait_for_job": True})
print("Predictions saved to Hopsworks Feature Group: citibike_predictions_v1")