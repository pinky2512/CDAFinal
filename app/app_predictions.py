import streamlit as st
import pandas as pd
import os
import hopsworks
from dotenv import load_dotenv

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
os.environ["HOPSWORKS_API_KEY"] = os.getenv("HOPSWORKS_API_KEY")

# ---------------------------
# Connect to Hopsworks
# ---------------------------
project = hopsworks.login(project=os.getenv("HOPSWORKS_PROJECT"))
fs = project.get_feature_store()

# ---------------------------
# Load prediction and actuals data
# ---------------------------
pred_fg = fs.get_feature_group("citibike_predictions", version=1)
actual_fg = fs.get_feature_group("citibike_lag_features", version=1)

df_pred = pred_fg.read()
df_actual = actual_fg.read()

df_pred["datetime"] = pd.to_datetime(df_pred["datetime"])
df_actual["datetime"] = pd.to_datetime(df_actual["datetime"])

# ---------------------------
# Setup Streamlit UI
# ---------------------------
st.set_page_config(page_title="Predictions vs Ground Truth", layout="wide")
st.title("📊 Citi Bike: Predictions vs Ground Truth")

stations = sorted(df_pred["start_station_name"].unique().tolist())
selected_station = st.selectbox("Select a Station", stations)

# ---------------------------
# Filter and align data
# ---------------------------
pred_df = df_pred[df_pred["start_station_name"] == selected_station]
actual_df = df_actual[df_actual["start_station_name"] == selected_station]

# Merge on datetime
merged_df = pd.merge(
    pred_df[["datetime", "prediction"]],
    actual_df[["datetime", "trip_count"]],
    on="datetime",
    how="inner"
).sort_values("datetime")

# ---------------------------
# Plot results
# ---------------------------
st.subheader(f"Predicted vs Actual Trips for {selected_station}")

st.line_chart(
    merged_df.set_index("datetime")[["trip_count", "prediction"]],
    use_container_width=True
)