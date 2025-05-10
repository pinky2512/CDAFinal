import streamlit as st
import hopsworks
import mlflow
from dotenv import load_dotenv
import os
import pandas as pd

# ---------------------------
# Load secrets from env or Streamlit
# ---------------------------
load_dotenv()
os.environ["HOPSWORKS_API_KEY"] = os.getenv("HOPSWORKS_API_KEY")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

# ---------------------------
# UI setup
# ---------------------------
st.set_page_config(page_title="Citi Bike Model Monitoring", layout="wide")
st.title("Citi Bike Model Monitoring Dashboard")

# ---------------------------
# MLflow from DagsHub
# ---------------------------
try:
    st.subheader("MAE by Model Version (from MLflow @ DagsHub)")

    # Connect to MLflow
    mlflow.set_tracking_uri(f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO')}.mlflow")
    mlflow.set_experiment("citibike_trip_prediction_lag28")

    # Get recent runs
    runs = mlflow.search_runs(order_by=["start_time DESC"]).head(10)

    # Use run name as version label (fallback: short run_id)
    runs["Version"] = runs["tags.mlflow.runName"].fillna(runs["run_id"].str[:8])
    runs = runs[["Version", "metrics.mae"]].rename(columns={"metrics.mae": "MAE"})

    # Drop any missing values and sort by Version label
    runs = runs.dropna().sort_values("Version")

    # Display bar chart
    st.bar_chart(runs.set_index("Version")["MAE"])

except Exception as e:
    st.error(f"Failed to load MLflow metrics: {e}")
