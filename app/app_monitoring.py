import streamlit as st
import hopsworks
import mlflow
from dotenv import load_dotenv
import os
import pandas as pd

# ---------------------------
# Load secrets from env
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
    st.subheader("Latest Training Metrics from MLflow (DagsHub)")

    mlflow.set_tracking_uri(f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO')}.mlflow")
    mlflow.set_experiment("citibike_trip_prediction_lag28")

    runs = mlflow.search_runs(order_by=["start_time DESC"]).head(10)
    runs = runs[["run_id", "start_time", "metrics.mae"]].rename(columns={"metrics.mae": "MAE"})

    st.line_chart(runs.set_index("start_time")["MAE"])
    st.dataframe(runs)

except Exception as e:
    st.error(f"Failed to load MLflow metrics: {e}")
