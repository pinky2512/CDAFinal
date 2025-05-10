# File: src/modeling/utils.py

import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
import hopsworks
# Load .env once
load_dotenv()

# -----------------------------------
# 1. Load Citi Bike data from Hopsworks
# -----------------------------------
def load_hourly_data_from_hopsworks(feature_group_name="citibike_hourly_trips", version=1):

    # Set environment variable for the API key
    os.environ["HOPSWORKS_API_KEY"] = os.getenv("HOPSWORKS_API_KEY")

    # Login will now pick up the API key and use the default host
    project = hopsworks.login(project=os.getenv("HOPSWORKS_PROJECT"))

    fs = project.get_feature_store()
    fg = fs.get_feature_group(feature_group_name, version=version)
    df = fg.read()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['start_station_name', 'datetime']).reset_index(drop=True)

    return df

# -----------------------------------
# 2. Create lag features
# -----------------------------------
def create_lag_features(df, lags=[1], target_col="trip_count"):
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby('start_station_name')[target_col].shift(lag)
    return df.dropna()

# -----------------------------------
# 3. Time-based train/test split
# -----------------------------------
def train_test_split_by_time(df, test_fraction=0.2):
    split_index = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test

# -----------------------------------
# 4. Configure DagsHub MLflow from .env
# -----------------------------------
def setup_dagshub_mlflow(experiment_name):
    username = os.getenv("DAGSHUB_USERNAME")
    repo = os.getenv("DAGSHUB_REPO")
    token = os.getenv("DAGSHUB_TOKEN")

    if not all([username, repo, token]):
        raise ValueError("Missing DAGSHUB_USERNAME, DAGSHUB_REPO, or DAGSHUB_TOKEN in .env")

    tracking_uri = f"https://dagshub.com/{username}/{repo}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow is set to DagsHub ({tracking_uri})")

# -----------------------------------
# 5. Log metrics & model to MLflow
# -----------------------------------
def log_to_mlflow(model, y_true, y_pred, model_name=None):
    mae = mean_absolute_error(y_true, y_pred)

    with mlflow.start_run():
        mlflow.log_metric("mae", mae)
        if model_name:
            mlflow.sklearn.log_model(model, model_name)

    print(f"MLflow logged: MAE = {mae:.4f}")
    return mae
