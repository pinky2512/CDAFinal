import pandas as pd
from sklearn.linear_model import LinearRegression
from src.modeling.utils import (
    load_hourly_data_from_hopsworks,
    create_lag_features,
    train_test_split_by_time,
    setup_dagshub_mlflow,
    log_to_mlflow
)

# ---------------------------
# 1. Setup DagsHub MLflow
# ---------------------------
setup_dagshub_mlflow(experiment_name="citibike_trip_prediction_baseline")

# ---------------------------
# 2. Load and Prepare Data
# ---------------------------
df = load_hourly_data_from_hopsworks()
df_lagged = create_lag_features(df, lags=[1])

# We'll use just one station for baseline modeling
station = df_lagged['start_station_name'].unique()[0]
df_station = df_lagged[df_lagged['start_station_name'] == station].copy()

print(f"Using station: {station} — {len(df_station)} rows")

# ---------------------------
# 3. Train/Test Split
# ---------------------------
train_df, test_df = train_test_split_by_time(df_station)

X_train = train_df[["lag_1"]]
y_train = train_df["trip_count"]
X_test = test_df[["lag_1"]]
y_test = test_df["trip_count"]

# ---------------------------
# 4. Train Baseline Model (Lag-1 Linear Regression)
# ---------------------------
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)

# ---------------------------
# 5. Log to DagsHub MLflow
# ---------------------------
log_to_mlflow(model=baseline_model, y_true=y_test, y_pred=y_pred, model_name="baseline_model")

print("Baseline modeling complete and logged to DagsHub.")
