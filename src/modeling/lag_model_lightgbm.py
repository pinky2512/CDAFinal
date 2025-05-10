import pandas as pd
import lightgbm as lgb
from src.modeling.utils import (
    load_hourly_data_from_hopsworks,
    create_lag_features,
    train_test_split_by_time,
    setup_dagshub_mlflow,
    log_to_mlflow
)
import joblib
import os

def main():
    # ---------------------------
    # 1. Setup DagsHub MLflow
    # ---------------------------
    setup_dagshub_mlflow(experiment_name="citibike_trip_prediction_lag28")

    # ---------------------------
    # 2. Load and Prepare Data
    # ---------------------------
    df = load_hourly_data_from_hopsworks()
    df_lagged = create_lag_features(df, lags=list(range(1, 29)))

    # Use just one station for now
    station = df_lagged['start_station_name'].unique()[0]
    df_station = df_lagged[df_lagged['start_station_name'] == station].copy()

    print(f"Using station: {station} — {len(df_station)} rows")

    # ---------------------------
    # 3. Train/Test Split
    # ---------------------------
    train_df, test_df = train_test_split_by_time(df_station)

    X_train = train_df[[f"lag_{i}" for i in range(1, 29)]]
    y_train = train_df["trip_count"]
    X_test = test_df[[f"lag_{i}" for i in range(1, 29)]]
    y_test = test_df["trip_count"]

    # ---------------------------
    # 4. Train Model
    # ---------------------------
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------------------------
    # 5. Log to MLflow (DagsHub)
    # ---------------------------
    log_to_mlflow(model=model, y_true=y_test, y_pred=y_pred, model_name="lag28_lightgbm_model")

    # ---------------------------
    # 6. Save Model
    # ---------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    print("Model saved to models/best_model.pkl")

# ------------------------------------------
# Optional CLI entry point
# ------------------------------------------
if __name__ == "__main__":
    main()