import pandas as pd
import lightgbm as lgb
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
setup_dagshub_mlflow(experiment_name="citibike_trip_prediction_reduced")

# ---------------------------
# 2. Load Data and Generate Lag Features
# ---------------------------
df = load_hourly_data_from_hopsworks()
df_lagged = create_lag_features(df, lags=list(range(1, 29)))  # lag_1 to lag_28

# Use one station for training (same as before)
station = df_lagged['start_station_name'].unique()[0]
df_station = df_lagged[df_lagged['start_station_name'] == station].copy()

print(f"Using station: {station} — {len(df_station)} rows")

# ---------------------------
# 3. Train/Test Split
# ---------------------------
train_df, test_df = train_test_split_by_time(df_station)

X_train_full = train_df[[f"lag_{i}" for i in range(1, 29)]]
y_train = train_df["trip_count"]
X_test_full = test_df[[f"lag_{i}" for i in range(1, 29)]]
y_test = test_df["trip_count"]

# ---------------------------
# 4. Train Full Model to Get Feature Importances
# ---------------------------
full_model = lgb.LGBMRegressor(random_state=42)
full_model.fit(X_train_full, y_train)

# Get top 10 features
importances = pd.Series(full_model.feature_importances_, index=X_train_full.columns)
top_features = importances.sort_values(ascending=False).head(10).index.tolist()
print(f"Top 10 Features: {top_features}")

# ---------------------------
# 5. Retrain Model with Reduced Features
# ---------------------------
X_train_reduced = X_train_full[top_features]
X_test_reduced = X_test_full[top_features]

reduced_model = lgb.LGBMRegressor(random_state=42)
reduced_model.fit(X_train_reduced, y_train)
y_pred = reduced_model.predict(X_test_reduced)

# ---------------------------
# 6. Log to MLflow (DagsHub)
# ---------------------------
log_to_mlflow(model=reduced_model, y_true=y_test, y_pred=y_pred, model_name="top10_lag_lightgbm")

print("Feature-reduced LightGBM model complete and logged to DagsHub.")