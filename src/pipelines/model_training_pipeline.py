# File: pipelines/model_training_pipeline.py

import os
import pandas as pd
import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error
import hopsworks
import mlflow
import mlflow.sklearn
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# ---------------------------
# Step 1: Load environment variables
# ---------------------------
load_dotenv()

# For Hopsworks
os.environ["HOPSWORKS_API_KEY"] = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

# For DagsHub MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
mlflow.set_tracking_uri(f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO')}.mlflow")
mlflow.set_experiment("citibike_trip_prediction_lag28")

# ---------------------------
# Step 2: Connect to Hopsworks & load lagged data
# ---------------------------
project = hopsworks.login(project=project_name)
fs = project.get_feature_store()

fg = fs.get_feature_group("citibike_lag_features", version=1)
df = fg.read()
df['datetime'] = pd.to_datetime(df['datetime'])

# Focus on one station
station = df['start_station_name'].unique()[0]
df_station = df[df['start_station_name'] == station].copy()
df_station = df_station.sort_values("datetime")

# ---------------------------
# Step 3: Train/test split
# ---------------------------
split_index = int(len(df_station) * 0.8)
train = df_station.iloc[:split_index]
test = df_station.iloc[split_index:]

X_train = train[[f"lag_{i}" for i in range(1, 29)]]
y_train = train["trip_count"]
X_test = test[[f"lag_{i}" for i in range(1, 29)]]
y_test = test["trip_count"]

# ---------------------------
# Step 4: Train LightGBM model
# ---------------------------
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

# ---------------------------
# Step 5: Log metrics to MLflow (DagsHub)
# ---------------------------
with mlflow.start_run():
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model, "lightgbm_lag28_model")
    print(f"MAE logged to MLflow: {mae:.4f}")

# ---------------------------
# Step 6: Save model locally
# ---------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")

# ---------------------------
# Step 7: Register model to Hopsworks Model Registry
# ---------------------------
mr = project.get_model_registry()
input_example = X_test.iloc[:1]
schema = Schema(input_example)

model_registry_entry = mr.python.create_model(
    name="citibike_lag28_lightgbm",
    metrics={"mae": mae},
    description="LightGBM model with 28 lag features",
    input_example=input_example,
    model_schema=ModelSchema(schema)
)


model_registry_entry.save("models/best_model.pkl")
print("Model saved to Hopsworks Model Registry as 'citibike_lag28_lightgbm'")