"""
Pre-compute bench-risk predictions for all employees and save to CSV.
This avoids making 800 HTTP calls every time the dashboard loads.
"""

import json
import joblib
import pandas as pd

MODEL_PATH = "models/bench_risk_model.pkl"
METADATA_PATH = "models/model_metadata.json"

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

current_snapshot = pd.read_csv("data/processed/current_snapshot.csv")

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
experience_band_order = metadata["experience_band_order"]
skill_profile_order = metadata["skill_profile_order"]

X = current_snapshot[numeric_features + categorical_features].copy()

# Convert categorical columns to categorical dtype with proper ordering
X["experience_band"] = pd.Categorical(X["experience_band"], categories=experience_band_order)
X["skill_profile"] = pd.Categorical(X["skill_profile"], categories=skill_profile_order)

# Get predictions
y_pred_proba = model.predict_proba(X)[:, 1]

# Create results dataframe
results = current_snapshot[["employee_id", "experience_band", "skill_profile", "utilization_pct"]].copy()
results["bench_risk_probability"] = y_pred_proba
results["bench_risk_score"] = (y_pred_proba * 100).round(1)

results.to_csv("data/processed/bench_risk_predictions.csv", index=False)

print(f"Saved predictions for {len(results)} employees to data/processed/bench_risk_predictions.csv")