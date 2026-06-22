"""
Phase 4: FastAPI Backend
--------------------------
Serves bench-risk predictions and skill recommendations via REST API.

The dashboard will call these endpoints to populate the visualizations.

Endpoints:
  GET /health                       -> health check
  GET /predict-bench-risk/{emp_id}  -> bench risk probability
  GET /recommend-skills/{emp_id}    -> top 3 reskilling recommendations
"""

import json
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Workforce Intelligence Platform API", version="1.0.0")

# Load model and metadata
MODEL_PATH = "models/bench_risk_model.pkl"
METADATA_PATH = "models/model_metadata.json"

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

# Load current snapshot and skill recommendations
current_snapshot = pd.read_csv("data/processed/current_snapshot.csv")
skill_recommendations = pd.read_csv("data/processed/skill_recommendations.csv")

# Build lookups for fast access
snapshot_lookup = current_snapshot.set_index("employee_id").to_dict("index")
recommendations_lookup = skill_recommendations.set_index("employee_id").to_dict("index")


def prepare_model_input(emp_data: dict) -> pd.DataFrame:
    """
    Convert employee snapshot data to model-ready feature table.
    """
    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    features = {}
    for col in numeric_features:
        features[col] = [emp_data.get(col, 0)]

    for col in categorical_features:
        features[col] = [emp_data.get(col, "")]

    return pd.DataFrame(features)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "AI Workforce Intelligence Platform"}


@app.get("/predict-bench-risk/{employee_id}")
def predict_bench_risk(employee_id: str):
    """
    Predict bench-risk probability for an employee.
    
    Returns:
      - bench_risk_probability: float between 0 and 1
      - bench_risk_score: 0 (low risk) to 100 (high risk)
      - recommendation: "LOW RISK", "MEDIUM RISK", or "HIGH RISK"
    """
    if employee_id not in snapshot_lookup:
        return JSONResponse(
            status_code=404,
            content={"error": f"Employee {employee_id} not found"}
        )

    emp_data = snapshot_lookup[employee_id]
    X = prepare_model_input(emp_data)

    # Get prediction probability
    bench_risk_prob = model.predict_proba(X)[0][1]
    bench_risk_score = round(bench_risk_prob * 100, 1)

    # Risk categorization
    if bench_risk_prob < 0.2:
        recommendation = "LOW RISK"
    elif bench_risk_prob < 0.4:
        recommendation = "MEDIUM RISK"
    else:
        recommendation = "HIGH RISK"

    return {
        "employee_id": employee_id,
        "bench_risk_probability": round(bench_risk_prob, 3),
        "bench_risk_score": bench_risk_score,
        "recommendation": recommendation,
        "experience_band": emp_data.get("experience_band"),
        "skill_profile": emp_data.get("skill_profile"),
        "utilization_pct": emp_data.get("utilization_pct"),
    }


@app.get("/recommend-skills/{employee_id}")
def recommend_skills(employee_id: str):
    """
    Get top 3 skill recommendations for an employee.
    
    Returns:
      - recommendations: list of 3 recommended skills with demand scores
    """
    if employee_id not in recommendations_lookup:
        return JSONResponse(
            status_code=404,
            content={"error": f"Employee {employee_id} not found"}
        )

    emp_recs = recommendations_lookup[employee_id]

    recommendations = []
    for i in range(1, 4):
        skill_name = emp_recs.get(f"recommended_skill_{i}_name")
        if pd.notna(skill_name):
            recommendations.append(
                {
                    "rank": i,
                    "skill_id": emp_recs.get(f"recommended_skill_{i}"),
                    "skill_name": skill_name,
                    "demand_score": emp_recs.get(f"recommended_skill_{i}_demand_score"),
                    "category": emp_recs.get(f"recommended_skill_{i}_category"),
                }
            )

    return {
        "employee_id": employee_id,
        "skill_profile": emp_recs.get("skill_profile"),
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)