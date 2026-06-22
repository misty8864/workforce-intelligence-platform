"""
Phase 2 (model): Bench Risk Prediction Model
-----------------------------------------------
Trains an XGBoost classifier on the feature table from build_features.py
to predict whether an employee will go on the bench after their current
project ends.

Split strategy: EMPLOYEE-LEVEL train/test split, not a random row split.
Each employee has multiple rows (one per past assignment), so a random
row split could let the same employee leak into both train and test,
inflating the test score. Splitting by employee_id first guarantees the
test score reflects real generalization to employees never seen before.

Why we don't lead with accuracy: only ~21% of rows are positive, so a
model that always predicts "no risk" scores ~79% accuracy while being
useless. Precision/recall/F1 on the bench-risk class and ROC-AUC are
the numbers that actually matter here.

Inputs:  data/processed/training_features.csv
Outputs: models/bench_risk_model.pkl
         models/model_metadata.json
"""

import json
import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

NUMERIC_FEATURES = [
    "utilization_pct",
    "days_remaining_in_project",
    "experience_years",
    "tenure_at_company_days",
    "historical_bench_count",
    "historical_bench_days_total",
    "skill_demand_score",
]
CATEGORICAL_FEATURES = ["experience_band", "skill_profile"]
TARGET = "bench_risk_label"

EXPERIENCE_BAND_ORDER = ["Junior", "Mid", "Senior"]
SKILL_PROFILE_ORDER = ["legacy", "balanced", "modern"]


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(f"{PROCESSED_DIR}/training_features.csv")
    df["experience_band"] = pd.Categorical(df["experience_band"], categories=EXPERIENCE_BAND_ORDER)
    df["skill_profile"] = pd.Categorical(df["skill_profile"], categories=SKILL_PROFILE_ORDER)
    return df


def employee_level_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Split by employee_id so no single employee's rows end up in both sets."""
    unique_employees = df["employee_id"].unique()
    train_ids, test_ids = train_test_split(
        unique_employees, test_size=test_size, random_state=random_state
    )
    train_df = df[df["employee_id"].isin(train_ids)]
    test_df = df[df["employee_id"].isin(test_ids)]
    return train_df, test_df


def main():
    df = load_training_data()
    train_df, test_df = employee_level_split(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    print(f"Train rows: {len(X_train)} ({train_df['employee_id'].nunique()} employees)")
    print(f"Test rows:  {len(X_test)} ({test_df['employee_id'].nunique()} employees)")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(
        f"Training class balance -> negative: {neg}, positive: {pos}, "
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Test set performance (employees the model never trained on) ---")
    print(classification_report(y_test, y_pred, target_names=["No bench risk", "Bench risk"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importances.round(3))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, f"{MODELS_DIR}/bench_risk_model.pkl")

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "experience_band_order": EXPERIENCE_BAND_ORDER,
        "skill_profile_order": SKILL_PROFILE_ORDER,
        "target": TARGET,
    }
    with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODELS_DIR}/bench_risk_model.pkl")
    print(f"Metadata saved to {MODELS_DIR}/model_metadata.json")


if __name__ == "__main__":
    main()