"""
Phase 2: Feature Engineering
-----------------------------
Turns the 5 raw tables from Phase 1 into ML-ready feature tables.

Key idea: for each employee, every assignment that has already ENDED is
a historical training example. We snapshot the employee's situation
at a VARIABLE point before that assignment ended (15-45 days), and 
label it 1 if a bench event followed, 0 if it didn't. Each employee's 
CURRENT, still-ongoing assignment is held out separately as a "live 
snapshot" -- that's not for training, it's what the API will score 
later in Phase 4.

Inputs:  data/raw/{employees,skills_taxonomy,projects,assignments,bench_events}.csv
Outputs: data/processed/training_features.csv   (has bench_risk_label)
         data/processed/current_snapshot.csv     (no label -- for live scoring)
"""

import os

import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LOOKAHEAD_MIN = 15
LOOKAHEAD_MAX = 45


def load_data():
    employees = pd.read_csv(f"{RAW_DIR}/employees.csv", parse_dates=["join_date"])
    skills = pd.read_csv(f"{RAW_DIR}/skills_taxonomy.csv")
    assignments = pd.read_csv(
        f"{RAW_DIR}/assignments.csv", parse_dates=["start_date", "end_date"]
    )
    bench_events = pd.read_csv(
        f"{RAW_DIR}/bench_events.csv", parse_dates=["bench_start", "bench_end"]
    )
    return employees, skills, assignments, bench_events


def compute_skill_demand_scores(employees: pd.DataFrame, skills: pd.DataFrame) -> pd.Series:
    """Average demand_score across each employee's listed skills."""
    demand_lookup = skills.set_index("skill_id")["demand_score"]

    def avg_demand(skill_string: str) -> float:
        skill_ids = skill_string.split("|")
        return demand_lookup.reindex(skill_ids).mean()

    return employees["primary_skills"].apply(avg_demand)


def build_feature_table(employees, skills, assignments, bench_events):
    employees = employees.copy()
    employees["skill_demand_score"] = compute_skill_demand_scores(employees, skills)

    # Fast lookup: does a bench event start exactly on this date for this employee?
    bench_lookup = set(
        zip(bench_events["employee_id"], bench_events["bench_start"])
    )

    # Sort each employee's assignments chronologically so we can tell
    # "is this their last (current) assignment, or a completed one?"
    assignments = assignments.sort_values(["employee_id", "start_date"])
    assignments["is_last_for_employee"] = (
        assignments.groupby("employee_id")["start_date"].rank(method="first", ascending=False) == 1
    )

    training_rows = []
    snapshot_rows = []

    emp_lookup = employees.set_index("employee_id")

    for _, row in assignments.iterrows():
        emp = emp_lookup.loc[row["employee_id"]]

        # Vary the lookahead window per assignment so days_remaining_in_project
        # is a meaningful, non-constant feature. Pick a random point 15-45 days
        # before the end.
        lookahead_days = pd.Series([pd.Timestamp(row["end_date"])])._values[0]
        lookahead = (pd.Timestamp(row["end_date"]) - pd.Timestamp(row["start_date"])).days
        lookahead = min(lookahead, 45)
        lookahead = max(lookahead, LOOKAHEAD_MIN)
        # Actually, simpler: just use the LOOKAHEAD_MIN to LOOKAHEAD_MAX range,
        # but randomized per row using the employee_id and assignment_id as a seed
        hash_val = hash((row["employee_id"], row["assignment_id"])) % (LOOKAHEAD_MAX - LOOKAHEAD_MIN + 1)
        actual_lookahead = LOOKAHEAD_MIN + hash_val

        snapshot_date = row["end_date"] - pd.Timedelta(days=actual_lookahead)
        if snapshot_date < row["start_date"]:
            snapshot_date = row["start_date"]

        prior_bench = bench_events[
            (bench_events["employee_id"] == row["employee_id"])
            & (bench_events["bench_start"] < snapshot_date)
        ]

        days_remaining_in_project = (row["end_date"] - snapshot_date).days
        tenure_at_company_days = (snapshot_date - emp["join_date"]).days

        feature_row = {
            "employee_id": row["employee_id"],
            "assignment_id": row["assignment_id"],
            "snapshot_date": snapshot_date.date(),
            "utilization_pct": row["utilization_pct"],
            "days_remaining_in_project": days_remaining_in_project,
            "experience_years": emp["experience_years"],
            "experience_band": emp["experience_band"],
            "tenure_at_company_days": max(tenure_at_company_days, 0),
            "historical_bench_count": len(prior_bench),
            "historical_bench_days_total": int(prior_bench["bench_days"].sum()),
            "skill_demand_score": round(emp["skill_demand_score"], 1),
            "skill_profile": emp["skill_profile"],
        }

        if row["is_last_for_employee"]:
            snapshot_rows.append(feature_row)
        else:
            label = 1 if (row["employee_id"], row["end_date"]) in bench_lookup else 0
            feature_row["bench_risk_label"] = label
            training_rows.append(feature_row)

    return pd.DataFrame(training_rows), pd.DataFrame(snapshot_rows)


def main():
    employees, skills, assignments, bench_events = load_data()
    training_df, snapshot_df = build_feature_table(employees, skills, assignments, bench_events)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    training_df.to_csv(f"{PROCESSED_DIR}/training_features.csv", index=False)
    snapshot_df.to_csv(f"{PROCESSED_DIR}/current_snapshot.csv", index=False)

    print("Feature engineering complete.")
    print(f"  training_features.csv -> {len(training_df)} rows")
    print(f"  current_snapshot.csv  -> {len(snapshot_df)} rows")

    positive_rate = training_df["bench_risk_label"].mean()
    print(f"\nOverall positive label rate (will go on bench): {positive_rate:.1%}")

    print("\nLabel rate by skill profile (sanity check -- legacy should be highest):")
    print(training_df.groupby("skill_profile")["bench_risk_label"].mean().round(3))

    print("\nDays remaining distribution (should vary, not be constant 30):")
    print(training_df["days_remaining_in_project"].describe().round(1))


if __name__ == "__main__":
    main()