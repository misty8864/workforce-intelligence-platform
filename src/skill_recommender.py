"""
Phase 3: Skill Recommendation Engine
--------------------------------------
For each employee (especially those flagged as bench-risk), recommends
the top 3 missing high-demand skills they should reskill into.

Simple approach: compare their current skills against all high-demand
skills, find the gaps, rank by demand score. No ML needed -- just
practical, explainable recommendations.

Inputs:  data/raw/skills_taxonomy.csv
         data/raw/employees.csv
         data/processed/current_snapshot.csv
Outputs: data/processed/skill_recommendations.csv
"""

import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DEMAND_THRESHOLD = 60  # only recommend skills with demand_score >= this


def load_data():
    skills = pd.read_csv(f"{RAW_DIR}/skills_taxonomy.csv")
    employees = pd.read_csv(f"{RAW_DIR}/employees.csv")
    snapshots = pd.read_csv(f"{PROCESSED_DIR}/current_snapshot.csv")
    return skills, employees, snapshots


def get_employee_current_skills(skill_string: str) -> set:
    """Parse the pipe-separated skill IDs and return as a set."""
    if pd.isna(skill_string):
        return set()
    return set(skill_string.split("|"))


def recommend_skills(
    employee_skills: set, skills_df: pd.DataFrame, num_recommendations: int = 3
) -> list:
    """
    Find the top N high-demand skills the employee doesn't have.
    Returns list of tuples: (skill_id, skill_name, demand_score, category)
    """
    # Filter to high-demand skills only
    high_demand = skills_df[skills_df["demand_score"] >= DEMAND_THRESHOLD]

    # Find skills employee doesn't have
    missing = high_demand[~high_demand["skill_id"].isin(employee_skills)]

    # Sort by demand score descending, take top N
    top = missing.nlargest(num_recommendations, "demand_score")

    recommendations = []
    for _, row in top.iterrows():
        recommendations.append(
            {
                "skill_id": row["skill_id"],
                "skill_name": row["skill_name"],
                "demand_score": row["demand_score"],
                "category": row["category"],
            }
        )
    return recommendations


def main():
    skills_df, employees_df, snapshots_df = load_data()

    # Create lookup: employee_id -> primary_skills
    employee_skills_lookup = employees_df.set_index("employee_id")["primary_skills"].to_dict()

    # Pad recommendations to always 3, filling missing slots with empty values
    recommendation_cols = []
    for i in range(1, 4):
        recommendation_cols.extend(
            [
                f"recommended_skill_{i}",
                f"recommended_skill_{i}_name",
                f"recommended_skill_{i}_demand_score",
                f"recommended_skill_{i}_category",
            ]
        )

    results = []

    for _, snap in snapshots_df.iterrows():
        emp_id = snap["employee_id"]
        primary_skills_str = employee_skills_lookup.get(emp_id, "")
        current_skills = get_employee_current_skills(primary_skills_str)

        recommendations = recommend_skills(current_skills, skills_df, num_recommendations=3)

        # Pad to 3 recommendations
        while len(recommendations) < 3:
            recommendations.append(
                {
                    "skill_id": None,
                    "skill_name": None,
                    "demand_score": None,
                    "category": None,
                }
            )

        row = {"employee_id": emp_id, "skill_profile": snap["skill_profile"]}

        for i, rec in enumerate(recommendations, start=1):
            row[f"recommended_skill_{i}"] = rec["skill_id"]
            row[f"recommended_skill_{i}_name"] = rec["skill_name"]
            row[f"recommended_skill_{i}_demand_score"] = rec["demand_score"]
            row[f"recommended_skill_{i}_category"] = rec["category"]

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{PROCESSED_DIR}/skill_recommendations.csv", index=False)

    print("Skill recommendations complete.")
    print(f"  skill_recommendations.csv -> {len(results_df)} rows")

    # Show a few examples
    print("\nSample recommendations (first 5 employees):")
    for _, row in results_df.head(5).iterrows():
        print(f"\n  {row['employee_id']} (profile: {row['skill_profile']}):")
        for i in range(1, 4):
            skill_name = row[f"recommended_skill_{i}_name"]
            demand = row[f"recommended_skill_{i}_demand_score"]
            if pd.notna(skill_name):
                print(f"    {i}. {skill_name} (demand: {demand})")


if __name__ == "__main__":
    main()