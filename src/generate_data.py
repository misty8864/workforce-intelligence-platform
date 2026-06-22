"""
Phase 1: Synthetic Workforce Data Generator
--------------------------------------------
Generates realistic, internally-correlated workforce data for the
AI Workforce Intelligence Platform.

Why the correlations matter: employees skewed toward legacy skills
(SAP, mainframe, COBOL) are deliberately given a higher chance of long
bench gaps, while employees skewed toward AI/Cloud skills cycle into
new projects faster. This is what gives the later ML model real signal
to learn from, instead of fitting to noise.

Output: 5 CSV files written to data/raw/
  - employees.csv
  - skills_taxonomy.csv
  - projects.csv
  - assignments.csv
  - bench_events.csv
"""

import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 42
NUM_EMPLOYEES = 800
NUM_PROJECTS = 250
TODAY = pd.Timestamp("2026-06-18")
OUTPUT_DIR = "data/raw"

random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ---------------------------------------------------------------------------
# SKILLS TAXONOMY
# Demand score (0-100) reflects how "in demand" a skill is right now.
# AI/GenAI and Cloud skills score high; legacy/ERP skills score low.
# ---------------------------------------------------------------------------
SKILLS = [
    ("LLM Application Development", "AI/GenAI", 95),
    ("Prompt Engineering", "AI/GenAI", 88),
    ("MLOps", "AI/GenAI", 90),
    ("Generative AI Solutions", "AI/GenAI", 93),
    ("AWS Cloud Architecture", "Cloud", 85),
    ("Azure Cloud Administration", "Cloud", 80),
    ("GCP Data Engineering", "Cloud", 78),
    ("Cloud Infrastructure Management", "Cloud", 75),
    ("Data Engineering (Spark)", "Data", 72),
    ("SQL & Data Warehousing", "Data", 65),
    ("Data Science & Analytics", "Data", 70),
    ("Power BI / Tableau", "Data", 60),
    ("Python Development", "Core Programming", 68),
    ("Java / Spring Boot", "Core Programming", 62),
    (".NET Development", "Core Programming", 55),
    ("JavaScript / React", "Core Programming", 60),
    ("SAP ERP", "Legacy/ERP", 30),
    ("Mainframe / COBOL", "Legacy/ERP", 18),
    ("Oracle ERP", "Legacy/ERP", 28),
    ("Legacy Systems Maintenance", "Legacy/ERP", 22),
    ("Banking Domain Expertise", "Domain", 50),
    ("Insurance Domain Expertise", "Domain", 45),
    ("Healthcare Domain Expertise", "Domain", 48),
    ("Business Analysis", "Functional", 52),
    ("Project Management", "Functional", 50),
    ("QA & Test Automation", "Functional", 47),
]

skills_df = pd.DataFrame(SKILLS, columns=["skill_name", "category", "demand_score"])
skills_df.insert(0, "skill_id", ["SK" + str(i + 1).zfill(3) for i in range(len(skills_df))])

ROLES_BY_EXPERIENCE = {
    "Junior": ["Software Engineer", "QA Engineer", "Business Analyst"],
    "Mid": ["Senior Software Engineer", "Data Engineer", "Systems Analyst", "ML Engineer"],
    "Senior": ["Technical Lead", "Senior ML Engineer", "Project Manager", "Solutions Architect"],
}

LOCATIONS = ["Bengaluru", "Hyderabad", "Pune", "Chennai", "Hubli", "Mysuru", "Noida"]
CLIENT_TYPES = ["BFSI", "Retail", "Healthcare", "Manufacturing", "Telecom", "GenAI Innovation"]


# ---------------------------------------------------------------------------
# EMPLOYEES
# ---------------------------------------------------------------------------
def generate_employees(n: int) -> pd.DataFrame:
    records = []
    for i in range(n):
        emp_id = "EMP" + str(i + 1).zfill(5)

        experience_years = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            p=[0.12, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.06, 0.06, 0.05, 0.04],
        )
        if experience_years <= 2:
            band = "Junior"
        elif experience_years <= 6:
            band = "Mid"
        else:
            band = "Senior"

        role = random.choice(ROLES_BY_EXPERIENCE[band])

        if band == "Junior":
            num_skills = np.random.randint(2, 4)
        elif band == "Mid":
            num_skills = np.random.randint(3, 6)
        else:
            num_skills = np.random.randint(4, 7)

        # Skill profile bias: this is the key correlation lever for bench risk later.
        profile = np.random.choice(["modern", "balanced", "legacy"], p=[0.35, 0.40, 0.25])
        if profile == "modern":
            weights = skills_df["category"].isin(["AI/GenAI", "Cloud", "Data"]).astype(int) * 3 + 1
        elif profile == "legacy":
            weights = skills_df["category"].isin(["Legacy/ERP", "Functional", "Domain"]).astype(int) * 3 + 1
        else:
            weights = np.ones(len(skills_df))
        weights = weights / weights.sum()

        chosen_skills = np.random.choice(
            skills_df["skill_id"], size=num_skills, replace=False, p=weights
        )

        join_date = TODAY - timedelta(days=int(experience_years * 365 + np.random.randint(0, 365)))

        records.append(
            {
                "employee_id": emp_id,
                "name": fake.name(),
                "experience_years": int(experience_years),
                "experience_band": band,
                "role": role,
                "location": random.choice(LOCATIONS),
                "skill_profile": profile,
                "primary_skills": "|".join(chosen_skills),
                "join_date": join_date.date(),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# PROJECTS
# ---------------------------------------------------------------------------
def generate_projects(n: int) -> pd.DataFrame:
    records = []
    for i in range(n):
        proj_id = "PRJ" + str(i + 1).zfill(4)
        client_type = random.choice(CLIENT_TYPES)

        if client_type == "GenAI Innovation":
            req_pool = skills_df[skills_df["category"].isin(["AI/GenAI", "Cloud", "Data"])]
        else:
            req_pool = skills_df

        required_skills = "|".join(
            np.random.choice(req_pool["skill_id"], size=min(3, len(req_pool)), replace=False)
        )
        duration_days = int(np.random.randint(60, 270))
        start_date = TODAY - timedelta(days=int(np.random.randint(0, 700)))
        end_date = start_date + timedelta(days=duration_days)

        records.append(
            {
                "project_id": proj_id,
                "project_name": f"{client_type} Engagement {i + 1}",
                "client_type": client_type,
                "required_skills": required_skills,
                "start_date": start_date.date(),
                "end_date": end_date.date(),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ASSIGNMENTS + BENCH EVENTS
# Walk each employee's timeline from join_date to TODAY, alternating
# "on a project" and "on the bench". Gap length is driven by the
# employee's demand percentile (not raw score), which ensures risk
# is spread evenly across the population rather than clustering.
# ---------------------------------------------------------------------------
def generate_assignments_and_bench(employees_df: pd.DataFrame, projects_df: pd.DataFrame):
    assignment_records = []
    bench_records = []
    assignment_counter = 1
    project_ids = projects_df["project_id"].values

    # Precompute each employee's skill-demand percentile. Mapping bench
    # probability from percentile (not raw score) guarantees risk spreads
    # evenly across the population, which is also a standard technique
    # in credit/risk scorecards.
    employees_df = employees_df.copy()

    def avg_demand(skill_string: str) -> float:
        skill_ids = skill_string.split("|")
        return skills_df[skills_df["skill_id"].isin(skill_ids)]["demand_score"].mean()

    employees_df["_demand_score"] = employees_df["primary_skills"].apply(avg_demand)
    employees_df["_demand_percentile"] = employees_df["_demand_score"].rank(pct=True)

    for _, emp in employees_df.iterrows():
        demand_percentile = emp["_demand_percentile"]

        # Low percentile (legacy skills) -> high bench probability.
        # High percentile (AI/cloud skills) -> low bench probability.
        bench_probability = np.interp(demand_percentile, [0, 1], [0.45, 0.05])

        # Low percentile -> longer average bench gaps. High percentile -> shorter gaps.
        base_gap = np.interp(demand_percentile, [0, 1], [55, 5])

        cursor = pd.Timestamp(emp["join_date"])
        while cursor < TODAY:
            duration = int(np.random.randint(60, 270))
            assignment_start = cursor
            assignment_end = assignment_start + timedelta(days=duration)
            hit_today = assignment_end >= TODAY
            if hit_today:
                assignment_end = TODAY

            utilization = int(np.random.randint(70, 101))

            assignment_records.append(
                {
                    "assignment_id": "ASG" + str(assignment_counter).zfill(6),
                    "employee_id": emp["employee_id"],
                    "project_id": random.choice(project_ids),
                    "start_date": assignment_start.date(),
                    "end_date": assignment_end.date(),
                    "utilization_pct": utilization,
                }
            )
            assignment_counter += 1

            if hit_today:
                break

            # Bench probability: this employee might go straight to next
            # project, or might sit on the bench. The probability is based
            # on demand percentile, not raw demand score.
            goes_on_bench = np.random.random() < bench_probability
            gap_days = (
                max(4, int(np.random.normal(loc=base_gap, scale=12)))
                if goes_on_bench
                else 0
            )
            if gap_days > 3:
                bench_start = assignment_end
                bench_end = bench_start + timedelta(days=gap_days)
                bench_records.append(
                    {
                        "employee_id": emp["employee_id"],
                        "bench_start": bench_start.date(),
                        "bench_end": bench_end.date(),
                        "bench_days": gap_days,
                    }
                )
            cursor = assignment_end + timedelta(days=gap_days)

    return pd.DataFrame(assignment_records), pd.DataFrame(bench_records)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    employees_df = generate_employees(NUM_EMPLOYEES)
    projects_df = generate_projects(NUM_PROJECTS)
    assignments_df, bench_events_df = generate_assignments_and_bench(employees_df, projects_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    skills_df.to_csv(f"{OUTPUT_DIR}/skills_taxonomy.csv", index=False)
    employees_df.to_csv(f"{OUTPUT_DIR}/employees.csv", index=False)
    projects_df.to_csv(f"{OUTPUT_DIR}/projects.csv", index=False)
    assignments_df.to_csv(f"{OUTPUT_DIR}/assignments.csv", index=False)
    bench_events_df.to_csv(f"{OUTPUT_DIR}/bench_events.csv", index=False)

    print("Data generation complete.")
    print(f"  employees.csv       -> {len(employees_df)} rows")
    print(f"  skills_taxonomy.csv -> {len(skills_df)} rows")
    print(f"  projects.csv        -> {len(projects_df)} rows")
    print(f"  assignments.csv     -> {len(assignments_df)} rows")
    print(f"  bench_events.csv    -> {len(bench_events_df)} rows")

    # Quick sanity check on the correlation that matters most downstream
    merged = bench_events_df.merge(employees_df[["employee_id", "skill_profile"]], on="employee_id")
    print("\nAverage bench days by skill profile (sanity check):")
    print(merged.groupby("skill_profile")["bench_days"].mean().round(1))


if __name__ == "__main__":
    main()