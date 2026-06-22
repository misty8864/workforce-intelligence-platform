"""
Phase 5: Streamlit Dashboard - COMPLETE WITH PHASE 2 & PHASE 3
==============================================================
Interactive frontend for the AI Workforce Intelligence Platform.

Multi-page app with:
  - Executive Dashboard: org-level KPIs + business impact + executive insights
  - Employee Risk Analysis: individual employee predictions
  - Skill Gap Analysis: reskilling recommendations
"""

import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="AI Workforce Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for professional styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .high-risk { color: #ef4444; font-weight: bold; }
    .medium-risk { color: #f59e0b; font-weight: bold; }
    .low-risk { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# API base URL
API_URL = "http://localhost:8000"

# Load data
@st.cache_data
def load_current_snapshot():
    return pd.read_csv("data/processed/current_snapshot.csv")

@st.cache_data
def load_predictions():
    return pd.read_csv("data/processed/bench_risk_predictions.csv")

@st.cache_data
def load_skill_recommendations():
    return pd.read_csv("data/processed/skill_recommendations.csv")

@st.cache_data
def load_employees():
    return pd.read_csv("data/raw/employees.csv")

# ============================================================================
# PHASE 3: EXECUTIVE INSIGHTS ENGINE
# ============================================================================
def generate_executive_insights(preds, recs):
    """
    Generate 5 dynamic, data-driven insights from workforce data.
    """
    insights = []
    
    # Insight 1: Bench Risk Overview
    high_risk = len(preds[preds["bench_risk_probability"] >= 0.4])
    total = len(preds)
    pct = (high_risk / total) * 100
    insights.append(
        f"⚠️ **{high_risk} employees ({pct:.0f}% of workforce) are at elevated bench risk** and should be prioritized for immediate redeployment or reskilling."
    )
    
    # Insight 2: Skill Profile Risk Disparity
    legacy_risk = preds[preds["skill_profile"] == "legacy"]["bench_risk_probability"].mean()
    modern_risk = preds[preds["skill_profile"] == "modern"]["bench_risk_probability"].mean()
    multiplier = legacy_risk / modern_risk if modern_risk > 0 else 0
    insights.append(
        f"🔴 **Legacy skill profiles carry {multiplier:.1f}x higher bench risk** than modern profiles. "
        f"Employees with legacy skills average {legacy_risk*100:.0f}% risk vs. {modern_risk*100:.0f}% for modern skills."
    )
    
    # Insight 3: Top Reskilling Skills
    skill_counts = {}
    for idx, row in recs.iterrows():
        for i in range(1, 4):
            skill = row.get(f"recommended_skill_{i}_name")
            if pd.notna(skill):
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    if skill_counts:
        top_3_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        skills_str = ", ".join([f"**{skill}**" for skill, count in top_3_skills])
        insights.append(
            f"📚 **Top reskilling pathways:** {skills_str}. "
            f"These 3 skills account for {sum([count for _, count in top_3_skills])} recommendation slots across the workforce."
        )
    
    # Insight 4: Experience Band Risk Pattern
    senior_high_risk = len(preds[(preds["experience_band"] == "Senior") & (preds["bench_risk_probability"] >= 0.4)])
    senior_total = len(preds[preds["experience_band"] == "Senior"])
    junior_high_risk = len(preds[(preds["experience_band"] == "Junior") & (preds["bench_risk_probability"] >= 0.4)])
    junior_total = len(preds[preds["experience_band"] == "Junior"])
    
    senior_pct = (senior_high_risk / senior_total * 100) if senior_total > 0 else 0
    junior_pct = (junior_high_risk / junior_total * 100) if junior_total > 0 else 0
    
    insights.append(
        f"👔 **Senior employees represent {senior_pct:.0f}% of high-risk cases** ({senior_high_risk} of {senior_total}), "
        f"while junior employees account for {junior_pct:.0f}%. "
        f"This suggests legacy skill concentration at senior levels."
    )
    
    # Insight 5: Utilization & Risk Correlation
    high_util = preds[preds["utilization_pct"] >= 90]["bench_risk_probability"].mean()
    low_util = preds[preds["utilization_pct"] < 70]["bench_risk_probability"].mean()
    insights.append(
        f"📊 **Utilization paradox:** Highly utilized employees (≥90%) average {high_util*100:.0f}% bench risk, "
        f"while lower-utilized employees (<70%) average {low_util*100:.0f}%. This suggests short project cycles over sustained allocation."
    )
    
    return insights

# ============================================================================
# Page: Executive Dashboard
# ============================================================================
def page_executive_dashboard():
    st.title("📊 Executive Dashboard")
    st.markdown("---")
    
    preds = load_predictions()
    recs = load_skill_recommendations()
    
    # BASIC KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_employees = len(preds)
        st.metric("Total Employees", total_employees)
    
    with col2:
        avg_bench_risk = preds["bench_risk_probability"].mean() * 100
        st.metric("Avg Bench Risk", f"{avg_bench_risk:.1f}%")
    
    with col3:
        high_risk_count = (preds["bench_risk_probability"] >= 0.4).sum()
        st.metric("High Risk Employees", high_risk_count)
    
    with col4:
        avg_utilization = preds["utilization_pct"].mean()
        st.metric("Avg Utilization", f"{avg_utilization:.0f}%")
    
    st.markdown("---")
    
    # PHASE 2: BUSINESS IMPACT OVERVIEW
    st.subheader("💼 Business Impact Overview")
    
    # KPI 1: Potential Redeployments
    redeployments = len(preds[preds["bench_risk_probability"] >= 0.4])
    
    # KPI 2: Estimated Annual Cost Avoided
    high_risk_count = redeployments
    avg_bench_days = 34
    times_per_year = 1.5
    daily_rate = 2500
    cost_avoided = high_risk_count * daily_rate * avg_bench_days * times_per_year
    cost_avoided_millions = cost_avoided / 1_000_000
    
    # KPI 3: Reskilling Coverage
    high_risk_ids = preds[preds["bench_risk_probability"] >= 0.4]["employee_id"].values
    recs_high_risk = recs[recs["employee_id"].isin(high_risk_ids)]
    has_all_3_recs = recs_high_risk[
        (recs_high_risk["recommended_skill_1_name"].notna()) &
        (recs_high_risk["recommended_skill_2_name"].notna()) &
        (recs_high_risk["recommended_skill_3_name"].notna())
    ]
    reskilling_coverage = (len(has_all_3_recs) / len(high_risk_ids) * 100) if len(high_risk_ids) > 0 else 0
    
    # KPI 4: Critical Skill Gap
    legacy_and_balanced = len(preds[preds["skill_profile"].isin(["legacy", "balanced"])])
    total_employees = len(preds)
    skill_gap_pct = (legacy_and_balanced / total_employees) * 100
    
    # Display Business Impact KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Potential Redeployments",
            f"{redeployments}",
            delta="Immediate action",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "💰 Cost Avoided (Annual)",
            f"${cost_avoided_millions:.1f}M",
            delta="If reskilled",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "✅ Reskilling Coverage",
            f"{reskilling_coverage:.0f}%",
            delta="High-risk employees",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            "⚠️ Skill Gap",
            f"{legacy_and_balanced}",
            delta=f"{skill_gap_pct:.0f}% of workforce",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # PHASE 3: EXECUTIVE INSIGHTS
    st.subheader("🧠 Executive Insights")
    
    insights = generate_executive_insights(preds, recs)
    
    for i, insight in enumerate(insights, 1):
        with st.container():
            st.markdown(f"**{i}.** {insight}")
            st.markdown("")
    
    st.markdown("---")
    
    # Bench Risk by Experience Band
    st.subheader("Bench Risk by Experience Band")
    risk_by_band = preds.groupby("experience_band")["bench_risk_probability"].mean() * 100
    st.bar_chart(risk_by_band)
    
    # Bench Risk by Skill Profile
    st.subheader("Bench Risk by Skill Profile")
    risk_by_profile = preds.groupby("skill_profile")["bench_risk_probability"].mean() * 100
    st.bar_chart(risk_by_profile)
    
    st.markdown("---")
    
    # High-Risk Leaderboard
    st.subheader("🚨 High-Risk Employees (Bench Risk ≥ 40%)")
    high_risk = preds[preds["bench_risk_probability"] >= 0.4].sort_values(
        "bench_risk_probability", ascending=False
    ).head(15)
    
    if not high_risk.empty:
        display_cols = ["employee_id", "bench_risk_score", "experience_band", "skill_profile"]
        st.dataframe(
            high_risk[display_cols].rename(columns={
                "employee_id": "Employee ID",
                "bench_risk_score": "Risk Score (0-100)",
                "experience_band": "Experience",
                "skill_profile": "Skill Profile"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No high-risk employees detected.")

# ============================================================================
# Page: Employee Risk Analysis
# ============================================================================
def page_employee_analysis():
    st.title("👤 Employee Risk Analysis")
    st.markdown("---")
    
    employees = load_employees()
    preds = load_predictions()
    
    # Employee lookup
    emp_id = st.selectbox(
        "Select Employee",
        options=employees["employee_id"].values,
        format_func=lambda x: f"{x} - {employees[employees['employee_id']==x]['name'].values[0]}"
    )
    
    if emp_id:
        try:
            recs = requests.get(f"{API_URL}/recommend-skills/{emp_id}", timeout=5).json()
        except Exception as e:
            st.error(f"Could not fetch skill recommendations: {e}")
            recs = {"recommendations": []}
        
        emp = employees[employees["employee_id"] == emp_id].iloc[0]
        pred = preds[preds["employee_id"] == emp_id].iloc[0]
        
        # Employee info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Name:** {emp['name']}")
            st.write(f"**Role:** {emp['role']}")
        with col2:
            st.write(f"**Experience:** {emp['experience_years']} years")
            st.write(f"**Band:** {emp['experience_band']}")
        with col3:
            st.write(f"**Skill Profile:** {emp['skill_profile']}")
            st.write(f"**Location:** {emp['location']}")
        
        st.markdown("---")
        
        # Bench Risk Prediction
        st.subheader("Bench Risk Prediction")
        risk_prob = pred["bench_risk_probability"]
        risk_score = pred["bench_risk_score"]
        
        col1, col2 = st.columns(2)
        with col1:
            if risk_prob < 0.2:
                risk_level = "LOW RISK"
            elif risk_prob < 0.4:
                risk_level = "MEDIUM RISK"
            else:
                risk_level = "HIGH RISK"
            
            st.metric("Risk Level", risk_level, delta=f"{risk_score:.1f}%")
        
        with col2:
            st.metric("Risk Probability", f"{risk_prob:.3f}", delta=f"{risk_prob*100:.1f}%")
        
        st.write(f"**Risk Score: {risk_score:.1f}/100**")
        st.progress(min(risk_score / 100, 1.0))
        
        st.markdown("---")
        
        # Skill Recommendations
        st.subheader("💡 Recommended Reskilling Paths")
        if recs["recommendations"]:
            for i, rec in enumerate(recs["recommendations"], 1):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{i}. {rec['skill_name']}**")
                with col2:
                    st.write(f"Demand: {rec['demand_score']}")
                with col3:
                    st.write(f"`{rec['category']}`")
        else:
            st.info("No recommendations available.")

# ============================================================================
# Page: Skill Gap Analysis
# ============================================================================
def page_skill_gap_analysis():
    st.title("🎯 Skill Gap Analysis")
    st.markdown("---")
    
    recs = load_skill_recommendations()
    preds = load_predictions()
    
    # Merge predictions with recommendations
    merged = preds.merge(recs, on="employee_id", how="left")
    
    st.subheader("Reskilling Priority by Risk Level")
    
    # Filter by risk level
    risk_level = st.radio(
        "Filter by Risk Level",
        ["All", "Low Risk (< 20%)", "Medium Risk (20-40%)", "High Risk (≥ 40%)"],
        horizontal=True
    )
    
    if risk_level == "Low Risk (< 20%)":
        filtered = merged[merged["bench_risk_probability"] < 0.2]
    elif risk_level == "Medium Risk (20-40%)":
        filtered = merged[(merged["bench_risk_probability"] >= 0.2) & (merged["bench_risk_probability"] < 0.4)]
    elif risk_level == "High Risk (≥ 40%)":
        filtered = merged[merged["bench_risk_probability"] >= 0.4]
    else:
        filtered = merged
    
    st.write(f"**Employees in this category:** {len(filtered)}")
    
    # Top recommended skills
    st.subheader("Top Recommended Skills")
    skill_counts = {}
    for idx, row in filtered.iterrows():
        for i in range(1, 4):
            skill = row.get(f"recommended_skill_{i}_name")
            if pd.notna(skill):
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    if skill_counts:
        skills_df = pd.DataFrame(
            list(skill_counts.items()),
            columns=["Skill", "Employees to Reskill"]
        ).sort_values("Employees to Reskill", ascending=False).head(10)
        
        st.bar_chart(skills_df.set_index("Skill"))
    else:
        st.info("No recommendations in this category.")
    
    st.markdown("---")
    
    # Skill recommendations table
    st.subheader("Employee Reskilling Plans")
    display_cols = [
        "employee_id",
        "bench_risk_score",
        "recommended_skill_1_name",
        "recommended_skill_2_name",
        "recommended_skill_3_name"
    ]
    available_cols = [col for col in display_cols if col in filtered.columns]
    
    table_data = filtered[available_cols].head(20).copy()
    st.dataframe(table_data, use_container_width=True, hide_index=True)

# ============================================================================
# Main App
# ============================================================================
def main():
    st.sidebar.image(
        "https://via.placeholder.com/200x60?text=AI+Workforce+Intel",
        use_column_width=True
    )
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        ["Executive Dashboard", "Employee Risk Analysis", "Skill Gap Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.write("**API:** http://localhost:8000")
    
    if page == "Executive Dashboard":
        page_executive_dashboard()
    elif page == "Employee Risk Analysis":
        page_employee_analysis()
    elif page == "Skill Gap Analysis":
        page_skill_gap_analysis()

if __name__ == "__main__":
    main()