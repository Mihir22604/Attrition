"""
AttritionIQ — Intelligent Employee Attrition Prediction System
Streamlit Production Decision-Support Web Application
Clean Presentation-Optimized Interface
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AttritionIQ — Employee Attrition Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Modern Presentation Styling (Clean, Professional, Focused)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 20px;
        color: white;
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: #3B82F6;
        color: white;
        margin-bottom: 8px;
    }
    
    .result-card-low {
        background: #F0FDF4;
        border: 2px solid #86EFAC;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .result-card-high {
        background: #FEF2F2;
        border: 2px solid #FCA5A5;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .risk-tag-low {
        color: #166534;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    
    .risk-tag-high {
        color: #991B1B;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    
    .metric-value {
        font-size: 38px;
        font-weight: 800;
        line-height: 1.1;
        margin: 8px 0;
    }
    
    .disclaimer-footer {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 12px;
        color: #64748B;
        margin-top: 24px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Load Final Production Model Pipeline
# ------------------------------------------------------------------------------
PRODUCTION_PIPELINE_PATH = "models/attrition_model_pipeline.joblib"

@st.cache_resource
def load_champion_pipeline():
    """Load the final champion Random Forest pipeline."""
    if not os.path.exists(PRODUCTION_PIPELINE_PATH):
        st.error(f"Production model pipeline not found at `{PRODUCTION_PIPELINE_PATH}`.")
        return None
    return joblib.load(PRODUCTION_PIPELINE_PATH)

pipeline = load_champion_pipeline()

# ------------------------------------------------------------------------------
# Sidebar: System Status & Sensitivity Control (No Model Switching)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💼 AttritionIQ System")
    st.markdown("""
    **Production Classifier:**  
    `Random Forest (Tuned Champion)`
    
    **Validated Performance:**
    - Test Accuracy: **82.65%**
    - ROC-AUC Score: **0.7638**
    - Low False Alarm Rate
    """)
    st.markdown("---")
    
    st.markdown("### 🎯 Decision Sensitivity")
    threshold = st.slider(
        "Risk Cutoff (τ)",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.02,
        help="Cutoff probability for classifying an employee as Attrition Risk. Default is 0.50 (balanced) or 0.42 (high recall)."
    )
    if threshold < 0.45:
        st.caption("🟡 **High Sensitivity Mode:** Flags emerging risk early.")
    else:
        st.caption("🔵 **Balanced Mode:** Standard operational threshold.")

    st.markdown("---")
    st.caption("AttritionIQ v2.0 • Final Production Deployment")

# ------------------------------------------------------------------------------
# Main Hero Section
# ------------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <span class="badge-pill">Production Analytics Engine</span>
    <h1 class="hero-title">AttritionIQ — Single Employee Risk Assessment</h1>
    <p class="hero-subtitle">
        Intelligent employee attrition risk prediction powered by our final validated Random Forest model.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Employee Profile Inputs (Organized Cleanly into 2 Columns)
# ------------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### 💼 Role & Compensation Profile")
    age = st.slider("Age", 18, 60, 32)
    
    dept_options = ["Sales", "Research & Development", "Human Resources"]
    department = st.selectbox("Department", dept_options, index=0)
    
    role_options = [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ]
    job_role = st.selectbox("Job Role", role_options, index=0)
    
    monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 4500, step=250)
    overtime = st.radio("Compulsory Overtime?", ["No", "Yes"], index=0, horizontal=True)
    stock_options = st.selectbox("Stock Option Level (0 = None, 3 = Max)", [0, 1, 2, 3], index=1)

with col_right:
    st.markdown("##### 🧠 Workload, Tenure & Sentiment")
    
    sat_map = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
    job_satisfaction = st.select_slider(
        "Job Satisfaction",
        options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: sat_map[x]
    )
    
    wlb_map = {1: "1 - Poor", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
    work_life_balance = st.select_slider(
        "Work-Life Balance",
        options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: wlb_map[x]
    )
    
    years_at_company = st.slider("Tenure at Company (Years)", 0, 40, 5)
    total_working_years = st.slider("Total Working Experience (Years)", 0, 40, 8)
    distance_home = st.slider("Commute Distance (Miles)", 1, 30, 8)
    
    marital_options = ["Single", "Married", "Divorced"]
    marital_status = st.selectbox("Marital Status", marital_options, index=1)

# ------------------------------------------------------------------------------
# Collapsible Optional Secondary Parameters (Keeps Main View Clean & Simple)
# ------------------------------------------------------------------------------
with st.expander("⚙️ Additional Parameters (Optional / Advanced)", expanded=False):
    col_adv1, col_adv2 = st.columns(2)
    with col_adv1:
        travel_options = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        business_travel = st.selectbox("Business Travel", travel_options, index=0)
        env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
        salary_hike = st.slider("Last Salary Hike (%)", 11, 25, 14)
    with col_adv2:
        years_with_manager = st.slider("Years with Current Manager", 0, 20, min(2, years_at_company))
        num_companies = st.slider("Num Prior Companies", 0, 9, 2)
        daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=2000, value=800)
        hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=150, value=65)

# Ensure experience is logically consistent
if years_at_company > total_working_years:
    total_working_years = years_at_company

# ------------------------------------------------------------------------------
# Assemble Complete 30-Attribute Feature Vector for Production Pipeline
# ------------------------------------------------------------------------------
input_df = pd.DataFrame({
    'Age': [age],
    'BusinessTravel': [business_travel],
    'DailyRate': [daily_rate],
    'Department': [department],
    'DistanceFromHome': [distance_home],
    'Education': [3],
    'EducationField': ['Life Sciences'],
    'EnvironmentSatisfaction': [env_satisfaction],
    'Gender': ['Male'],
    'HourlyRate': [hourly_rate],
    'JobInvolvement': [job_involvement],
    'JobLevel': [1 if monthly_income < 3500 else (2 if monthly_income < 6000 else (3 if monthly_income < 10000 else 4))],
    'JobRole': [job_role],
    'JobSatisfaction': [job_satisfaction],
    'MaritalStatus': [marital_status],
    'MonthlyIncome': [monthly_income],
    'MonthlyRate': [14000],
    'NumCompaniesWorked': [num_companies],
    'OverTime': [overtime],
    'PercentSalaryHike': [salary_hike],
    'PerformanceRating': [3],
    'RelationshipSatisfaction': [3],
    'StockOptionLevel': [stock_options],
    'TotalWorkingYears': [total_working_years],
    'TrainingTimesLastYear': [3],
    'WorkLifeBalance': [work_life_balance],
    'YearsAtCompany': [years_at_company],
    'YearsInCurrentRole': [min(2, years_at_company)],
    'YearsSinceLastPromotion': [min(1, years_at_company)],
    'YearsWithCurrManager': [years_with_manager]
})

# ------------------------------------------------------------------------------
# Real-Time Risk Inference & Presentation Results Card
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Prediction Result")

if pipeline is not None:
    # Obtain prediction probability from the final champion pipeline
    prob_depart = pipeline.predict_proba(input_df)[0][1]
    is_at_risk = prob_depart >= threshold

    # Result Card
    card_class = "result-card-high" if is_at_risk else "result-card-low"
    risk_title = "🔴 ELEVATED ATTRITION RISK" if is_at_risk else "🟢 LOW ATTRITION RISK (RETAINED)"
    risk_tag_class = "risk-tag-high" if is_at_risk else "risk-tag-low"
    
    st.markdown(f"""
    <div class="{card_class}">
        <span class="{risk_tag_class}">{risk_title}</span>
        <div class="metric-value" style="color: {'#DC2626' if is_at_risk else '#16A34A'};">
            {prob_depart * 100:.1f}% Propensity
        </div>
        <p style="margin: 0; color: #475569; font-size: 14px;">
            Predicted Status: <strong>{'Likely to Leave' if is_at_risk else 'Likely to Stay'}</strong> 
            (Decision Threshold: τ = {threshold:.2f})
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Bar Indicator
    st.progress(prob_depart)

    # 2 Columns for Drivers and Recommended Action
    col_out1, col_out2 = st.columns(2)

    with col_out1:
        st.markdown("##### 🔍 Key Risk Drivers Detected")
        signals = []
        if overtime == "Yes":
            signals.append("⚠️ **Compulsory Overtime Active** (Major Turnover Factor)")
        if stock_options == 0:
            signals.append("⚠️ **Zero Equity / Stock Options** (Low Retention Anchor)")
        if monthly_income < 3500:
            signals.append(f"⚠️ **Low Monthly Income ($ {monthly_income:,})** (Below Peer Median)")
        if job_satisfaction <= 2:
            signals.append(f"⚠️ **Low Job Satisfaction Rating ({job_satisfaction}/4)**")
        if work_life_balance <= 2:
            signals.append(f"⚠️ **Poor Work-Life Balance Rating ({work_life_balance}/4)**")
        if distance_home > 15:
            signals.append(f"⚠️ **Long Commuting Distance ({distance_home} miles)**")
        if marital_status == "Single":
            signals.append("⚠️ **Single Demographic Cohort** (Higher Mobility Profile)")

        if signals:
            for s in signals:
                st.write(s)
        else:
            st.success("✅ No primary risk triggers detected. Profile demonstrates healthy retention signals.")

    with col_out2:
        st.markdown("##### 💡 Recommended Retention Action")
        if is_at_risk:
            st.warning("""
            **Priority HR Retention Playbook:**
            1. **Workload Relief:** Review overtime schedule and rebalance project commitments.
            2. **Compensation Review:** Benchmark salary against external market percentiles.
            3. **Retention Grant:** Explore Tier 1/2 stock option grant to build long-term alignment.
            4. **Stay Interview:** Conduct a 1-on-1 career development and satisfaction dialogue.
            """)
        else:
            st.info("""
            **Standard Talent Stewardship:**
            1. Maintain quarterly career milestone check-ins.
            2. Continue standard performance recognition and leadership development pathways.
            """)

# ------------------------------------------------------------------------------
# Persistent Ethical & Governance Disclaimer
# ------------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-footer">
    <strong>⚖️ Responsible AI Notice:</strong> 
    Predictions denote statistical propensity based on aggregate historical patterns. 
    This system serves strictly as advisory decision-support and must <strong>never</strong> be used as the sole basis 
    for termination, compensation changes, or withholding promotion opportunities. All decisions require human dialogue and oversight.
</div>
""", unsafe_allow_html=True)
