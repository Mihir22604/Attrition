"""
AttritionIQ — Intelligent Employee Attrition Prediction System
Streamlit Production Decision-Support Web Application
Connected with FastAPI Backend + Local Fallback Architecture
"""

import os
import requests
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
# API Configuration (Streamlit Secrets / Environment Variable)
# ------------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "").strip().rstrip("/")
if not API_URL:
    try:
        API_URL = st.secrets.get("API_URL", "").strip().rstrip("/")
    except Exception:
        API_URL = ""

# ------------------------------------------------------------------------------
# Modern Presentation Styling
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
    
    .service-badge-api {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    .service-badge-local {
        display: inline-block;
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
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
# Load Local Production Model Pipeline (Used for Local Mode & Fallback)
# ------------------------------------------------------------------------------
PRODUCTION_PIPELINE_PATH = "models/attrition_model_pipeline.joblib"

@st.cache_resource
def load_champion_pipeline():
    """Load the final champion Random Forest pipeline for local inference/fallback."""
    if not os.path.exists(PRODUCTION_PIPELINE_PATH):
        return None
    try:
        return joblib.load(PRODUCTION_PIPELINE_PATH)
    except Exception as e:
        st.warning(f"Note: Local pipeline could not be loaded ({e}).")
        return None

local_pipeline = load_champion_pipeline()

# ------------------------------------------------------------------------------
# Sidebar: System Status & Sensitivity Control
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💼 AttritionIQ System")
    st.markdown("""
    **Production Classifier:**  
    `Random Forest (Tuned Champion)`
    
    **Validated Performance:**
    - Test Accuracy: **82.65%**
    - ROC-AUC Score: **0.7638**
    - Lowest False Alarm Rate
    """)
    st.markdown("---")
    
    st.markdown("### 🌐 Backend Integration")
    if API_URL:
        st.success(f"🔗 **FastAPI URL Configured:**\n`{API_URL}`")
    else:
        st.info("💻 **Running in Local Mode**\n*(Set `API_URL` environment variable or Streamlit Secret to connect to Render backend)*")
        
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
    st.caption("AttritionIQ v2.0 • Full-Stack Cloud Architecture")

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
# Collapsible Optional Secondary Parameters
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
# Assemble Complete 30-Attribute Payload / DataFrame
# ------------------------------------------------------------------------------
employee_payload = {
    'Age': age,
    'BusinessTravel': business_travel,
    'DailyRate': float(daily_rate),
    'Department': department,
    'DistanceFromHome': float(distance_home),
    'Education': 3,
    'EducationField': 'Life Sciences',
    'EnvironmentSatisfaction': env_satisfaction,
    'Gender': 'Male',
    'HourlyRate': float(hourly_rate),
    'JobInvolvement': job_involvement,
    'JobLevel': 1 if monthly_income < 3500 else (2 if monthly_income < 6000 else (3 if monthly_income < 10000 else 4)),
    'JobRole': job_role,
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_status,
    'MonthlyIncome': float(monthly_income),
    'MonthlyRate': 14000.0,
    'NumCompaniesWorked': num_companies,
    'OverTime': overtime,
    'PercentSalaryHike': float(salary_hike),
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': stock_options,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': 3,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': min(2, years_at_company),
    'YearsSinceLastPromotion': min(1, years_at_company),
    'YearsWithCurrManager': years_with_manager
}

# ------------------------------------------------------------------------------
# Execute Prediction (FastAPI Backend with Graceful Local Fallback)
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Prediction Result")

result_data = None
prediction_source = "local"
api_warning = None

# Attempt API Prediction if API_URL is configured
if API_URL:
    try:
        api_endpoint = f"{API_URL}/predict"
        res = requests.post(
            api_endpoint,
            json=employee_payload,
            params={"threshold": threshold},
            timeout=8
        )
        if res.status_code == 200:
            result_data = res.json()
            prediction_source = "api"
        else:
            api_warning = f"Backend returned HTTP {res.status_code}: {res.text}"
    except Exception as e:
        api_warning = f"Unable to reach FastAPI backend at `{API_URL}`: {str(e)}"

# Fallback to local Scikit-Learn pipeline if API prediction was not achieved
if result_data is None:
    if local_pipeline is not None:
        try:
            input_df = pd.DataFrame([employee_payload])
            prob_val = float(local_pipeline.predict_proba(input_df)[0][1])
            is_at_risk_local = prob_val >= threshold
            
            # Risk factor signals
            signals = []
            if overtime == "Yes":
                signals.append("Compulsory Overtime Active (Primary Turnover Risk)")
            if stock_options == 0:
                signals.append("Zero Stock Options / Equity (Lack of Retention Anchor)")
            if monthly_income < 3500:
                signals.append(f"Low Monthly Income (${monthly_income:,} - Below Peer Median)")
            if job_satisfaction <= 2:
                signals.append(f"Depressed Job Satisfaction Rating ({job_satisfaction}/4)")
            if work_life_balance <= 2:
                signals.append(f"Low Work-Life Balance Rating ({work_life_balance}/4)")
            if distance_home > 15:
                signals.append(f"Long Commute Distance ({distance_home} miles)")
            if marital_status == "Single":
                signals.append("Single Demographic Cohort (Elevated Baseline Mobility)")

            result_data = {
                "predicted_class": 1 if is_at_risk_local else 0,
                "attrition_label": "Yes" if is_at_risk_local else "No",
                "attrition_probability": round(prob_val, 4),
                "risk_category": "Low" if prob_val < 0.30 else ("Moderate" if prob_val < 0.60 else "Elevated"),
                "decision_threshold": threshold,
                "risk_factors": signals,
                "recommended_action": (
                    "Initiate proactive retention playbook: schedule confidential stay interview, "
                    "review compensation competitiveness against external benchmarks, and explore "
                    "workload / overtime rebalancing."
                ) if is_at_risk_local else (
                    "Maintain standard talent stewardship: continue regular quarterly check-ins "
                    "and ongoing professional development opportunities."
                )
            }
            prediction_source = "local"
        except Exception as e:
            st.error(f"Local inference failed: {str(e)}")
    else:
        st.error("No inference engine available. Both the backend API and local pipeline failed.")

# ------------------------------------------------------------------------------
# Render Prediction Cards & Insights
# ------------------------------------------------------------------------------
if result_data is not None:
    # Display backend source status indicator
    if prediction_source == "api":
        st.markdown(f'<div class="service-badge-api">⚡ Inferred via FastAPI Cloud Backend ({API_URL})</div>', unsafe_allow_html=True)
    else:
        if api_warning:
            st.warning(f"⚠️ **FastAPI Backend Notice:** {api_warning}\n\n*Gracefully fell back to in-memory local model pipeline.*")
        st.markdown('<div class="service-badge-local">💻 Inferred via Local Model Pipeline</div>', unsafe_allow_html=True)

    prob_depart = result_data["attrition_probability"]
    is_at_risk = result_data["predicted_class"] == 1

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
            (Decision Threshold: τ = {threshold:.2f} | Risk Tier: {result_data['risk_category']})
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Bar Indicator
    st.progress(prob_depart)

    # 2 Columns for Drivers and Recommended Action
    col_out1, col_out2 = st.columns(2)

    with col_out1:
        st.markdown("##### 🔍 Key Risk Drivers Detected")
        signals = result_data.get("risk_factors", [])
        if signals:
            for s in signals:
                st.write(f"⚠️ **{s}**")
        else:
            st.success("✅ No primary risk triggers detected. Profile demonstrates healthy retention signals.")

    with col_out2:
        st.markdown("##### 💡 Recommended Retention Action")
        if is_at_risk:
            st.warning(result_data["recommended_action"])
        else:
            st.info(result_data["recommended_action"])

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
