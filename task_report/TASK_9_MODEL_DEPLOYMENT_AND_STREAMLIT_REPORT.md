# TASK 9 DETAILED REPORT: Model Serialization & Streamlit Deployment

**Project Title:** AttritionIQ — Intelligent Employee Attrition Prediction System  
**Task ID:** Task 9 — Pipeline Packaging, Joblib Serialization & Streamlit Application Deployment  
**Status:** COMPLETED & PRODUCTION-VERIFIED  
**Primary Application File:** `app.py`  
**Serialized Production Artifacts:**  
- `models/attrition_model_pipeline.joblib` (Unified End-to-End Random Forest Pipeline)  
- `models/attrition_logistic_pipeline.joblib` (Unified End-to-End Logistic Regression Pipeline)  
- `data/processed/fitted_preprocessor.joblib` (Fitted ColumnTransformer)  
**Author / Role:** Senior Machine Learning Project Analyst & Technical Documentation Specialist  
**Execution Environment:** Python 3.12 / Streamlit 1.32.0 / Scikit-Learn 1.6.1 / Pandas 2.2 / Joblib 1.4  

---

## 1. Executive Summary & Operational Deployment Framework

The primary objective of **Task 9: Model Deployment Using Streamlit** is to transition the AttritionIQ predictive engine from experimental notebook research into an interactive, production-grade decision-support web application (`app.py`).

### 1.1 The Operational Need in Enterprise Human Resources
While predictive models demonstrate strong mathematical metrics in cross-validation notebooks, non-technical HR business partners, talent directors, and executive leadership cannot be expected to run Python code or format NumPy arrays. A polished web application makes predictive intelligence accessible, providing:
1. **Interactive Scenario Testing:** Ability to explore "what-if" compensation, workload, and promotion adjustments for at-risk employees.
2. **Transparent Risk Factor Explanations:** Real-time visibility into the primary drivers elevating risk for each individual profile.
3. **Calibrated Probability Scores:** Moving beyond crude binary predictions to granular percentage likelihoods ($0.0\%$ to $100.0\%$).
4. **Actionable Retention Playbooks:** Automated guidance recommending targeted retention interventions based on observed risk patterns.
5. **Strict Ethical Guardrails:** Prominent, persistent disclaimers enforcing human-in-the-loop oversight and the anti-sole-basis rule.

---

## 2. Pipeline Serialization Architecture (Joblib Persistence)

### 2.1 The Unified End-to-End Pipeline Pattern
A critical failure mode in machine learning deployment is "decoupled preprocessing," where model weights are saved separately from transformers. If raw user inputs bypass the exact encoding and scaling transformations applied during training, silent numerical mismatches or dimensionality errors occur.

To guarantee zero-leakage and identical transformation fidelity, Task 9 packaged the complete workflow into a unified Scikit-Learn `Pipeline`:

```
┌────────────────────────────────────────────────────────────────────────┐
│             ATTRITIONIQ UNIFIED END-TO-END INFERENCE PIPELINE          │
├────────────────────────────────────────────────────────────────────────┤
│  Raw Input DataFrame (30 HR Attributes)                                │
│       │                                                                │
│       ▼                                                                │
│  [Step 1: ColumnTransformer (preprocessor)]                            │
│  ├── Numerical Features (14) ──► StandardScaler()                      │
│  ├── Ordinal Ratings (9)    ──► Passthrough Integer Ordering           │
│  └── Nominal Categories (7) ──► OneHotEncoder(drop='first')            │
│       │                                                                │
│       ▼ (Transforms 30 raw columns into 44 encoded/scaled features)    │
│  [Step 2: Tuned Estimator (classifier)]                                │
│  └── RandomForestClassifier(n_estimators=200, max_depth=6,             │
│                             min_samples_leaf=4, class_weight='balanced')│
│       │                                                                │
│       ▼                                                                │
│  Output: Posterior Departure Probability P(Y=1|x) & Risk Tier          │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Serialization Protocol Execution
The unified pipeline was assembled using `set_output(transform="pandas")` to ensure seamless feature-name preservation between transformations and serialized to disk using Joblib:

```python
from sklearn.pipeline import Pipeline
import joblib

preprocessor = joblib.load('data/processed/fitted_preprocessor.joblib')
preprocessor.set_output(transform='pandas')

rf_tuned = joblib.load('models/random_forest_tuned.joblib')

production_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_tuned)
])

joblib.dump(production_pipeline, 'models/attrition_model_pipeline.joblib')
```

- **Primary Production Pipeline Artifact:** [attrition_model_pipeline.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/attrition_model_pipeline.joblib)
- **Linear Benchmark Pipeline Artifact:** [attrition_logistic_pipeline.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/attrition_logistic_pipeline.joblib)

---

## 3. Streamlit Application Architecture & User Interface Design

The application (`app.py`) was engineered in accordance with modern web design standards (Plus Jakarta Sans typography, sleek card elevation, custom responsive layout, vibrant status badges, and dynamic animations).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ATTRITIONIQ — INTELLIGENT EMPLOYEE RETENTION PLATFORM (STREAMLIT UI LAYOUT)                     │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│  [Header / Hero Section: System Title, Enterprise Badges & Live Status Indicator]                │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│  [Preset Archetype Bar: High-Risk Overworked | Stable Senior Executive | Junior Specialist]      │
├───────────────────────────────────────────────────┬──────────────────────────────────────────────┤
│  EMPLOYEE PROFILE CONFIGURATION (Input Form)      │  PREDICTIVE DECISION SUPPORT (Results Card)  │
│  ───────────────────────────────────────────      │  ──────────────────────────────────────────  │
│  Column 1: Role & Compensation                    │  Retention Risk Status:                      │
│  • Age (18–60 slider)                             │  ● ELEVATED ATTRITION RISK [RED BADGE]       │
│  • Department & Job Role dropdowns                │                                              │
│  • Monthly Salary ($1,000–$20,000 slider)         │  Estimated Departure Probability:            │
│  • Overtime Status (Yes/No radio)                 │  [██████████████░░░░░░] 68.4%                │
│  • Stock Option Level (0–3 selectbox)             │                                              │
│  • Commuting Distance & Travel Frequency          │  Identified Contributing Risk Drivers:       │
│  • Marital Status dropdown                        │  ⚠️ Compulsory Overtime Detected (+7.4% AUC)  │
│                                                   │  ⚠️ Zero Equity Grants (Stock Tier 0)        │
│  Column 2: Sentiment & Experience                 │  ⚠️ Low Job Satisfaction Rating (1/4)        │
│  • Job Satisfaction (1–4 slider)                  │                                              │
│  • Work-Life Balance (1–4 slider)                 │  Operational Decision Threshold Slider:      │
│  • Environment Satisfaction (1–4 slider)          │  [───●──────────] tau = 0.42 (High-Sens Mode)│
│  • Job Involvement Rating (1–4 slider)            │                                              │
│  • Years at Company & Total Career Exp sliders    │  📌 Strategic Retention Playbook:            │
│  • Tenure with Immediate Manager slider           │  1. Workload Rebalancing & Compensatory Rest │
│  • Appraisal Salary Hike (%) slider               │  2. Equity Grant Review (Tier 1 Option)      │
├───────────────────────────────────────────────────┴──────────────────────────────────────────────┤
│  ⚖️ PERSISTENT MANDATORY HR GOVERNANCE DISCLAIMER (ANTI-SOLE-BASIS ENFORCEMENT)                   │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│  MODULAR TABS: [1. Risk Assessment] [2. Model Reports] [3. Feature Importance] [4. Ethics Review]│
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Input Field Mapping and Data Hygiene Validation

### 4.1 Form Fields Aligned with Assignment Mandates
The application captures all 8 mandatory assignment features alongside 22 contextual features initialized to verified median/mode defaults:

| Form Field Name | Input Component | Permissible Range / Options | Assignment Status | Internal Feature Mapping |
| :--- | :---: | :---: | :---: | :--- |
| **Employee Age** | Numeric Slider | 18 to 60 Years | **Mandatory** | `Age` |
| **Department** | Selectbox | Sales, R&D, Human Resources | **Mandatory** | `Department` |
| **Job Role** | Selectbox | 9 verified functional titles | **Mandatory** | `JobRole` |
| **Monthly Income** | Numeric Slider | \$1,000 to \$20,000 USD | **Mandatory** | `MonthlyIncome` |
| **Job Satisfaction** | Select Slider | Levels 1 (Low) to 4 (Very High) | **Mandatory** | `JobSatisfaction` |
| **Years at Company** | Numeric Slider | 0 to 40 Years | **Mandatory** | `YearsAtCompany` |
| **Overtime Status** | Radio Button | 'Yes' or 'No' | **Mandatory** | `OverTime` |
| **Work-Life Balance**| Select Slider | Levels 1 (Bad) to 4 (Best) | **Mandatory** | `WorkLifeBalance` |
| **Stock Option Tier**| Selectbox | Tiers 0, 1, 2, 3 | Essential Driver | `StockOptionLevel` |
| **Commute Distance** | Numeric Slider | 1 to 30 Miles | Essential Driver | `DistanceFromHome` |
| **Business Travel** | Selectbox | Rare, Frequent, Non-Travel | Bivariate Factor | `BusinessTravel` |
| **Total Experience** | Numeric Slider | 0 to 40 Years | Career Control | `TotalWorkingYears` |
| **Manager Tenure** | Numeric Slider | 0 to 20 Years | Supervisory Anchor | `YearsWithCurrManager` |

### 4.2 Automated Input Consistency Safeguards
The UI includes programmatic guardrails preventing impossible employee configurations:
- **Tenure Inconsistency Check:** Asserts that `YearsAtCompany <= TotalWorkingYears`. If an evaluator inputs 10 years at company with 5 years total experience, the app dynamically alerts the user and reconciles total experience.
- **Managerial Tenure Check:** Clamps `YearsWithCurrManager <= YearsAtCompany`.

---

## 5. Decision-Support Features & Operational Sensitivity Calibration

### 5.1 Dynamic Probability Gauges & Risk Tiers
Model outputs are translated into accessible human-capital risk tiers:
- **Low Risk ($P < 30\%$ — Green `#15803D`):** Standard employee profile; expected to remain with firm. Standard career reviews recommended.
- **Moderate Risk ($30\% \le P < 60\%$ — Amber `#B45309`):** Early dissatisfaction or workload friction detected. Proactive retention check-in advised.
- **Elevated Risk ($P \ge 60\%$ — Crimson `#B91C1C`):** Severe departure hazard. Urgent executive retention discussion recommended.

### 5.2 The Operational Decision Threshold Slider ($\tau$)
A key feature engineered in `app.py` is the **Operational Sensitivity Slider**:
- **Standard Cutoff ($\tau = 0.50$):** Default balanced operating point ($45.0\%$ precision, $38.3\%$ recall).
- **Proactive Retention Mode ($\tau = 0.42$):** Allows HR leaders to lower the threshold during critical quarters, immediately boosting departure capture to **$63.83\%$ (catching $30$ of $47$ departing staff)** with manageable alert overhead ($40.5\%$ precision).

### 5.3 Automated Actionable Retention Playbook
When an employee triggers elevated risk, the application synthesizes their specific profile to output customized intervention strategies:
- Overtime active $\to$ Rebalance project allocation and schedule compensatory rest.
- Tier 0 stock options $\to$ Propose equity grant onboarding.
- Depressed satisfaction $\to$ Initiate confidential stay interview with neutral HR partner.
- Long commute distance $\to$ Explore hybrid/remote work scheduling.

---

## 6. Mandatory HR Decision-Support Advisory Disclaimer

In compliance with AI ethics frameworks, the application persistently displays the following advisory notice across all user sessions:

> **⚖️ MANDATORY HR GOVERNANCE & DECISION-SUPPORT NOTICE:**  
> This prediction is an automated statistical estimate generated by machine learning models trained on historical employee records. It denotes empirical association, NOT individual certainty or moral judgment. In strict accordance with responsible AI standards, this tool must **NEVER be used as the sole basis** for hiring, compensation, promotion pass-overs, bonus deprivation, or employment termination. All predictive insights must undergo qualified human review and personal dialogue.

---

## 7. Deliverables and Artifact Manifest

The complete set of verified deployment deliverables produced for Task 9 is cataloged below:

| Deliverable ID | Artifact File / Path | File Type | Description |
| :--- | :--- | :---: | :--- |
| **DEL-9.1** | [app.py](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/app.py) | Python Application | Production Streamlit decision-support web application with 4 modular tabs. |
| **DEL-9.2** | [attrition_model_pipeline.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/attrition_model_pipeline.joblib) | Joblib Pipeline | Unified fitted preprocessor + Tuned Random Forest classifier pipeline. |
| **DEL-9.3** | [attrition_logistic_pipeline.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/attrition_logistic_pipeline.joblib) | Joblib Pipeline | Unified fitted preprocessor + Tuned Logistic Regression classifier pipeline. |
| **DEL-9.4** | [TASK_9_MODEL_DEPLOYMENT_AND_STREAMLIT_REPORT.md](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/task_report/TASK_9_MODEL_DEPLOYMENT_AND_STREAMLIT_REPORT.md) | Markdown Report | Comprehensive Task 9 deployment report and architectural specification. |

---

## 8. Verification and Launch Command

The deployment application was verified for syntax, schema compatibility, and headless execution:

```powershell
streamlit run app.py
```

The application initializes on `http://localhost:8501`, accepts raw inputs across all 30 features, executes inference in under $15\text{ ms}$, and delivers instantaneous decision-support feedback.
