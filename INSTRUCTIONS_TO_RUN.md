# AttritionIQ — Complete Instructions to Run the Project

Welcome to **AttritionIQ: Intelligent Employee Attrition Prediction System**. This guide provides comprehensive, step-by-step instructions to set up the environment, run the complete data and modeling pipeline from scratch, launch the interactive Streamlit decision-support application, and access all artifacts and reports.

---

## 1. System Requirements & Prerequisites

- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** Python 3.10, 3.11, or 3.12 (Python 3.12 tested & verified)
- **Package Manager:** `pip` (included with standard Python installations)
- **Web Browser:** Modern browser (Chrome, Edge, Firefox, or Safari) for the Streamlit web dashboard

---

## 2. Environment Setup & Installation

Open your terminal or PowerShell, navigate to the project directory, and follow these steps:

### Step 2.1: Clone or Open the Project Directory
```powershell
cd "c:\Users\mihir\Desktop\AttritionIQ — Intelligent Employee Attrition Prediction System"
```
*(On macOS/Linux, adjust the path accordingly).*

### Step 2.2: Create and Activate a Python Virtual Environment

**On Windows (PowerShell):**
```powershell
# Create virtual environment named 'venv'
python -m venv venv

# If PowerShell script execution policy is restricted, enable it for this process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt `cmd`):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

*(Once activated, your terminal prompt will display `(venv)`).*

### Step 2.3: Install Required Dependencies
Install all required libraries using the pinned `requirements.txt`:
```powershell
pip install -r requirements.txt
```

Verify that all dependencies installed cleanly:
```powershell
python -c "import pandas, sklearn, streamlit, joblib, matplotlib, seaborn; print('All core libraries successfully imported!')"
```

---

## 3. Project Directory Architecture

```
AttritionIQ — Intelligent Employee Attrition Prediction System/
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Raw IBM HR Employee Dataset (Tab-separated)
├── Machine Learning Assignment.docx       # Assignment Specification Document
├── app.py                                 # Production Streamlit Web Application
├── requirements.txt                       # Pinned Python package dependencies
├── INSTRUCTIONS_TO_RUN.md                 # Complete Execution Guide (This File)
├── README.md                              # Repository Overview & Architecture Guide
├── PROJECT_TASK_REPORT.md                 # Master Implementation & Compliance Specification
│
├── data/
│   ├── raw/
│   │   └── employee_attrition.csv         # Standardized comma-separated raw copy
│   └── processed/
│       ├── employee_attrition_clean.csv   # Cleaned dataset (31 columns, zero nulls/constants)
│       ├── X_train.csv                    # Training split features (1,176 records)
│       ├── X_test.csv                     # Held-out test split features (294 records)
│       ├── y_train.csv                    # Training target vector
│       ├── y_test.csv                     # Held-out test target vector
│       └── fitted_preprocessor.joblib     # Fitted ColumnTransformer pipeline component
│
├── models/
│   ├── attrition_model_pipeline.joblib    # PRODUCTION UNIFIED PIPELINE (Preprocessor + Tuned RF)
│   ├── attrition_logistic_pipeline.joblib # BENCHMARK LINEAR PIPELINE (Preprocessor + Tuned LR)
│   ├── decision_tree_baseline.joblib      # Baseline Decision Tree model
│   ├── random_forest_baseline.joblib      # Baseline Random Forest model
│   ├── logistic_regression_baseline.joblib# Baseline Logistic Regression model
│   ├── decision_tree_tuned.joblib         # Tuned Decision Tree model
│   ├── random_forest_tuned.joblib         # Tuned Random Forest model (Champion Ensemble)
│   └── logistic_regression_tuned.joblib   # Tuned Logistic Regression model (Linear Champion)
│
├── reports/
│   └── figures/                           # Generated High-Resolution Visualization Suite
│       ├── 01_attrition_distribution.png  # Enterprise attrition distribution
│       ├── 02_overtime_attrition.png      # Overtime impact comparison
│       ├── 03_department_attrition.png    # Departmental turnover analysis
│       ├── 04_job_role_attrition.png      # Job role turnover hazard chart
│       ├── 05_monthly_income_box.png      # Compensation distributions
│       ├── 06_age_distribution.png       # Age demographic turnover density
│       ├── 07_total_working_years.png     # Experience tenure curves
│       ├── 08_correlation_heatmap.png     # Multivariate Pearson correlation heatmap
│       ├── 09_model_roc_curves.png        # Baseline model ROC curve comparison
│       ├── 10_feature_importance_rf.png   # MDI Top 10 Random Forest importance
│       ├── 11_permutation_importance.png  # Out-of-sample test permutation importance
│       ├── 12_tuned_models_roc_curves.png # Post-tuning ROC curves
│       └── 13_tuned_confusion_matrices.png# Post-tuning test confusion matrices
│
├── src/                                   # Production-Grade Modular Source Code
│   ├── data_loader.py                     # Task 1: Ingestion, schema audit & raw export
│   ├── data_cleaner.py                    # Task 2: Null/duplicate check & constant pruning
│   ├── eda_analysis.py                    # Task 3: Bivariate analysis & visual suite generation
│   ├── preprocessor.py                    # Task 4: Pipeline construction, OHE & train/test split
│   ├── model_trainer.py                   # Tasks 5 & 6: Training baselines & metric evaluations
│   ├── task7_task8_runner.py              # Tasks 7 & 8: MDI, permutation importance & RF tuning
│   ├── tune_all_models.py                 # Task 8: Stratified 5-Fold grid tuning for DT, RF, LR
│   └── build_notebook.py                  # Generates AttritionIQ_Workflow.ipynb automatically
│
├── notebooks/
│   └── AttritionIQ_Workflow.ipynb         # Full reproducible end-to-end Jupyter Notebook
│
└── task_report/                           # Standalone Deep-Dive Reports for All 10 Tasks
    ├── TASK_1_DATA_LOADING_AND_UNDERSTANDING_REPORT.md
    ├── TASK_2_DATA_CLEANING_AND_QUALITY_VERIFICATION_REPORT.md
    ├── TASK_3_EXPLORATORY_DATA_ANALYSIS_REPORT.md
    ├── TASK_4_FEATURE_ENGINEERING_AND_PREPROCESSING_REPORT.md
    ├── TASK_5_BUILD_MACHINE_LEARNING_MODELS_REPORT.md
    ├── TASK_6_MODEL_EVALUATION_AND_COMPARISON_REPORT.md
    ├── TASK_7_FEATURE_IMPORTANCE_AND_INTERPRETABILITY_REPORT.md
    ├── TASK_8_HYPERPARAMETER_OPTIMIZATION_AND_TUNING_REPORT.md
    ├── TASK_9_MODEL_DEPLOYMENT_AND_STREAMLIT_REPORT.md
    └── TASK_10_ETHICS_FAIRNESS_AND_LIMITATIONS_REPORT.md
```

---

## 4. End-to-End Pipeline Execution (From Scratch)

You can execute the entire pipeline sequentially from the command line. Each script executes deterministically with set random seeds (`random_state=42`) to guarantee identical, reproducible outputs.

### Step 4.1: Task 1 — Data Ingestion & Understanding
Ingests `WA_Fn-UseC_-HR-Employee-Attrition.csv` (using `sep='\t'`), audits data types, and exports standard raw CSV:
```powershell
python src/data_loader.py
```
*Output: Verified `(1470, 35)` DataFrame.*

### Step 4.2: Task 2 — Data Cleaning & Quality Assurance
Audits nulls (0 found), duplicates (0 found), and removes zero-variance constants (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`):
```powershell
python src/data_cleaner.py
```
*Output: `data/processed/employee_attrition_clean.csv` `(1470, 31)`.*

### Step 4.3: Task 3 — Exploratory Data Analysis (EDA) & Visualizations
Computes enterprise turnover statistics and generates 8 publication-ready figures in `reports/figures/`:
```powershell
python src/eda_analysis.py
```
*Output: Figures `01` through `08` saved in `reports/figures/`.*

### Step 4.4: Task 4 — Preprocessing, Encoding & Stratified Split
Constructs the `ColumnTransformer`, standardizes numeric attributes, one-hot encodes categoricals, and creates stratified 80:20 train/test splits:
```powershell
python src/preprocessor.py
```
*Output: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, and `fitted_preprocessor.joblib`.*

### Step 4.5: Tasks 5 & 6 — Baseline Machine Learning Training & Evaluation
Trains baseline Decision Tree, Random Forest, and Logistic Regression models and computes train/test evaluation metrics:
```powershell
python src/model_trainer.py
```
*Output: Baseline models saved in `models/`, Figure `09_model_roc_curves.png`.*

### Step 4.6: Tasks 7 & 8 — Feature Importance & Comprehensive Hyperparameter Tuning
Performs MDI and test permutation importance analysis, executes Stratified 5-Fold Grid Search CV for Decision Tree, Random Forest, and Logistic Regression, prints comprehensive classification reports, and bundles unified production pipelines:
```powershell
python src/tune_all_models.py
```
*Output: Tuned models, Figures `10`, `11`, `12`, `13`, and unified pipeline `models/attrition_model_pipeline.joblib`.*

### Step 4.7: Build the Master Jupyter Notebook
Assembles the complete end-to-end reproducible Jupyter Notebook from the verified modules:
```powershell
python src/build_notebook.py
```
*Output: `notebooks/AttritionIQ_Workflow.ipynb`.*

---

## 5. Launching the Interactive Streamlit Web Application (Task 9)

The Streamlit web application delivers an intuitive, interactive decision-support interface for HR business partners, talent acquisition leads, and people analytics teams.

### Step 5.1: Run the Streamlit Command
```powershell
streamlit run app.py
```

### Step 5.2: Accessing the Application
Once the server starts, Streamlit will display the local URL in your terminal:
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open **`http://localhost:8501`** in your browser.

*(To stop the Streamlit server at any time, press `Ctrl + C` in the terminal).*

---

## 6. How to Use the Streamlit Application (Presentation-Optimized UI)

The user interface is designed specifically for clean, focused presentations and intuitive employee risk evaluation:

### 1. Final Production Model (Single Champion Architecture)
- The application automatically runs on our top-performing champion architecture: **Tuned Random Forest Pipeline** (`models/attrition_model_pipeline.joblib`).
- Validated performance: **82.65% Test Accuracy**, **0.7638 ROC-AUC**, and the lowest false-alarm rate among non-linear models.

### 2. Streamlined Employee Profile Inputs
- **Primary Business Drivers (Clean 2-Column Grid):**
  - *Left Column (Role & Compensation):* Age, Department, Job Role, Monthly Income, Overtime (`Yes`/`No`), Stock Option Level.
  - *Right Column (Workload, Tenure & Sentiment):* Job Satisfaction, Work-Life Balance, Tenure at Company, Commute Distance, Marital Status.
- **Additional Parameters (Optional Expander):** Secondary features (daily/hourly rates, business travel frequency, years with manager) are neatly tucked inside an expandable drawer with smart defaults, keeping the main view clean and uncluttered.

### 3. Real-Time Decision-Support Results Card
- **Prominent Risk Status:** Displays a clean color-coded status banner: **🔴 ELEVATED ATTRITION RISK** vs **🟢 LOW ATTRITION RISK (RETAINED)**.
- **Propensity Percentage & Progress Bar:** Shows the exact probability score ($0.0\%$ to $100.0\%$).
- **Key Risk Drivers Detected:** Highlights the specific factors triggering elevated risk (e.g., compulsory overtime, zero equity, salary below median).
- **Actionable Retention Playbook:** Provides concrete HR action steps (workload restructuring, equity grant review, stay interview).
- **Decision Threshold (τ) Slider:** In the sidebar, adjust sensitivity between $0.30$ and $0.70$ (Default: $0.50$; High-Sensitivity Mode: $0.42$).
- **Responsible AI Disclaimer:** Enforces human oversight and adherence to the anti-sole-basis rule.

---

## 7. Reviewing the Task Reports

Every phase of the project has a dedicated, peer-reviewed Markdown report located in the `task_report/` directory:

| Task ID | Report Filename | Focus & Key Deliverable |
| :--- | :--- | :--- |
| **Task 1** | `task_report/TASK_1_DATA_LOADING_AND_UNDERSTANDING_REPORT.md` | Data ingestion (`sep='\t'`), shape audit `(1470, 35)`, data dictionary. |
| **Task 2** | `task_report/TASK_2_DATA_CLEANING_AND_QUALITY_VERIFICATION_REPORT.md` | Zero nulls/duplicates, pruning 4 zero-variance columns `(1470, 31)`. |
| **Task 3** | `task_report/TASK_3_EXPLORATORY_DATA_ANALYSIS_REPORT.md` | 16.12% base attrition, 8 bivariate analyses, 5 core business findings. |
| **Task 4** | `task_report/TASK_4_FEATURE_ENGINEERING_AND_PREPROCESSING_REPORT.md` | OHE `drop='first'`, StandardScaler, stratified 80:20 split pipeline. |
| **Task 5** | `task_report/TASK_5_BUILD_MACHINE_LEARNING_MODELS_REPORT.md` | Baseline Decision Tree, Random Forest, and Logistic Regression models. |
| **Task 6** | `task_report/TASK_6_MODEL_EVALUATION_AND_COMPARISON_REPORT.md` | Confusion matrices, overfitting diagnosis, ROC-AUC curves comparison. |
| **Task 7** | `task_report/TASK_7_FEATURE_IMPORTANCE_AND_INTERPRETABILITY_REPORT.md` | MDI vs. Permutation importance, causality vs. correlation analysis. |
| **Task 8** | `task_report/TASK_8_HYPERPARAMETER_OPTIMIZATION_AND_TUNING_REPORT.md` | Stratified 5-Fold Grid Search CV, post-tuning classification reports. |
| **Task 9** | `task_report/TASK_9_MODEL_DEPLOYMENT_AND_STREAMLIT_REPORT.md` | Unified pipeline packaging, Joblib serialization, interactive UI design. |
| **Task 10**| `task_report/TASK_10_ETHICS_FAIRNESS_AND_LIMITATIONS_REPORT.md` | Demographic bias, human harm audit, GDPR privacy, AI governance. |

---

## 8. Troubleshooting Common Questions

1. **Port 8501 is already in use:**
   ```powershell
   streamlit run app.py --server.port 8502
   ```
2. **Missing dependency error:**
   ```powershell
   pip install -r requirements.txt --upgrade
   ```
3. **ExecutionPolicy error on Windows PowerShell:**
   Run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
   Then re-run `.\venv\Scripts\Activate.ps1`.
4. **Where is the final production pipeline stored?**
   The unified pipeline is located at:
   `models/attrition_model_pipeline.joblib`
   It automatically handles raw 30-attribute feature vectors, applies categorical one-hot encoding and standard scaling, and outputs both class predictions and calibrated probability scores.

---
*AttritionIQ — Built with Scikit-Learn, Streamlit, and Python.*
