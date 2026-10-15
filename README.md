# AttritionIQ — Intelligent Employee Attrition Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7.svg)](https://render.com/)
[![Status](https://img.shields.io/badge/Project%20Status-Completed%20%26%20Verified-brightgreen.svg)](#)

An enterprise-grade, end-to-end Machine Learning system and interactive web application designed to proactively predict voluntary employee attrition, uncover organizational turnover drivers, and recommend high-impact talent retention interventions.

---

## 🚀 Quick Start (Local Services)

1. **Activate your Python environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1   # Windows PowerShell
   # or source venv/bin/activate  # macOS / Linux
   ```

2. **Install all dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Cloud Backend (Terminal 1):**
   ```powershell
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   - Swagger Documentation: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

4. **Start the Streamlit Frontend (Terminal 2):**
   ```powershell
   $env:API_URL = "http://localhost:8000"
   streamlit run app.py
   ```
   Open **`http://localhost:8501`** in your browser.

> ☁️ **Cloud Deployment (Render + Streamlit Cloud):**  
> - **Backend on Render:** Deploy using `render.yaml` or Web Service with start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.  
> - **Frontend on Streamlit Community Cloud:** Configure `API_URL = "https://your-backend.onrender.com"` in Streamlit Secrets.  
> 📖 For full setup and testing details, read [INSTRUCTIONS_TO_RUN.md](INSTRUCTIONS_TO_RUN.md).

---

## 📌 Project Architecture & Key Results

| Metric / Dimension | Decision Tree (Tuned) | Random Forest (Tuned Champion) | Logistic Regression (Linear Benchmark) |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | $72.79\%$ | **$82.65\%$** | **$83.67\%$** |
| **Test Recall (Attrition=Yes)** | $51.06\%$ | **$38.30\%$** *(or $63.83\%$ at $\tau=0.42$)* | **$53.19\%$** |
| **Test Precision** | $29.63\%$ | **$45.00\%$** | **$49.02\%$** |
| **Test F1-Score** | $0.3750$ | **$0.4138$** *(or $0.4959$ at $\tau=0.42$)* | **$0.5102$** |
| **Test ROC-AUC** | $0.7032$ | **$0.7638$** | **$0.7994$** |
| **False Positives (Held-out Test)** | $57$ false alarms | **$22$ false alarms** (Lowest FP) | $26$ false alarms |
| **Production Role** | Benchmarking | **Champion Non-Linear Classifier** | **Calibrated Linear Benchmark** |

### Top Empirical Turnover Drivers
1. **Compulsory Overtime (`OverTime_Yes`):** Single strongest feature across Gini impurity ($10.5\%$) and out-of-sample Permutation Importance ($\Delta\text{ROC-AUC} = +0.0270$). Employees working compulsory overtime depart at $30.5\%$ vs $10.4\%$ without overtime ($3.0\times$ turnover hazard).
2. **Equity Retention Incentives (`StockOptionLevel`):** Zero stock options correlates with a $24.4\%$ attrition rate vs $7.6\%$ with level 1–2 options.
3. **Monthly Compensation (`MonthlyIncome`):** Strongest continuous signal; turnover concentrates heavily in the junior $<\$3,000/\text{mo}$ compensation band.
4. **Marital Status & Geographic Mobility (`MaritalStatus_Single`):** Single employees have a $25.5\%$ turnover rate vs $12.5\%$ for married employees.
5. **Career Velocity & Stagnation:** Imbalances between `YearsInCurrentRole` and `YearsSinceLastPromotion`.

---

## 📁 Repository Directory Structure

- **`app.py`**: Production interactive Streamlit application with custom styling, preset archetypes, dynamic threshold slider, risk factor attribution, and retention playbooks.
- **`api/main.py`**: FastAPI cloud REST API service with `/health`, `/predict`, and `/docs` interactive Swagger endpoints.
- **`render.yaml`**: Infrastructure-as-code deployment blueprint for automated Render Web Service provisioning.
- **`tests/test_api.py`**: Automated unit and consistency test suite validating API endpoints and pipeline output alignment.
- **`src/`**: Modular, reproducible Python source scripts:
  - `data_loader.py` — Task 1: Tab-separated ingestion and schema audit.
  - `data_cleaner.py` — Task 2: Null/duplicate verification and constant pruning.
  - `eda_analysis.py` — Task 3: Bivariate analysis and visual figure suite generation.
  - `preprocessor.py` — Task 4: ColumnTransformer pipeline, scaling, OHE, stratified split.
  - `model_trainer.py` — Tasks 5 & 6: Baseline model training and evaluation.
  - `task7_task8_runner.py` — Tasks 7 & 8: Feature importance (MDI/PFI), Stratified 5-Fold grid tuning for RF & LR, post-tuning classification reports, and production pipeline serialization.
  - `build_notebook.py` — Assembles the master Jupyter Notebook.
- **`models/`**: Joblib serialized models, including the unified production pipeline `attrition_model_pipeline.joblib`.
- **`reports/figures/`**: 13 high-resolution visualization charts generated during analysis.
- **`notebooks/AttritionIQ_Workflow.ipynb`**: Master self-contained research notebook.
- **`task_report/`**: 10 comprehensive task reports covering Tasks 1 through 10.
- **`INSTRUCTIONS_TO_RUN.md`**: Complete operational instructions to run every script and tool.
- **`PROJECT_TASK_REPORT.md`**: Master project specification, compliance matrix, and methodology blueprint.

---

## 📚 Master Task Reports

All 10 project tasks have standalone, deep-dive analytical reports:

- [Task 1: Data Loading & Schema Understanding](task_report/TASK_1_DATA_LOADING_AND_UNDERSTANDING_REPORT.md)
- [Task 2: Data Cleaning & Quality Assurance](task_report/TASK_2_DATA_CLEANING_AND_QUALITY_VERIFICATION_REPORT.md)
- [Task 3: Exploratory Data Analysis & Visualizations](task_report/TASK_3_EXPLORATORY_DATA_ANALYSIS_REPORT.md)
- [Task 4: Feature Engineering, Preprocessing & Stratified Splitting](task_report/TASK_4_FEATURE_ENGINEERING_AND_PREPROCESSING_REPORT.md)
- [Task 5: Building Machine Learning Models](task_report/TASK_5_BUILD_MACHINE_LEARNING_MODELS_REPORT.md)
- [Task 6: Model Evaluation, Overfitting Diagnostics & Comparison](task_report/TASK_6_MODEL_EVALUATION_AND_COMPARISON_REPORT.md)
- [Task 7: Feature Importance, Permutation Importance & Interpretability](task_report/TASK_7_FEATURE_IMPORTANCE_AND_INTERPRETABILITY_REPORT.md)
- [Task 8: Hyperparameter Optimization, Tuning & Post-Tuning Classification Reports](task_report/TASK_8_HYPERPARAMETER_OPTIMIZATION_AND_TUNING_REPORT.md)
- [Task 9: Model Deployment & Streamlit Web Application](task_report/TASK_9_MODEL_DEPLOYMENT_AND_STREAMLIT_REPORT.md)
- [Task 10: Ethics, Fairness, Data Privacy & Systemic Limitations](task_report/TASK_10_ETHICS_FAIRNESS_AND_LIMITATIONS_REPORT.md)

---

## ⚖️ Ethical Governance & Human-in-the-Loop Protocol

1. **Anti-Sole-Basis Mandate:** Predictive scores generated by AttritionIQ must never serve as the sole justification for adverse employment decisions (termination, withholding compensation, or bypassing promotions).
2. **Supportive-Only Retention Policy:** High-risk indications must trigger supportive retention dialogues (e.g., workload rebalancing, career pathing, compensation benchmarking).
3. **Four-Fifths Fairness Audits:** Continuous demographic parity audits across age, gender, and marital status to avoid systemic disparate impact.
4. **GDPR / Privacy Compliance:** Role-based access control (RBAC), end-to-end encryption, and full right-to-explanation compliance for affected employees.

---
*AttritionIQ — Built with Scikit-Learn, Streamlit, and Python.*
